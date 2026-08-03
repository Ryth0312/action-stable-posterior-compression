"""Re-derive the R2 certificate from the committed paired-coupling run (torch-free, no re-simulation).

Two defects in the committed ``r2_paired_coupling_*.json`` are repaired here without touching the expensive
solver draws. (1) The original ``classification_safe`` test was ``bound < min(pX, 1-pX)``, which can never pass
at ``pX = 1.0``. (2) More seriously, the boundary-tube term was evaluated under the UNTRUNCATED Gaussian while
the paired draws are clipped to the prior's physical support (22-87% of draws here), so the two terms of the
coupling bound referred to different laws; on mAb A that understated the tube by three orders of magnitude,
anti-conservatively.

The tube is a function of the posterior covariance, the decision Jacobian and the physical bounds only -- no
solver -- so it is recomputed here under Y's own (clipped) law by direct sampling, and both terms are bounded by
exact one-sided Clopper-Pearson rates, union-bounded over both terms and the whole pre-fixed grid. The paired
tail rate is taken at the STORED t* from the committed run; re-optimising the grid point needs a re-run of
``step4_r2_paired_coupling.py``, whose patched version does all of this directly.

Usage:  python scripts/step4b_r2_reverdict.py [--in results/bayes/r2_paired_coupling_*.json]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from scipy.stats import multivariate_normal as _mvn

P_HI, P_LO = 0.95, 0.05
_RAW = ("P_X_meet_nonlinear", "P_Y_meet_linear", "certified_bound", "t_star")
# prior physical support, mirrored from cex_model.bayes.prior so this stays torch-free
_BOUNDS = {"keq": (1e-7, 1e-1), "kkin": (1e-11, 1e-3), "nu": (1.0, 18.0), "sigma": (1.0, 100.0)}
_ROWS, _LOG = ("keq", "kkin", "nu", "sigma"), {"keq", "kkin"}


def _u_bounds(n):
    lo, hi = [], []
    for r in _ROWS:
        a, b = _BOUNDS[r]
        if r in _LOG:
            a, b = np.log10(a), np.log10(b)
        lo += [a] * n
        hi += [b] * n
    return np.asarray(lo, float), np.asarray(hi, float)


def _cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def recompute(rec: dict, in_dir: Path, spec, m_grid: int = 50, delta: float = 0.05,
              n_big: int = 4_000_000, n_protein: int = 5, seed: int = 7) -> dict:
    """Recompute the certificate at the STORED t*, under Y's true (clipped) law, with exact binomial rates.

    The tube term needs no solver -- only the posterior covariance, the decision Jacobian and the physical
    bounds -- so it is reproducible offline. The paired tail is taken at the stored t* from the committed run;
    re-optimising over the whole grid requires re-running step4_r2_paired_coupling.py.
    """
    missing = [k for k in _RAW if k not in rec]
    if missing:
        raise KeyError(f"{rec.get('product', '?')}: missing raw fields {missing}; re-run step4_r2_paired_coupling.py")
    prod = rec["product"]
    dj = json.loads((in_dir / f"{prod}_decision.json").read_text())
    G = np.atleast_2d(np.asarray(dj["decision_jacobian"], float))
    tol = np.asarray(dj["tol"], float)
    g_map = np.array([dj["decision"]["g_map"]["pool_purity"], dj["decision"]["g_map"]["pool_yield"]])
    z = np.load(in_dir / f"{prod}_posterior.npz", allow_pickle=True)

    # Familywise budget: one CP test per grid point for each of the two coupling terms, plus the two tails of
    # the support-gap bridge. Budgeting delta/(2m) covers only the first 2m and leaves the guarantee at
    # 1 - delta*(2m+2)/(2m); count the bridge explicitly so the stated level is the level delivered.
    n_tests = 2 * m_grid + 2
    alpha = delta / n_tests
    lo, hi = _u_bounds(n_protein)
    rng = np.random.default_rng(seed)
    us = rng.multivariate_normal(np.asarray(z["mean"], float), np.asarray(z["cov"], float), size=n_big)
    usc = np.clip(us, lo, hi)
    clip_frac = float(np.mean(np.any(usc != us, axis=1)))
    Yw = (g_map + (usc - np.asarray(z["u_map"], float)) @ G.T) / tol
    sb = np.asarray(spec, float) / tol
    marg = Yw - sb
    inside = np.all(marg >= 0, axis=1)
    d = np.where(inside, marg.min(axis=1), np.clip(-marg, 0, None).max(axis=1))

    t = float(rec["t_star"])
    k1 = int(np.sum(d <= t))
    t1_hat, t1_up = k1 / n_big, _cp_upper(k1, n_big, alpha)
    N = int(rec["n_draws"])
    if "term2_tail_hat" in rec:                                          # already re-verdicted: rerun is idempotent
        t2_hat = float(rec["term2_tail_hat"])
    elif "term2_upper" in rec and "union_slack" in rec:
        t2_hat = float(rec["term2_upper"]) - float(rec["union_slack"])   # strip the old Hoeffding slack
    else:
        raise KeyError(f"{prod}: no paired-tail rate; re-run step4_r2_paired_coupling.py")
    t2_up = _cp_upper(int(round(t2_hat * N)), N, alpha)

    # Production evaluates P(meet) under the UNTRUNCATED Gaussian, while both coupling terms above are taken
    # under Y's clipped law. Bridge the two laws with the measured support gap so the certificate applies to
    # the probability the pipeline actually reports.
    C = G @ np.asarray(z["cov"], float) @ G.T
    p_lin_gauss = float(_mvn(mean=-g_map / tol, cov=C / tol[:, None] / tol[None, :] + 1e-15 * np.eye(len(tol)),
                             allow_singular=True).cdf(-sb))
    k_meet = int(np.sum(inside))
    gap_up = max(abs(_cp_upper(k_meet, n_big, alpha) - p_lin_gauss),
                 abs((1.0 - _cp_upper(n_big - k_meet, n_big, alpha)) - p_lin_gauss))

    b = t1_up + t2_up + gap_up
    lo_i, hi_i = max(0.0, p_lin_gauss - b), min(1.0, p_lin_gauss + b)
    if p_lin_gauss >= P_HI:
        state, safe = "cond-decisive-meet", bool(lo_i >= P_HI)
    elif p_lin_gauss <= P_LO:
        state, safe = "cond-decisive-miss", bool(hi_i <= P_LO)
    else:
        state, safe = "already-ambiguous", False
    floor = t1_hat + t2_hat
    rec.update({
        "certified_bound": b, "certified_interval": [lo_i, hi_i],
        "P_lin_untruncated": p_lin_gauss, "P_lin_clipped": float(k_meet / n_big),
        "support_gap_upper": gap_up,
        "term1_tube_hat": t1_hat, "term1_tube_upper": t1_up,
        "term2_tail_hat": t2_hat, "term2_tail_upper": t2_up,
        "large_sample_floor": floor, "finite_sample_share": (b - floor) / b if b > 0 else 0.0,
        "clip_fraction_tube": clip_frac, "alpha_per_test": alpha, "n_tests_familywise": n_tests,
        "familywise_level": 1.0 - delta,
        "classification_state": state, "classification_safe": safe,
        "verdict_note": ("certificate recomputed at the stored t* by step4b_r2_reverdict.py: tube under Y's "
                         "true (clipped) law, exact Clopper-Pearson rates on both terms, plus the measured "
                         "support-gap bridge to the untruncated P_lin the pipeline reports; grid "
                         "re-optimisation requires re-running step4_r2_paired_coupling.py"),
    })
    for stale in ("term1_tube", "term2_upper", "union_slack", "empirical_tail_excess", "limited_by"):
        rec.pop(stale, None)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/bayes/r2_paired_coupling_*.json")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--spec", type=float, nargs=2, default=[0.70, 0.50])
    args = ap.parse_args()
    paths = sorted(glob.glob(args.inp))
    if not paths:
        raise SystemExit(f"no file matched {args.inp}")
    for path in paths:
        rows = [recompute(r, Path(args.in_dir), args.spec) for r in json.loads(Path(path).read_text())]
        Path(path).write_text(json.dumps(rows, indent=2))
        print(f"{path}:")
        for r in rows:
            iv = r["certified_interval"]
            print(f"  {r['product']:14} P_lin={r['P_Y_meet_linear']:.3f} b={r['certified_bound']:.4f} "
                  f"interval=[{iv[0]:.3f},{iv[1]:.3f}] {r['classification_state']:19} "
                  f"safe={str(r['classification_safe']):5} floor={r['large_sample_floor']:.3f} "
                  f"clip={r['clip_fraction_tube']:.3f}")


if __name__ == "__main__":
    main()
