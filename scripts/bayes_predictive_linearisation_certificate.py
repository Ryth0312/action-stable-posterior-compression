"""Carry the candidate-level linearisation certificate from the conditional law to the DEPLOYED law.

``step4_r2_paired_coupling.py`` certifies |P_nonlin - P_lin| under theta|M.  The recommendation is taken
under the predictive mixture of the article's (2), which adds a hierarchy draw, a product bias, a decision
discrepancy and assay noise on top of the mechanistic map.  Those enter ADDITIVELY and are shared by the
nonlinear and the linearised arm, so with U the predictive-layer draw the paired difference is invariant,

    (X + U) - (Y + U) = X - Y,

and the second coupling term P{|X-Y|_inf > t} -- the only term that needs the solver -- carries over from the
stored conditional artifact unchanged.  What has to be recomputed is the boundary tube under the predictive
law of Y + U and the support-gap bridge, both pure linear algebra.  The certified interval is then centred on
the deployed predictive P(meet), the quantity Section 6 reads, and classified against tau_dec.

Run (needs the solver only for the Jacobian at each condition, ~17 s each):
  python scripts/bayes_predictive_linearisation_certificate.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, multivariate_normal

import cex_model.app_support as A
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_jacobian
from cex_model.bayes.posterior import Posterior
from cex_model.bayes.prior import solver_guard_bounds

_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
SPEC = np.array([0.70, 0.50])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
TAU_HI, TAU_LO = 0.95, 0.05
N_BIG = 4_000_000
DELTA, M_GRID = 0.05, 50


def _cp_upper(k, n, alpha):
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def _dist_to_orthant_boundary(Yw, sw):
    margin = Yw - sw
    inside = np.all(margin >= 0, axis=1)
    return np.where(inside, margin.min(axis=1), np.clip(-margin, 0, None).max(axis=1))


def _deployed_p_meet(g_map, C, bias, sd_draws):
    """P(meet) under the predictive mixture -- the same closed form the deployed scan uses."""
    return float(np.mean([
        multivariate_normal(mean=-(g_map + bias[k]), cov=C + sd_draws[k] + C_MEAS + 1e-12 * np.eye(2),
                            allow_singular=True).cdf(-SPEC)
        for k in range(len(sd_draws))]))


def _predictive_tube(post, G, g_map, tol, t_grid, n_protein, bias, sd_draws, alpha, rng):
    """P{d_inf(Y + U, boundary) <= t} under the predictive law of the LINEARISED arm."""
    lo, hi = solver_guard_bounds(n_protein)
    L = np.linalg.cholesky(sd_draws + C_MEAS + 1e-14 * np.eye(2))          # (K, 2, 2)
    sw = SPEC / tol
    n_clip = n_meet = 0
    counts = np.zeros(len(t_grid))
    for start in range(0, N_BIG, 500_000):                                 # chunked: 4e6 x 20 will not fit
        m = min(500_000, N_BIG - start)
        us = rng.multivariate_normal(np.asarray(post.mean, float), np.asarray(post.cov, float), size=m)
        usc = np.clip(us, lo, hi)
        n_clip += int(np.sum(np.any(usc != us, axis=1)))
        Y = g_map + (usc - np.asarray(post.u_map, float)) @ G.T
        k = rng.integers(0, len(sd_draws), m)
        Yw = (Y + bias[k] + np.einsum("nij,nj->ni", L[k], rng.standard_normal((m, 2)))) / tol
        d = _dist_to_orthant_boundary(Yw, sw)
        counts += np.array([float(np.sum(d <= t)) for t in t_grid])
        n_meet += int(np.sum(np.all(Yw >= sw, axis=1)))
    hat = counts / N_BIG
    up = np.array([_cp_upper(int(c), N_BIG, alpha) for c in counts])
    return hat, up, float(n_clip / N_BIG), n_meet


def certify(cert, in_dir, hier, rng):
    product, op = cert["product"], np.asarray(cert["op"], float)
    post = Posterior.load(Path(in_dir) / f"{product}_{cert['posterior']}.npz")
    tol = np.asarray(json.loads((Path(in_dir) / f"{product}_decision.json").read_text())
                     .get("tol", [0.02, 0.05]), float)
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)

    G, g_map, _ = decision_jacobian(bundle, list(map(float, op)), post.u_map, n_steps=300, return_extra=True)
    G = np.atleast_2d(G)
    C = G @ post.cov @ G.T
    bias, sd_draws = hier[f"b_{product}"], hier["Sd_draws"]

    t_grid = np.asarray(cert["t_grid"], float)
    alpha = DELTA / (2 * M_GRID + 2)
    N = cert["n_draws"]
    # term 2 is invariant under the shared additive predictive draw: reuse the stored solver-based rate
    t2h = np.asarray(cert["term2_hat_curve"], float)
    t2u = np.array([_cp_upper(int(round(h * N)), N, alpha) for h in t2h])
    t1h, t1u, clip_frac, k_meet = _predictive_tube(post, G, g_map, tol, t_grid,
                                                   bundle.components.n_protein, bias, sd_draws, alpha, rng)
    p_dep = _deployed_p_meet(g_map, C, bias, sd_draws)
    gap = max(abs(_cp_upper(k_meet, N_BIG, alpha) - p_dep),
              abs((1.0 - _cp_upper(N_BIG - k_meet, N_BIG, alpha)) - p_dep))

    total = t1u + t2u + gap
    j = int(np.argmin(total))
    b = float(total[j])
    floor = float(t1h[j] + t2h[j])
    lo_i, hi_i = max(0.0, p_dep - b), min(1.0, p_dep + b)
    if p_dep >= TAU_HI:
        safe, state = bool(lo_i >= TAU_HI), "deployed-decisive-meet"
    elif p_dep <= TAU_LO:
        safe, state = bool(hi_i <= TAU_LO), "deployed-decisive-miss"
    else:
        safe, state = False, "deployed-ambiguous"
    return {
        "product": product, "op": [float(v) for v in op], "posterior": cert["posterior"],
        "law": "predictive", "n_draws": N,
        "P_meet_deployed": p_dep, "P_meet_conditional": cert["P_lin_untruncated"],
        "certified_bound": b, "t_star": float(t_grid[j]),
        "term1_tube_hat": float(t1h[j]), "term1_tube_upper": float(t1u[j]),
        "term2_tail_hat": float(t2h[j]), "term2_tail_upper": float(t2u[j]),
        "support_gap_upper": float(gap), "large_sample_floor": floor,
        "finite_sample_share": float((b - floor) / b) if b > 0 else 0.0,
        "clip_fraction_tube": clip_frac, "alpha_per_test": alpha,
        "bound_conditional": cert["certified_bound"], "safe_conditional": cert["classification_safe"],
        "certified_interval": [lo_i, hi_i], "classification_state": state, "classification_safe": safe,
        "non_vacuous": bool(b < 1.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/bayes/r2_paired_coupling_*correlated*_op*.json")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--hier-draws", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/bayes/predictive_linearisation_certificate.json")
    args = ap.parse_args()

    hier = np.load(args.hier_draws)
    rng = np.random.default_rng(args.seed)
    rows = []
    for f in sorted(glob.glob(args.glob)):
        r = certify(json.load(open(f))[0], args.in_dir, hier, rng)
        r["source"] = Path(f).name
        rows.append(r)
        print(f"{r['product']:14s} {r['op'][0]:8.4f}  P_dep={r['P_meet_deployed']:.4f} "
              f"b={r['certified_bound']:.4f} [{r['certified_interval'][0]:.3f},"
              f"{r['certified_interval'][1]:.3f}] {r['classification_state']} safe={r['classification_safe']}")
    Path(args.out).write_text(json.dumps(
        {"hier_draws": args.hier_draws, "n_hier_draws": int(len(hier["Sd_draws"])),
         "n_tube_draws": N_BIG, "delta": DELTA, "rows": rows}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
