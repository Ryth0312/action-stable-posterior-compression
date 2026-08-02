#!/usr/bin/env python3
"""Direct empirical-convolution meet-probability vs the moment-matched Gaussian (reviewer item 2).

For the one product whose decision linearization trips the validity flag (mAb C / HLXSYN, rel-Frob
0.70), the deployed predictive P(meet) is computed by moment-matching: the nonlinear Monte-Carlo
pushforward COVARIANCE is taken and P(meet) is read from a Gaussian with that covariance. Matching
the first two moments need not reproduce a JOINT threshold probability when the pushforward is
non-Gaussian. This script settles the question by computing P(meet) two ways at both the historical
operating point and the deployed lower-loading candidate (~19.7 g/L):

  (A) EMPIRICAL CONVOLUTION -- draw theta^(s) ~ correlated posterior, evaluate the FULL nonlinear
      g(theta^(s), c) on the solver, then draw b_p^(s), Sigma_delta^(s) from the hierarchy Gibbs
      draws and a measurement/discrepancy realization eps ~ N(0, Sigma_delta^(s) + C_meas), and
      COUNT the joint meet indicator  1{ g0+b0+eps0 >= spec0  AND  g1+b1+eps1 >= spec1 }.
  (B) MOMENT-MATCHED GAUSSIAN -- fit mean and covariance of the same mixture and read the joint
      P(meet) from that Gaussian.

Reports both, their difference, and the Monte-Carlo standard error of (A). If they agree, the
deployed moment-matched read is validated; if not, (A) is authoritative and the paper should carry
it.

REQUIRES the differentiable solver (torch) and the real product data -> run on Colab, not in a
light post-processing environment. Writes results/bayes/empirical_convolution_{product}.json.

Usage (2-op cross-check, default):
  OMP_NUM_THREADS=4 python scripts/bayes_empirical_convolution.py --product HLXSYN \
      --n-steps 300 --n-theta 2000 --n-inner 40000

Usage (reviewer item 4: full nonlinear empirical-convolution OPERATING-WINDOW scan for C):
  OMP_NUM_THREADS=4 python scripts/bayes_empirical_convolution.py --product HLXSYN \
      --empirical-window --n-candidates 24 --n-steps 300 \
      --n-theta 2000 --n-inner 40000 --load-cap 35.0 --seed 0
  -> results/bayes/empirical_convolution_window_HLXSYN.json
     (per-op honest Phat_meet with theta-CLUSTERED batch-means MCSE; best in-domain candidate,
      its decisiveness under the 0.95 threshold, and the conditional-rule move frequency).

WHICH HIERARCHY DRAWS. The deployed scan and step2d_meet_margin_certificate.py integrate the
predictive layer over hier_draws_capB.npz; the reads committed before this flag existed used
hier_draws.npz. A read is comparable with Table 4 only when --hier-draws matches, so pass it
explicitly. The two files differ: b_HLXSYN means differ by up to 0.0025.

CACHING. The hierarchy draws enter only the predictive layer, never the solver, so --gs-cache stores
each op's nonlinear pushforward and any later change of draws file, spec or threshold is free:

  # populate the cache and report under capB (the expensive pass)
  OMP_NUM_THREADS=4 python scripts/bayes_empirical_convolution.py --product HLXSYN \
      --empirical-window --n-candidates 24 --n-steps 300 --n-theta 1000 --n-inner 40000 \
      --hier-draws results/bayes/hier_draws_capB.npz \
      --compare-hier-draws results/bayes/hier_draws.npz \
      --out results/bayes/empirical_convolution_window_HLXSYN_capB.json
  # anything else you want from the same pushforwards: no solves
  python scripts/bayes_empirical_convolution.py --product HLXSYN --empirical-window --analysis-only ...
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent / "results" / "bayes"
SPEC = np.array([0.70, 0.50])
TOL = np.array([0.02, 0.05])                 # illustrative purity/yield tolerances (for expected regret)
C_MEAS = np.diag([0.005 ** 2, 0.008 ** 2])


def _gs_cached(post, bundle, op, a, tag):
    """Nonlinear pushforward g(theta^s, op), cached on disk.

    The hierarchy draws enter only the predictive layer below, never the solver, so once an op's
    pushforward is stored, re-reading it under a different hierarchy draws file (or a different spec
    or threshold) costs no solve at all. Returns None only under --analysis-only on a cache miss.

    The solver module is imported only on a miss; the product setup above it still is not, so a
    cached re-analysis costs no solve but still needs the environment the run was made in.
    """
    cf = None
    if a.gs_cache:
        key = hashlib.md5(np.asarray(op, float).round(6).tobytes()).hexdigest()[:10]
        cf = Path(a.gs_cache) / (f"gs_{a.product}_corr_op{key}_n{a.n_theta}"
                                 f"_s{a.seed}_st{a.n_steps}.npz")
        if cf.exists():
            z = np.load(cf)
            if np.allclose(z["op"], np.asarray(op, float)):
                print(f"  [{tag}] cached ({a.n_theta} solves reused)")
                return np.asarray(z["gs"], float)
            print(f"  [{tag}] key collision in {cf.name}; recomputing")
    if a.analysis_only:
        return None
    from cex_model.bayes.decision import decision_covariance_mc
    mc = decision_covariance_mc(post, bundle, op, n_samples=a.n_theta, seed=a.seed,
                                n_steps=a.n_steps, return_gs=True)
    gs = np.asarray(mc["gs"], float)
    gs = gs[np.isfinite(gs).all(1)]
    if cf is not None:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, gs=gs, op=np.asarray(op, float))
    return gs


def _chol_all(Sd_draws):
    """Cholesky of Sigma_delta + C_meas for every hierarchy draw, once per run instead of once per
    (theta, inner) pair."""
    return np.linalg.cholesky(Sd_draws + C_MEAS + 1e-15 * np.eye(2))


def _inner_draws(ng, nh, k_inner, rng):
    """The (hierarchy index, inner noise) draws one op consumes.

    Drawn per theta, in the order the scalar loops used, so the random stream is unchanged; sharing
    the return value across two hierarchy draw files gives a common-random-numbers paired contrast.
    """
    idx = np.empty((ng, k_inner), dtype=np.int64)
    z = np.empty((ng, k_inner, 2))
    for i in range(ng):
        idx[i] = rng.integers(nh, size=k_inner)
        z[i] = rng.standard_normal((k_inner, 2))
    return idx, z


def _convolve(gs, b_draws, chol, idx, z):
    """x = g + b_h + L_h eps over every (theta, inner) pair; (ng, k_inner, 2)."""
    return gs[:, None, :] + b_draws[idx] + np.einsum("ijkl,ijl->ijk", chol[idx], z)


def _p_meet_empirical(gs, b_draws, Sd_draws, spec, rng, n_inner):
    """Empirical joint-meet convolution: nonlinear g-samples (x) hierarchy draws (x) inner noise."""
    hit = tot = 0
    ng = len(gs)
    nh = len(b_draws)
    for _ in range(n_inner):
        g = gs[rng.integers(ng)]
        j = rng.integers(nh)
        b = b_draws[j]
        Sd = Sd_draws[j]
        L = np.linalg.cholesky(Sd + C_MEAS + 1e-15 * np.eye(2))
        eps = L @ rng.standard_normal(2)
        x = g + b + eps
        hit += int(x[0] >= spec[0] and x[1] >= spec[1])
        tot += 1
    p = hit / tot
    se = np.sqrt(p * (1 - p) / tot)
    return p, se, tot


def _p_meet_empirical_clustered(gs, b_draws, Sd_draws, spec, rng, k_inner, draws=None, chol=None):
    """theta-CLUSTERED batch-means empirical convolution (honest MCSE).

    Each nonlinear pushforward g(theta^i,c) is one independent replicate (cluster). Per cluster we
    average k_inner (hierarchy index, inner-noise) draws of the joint meet indicator; the estimator is
    the mean over the M = len(gs) clusters and its MCSE is the batch-means s.d. over clusters / sqrt(M).
    This captures BOTH the between-theta variance (which dominates for a nonlinear g) and the inner
    variance, unlike the flat binomial sqrt(p(1-p)/n_inner) which treats all inner draws as independent.

    Returns (phat, mcse, phat_per_cluster). Pass ``draws`` to reuse another call's (index, noise)
    draws, which is what makes a two-draw-file contrast paired.
    """
    idx, z = _inner_draws(len(gs), len(b_draws), k_inner, rng) if draws is None else draws
    chol = _chol_all(Sd_draws) if chol is None else chol
    x = _convolve(gs, b_draws, chol, idx, z)
    phat_i = ((x[..., 0] >= spec[0]) & (x[..., 1] >= spec[1])).mean(1)
    phat = float(phat_i.mean())
    mcse = float(phat_i.std(ddof=1) / np.sqrt(len(gs))) if len(gs) > 1 else float("nan")
    return phat, mcse, phat_i


def _regret_empirical_clustered(gs, b_draws, Sd_draws, spec, tol, rng, k_inner, draws=None, chol=None):
    """Expected tolerance-weighted regret under the SAME empirical-convolution law as P(meet).

    Regret(c) = E[ sum_q max(0, (spec_q - x_q)/tol_q) ], x = g(theta,c) + b_p + delta + eps_meas, the
    honest (Jensen-corrected) posterior-predictive read -- the empirical-convolution counterpart of the
    deployed moment-matched regret map. theta-clustered batch-means MCSE (as in _p_meet_empirical_clustered).
    Returns (regret_mean, mcse).
    """
    idx, z = _inner_draws(len(gs), len(b_draws), k_inner, rng) if draws is None else draws
    chol = _chol_all(Sd_draws) if chol is None else chol
    x = _convolve(gs, b_draws, chol, idx, z)
    reg_i = np.maximum(0.0, (spec - x) / tol).sum(-1).mean(1)
    reg = float(reg_i.mean())
    mcse = float(reg_i.std(ddof=1) / np.sqrt(len(gs))) if len(gs) > 1 else float("nan")
    return reg, mcse


def _cond_pmeet_per_h(gs, b_draws, Sd_draws, spec, rng, h_idx, chol=None):
    """Conditional P(meet | h) under the empirical law, for each hierarchy draw in h_idx.

    For a fixed hierarchy draw h = (b_p, Sigma_delta), average the joint meet indicator over the
    nonlinear theta pushforward gs (one inner-noise draw per theta). Returns an array over h_idx.
    """
    ng = len(gs)
    chol = _chol_all(Sd_draws) if chol is None else chol
    out = np.empty(len(h_idx))
    for k, j in enumerate(h_idx):
        b = b_draws[j]
        L = chol[j]
        z = rng.standard_normal((ng, 2))
        x = gs + b + z @ L.T
        out[k] = np.mean((x[:, 0] >= spec[0]) & (x[:, 1] >= spec[1]))
    return out


def _p_meet_moment_matched(gs, b_draws, Sd_draws, spec, rng, n_inner):
    """Gaussian with the mixture's mean and covariance (the deployed moment-matched read)."""
    mean = gs.mean(0) + b_draws.mean(0)
    cov = np.cov(gs.T) + Sd_draws.mean(0) + C_MEAS + np.cov(b_draws.T)
    L = np.linalg.cholesky(cov + 1e-15 * np.eye(2))
    z = rng.standard_normal((n_inner, 2))
    x = mean + z @ L.T
    hit = np.sum((x[:, 0] >= spec[0]) & (x[:, 1] >= spec[1]))
    return hit / n_inner, mean.tolist(), cov.tolist()


def run_empirical_window(a):
    """Reviewer item 4: full NONLINEAR empirical-convolution operating-window scan for mAb C.

    For every candidate operating condition c in [historical] + candidate_ops_for(bundle) evaluate BOTH
    Phat_meet(c) = (1/S) sum_s 1{ g(theta^s,c) + b_p^s + delta^s + eps_meas^s >= spec } and the expected
    tolerance-weighted regret E[sum_q max(0,(spec_q - x_q)/tol_q)] under the SAME honest law (so the
    historical point, the window scan, P(meet) AND regret are all read off the empirical-convolution law,
    not a mix of laws), directly on the
    nonlinear pushforward, attach a theta-CLUSTERED batch-means MCSE, restrict to the in-domain pool
    (loading <= load_cap), relocate the best in-domain candidate by the honest Phat_meet, and report its
    conditional-rule MOVE frequency Pr(move|D) under THIS law. This replaces the MAP-centered moment
    match (which reads ~0.985 at C's candidate) with the Jensen-corrected posterior-predictive read.
    """
    from cex_model.bayes.posterior import Posterior
    from cex_model.bayes.loading_sweep import _load_product_setup
    from cex_model.bayes.decision_window import candidate_ops_for

    post = Posterior.load(f"{a.in_dir}/{a.product}_correlated_posterior.npz")
    _iid, base_op, _tol, _n, _prior, bundle = _load_product_setup(a.product, a.in_dir)
    hpath = a.hier_draws or f"{a.in_dir}/hier_draws.npz"
    hd = np.load(hpath, allow_pickle=True)
    bkey = f"b_{a.product}"
    if bkey not in hd.files:
        raise SystemExit(f"{bkey} not in {hpath} (have {list(hd.files)})")
    b_draws, Sd_draws = hd[bkey], hd["Sd_draws"]
    chol = _chol_all(Sd_draws)

    alt = None
    if a.compare_hier_draws:
        hd2 = np.load(a.compare_hier_draws, allow_pickle=True)
        if bkey not in hd2.files:
            raise SystemExit(f"{bkey} not in {a.compare_hier_draws}")
        if len(hd2[bkey]) != len(b_draws):
            raise SystemExit("paired contrast needs the two draw files to have the same number of "
                             f"draws ({len(hd2[bkey])} vs {len(b_draws)})")
        alt = (hd2[bkey], hd2["Sd_draws"], _chol_all(hd2["Sd_draws"]))

    cands = [list(base_op)] + [list(c) for c in candidate_ops_for(bundle, n_candidates=a.n_candidates, seed=a.seed)]
    if a.only_loadings:
        # Screen-then-confirm: the nonlinear read is only needed where the cheap moment-matched scan puts a
        # candidate within reach of the 0.95 threshold. Keep the historical op (index 0, used as the move-
        # frequency reference) plus the pool candidates nearest the requested loadings, preserving pool
        # identity so the rows stay comparable with the moment-matched scan.
        keep = {0}
        for want in a.only_loadings:
            keep.add(min(range(1, len(cands)), key=lambda i: abs(cands[i][0] - want)))
        cands = [cands[i] for i in sorted(keep)]
        print(f"[subset] evaluating {len(cands)} ops (historical + "
              f"{', '.join(f'{c[0]:.2f}' for c in cands[1:])} g/L) instead of {a.n_candidates + 1}")
    k_inner = max(1, a.n_inner // a.n_theta)
    n_h_move = min(len(b_draws), a.n_h_move)
    rng = np.random.default_rng(a.seed)
    DECISIVE = 0.95

    rows = []
    cond_hist = None
    print("=" * 96)
    print(f"EMPIRICAL-CONVOLUTION OPERATING-WINDOW SCAN  --  {a.product}  (load_cap={a.load_cap} g/L)")
    print("=" * 96)
    for k, op in enumerate(cands):
        tag = "hist" if k == 0 else f"cand{k:02d}"
        gs = _gs_cached(post, bundle, op, a, tag)
        if gs is None:
            raise SystemExit(f"[{tag}] load={op[0]:.2f}: no cached pushforward under --analysis-only; "
                             f"run once without it to populate {a.gs_cache}")
        # Drawn here rather than inside each estimator so the alternative hierarchy file below is
        # evaluated on the SAME (theta, index, noise) draws; the stream order is the estimators' own.
        d_p = _inner_draws(len(gs), len(b_draws), k_inner, rng)
        d_r = _inner_draws(len(gs), len(b_draws), k_inner, rng)
        phat, mcse, _ = _p_meet_empirical_clustered(gs, b_draws, Sd_draws, SPEC, rng, k_inner, d_p, chol)
        regret, regret_mcse = _regret_empirical_clustered(gs, b_draws, Sd_draws, SPEC, TOL, rng, k_inner, d_r, chol)
        h_idx = rng.integers(len(b_draws), size=n_h_move)
        cond = _cond_pmeet_per_h(gs, b_draws, Sd_draws, SPEC, rng, h_idx, chol)
        paired = {}
        if alt is not None:
            b2, Sd2, ch2 = alt
            p2, _, pi2 = _p_meet_empirical_clustered(gs, b2, Sd2, SPEC, rng, k_inner, d_p, ch2)
            _, _, pi1 = _p_meet_empirical_clustered(gs, b_draws, Sd_draws, SPEC, rng, k_inner, d_p, chol)
            d = pi1 - pi2
            paired = {"phat_meet_alt": p2, "d_phat": phat - p2,
                      "d_mcse": float(d.std(ddof=1) / np.sqrt(len(gs)))}
        loading = float(op[0])
        row = {"op": [float(x) for x in op], "loading": loading,
               "in_domain": loading <= a.load_cap, "n_g_used": int(len(gs)),
               "phat_meet": phat, "mcse_clustered": mcse,
               "regret": regret, "regret_mcse": regret_mcse,
               "mcse_naive_binomial": float(np.sqrt(phat * (1 - phat) / max(1, len(gs) * k_inner))),
               "ci95": [phat - 1.96 * mcse, phat + 1.96 * mcse],
               "cond_decisive_frac": float(np.mean(cond >= DECISIVE)),
               **paired,
               "_cond_h_idx": h_idx.tolist(), "_cond_pmeet": cond.tolist()}
        rows.append(row)
        if k == 0:
            cond_hist = cond  # historical is candidate 0
        print(f"[{tag}] load={loading:6.2f}  in_dom={row['in_domain']!s:5}  "
              f"Phat_meet={phat:.3f} +/- {mcse:.3f} (clustered)  regret={regret:.3f}  "
              f"decisive_frac={row['cond_decisive_frac']:.3f}"
              + (f"  d_vs_alt={paired['d_phat']:+.4f} +/- {paired['d_mcse']:.4f}" if paired else ""))

    # best in-domain candidate (exclude the historical point itself, k==0)
    in_dom = [r for r in rows[1:] if r["in_domain"]]
    best = max(in_dom, key=lambda r: r["phat_meet"]) if in_dom else None
    rec = None
    if best is not None:
        # move frequency vs the FIXED best candidate: hist non-decisive AND candidate decisive, per h
        cand_cond = np.asarray(best["_cond_pmeet"])
        hist_cond = np.asarray(cond_hist)
        m = min(len(cand_cond), len(hist_cond))
        p_move = float(np.mean((cand_cond[:m] >= DECISIVE) & (hist_cond[:m] < DECISIVE)))
        rec = {"op": best["op"], "loading": best["loading"], "phat_meet": best["phat_meet"],
               "mcse_clustered": best["mcse_clustered"], "ci95": best["ci95"],
               "decisive": best["phat_meet"] >= DECISIVE, "p_move_given_D": p_move,
               "note": "empirical-convolution (Jensen-corrected) argmax over in-domain pool; "
                       "decisive iff Phat_meet >= 0.95"}
        print("-" * 96)
        print(f"BEST IN-DOMAIN CANDIDATE: load={best['loading']:.2f}  Phat_meet={best['phat_meet']:.3f}"
              f"  decisive={rec['decisive']}  p_move={p_move:.3f}")

    for r in rows:  # drop bulky per-draw arrays from the persisted summary
        r.pop("_cond_h_idx", None)
        r.pop("_cond_pmeet", None)
    out = {"product": a.product, "spec": SPEC.tolist(), "tol": TOL.tolist(),
           "n_theta": a.n_theta, "n_inner": a.n_inner,
           "k_inner": k_inner, "load_cap": a.load_cap, "decisive_threshold": DECISIVE,
           "law": "direct-empirical-convolution", "rows": rows, "recommended_candidate": rec,
           "historical": rows[0],
           "n_candidates_pool": a.n_candidates, "n_ops_evaluated": len(cands),
           "only_loadings": a.only_loadings,
           # provenance: which hierarchy Gibbs draws the predictive layer was integrated over. The deployed
           # scan and step2d use hier_draws_capB.npz; the committed reads here used hier_draws.npz, so the
           # two are only comparable when this field matches theirs.
           "hier_draws": hpath,
           "compare_hier_draws": a.compare_hier_draws,
           "scan_complete": a.only_loadings is None}
    outp = a.out or f"{a.in_dir}/empirical_convolution_window_{a.product}.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 96)
    if alt is not None:
        worst = max(rows, key=lambda r: abs(r["d_phat"]))
        print(f"paired contrast vs {a.compare_hier_draws}: largest |d_phat| = "
              f"{abs(worst['d_phat']):.4f} +/- {worst['d_mcse']:.4f} at {worst['loading']:.2f} g/L")
    print(f"wrote {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--n-theta", type=int, default=2000, help="posterior draws for the nonlinear g pushforward")
    ap.add_argument("--n-inner", type=int, default=40000, help="convolution draws for the meet count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in-dir", default=str(RES))
    ap.add_argument("--empirical-window", action="store_true",
                    help="reviewer item 4: full nonlinear empirical-convolution operating-window scan")
    ap.add_argument("--n-candidates", type=int, default=24, help="Sobol candidate pool size (matches the hier scan)")
    ap.add_argument("--load-cap", type=float, default=35.0, help="in-domain loading cap (g/L)")
    ap.add_argument("--n-h-move", type=int, default=1000, help="hierarchy draws for the conditional move frequency")
    ap.add_argument("--only-loadings", nargs="*", type=float, default=None,
                    help="evaluate only the pool candidates nearest these loadings (g/L), plus the historical "
                         "op. The full 25-op scan is ~50k nonlinear solves; when the cheap moment-matched scan "
                         "leaves only a few candidates within reach of the decision threshold, only those need "
                         "the nonlinear read.")
    ap.add_argument("--hier-draws", default=None,
                    help="hierarchy Gibbs draws .npz; default <in-dir>/hier_draws.npz. The deployed scan and "
                         "step2d_meet_margin_certificate.py use results/bayes/hier_draws_capB.npz, so pass "
                         "that to make this read comparable with Table 4.")
    ap.add_argument("--compare-hier-draws", default=None,
                    help="a second hierarchy draws .npz, evaluated on the same pushforwards and the same "
                         "(index, noise) draws as --hier-draws. Reports the paired difference per op and its "
                         "MCSE, which is far tighter than differencing two independent runs. Diagnostic only: "
                         "the reported read stays the --hier-draws one.")
    ap.add_argument("--gs-cache", default=str(RES / "convolution_gs_cache"),
                    help="directory of cached nonlinear pushforwards, keyed by product, op, n-theta, seed and "
                         "n-steps. The hierarchy draws never enter the solver, so a cached op costs no solve "
                         "when only the predictive layer changes. Pass '' to disable.")
    ap.add_argument("--analysis-only", action="store_true",
                    help="re-read the cache and redo only the (free) convolution arithmetic; errors on a miss")
    ap.add_argument("--out", default=None,
                    help="output path; default <in-dir>/empirical_convolution[_window]_<product>.json. Pass an "
                         "explicit path to avoid overwriting a committed artifact.")
    a = ap.parse_args()

    if a.empirical_window:
        run_empirical_window(a)
        return

    from cex_model.bayes.posterior import Posterior
    from cex_model.bayes.loading_sweep import _load_product_setup

    post = Posterior.load(f"{a.in_dir}/{a.product}_correlated_posterior.npz")
    # _load_product_setup returns (post_iid, base_op, tol, n, prior, bundle); we only need the bundle
    # (the correlated posterior loaded above is what we sample theta from).
    _iid_post, _base_op, _tol, _n, _prior, bundle = _load_product_setup(a.product, a.in_dir)

    # hierarchy Gibbs draws (b_p for this product, shared Sigma_delta)
    hpath = a.hier_draws or f"{a.in_dir}/hier_draws.npz"
    hd = np.load(hpath, allow_pickle=True)
    bkey = f"b_{a.product}"
    if bkey not in hd.files:
        raise SystemExit(f"{bkey} not in {hpath} (have {list(hd.files)})")
    b_draws = hd[bkey]
    Sd_draws = hd["Sd_draws"]

    # operating points: historical + committed lower-loading candidate
    scan = json.load(open(f"{a.in_dir}/hier_scan/{a.product}_decision_window_predictive.json"))
    ops = {"historical": scan["decision_op"], "candidate": scan["predictive_scan"]["argmax_op"]}

    rng = np.random.default_rng(a.seed)
    out = {"product": a.product, "spec": SPEC.tolist(), "n_theta": a.n_theta, "n_inner": a.n_inner,
           "hier_draws": hpath,
           "sigma_meas": [0.005, 0.008], "ops": {}}
    print("=" * 92)
    print(f"EMPIRICAL CONVOLUTION vs MOMENT-MATCHED GAUSSIAN  --  {a.product}")
    print("=" * 92)
    for tag, op in ops.items():
        gs = _gs_cached(post, bundle, op, a, tag)
        if gs is None:
            raise SystemExit(f"[{tag}]: no cached pushforward under --analysis-only; run once without it")
        p_emp, se, tot = _p_meet_empirical(gs, b_draws, Sd_draws, SPEC, rng, a.n_inner)
        p_mm, mean_mm, cov_mm = _p_meet_moment_matched(gs, b_draws, Sd_draws, SPEC, rng, a.n_inner)
        row = {"op": list(op), "n_g_used": int(len(gs)),
               "p_meet_empirical": p_emp, "mc_se": se,
               "p_meet_moment_matched": p_mm, "diff": p_emp - p_mm,
               "mean_g_mixture": mean_mm, "cov_mixture": cov_mm}
        out["ops"][tag] = row
        print(f"\n[{tag}] op={[round(x,1) for x in op]}  (n_g={len(gs)})")
        print(f"  empirical convolution P(meet) = {p_emp:.4f} +/- {se:.4f} (MC SE)")
        print(f"  moment-matched Gaussian P(meet) = {p_mm:.4f}")
        print(f"  difference (emp - gauss) = {p_emp - p_mm:+.4f}"
              f"   {'AGREE (within ~2 SE)' if abs(p_emp - p_mm) < 2 * se + 0.01 else 'DISAGREE -> use empirical'}")

    outp = a.out or f"{a.in_dir}/empirical_convolution_{a.product}.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 92)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
