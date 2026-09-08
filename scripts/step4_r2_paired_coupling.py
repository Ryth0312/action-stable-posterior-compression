"""R2 pilot (Colab / torch): a joint decision-region linearization certificate, conditional on the posterior law.

Certifies the gap between the NONLINEAR posterior pushforward and its LINEARIZED (delta-method) pushforward on the
JOINT meet region, under the SAME posterior law. Everything is in TOLERANCE-WHITENED units, so the
region is Abar = {x : x_q >= spec_q / tol_q}. Paired-coupling bound (NOT a
first-order Lipschitz-constant bound): for the same posterior draw theta,

    X = T^{-1} g(theta)              (nonlinear pushforward, on the fixed MAP window)
    Y = T^{-1} [g(mu) + G(theta-mu)] (linearized pushforward)

and for any t > 0,

    |P(X in A) - P(Y in A)|  <=  P{ d_inf(Y, dA) <= t }  +  P{ |X - Y|_inf > t }.        (*)

Both terms are bounded under the SAME law. Posterior draws are clamped componentwise to the physical box before
the solver sees them, so Y follows a CLIPPED (censored) rather than an untruncated Gaussian law; the untruncated
closed form for the tube is wrong by up to three orders of magnitude here, anti-conservatively. Term 1 is
therefore sampled under Y's own law (4e6 linear-algebra draws, no solver) and term 2 from the N paired solver
draws, each bounded by an exact one-sided Clopper-Pearson rate, union-bounded over both terms and a pre-fixed
t-grid. Production evaluates P(meet) under the UNTRUNCATED Gaussian, so a third measured term bridges the two
laws, |P(X_clip)-P(Y_G)| <= |P(X_clip)-P(Y_clip)| + |P(Y_clip)-P(Y_G)|, and the certified interval is centred on
that untruncated P_lin. The nonlinear Monte-Carlo estimate and its standard error are the observed-gap check.

SCOPE (honest): this certifies the linearization gap CONDITIONAL ON the posterior law. It does NOT prove the
Laplace posterior equals the true posterior (that stays with the NUTS anchor / multistart / empirical convolution).
Hence "conditional on the posterior law", not "posterior coverage theorem".

go/no-go: non-vacuous iff the certified bound is small enough not to change the wdec<1 / P(meet) threshold
classification. Ideal: mAb A/B tight; mAb C wider (auto-requires empirical convolution); mAb D/E wide/refused.

SCOPE OF THE CONDITION.  By default this certifies each product's HISTORICAL operating condition, which is
what the article reports. ``--op`` certifies any other candidate, which is what the pool-wide question needs:
the deployed reads are taken across a 24-candidate pool spanning a four-dimensional box, and a linearisation
validated at one condition does not automatically hold at another. ``--posterior correlated_posterior`` takes
it at the law the deployment reads use rather than at the independent-residual fit. Neither can overwrite the
committed historical artifact: the output path carries both tags.

Run (Colab, after `pip install -e ".[bayes]"`):
  OMP_NUM_THREADS=4 python scripts/step4_r2_paired_coupling.py --products HLXSYN HLXSYN HLXSYN --n-draws 4000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

import cex_model.app_support as A
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_covariance_mc, decision_jacobian
from cex_model.bayes.posterior import Posterior
from cex_model.bayes.prior import physical_u_bounds, solver_guard_bounds

_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
C_MEAS_DEFAULT = np.diag(np.array([0.005, 0.008]) ** 2)


def _load(product, in_dir, posterior="posterior", op=None):
    """The posterior and the operating condition this certificate is taken at.

    ``op`` defaults to the product's historical condition, the one the article certifies. Passing an explicit
    operating condition certifies that candidate instead, which is what closes the pool-wide question: the
    deployed reads are taken across a 24-candidate pool, and nothing forces a linearisation validated at one
    condition to hold at another. ``posterior`` selects the fitted law; the deployed reads use the correlated
    refit, while the certificate as reported in the article is taken at the independent-residual fit.
    """
    post = Posterior.load(Path(in_dir) / f"{product}_{posterior}.npz")
    dj = json.loads((Path(in_dir) / f"{product}_decision.json").read_text())
    op = np.asarray(dj["decision_op"] if op is None else op, float)
    tol = np.asarray(dj.get("tol", [0.02, 0.05]), float)
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    return post, bundle, op, tol


def _dist_to_orthant_boundary(Yw, sw):
    """d_inf(y, dA) for A = {y >= sw} (whitened orthant): min_i(y_i - s_i) inside A, else max_i (s_i - y_i)_+."""
    margin = Yw - sw                                  # (N, 2)
    inside = np.all(margin >= 0, axis=1)
    d = np.where(inside, margin.min(axis=1), np.clip(-margin, 0, None).max(axis=1))
    return d


def _cp_upper(k, n, alpha):
    """Exact one-sided Clopper-Pearson upper bound on a binomial rate (tight where Hoeffding is loose)."""
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def _term1_tube(post, G, g_map, tol, spec, t_grid, n_protein, alpha, guard, n_big=4_000_000, seed=12345):
    """P{d_inf(Y, dAbar) <= t} under the ACTUAL law of Y, with an exact-binomial upper bound.

    Y is the linearised pushforward of posterior draws CLAMPED componentwise to the prior's physical support, so
    it follows a clipped (censored) law: on these products 22-87% of draws are clipped and the untruncated
    closed form is wrong by up to three orders of magnitude (and ANTI-conservatively so). We therefore sample
    Y's own law -- pure linear algebra, no solver -- and bound the tube rate exactly. Returns
    ``(tube_hat, tube_upper, clip_fraction)`` per grid point.
    """
    rng = np.random.default_rng(seed)
    lo, hi = guard(n_protein)
    us = rng.multivariate_normal(np.asarray(post.mean, float), np.asarray(post.cov, float), size=n_big)
    usc = np.clip(us, lo, hi)
    clip_frac = float(np.mean(np.any(usc != us, axis=1)))
    Yw = (g_map + (usc - np.asarray(post.u_map, float)) @ G.T) / tol
    d = _dist_to_orthant_boundary(Yw, spec / tol)
    hat = np.array([float(np.mean(d <= t)) for t in t_grid])
    up = np.array([_cp_upper(int(round(h * n_big)), n_big, alpha) for h in hat])
    k_meet = int(np.sum(np.all(Yw >= spec / tol, axis=1)))       # clipped-law meet rate, same big sample
    return hat, up, clip_frac, k_meet, n_big


def _paired_draws(post, bundle, op, n_draws, seed, guard):
    """Paired (nonlinear X, linear Y) whitened decision draws on the SAME posterior samples.

    Returns the Jacobian alongside them: the caller needs it for the tube term, and reading it from a
    notebook global is what let this run only inside the session it was written in.
    """
    G, g_map, _ = decision_jacobian(bundle, list(op), post.u_map, n_steps=300, return_extra=True)
    G = np.atleast_2d(G)
    # nonlinear pushforward on the fixed MAP window (reselect=False => all draws kept, index-aligned)
    mc = decision_covariance_mc(post, bundle, list(op), n_samples=n_draws, seed=seed, n_steps=300,
                                reselect=False, return_gs=True,
                                clip_bounds=guard(bundle.components.n_protein))
    X = np.asarray(mc["gs"], float)                   # (N, 2) nonlinear
    # same draws, linearized: regenerate posterior.samples(N, seed) with the identical clip
    n = bundle.components.n_protein
    lo, hi = guard(n)
    us_raw = post.samples(n_draws, seed=seed)
    us = np.clip(us_raw, lo, hi)
    clip_frac = float(np.mean(np.any(us != us_raw, axis=1)))   # Y is exactly Gaussian only if this is 0
    us = us[: X.shape[0]]
    Y = g_map + (us - np.asarray(post.u_map, float)) @ G.T   # (N, 2) linear
    C = G @ post.cov @ G.T
    return X, Y, g_map, C, clip_frac, G


def _predictive_flips(X, Y, tol, spec, hier_draws, product, alpha, rng):
    """Directional nonlinear-versus-linear flip rates at the DEPLOYED predictive law.

    The union bound of (*) charges the whole boundary tube, which at the deployed law is dominated by the
    discrepancy layer rather than by the paired mechanistic discrepancy and cannot resolve a classification
    whose predictive margin is a few thousandths. The two arms share every predictive draw, so the
    DISAGREEMENT event can be measured directly instead of bounded by two marginals:

        |P(X' in Abar) - P(Y' in Abar)| <= max{ P(Y' in Abar, X' notin Abar), P(X' in Abar, Y' notin Abar) },

    each an exact binomial rate on paired trials. Reuses the same solver draws; no further solver calls.
    """
    hd = np.load(hier_draws)
    bias, sd = hd[f"b_{product}"], hd["Sd_draws"]
    L = np.linalg.cholesky(sd + C_MEAS_DEFAULT + 1e-14 * np.eye(2))
    sw = spec / tol
    n = X.shape[0]
    # ONE predictive draw per solver draw, so the trials are i.i.d. Bernoulli and Clopper-Pearson is exact --
    # the same construction the threshold certificate of Corollary 3.7 uses. Replicating the predictive draw
    # within a solver draw would cluster the trials and make the limits anti-conservative.
    k = rng.integers(0, len(sd), n)
    U = bias[k] + np.einsum("nij,nj->ni", L[k], rng.standard_normal((n, 2)))
    inX = np.all((X + U) / tol >= sw, axis=1)
    inY = np.all((Y + U) / tol >= sw, axis=1)
    hi_cnt = int(np.sum(inX & ~inY))                 # nonlinear meets where the linearised one does not
    lo_cnt = int(np.sum(inY & ~inX))                 # and the reverse
    up_hi = _cp_upper(hi_cnt, n, alpha)
    up_lo = _cp_upper(lo_cnt, n, alpha)
    return {"n_paired_predictive": n,
            "P_meet_nonlinear_predictive": float(np.mean(inX)),
            "P_meet_linear_predictive": float(np.mean(inY)),
            "flip_up_rate": hi_cnt / n, "flip_down_rate": lo_cnt / n,
            "flip_up_upper": up_hi, "flip_down_upper": up_lo,
            "flip_bound": float(max(up_hi, up_lo))}


def analyze(product, in_dir, spec, n_draws, seed, guard, *, op=None, posterior="posterior",
            hier_draws=None, save_paired=None):
    hist = op is None
    post, bundle, op, tol = _load(product, in_dir, posterior, op)
    X, Y, g_map, C, clip_frac, G = _paired_draws(post, bundle, op, n_draws, seed, guard)
    N = X.shape[0]
    sw = spec / tol
    Xw, Yw = X / tol, Y / tol
    pX = float(np.mean(np.all(Xw >= sw, axis=1)))
    pY = float(np.mean(np.all(Yw >= sw, axis=1)))
    dinf = np.max(np.abs(Xw - Yw), axis=1)            # |X - Y|_inf per paired draw

    t_grid = np.linspace(0.02, 1.0, 50)
    m = len(t_grid)
    delta = 0.05
    alpha = delta / (2 * m)                           # union bound over BOTH terms at every grid point
    n_prot = bundle.components.n_protein
    term1_hat, term1_up, clip_frac_big, k_meet, n_big = _term1_tube(post, G, g_map, tol, spec, t_grid,
                                                                    n_prot, alpha, guard)
    # exact-binomial upper bound on term2 = P{|X-Y|_inf > t} from the N paired draws
    term2_hat = np.array([float(np.mean(dinf > t)) for t in t_grid])
    term2_up = np.array([_cp_upper(int(round(h * N)), N, alpha) for h in term2_hat])
    # The deployed pipeline computes P(meet) under the UNTRUNCATED Gaussian, while both paired terms live
    # under the clipped law. Bridge the two with the (measured) support gap so the certificate applies to the
    # production quantity: |P(X_clip)-P(Y_G)| <= |P(X_clip)-P(Y_clip)| + |P(Y_clip)-P(Y_G)|.
    from scipy.stats import multivariate_normal as _mvn
    p_lin_gauss = float(_mvn(mean=-g_map / tol, cov=C / tol[:, None] / tol[None, :] + 1e-15 * np.eye(2),
                             allow_singular=True).cdf(-sw))
    gap_up = max(abs(_cp_upper(k_meet, n_big, alpha) - p_lin_gauss),
                 abs((1.0 - _cp_upper(n_big - k_meet, n_big, alpha)) - p_lin_gauss))
    bound = term1_up + term2_up + gap_up
    j = int(np.argmin(bound))
    floor = float(term1_hat[j] + term2_hat[j])        # what the bound tends to as the samples grow
    mc_err = float(np.sqrt(max(pX * (1 - pX), pY * (1 - pY)) / N))

    # The bound certifies |P_nonlin - P_lin| <= b, so the certified interval is centred on the LINEARISED
    # probability (what the deployed pipeline computes); the nonlinear Monte-Carlo estimate and its standard
    # error are the separate observed-gap check. Thresholds are the i.i.d.-CONDITIONAL decisiveness
    # thresholds, not a deployment verdict (the deployed law is predictive).
    P_HI, P_LO = 0.95, 0.05
    b = float(bound[j])
    lo_i, hi_i = max(0.0, p_lin_gauss - b), min(1.0, p_lin_gauss + b)
    if p_lin_gauss >= P_HI:
        cls_safe, cls_state = bool(lo_i >= P_HI), "cond-decisive-meet"
    elif p_lin_gauss <= P_LO:
        cls_safe, cls_state = bool(hi_i <= P_LO), "cond-decisive-miss"
    else:
        cls_safe, cls_state = False, "already-ambiguous"
    if save_paired:
        Path(save_paired).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_paired, X=X, Y=Y, op=np.asarray(op, float), tol=tol)
    if hier_draws:
        pf = _predictive_flips(X, Y, tol, spec, hier_draws, product, alpha,
                               np.random.default_rng(seed + 1))
        pf["hier_draws"] = str(hier_draws)
    out = {
        "product": product, "op": [float(v) for v in op], "op_is_historical": bool(hist),
        "posterior": posterior,
        "n_draws": N, "P_X_meet_nonlinear": pX, "P_Y_meet_linear": pY,
        "observed_gap": abs(pX - pY), "certified_bound": b, "t_star": float(t_grid[j]),
        "term1_tube_hat": float(term1_hat[j]), "term1_tube_upper": float(term1_up[j]),
        "term2_tail_hat": float(term2_hat[j]), "term2_tail_upper": float(term2_up[j]),
        "P_lin_untruncated": p_lin_gauss, "support_gap_upper": float(gap_up),
        "large_sample_floor": floor, "finite_sample_share": float((b - floor) / b) if b > 0 else 0.0,
        "mc_stderr": mc_err, "alpha_per_test": float(alpha),
        "clip_fraction_paired": float(clip_frac), "clip_fraction_tube": float(clip_frac_big),
        "non_vacuous": bool(b < 1.0),
        "classification_state": cls_state, "classification_safe": cls_safe,
        "certified_interval": [lo_i, hi_i],
        "predictive_flips": pf if hier_draws else None,
        "t_grid": [float(t) for t in t_grid],
        "term1_hat_curve": [float(v) for v in term1_hat],
        "term2_hat_curve": [float(v) for v in term2_hat],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=['HLXSYN'])
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--spec", type=float, nargs=2, default=[0.70, 0.50])
    ap.add_argument("--n-draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard", choices=["solver", "prior-span"], default="solver",
                    help="support draws are clipped to before the solver. 'solver' is the "
                         "interval the model is defined on; 'prior-span' reuses the prior's "
                         "elicitation span, which is what the superseded certificate used")
    ap.add_argument("--op", type=float, nargs=4, default=None,
                    metavar=("LOADING", "GRAD_START", "GRAD_END", "ELUTION_CV"),
                    help="certify this operating condition instead of the product's historical one")
    ap.add_argument("--posterior", default="posterior",
                    help="posterior file tag; 'correlated_posterior' is the law the deployment reads use")
    ap.add_argument("--hier-draws", default=None,
                    help="committed hierarchy draws; enables the DEPLOYED-law directional flip certificate, "
                         "measured on the same paired solver draws (no extra solver calls)")
    ap.add_argument("--save-paired", default=None,
                    help="write the paired (X, Y) whitened decision draws to this .npz, so any later "
                         "predictive-layer question is post-processing rather than another solver run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = np.asarray(args.spec, float)
    # a non-default condition or posterior must not overwrite the committed historical artifact
    tag = "" if args.op is None else f"_op{args.op[0]:g}"
    ptag = "" if args.posterior == "posterior" else f"_{args.posterior}"
    out_path = args.out or f"results/bayes/r2_paired_coupling_{'_'.join(args.products)}{ptag}{tag}.json"

    rows = []
    for product in args.products:
        a = analyze(product, args.in_dir, spec, args.n_draws, args.seed,
                    physical_u_bounds if args.guard == "prior-span" else solver_guard_bounds,
                    op=args.op, posterior=args.posterior,
                    hier_draws=args.hier_draws,
                    save_paired=(args.save_paired.replace("PRODUCT", product) if args.save_paired else None))
        rows.append(a)
        iv = a["certified_interval"]
        print(f"{product:14} P(meet) nonlinear={a['P_X_meet_nonlinear']:.3f} linear={a['P_Y_meet_linear']:.3f} "
              f"| observed gap={a['observed_gap']:.4f}  certified<= {a['certified_bound']:.3f} @t*={a['t_star']:.2f} "
              f"(tube={a['term1_tube_upper']:.3f}+tail={a['term2_tail_upper']:.3f})")
        print(f"{'':14}   interval=[{iv[0]:.3f},{iv[1]:.3f}] {a['classification_state']} "
              f"safe={a['classification_safe']} | large-sample floor={a['large_sample_floor']:.3f} "
              f"({100*a['finite_sample_share']:.0f}% of the width is finite-sample) "
              f"clip_frac={a['clip_fraction_tube']:.3f}")
        if a.get("predictive_flips"):
            pf = a["predictive_flips"]
            print(f"{'':14}   deployed law: P(meet) nonlinear={pf['P_meet_nonlinear_predictive']:.4f} "
                  f"linear={pf['P_meet_linear_predictive']:.4f} | flips up={pf['flip_up_rate']:.5f} "
                  f"down={pf['flip_down_rate']:.5f} -> certified<= {pf['flip_bound']:.4f} "
                  f"on {pf['n_paired_predictive']:,} i.i.d. paired trials")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}")
    print("go/no-go: R2 graduates to a Contribution only if the certified bound is non-vacuous AND does not flip")
    print("  the threshold classification; otherwise keep direct empirical convolution. Scope: conditional on the")
    print("  posterior law (NOT a proof that the Laplace posterior equals the true posterior).")


if __name__ == "__main__":
    main()
