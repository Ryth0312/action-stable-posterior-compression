"""The action certificate at the law the recommendation is actually taken under.

The article certifies the shortfall action (Corollary 3.5) at the CONDITIONAL law, where the decision law is
Gaussian and the transport bound is a Bures distance in closed form, and it certifies the threshold rule
(Corollary 3.7) at the DEPLOYED predictive law by paired binomial rates.  What it does not do is carry the
action certificate itself to the deployed law, where the decision law is a finite mixture of Gaussians rather
than a Gaussian.  This does that, on all three products and both reductions.

Three quantities per candidate and reduction, none of which needs the solver, because the deployed map is the
linearised one and the Jacobians are already cached:

1. EXACT risks.  Conditional on a hierarchy draw the predictive law is Gaussian and the separable shortfall
   integrates in closed form, so ``R_j(a) = K^-1 sum_k E[ell | k]`` is exact over the committed draws -- no
   Monte Carlo at all.  That gives the action gap, both minimisers and whether the compressed one is the full
   one, at the deployed law.

2. The GAUSSIAN route, carried to the mixture.  The reduction leaves the predictive layer untouched, so the
   two arms share the mixing index and

       W2(P_a, P_a,j)^2  <=  K^-1 sum_k W2( N(mu_k, C_a + S_k + C_meas), N(mu_k, C_a,j + S_k + C_meas) )^2

   by convexity of the squared transport cost under a common mixing variable, each term a Bures distance in
   closed form.  So the closed form is not unavailable on the deployed law; it is available as an upper bound.
   Reporting it is what makes the comparison in 3 meaningful rather than rhetorical.

3. The DISTRIBUTION-FREE route of Lemma 3.2 and Corollary 3.3.  Under the canonical coupling the paired
   difference is ``Z_a - Z_a,j = T^-1 G_a (theta - theta_j)``, a centred Gaussian with covariance
   ``Delta_a,j = T^-1 (C_a - C_a,j) T^-1`` free of the predictive layer, and the shortfall is separable and
   1-Lipschitz per whitened coordinate, so

       eps_a,j  <=  sum_q w_q E|Delta Z_q|  =  sqrt(2/pi) sum_q w_q sqrt(Delta_a,j[q,q]),

   again closed form.  The paired Monte-Carlo value of eps is reported beside it to show how much of the bound
   is the Lipschitz step rather than the coupling.  That step is uniform over the predictive layer and can
   therefore be loose by an order of magnitude: eps itself depends on where the hinge sits relative to the
   predictive mean, since an increment only reaches the loss where at least one arm is below specification.
   The sharper form keeps that indicator,
   ``eps <= sum_q w_q E[|dZ_q| 1{min(Z_q, Z_j,q) < sbar_q}]``, and is not used, because the certificate is
   stated from the closed form.

The almost-sure range of 70 that Supplement A quotes, and that the article's Corollary 3.3 remark defers to
   it, belongs to the exact decision map,
whose coordinates are ratios in the unit interval.  It does NOT hold here: under the linearised surrogate the
paired difference is an unbounded Gaussian.  That is why the finite-sample layer is not the point of this run
-- everything reported is a closed form over the committed draws -- and where a Monte-Carlo value is given its
limit is built from the non-negativity of the paired difference and a bound on its second moment, which need
neither a range nor a Lipschitz constant.

One consequence of the two routes deserves stating, because it explains the numbers rather than only
recording them -- and it must be stated narrowly.  Since ``C_a - C_a,j`` is positive semi-definite,
componentwise ``Bures^2(A_k, B_k) <= tr(A_k) - tr(B_k) = tr(Delta_a,j)``, so the mixture-Bures bound is
structurally no larger than the RESIDUAL-L2 coupling surrogate ``L sqrt(tr Delta_a,j)`` and, through the same
comparison, than the coordinatewise bound used here.  That is a statement about one particular analytic
surrogate.  It is NOT a statement that transport dominates every coupling-based route: the surrogate exceeds
the Monte-Carlo value of eps by factors of 3 to 230 on these products, so a loss-aware bound that keeps the
sign, the hinge or the common randomness between candidates is not covered by the comparison and could be
sharper than either.

    python scripts/bayes_route_b_predictive_action.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from cex_model.bayes.decision_compression import (
    bures_w2,
    decision_cov,
    rotate_sigma_block,
    schur_cov,
    variant_d_cov,
)
from cex_model.bayes.posterior import Posterior

PRODUCTS = ['HLXSYN']
SPEC = np.array([0.70, 0.50])
TOL = np.array([0.02, 0.05])
W_LOSS = np.array([1.0, 1.0])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
DELTA = 0.05


def shortfall_gauss(mu, var):
    """E[sum_q w_q (sbar_q - Z_q)_+] for independent-marginal Gaussians; exact, vectorised over draws."""
    s = SPEC / TOL
    sd = np.sqrt(np.maximum(var, 0.0))
    z = np.where(sd > 0, (s - mu) / np.where(sd > 0, sd, 1.0), np.inf * np.sign(s - mu))
    contrib = np.where(sd > 0, (s - mu) * norm.cdf(z) + sd * norm.pdf(z), np.maximum(s - mu, 0.0))
    return contrib @ W_LOSS


def predictive_risk(g, C, bias, sd_draws):
    """Exact predictive risk at one candidate: the mixture is a mean over the committed draws."""
    mu = (g + bias) / TOL                                   # (K, 2)
    var = (np.diagonal(sd_draws + C + C_MEAS, axis1=1, axis2=2)) / TOL ** 2
    return float(shortfall_gauss(mu, var).mean())


def mixture_bures(C_full, C_j, sd_draws):
    """L * sqrt(K^-1 sum_k W2^2) -- the Gaussian closed form carried to the mixture by shared mixing."""
    Ti = np.diag(1.0 / TOL)
    tot = 0.0
    for S in sd_draws:
        A = Ti @ (C_full + S + C_MEAS) @ Ti
        B = Ti @ (C_j + S + C_MEAS) @ Ti
        tot += bures_w2(A, B) ** 2
    return float(np.linalg.norm(W_LOSS) * np.sqrt(tot / len(sd_draws)))


def eps_analytic(C_full, C_j):
    """sqrt(2/pi) sum_q w_q sqrt(Delta[q,q]) -- the canonical-coupling bound, free of the predictive layer."""
    Ti = np.diag(1.0 / TOL)
    D = Ti @ (C_full - C_j) @ Ti
    return float(np.sqrt(2.0 / np.pi) * (W_LOSS * np.sqrt(np.maximum(np.diag(D), 0.0))).sum())


def eps_paired_mc(g, C_full, C_j, bias, sd_draws, n, rng):
    """Paired Monte-Carlo eps and a one-sided upper limit that needs neither a range nor a Lipschitz step.

    The parameter increment is drawn from its own law ``N(0, Delta)``; the two arms share the hierarchy index,
    the discrepancy and the measurement draw, which is what makes the increment free of them.  Both reductions
    drop a non-negative term, so ``C_a - C_a,j >= 0`` and the compressed law is the less dispersed one: the
    base is drawn at the COMPRESSED variance and the increment is added to reach the full law.  Drawing the
    base at the full variance instead would give the compressed arm the variance ``V + Delta`` rather than
    ``V - Delta``, and would invert the sign of the risk shift.

    The limit rests on two facts about ``D = |ell - ell_j|`` and nothing else.  It is non-negative, and
    ``D <= sum_q w_q |dZ_q|`` pointwise, so by Minkowski ``E D^2 <= L2^2`` with
    ``L2 = sum_q w_q sqrt(Delta[q,q])``.  For non-negative ``D`` and ``lam > 0``,
    ``E exp(-lam D) <= 1 - lam eps + lam^2 E D^2 / 2 <= exp(-lam eps + lam^2 L2^2 / 2)``, and optimising the
    Chernoff bound on the LEFT tail of the mean gives ``P(mean <= eps - t) <= exp(-n t^2 / (2 L2^2))``, hence
    ``UCB = mean + L2 sqrt(2 log(1/alpha) / n)``.

    A Borell--TIS argument does not serve here, though it is the tempting one: the Lipschitz constant in the
    Gaussian behind the increment holds only with the base point frozen, while ``D`` depends on the base point
    too, so what it concentrates around is a conditional mean and not ``eps``; and its tail is the upper one,
    which is the wrong side for an upper confidence limit on a mean.
    """
    Ti = np.diag(1.0 / TOL)
    D = Ti @ (C_full - C_j) @ Ti
    D = 0.5 * (D + D.T)
    ev, V = np.linalg.eigh(D)
    A = V @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))
    Cf = Ti @ (C_full + C_MEAS) @ Ti
    k = rng.integers(0, len(sd_draws), n)
    Sk = np.diagonal(sd_draws[k], axis1=1, axis2=2) / TOL ** 2
    var_j = np.clip(np.diag(Cf) + Sk - np.diag(D), 0.0, None)          # the compressed arm is the base
    base = (g + bias[k]) / TOL + rng.standard_normal((n, 2)) * np.sqrt(var_j)
    dz = rng.standard_normal((n, 2)) @ A.T
    s = SPEC / TOL
    l1 = (np.maximum(s - (base + dz), 0.0) * W_LOSS).sum(1)
    l2 = (np.maximum(s - base, 0.0) * W_LOSS).sum(1)
    d = np.abs(l1 - l2)
    l2c = float((W_LOSS * np.sqrt(np.maximum(np.diag(D), 0.0))).sum())     # the second-moment constant
    return (float(d.mean()), float(d.mean() + l2c * np.sqrt(2.0 * np.log(1.0 / DELTA) / n)), l2c,
            float(l1.mean()), float(l2.mean()))                            # arm means: the direction guard


def run(product, *, in_dir, hier_draws, n_mc, seed):
    post = Posterior.load(Path(in_dir) / f"{product}_correlated_posterior.npz")
    z = np.load(Path(in_dir) / f"inflation_jacobians_{product}.npz")
    ops, Gs, gmaps = z["ops"], z["G"], z["g_map"]
    n_pool = len(ops) - 1                                   # the last row is the historical condition
    hd = np.load(hier_draws)
    bias, sd_draws = hd[f"b_{product}"], hd["Sd_draws"]
    n = post.n_protein
    rng = np.random.default_rng(seed)

    cov = {"full": [], "S": [], "F": []}
    for G in Gs[:n_pool]:
        Sr, Gr, ui, vi = rotate_sigma_block(post.cov, G, n)
        cov["full"].append(decision_cov(Gr, Sr))
        cov["S"].append(decision_cov(Gr, schur_cov(Sr, ui, vi)))
        cov["F"].append(variant_d_cov(Sr, Gr, ui, vi))

    R = {k: np.array([predictive_risk(gmaps[a], cov[k][a], bias, sd_draws) for a in range(n_pool)])
         for k in ("full", "S", "F")}
    order = np.argsort(R["full"])
    a_star = int(order[0])
    delta_act = float(R["full"][order[1]] - R["full"][order[0]])
    r_full = R["full"] - R["full"][a_star]

    out = {"product": product, "n_candidates": n_pool, "law": "deployed predictive mixture",
           "hier_draws": str(hier_draws), "n_hier_draws": int(len(sd_draws)),
           "a_star": a_star, "loading_a_star": float(ops[a_star, 0]), "Delta_act": delta_act,
           "R_full": R["full"].tolist(), "loading": ops[:n_pool, 0].tolist(), "by_reduction": {}}

    for j in ("S", "F"):
        b_mix = np.array([mixture_bures(cov["full"][a], cov[j][a], sd_draws) for a in range(n_pool)])
        e_an = np.array([eps_analytic(cov["full"][a], cov[j][a]) for a in range(n_pool)])
        mc = [eps_paired_mc(gmaps[a], cov["full"][a], cov[j][a], bias, sd_draws, n_mc, rng)
              for a in range(n_pool)]
        e_mc = np.array([m[0] for m in mc])
        e_ucb = np.array([m[1] for m in mc])
        # the guard on the direction of the coupling: each arm against its own closed form at a*
        marg = {"full": {"mc": mc[a_star][3], "closed_form": float(R["full"][a_star])},
                "comp": {"mc": mc[a_star][4], "closed_form": float(R[j][a_star])}}
        env_b = np.flatnonzero(r_full <= b_mix)
        env_e = np.flatnonzero(r_full <= e_an)
        a_j = int(np.argmin(R[j]))
        out["by_reduction"][j] = {
            "argmin_compressed": a_j, "preserved": bool(a_j == a_star),
            "max_obs_gap": float(np.max(np.abs(R["full"] - R[j]))),
            "bures_mixture": {"eta": float(b_mix.max()),
                              "ratio_onesided": float(b_mix.max() / delta_act) if delta_act > 0 else np.inf,
                              "envelope_size": int(env_b.size),
                              "certified": bool(env_b.size == 1 and env_b[0] == a_star)},
            "coupling_analytic": {"eta": float(e_an.max()),
                                  "ratio_onesided": float(e_an.max() / delta_act) if delta_act > 0 else np.inf,
                                  "envelope_size": int(env_e.size),
                                  "certified": bool(env_e.size == 1 and env_e[0] == a_star),
                                  "uniform_condition": bool(delta_act > e_an[a_star]
                                                            + np.max(np.delete(e_an, a_star)))},
            "marginal_check_a_star": marg,
            "eps_mc_mean": e_mc.tolist(), "eps_mc_ucb_max": float(e_ucb.max()),
            "eps_analytic_over_mc": float(np.max(e_an / np.maximum(e_mc, 1e-30))),
            "b_mix": b_mix.tolist(), "eps_analytic": e_an.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=PRODUCTS)
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--hier-draws", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--n-mc", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bayes/route_b_predictive_action.json")
    a = ap.parse_args()

    res = [run(p, in_dir=a.in_dir, hier_draws=a.hier_draws, n_mc=a.n_mc, seed=a.seed)
           for p in a.products]
    Path(a.out).write_text(json.dumps(res, indent=1))

    print(f"{'product':14}{'j':>3}{'a*':>4}{'a_j':>5}{'keep':>6}{'Delta_act':>12}"
          f"{'eta_bures':>11}{'ratio':>9}{'|C|':>5}{'eta_coup':>10}{'ratio':>9}{'|C|':>5}"
          f"{'eps_mc':>9}{'bnd/mc':>8}")
    for r in res:
        for j in ("S", "F"):
            b = r["by_reduction"][j]
            bm, cp = b["bures_mixture"], b["coupling_analytic"]
            print(f"{r['product']:14}{j:>3}{r['a_star']:>4}{b['argmin_compressed']:>5}"
                  f"{('yes' if b['preserved'] else 'NO'):>6}{r['Delta_act']:12.3e}"
                  f"{bm['eta']:11.4f}{bm['ratio_onesided']:9.2f}{bm['envelope_size']:5d}"
                  f"{cp['eta']:10.4f}{cp['ratio_onesided']:9.2f}{cp['envelope_size']:5d}"
                  f"{max(b['eps_mc_mean']):9.4f}{b['eps_analytic_over_mc']:8.2f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
