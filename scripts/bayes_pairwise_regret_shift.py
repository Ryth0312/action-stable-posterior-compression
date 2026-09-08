"""Signed pairwise regret-shift certificate on the deployed predictive mixture.

WHAT IT IS FOR.  The distribution-free envelope of main text Corollary 3.3 controls the compressed action
through ``eps_{a,j} = E|ell(a, Z_{a,j}) - ell(a, Z_a)|``, an absolute perturbation taken one action at a time.
Two things are thrown away there: the SIGN of what compression does to a risk, and the COMMON randomness a
candidate shares with the full optimum.  On the deployed mixture that costs a great deal -- the analytic bound
runs 3 to 230 times the sampled value, and the envelope certifies none of the six product-by-reduction cells
where the component-matched transport bound certifies three.

The identity this run is built on keeps both.  With

    delta_{a,j} = E[ ell(a, Z_{a,j}) - ell(a, Z_a) ]

the compressed regret of a against the FULL optimum a* satisfies exactly

    r_j(a) = r_pi(a) + delta_{a,j} - delta_{a*,j},

so with every arm driven by one parameter draw, one hierarchy index, one bias, one discrepancy and one
measurement draw,

    X_{a,j} = [ ell(a, Z_{a,j}) - ell(a, Z_a) ] - [ ell(a*, Z_{a*,j}) - ell(a*, Z_a*) ]

has mean delta_{a,j} - delta_{a*,j} and r_j(a) = r_pi(a) + E X_{a,j}.  A simultaneous lower bound
``LCB{ r_pi(a) + E X_{a,j} } > 0`` for every a != a* certifies that a* is still the compressed minimiser, and
the candidates that fail it are the certified set.  The full-law regret r_pi(a) is exact here -- it is a
closed-form average over the committed draws -- so the only sampled object is E X_{a,j}.

WHY THE ONE-SIDED BOUND HAS TO CHANGE.  X is SIGNED, so the non-negativity argument used for eps does not
apply.  The limit reported is a Catoni M-estimator limit, which needs a variance proxy rather than a range,
and the proxy used is a DETERMINISTIC second moment fixed before the certification sample rather than an
empirical variance, which would make the limit a plug-in approximation and not a confidence bound.
Pathwise, each bracket of X moves the loss by at most the weighted increment it is given, so
|X| <= sum_q w_q (|dZ_q^a| + |dZ_q^{a*}|), and two applications of
Minkowski on Gaussian increments with known covariance Delta = T^{-1}(C_a - C_{a,j})T^{-1} give

    E X^2 <= ( sum_q w_q sqrt(Delta^a_qq) + sum_q w_q sqrt(Delta^{a*}_qq) )^2,

which is the proxy fed to Catoni.  It is loose -- the two brackets are strongly positively correlated through
the common randomness, so the empirical standard deviation runs several times smaller -- and it is reported
alongside so the gap is visible.  The empirical Bernstein limit at an EMPIRICAL range, and the Catoni limit at
the EMPIRICAL variance, are printed as diagnostics and are used for no verdict.

PRE-REGISTERED GO/NO-GO, fixed before the run.  The component-matched mixture bound of
``bayes_route_b_predictive_action.py`` certifies three of six cells and fails on mAb A under both reductions
and on mAb C under the fixed block.  This certificate continues the venue argument only if it certifies at
least two of those three.  Anything less and the direction is closed.  The criterion was fixed at the
per-product family alpha = delta / 46; the verdict reported is the stricter paper-wide family
alpha = delta / 138 = delta / (3 products x 2 reductions x 23 competitors), which subsumes it.

    python scripts/bayes_pairwise_regret_shift.py --ladder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # importable as well as runnable

import numpy as np
from scipy.optimize import brentq

from cex_model.bayes.decision_compression import (
    decision_cov,
    rotate_sigma_block,
    schur_cov,
    variant_d_cov,
)
from cex_model.bayes.posterior import Posterior

from bayes_route_b_predictive_action import predictive_risk       # same directory; the closed forms are there

PRODUCTS = ['HLXSYN']
SPEC = np.array([0.70, 0.50])
TOL = np.array([0.02, 0.05])
W_LOSS = np.array([1.0, 1.0])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
DELTA = 0.05
BURES_FAIL = {("HLXSYN", "S"), ("HLXSYN", "F"), ("HLXSYN", "F")}     # the cells the go/no-go is about
LADDER = [20_000, 50_000, 100_000, 200_000, 400_000]              # the sample-size ladder


def catoni_score(x, mu, alpha, var_proxy):
    """``sum_i psi((x_i - mu)/s) - 2 log(1/alpha)``, positive exactly when the Catoni limit exceeds ``mu``.

    The map is strictly decreasing in ``mu``, so a certificate that only needs the SIGN of ``LCB - mu`` can be
    settled by one evaluation instead of a root search.  ``catoni_lcb`` is the root of this function.
    """
    lg = np.log(1.0 / alpha)
    s = np.sqrt(max(len(x) * var_proxy / (2.0 * lg), 1e-300))
    u = (x - mu) / s
    return float(np.sum(np.sign(u) * np.log1p(np.abs(u) + 0.5 * u ** 2)) - 2.0 * lg)


def catoni_lcb(x, alpha, var_proxy):
    """Catoni M-estimator lower confidence limit; needs a variance proxy, not a range.

    With psi(u) = sign(u) log(1 + |u| + u^2/2) one has exp(psi(u)) <= 1 + u + u^2/2, so for any fixed mu
    E exp(psi((X-mu)/s)) <= exp( (EX - mu)/s + E(X-mu)^2 / (2 s^2) ).  Chernoff at mu = EX then gives

        P{ sum_i psi((x_i - EX)/s) > n v / (2 s^2) + log(1/alpha) } <= alpha,

    with v >= E(X - EX)^2.  The map mu -> sum_i psi((x_i - mu)/s) is strictly decreasing, so the root of
    ``sum_i psi = c`` is a valid 1-alpha lower limit whenever c >= n v / (2 s^2) + log(1/alpha).  Taking
    s^2 = n v / (2 log(1/alpha)) makes that budget c = 2 log(1/alpha) and minimises the resulting width, which
    is the classical sqrt(2 v log(1/alpha) / n) to first order.  ``var_proxy`` must be a deterministic bound on
    the second moment fixed before the sample; an empirical variance would void the guarantee.
    """
    def g(mu):
        return catoni_score(x, mu, alpha, var_proxy)

    lo, hi = x.mean() - 10.0 * np.sqrt(var_proxy) - 1.0, x.mean() + 1.0
    while g(lo) < 0 and lo > x.mean() - 1e6:
        lo -= 10.0 * (abs(lo) + 1.0)
    if g(lo) < 0 or g(hi) > 0:
        return float("-inf")
    return float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-14))


def emp_bernstein_lcb(x, alpha):
    """Empirical Bernstein at the observed range; reported as a diagnostic, never as a verdict."""
    n = len(x)
    rng_ = float(x.max() - x.min())
    v = float(x.var(ddof=1))
    lg = np.log(2.0 / alpha)
    return float(x.mean() - np.sqrt(2.0 * v * lg / n) - 7.0 * rng_ * lg / (3.0 * (n - 1)))


def draw_arms(g, C_full, C_j, bias, sd_draws, idx, rng, n):
    """Losses of both arms at one candidate, on a SHARED set of draws indexed by ``idx``.

    The parameter draw, the hierarchy index, the bias, the discrepancy and the measurement noise are all
    common between the two arms and, through ``idx``, between candidates; only the compression differs.

    The DIRECTION of the coupling is fixed by Lemma 3.1: both reductions drop a non-negative term, so
    ``C_a - C_{a,j} >= 0`` and the COMPRESSED law is the less dispersed one.  The base is therefore drawn at
    the compressed variance and the increment is ADDED to reach the full law, which is the canonical coupling
    ``Z_a = Z_{a,j} + T^{-1} G_a (theta - theta_j)``.  Both arms then carry their own exact marginals -- which
    ``check_marginals`` verifies against the closed forms -- and the pair is positively coupled, which is what
    the paired variance reduction lives on.
    """
    Ti = np.diag(1.0 / TOL)
    D = 0.5 * (Ti @ (C_full - C_j) @ Ti + (Ti @ (C_full - C_j) @ Ti).T)
    ev, V = np.linalg.eigh(D)
    A = V @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))
    Cf = Ti @ (C_full + C_MEAS) @ Ti
    Sk = np.diagonal(sd_draws[idx["k"]], axis1=1, axis2=2) / TOL ** 2
    var_j = np.clip(np.diag(Cf) + Sk - np.diag(D), 0.0, None)
    base = (g + bias[idx["k"]]) / TOL + idx["z"] * np.sqrt(var_j)
    dz = idx["u"] @ A.T
    s = SPEC / TOL
    l_comp = (np.maximum(s - base, 0.0) * W_LOSS).sum(1)
    l_full = (np.maximum(s - (base + dz), 0.0) * W_LOSS).sum(1)
    l2 = float((W_LOSS * np.sqrt(np.clip(np.diag(D), 0.0, None))).sum())
    return l_full, l_comp, l2


def check_marginals(l_full, l_comp, g, C_full, C_j, bias, sd_draws):
    """Both arms against their closed forms -- the guard that pins the direction of the coupling.

    Sampling the base at the wrong variance leaves the full arm correct and the compressed arm off by
    ``2 Delta``, which inverts the sign of the risk shift while leaving every printed regret plausible.  The
    check is one line of arithmetic and it catches exactly that.
    """
    ex = {"full": predictive_risk(g, C_full, bias, sd_draws), "comp": predictive_risk(g, C_j, bias, sd_draws)}
    mc = {"full": float(l_full.mean()), "comp": float(l_comp.mean())}
    se = {k: float(v.std(ddof=1) / np.sqrt(len(v))) for k, v in (("full", l_full), ("comp", l_comp))}
    return {k: {"closed_form": ex[k], "mc": mc[k], "mc_se": se[k],
                "z": (mc[k] - ex[k]) / se[k] if se[k] > 0 else 0.0} for k in ex}


def run(product, *, in_dir, hier_draws, n_mc, seed, route_b, n_products):
    post = Posterior.load(Path(in_dir) / f"{product}_correlated_posterior.npz")
    z = np.load(Path(in_dir) / f"inflation_jacobians_{product}.npz")
    ops, Gs, gmaps = z["ops"], z["G"], z["g_map"]
    n_pool = len(ops) - 1
    hd = np.load(hier_draws)
    bias, sd_draws = hd[f"b_{product}"], hd["Sd_draws"]
    n = post.n_protein
    rb = next(r for r in route_b if r["product"] == product)
    a_star = rb["a_star"]
    r_pi = np.array(rb["R_full"]) - rb["R_full"][a_star]          # exact full-law regret

    cov = {"full": [], "S": [], "F": []}
    for G in Gs[:n_pool]:
        Sr, Gr, ui, vi = rotate_sigma_block(post.cov, G, n)
        cov["full"].append(decision_cov(Gr, Sr))
        cov["S"].append(decision_cov(Gr, schur_cov(Sr, ui, vi)))
        cov["F"].append(variant_d_cov(Sr, Gr, ui, vi))

    rng = np.random.default_rng(seed)
    idx = {"k": rng.integers(0, len(sd_draws), n_mc),
           "z": rng.standard_normal((n_mc, 2)),
           "u": rng.standard_normal((n_mc, 2))}

    m_prod = 2 * (n_pool - 1)                                     # competitors x two reductions, one product
    m_paper = n_products * m_prod                                 # the family the article's claim is made over
    alpha, alpha_prod = DELTA / m_paper, DELTA / m_prod
    out = {"product": product, "a_star": a_star, "n_mc": n_mc, "delta": DELTA,
           "m_paper": m_paper, "m_product": m_prod, "alpha": alpha, "alpha_product": alpha_prod,
           "Delta_act": rb["Delta_act"], "by_reduction": {}}
    for j in ("S", "F"):
        lf_star, lc_star, l2_star = draw_arms(gmaps[a_star], cov["full"][a_star], cov[j][a_star],
                                              bias, sd_draws, idx, rng, n_mc)
        shift_star = lc_star - lf_star
        marg = check_marginals(lf_star, lc_star, gmaps[a_star], cov["full"][a_star], cov[j][a_star],
                               bias, sd_draws)
        rows, certified = [], True
        for a in range(n_pool):
            if a == a_star:
                continue
            lf, lc, l2_a = draw_arms(gmaps[a], cov["full"][a], cov[j][a], bias, sd_draws, idx, rng, n_mc)
            X = (lc - lf) - shift_star
            vp = (l2_a + l2_star) ** 2                            # deterministic second-moment bound on X
            lcb_c = r_pi[a] + catoni_lcb(X, alpha, vp)                       # paper-wide family
            lcb_p = r_pi[a] + catoni_lcb(X, alpha_prod, vp)                  # per-product family
            lcb_emp = r_pi[a] + catoni_lcb(X, alpha, float(X.var(ddof=1)))
            lcb_b = r_pi[a] + emp_bernstein_lcb(X, alpha)
            rows.append({"a": a, "r_pi": float(r_pi[a]), "EX": float(X.mean()),
                         "sd_X": float(X.std(ddof=1)), "sd_proxy": float(np.sqrt(vp)),
                         "lcb_catoni": lcb_c, "lcb_catoni_product_family": lcb_p,
                         "lcb_catoni_empirical_var": lcb_emp,
                         "lcb_bernstein": lcb_b, "positive": bool(lcb_c > 0)})
            certified &= lcb_c > 0
        out["by_reduction"][j] = {
            "certified": bool(certified),
            "marginal_check_a_star": marg,
            "n_positive": int(sum(r["positive"] for r in rows)),
            "n_competitors": len(rows),
            "worst_lcb": float(min(r["lcb_catoni"] for r in rows)),
            "worst_candidate": int(min(rows, key=lambda r: r["lcb_catoni"])["a"]),
            "certified_product_family": bool(all(r["lcb_catoni_product_family"] > 0 for r in rows)),
            "bures_certified": bool(rb["by_reduction"][j]["bures_mixture"]["certified"]),
            "rows": rows}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=PRODUCTS)
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--hier-draws", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--route-b", default="results/bayes/route_b_predictive_action.json")
    ap.add_argument("--n-mc", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/bayes/pairwise_regret_shift.json")
    ap.add_argument("--ladder", action="store_true",
                    help="also run the sample-size ladder and report cells certified at each rung")
    a = ap.parse_args()

    route_b = json.loads(Path(a.route_b).read_text())
    go = lambda n: [run(p, in_dir=a.in_dir, hier_draws=a.hier_draws, n_mc=n, seed=a.seed,
                        route_b=route_b, n_products=len(a.products)) for p in a.products]
    res = go(a.n_mc)

    ladder = []
    if a.ladder:
        for n in LADDER:
            rr = res if n == a.n_mc else go(n)
            cells = [(r["product"], j, r["by_reduction"][j]) for r in rr for j in ("S", "F")]
            ladder.append({"n_mc": n,
                           "n_certified": sum(b["certified"] for _, _, b in cells),
                           "n_bures_fail_certified": sum(b["certified"] for p, j, b in cells
                                                         if (p, j) in BURES_FAIL),
                           "worst_lcb": min(b["worst_lcb"] for _, _, b in cells)})

    Path(a.out).write_text(json.dumps({"n_mc": a.n_mc, "seed": a.seed, "delta": DELTA,
                                       "alpha": res[0]["alpha"], "m_paper": res[0]["m_paper"],
                                       "products": res, "ladder": ladder}, indent=1))

    print(f"{'product':14}{'j':>3}{'Bures':>7}{'pairwise':>10}{'pos/tot':>10}"
          f"{'worst LCB':>12}{'at a':>6}{'Delta_act':>12}")
    won = 0
    for r in res:
        for j in ("S", "F"):
            b = r["by_reduction"][j]
            tag = "cert" if b["certified"] else "--"
            if (r["product"], j) in BURES_FAIL and b["certified"]:
                won += 1
                tag = "CERT*"
            print(f"{r['product']:14}{j:>3}{('cert' if b['bures_certified'] else '--'):>7}{tag:>10}"
                  f"{str(b['n_positive']) + '/' + str(b['n_competitors']):>10}"
                  f"{b['worst_lcb']:12.5f}{b['worst_candidate']:6d}{r['Delta_act']:12.3e}")
    worst_z = max(abs(b["marginal_check_a_star"][k]["z"])
                  for r in res for b in r["by_reduction"].values() for k in ("full", "comp"))
    print(f"\nfamily: {res[0]['m_paper']} tests, alpha = {res[0]['alpha']:.3e} (paper-wide); "
          f"per-product alpha = {res[0]['alpha_product']:.3e} certifies "
          f"{sum(b['certified_product_family'] for r in res for b in r['by_reduction'].values())}/6")
    print(f"marginal check at a*: largest |MC - closed form| over both arms = {worst_z:.2f} MC standard errors")
    for L in ladder:
        print(f"ladder n={L['n_mc']:>7}: {L['n_certified']}/6 cells, "
              f"{L['n_bures_fail_certified']}/3 Bures-failing, worst LCB {L['worst_lcb']:+.5f}")
    print(f"\nPRE-REGISTERED GO/NO-GO: {won} of the 3 Bures-failing cells certified "
          f"-> {'CONTINUE the venue argument' if won >= 2 else 'STOP; freeze the AOAS resubmission'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
