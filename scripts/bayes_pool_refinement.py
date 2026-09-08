"""Nested candidate-pool refinement: does the recommendation survive a denser discretisation?

WHY.  The $24$-point pool is a numerical discretisation of a four-dimensional engineering box, not an
engineering shortlist, so every finite-pool object in the article is in principle a statement about that
discretisation: the shortfall minimiser, the action gap, the certified envelope, whether a decisive candidate
exists in the deployment domain, and therefore which of the three branches a product lands in.  This runs the
whole chain again on a doubled, $48$-candidate pool and reports what moves.

WHY THE COMPARISON IS CLEAN.  The pool is a scrambled Sobol sequence taken at a fixed seed and truncated, so
the pools are NESTED -- the $24$ points are the first $24$ of the $48$ -- so refinement adds candidates
without moving the ones already there.  A verdict that changes therefore
changes because a genuinely new operating condition entered the box, not because the design moved under it.
The nesting check runs before anything else unless ``--skip-nesting-check`` is passed.

WHAT IS RECOMPUTED AT EACH SIZE.  Everything downstream of the Jacobians: the deployed predictive shortfall
risks under the full law and both reductions, the minimiser and the action gap, the deployed meet
probabilities and hence the decisive set and the branch, the mixture-Bures and distribution-free envelopes,
and the paired compression-regret certificate at its own family size, which grows with the pool.  The only
solver work is the decision Jacobian at each new candidate, cached separately from the deployed
24-candidate one so that nothing the article already reports is overwritten.

    python scripts/bayes_covariance_inflation_sensitivity.py --stage jacobians \
        --n-steps 300 --n-candidates 48 --seed 0 --jac-tag _N48
    python scripts/bayes_pool_refinement.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayes_pairwise_regret_shift import catoni_score, draw_arms
from bayes_route_b_predictive_action import eps_analytic, mixture_bures, predictive_risk

from cex_model.bayes.decision_compression import (
    decision_cov,
    rotate_sigma_block,
    schur_cov,
    variant_d_cov,
)
from cex_model.bayes.decision_window import candidate_ops_for, gaussian_meet_prob
from cex_model.bayes.design import _bundle_op_matrix
from cex_model.bayes.posterior import Posterior

PRODUCTS = ['HLXSYN']
SPEC = np.array([0.70, 0.50])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
DELTA, TAU_DEC = 0.05, 0.95
SIZES = [24, 48]
DEPLOY = {'HLXSYN': (25.0, 35.0)}
INFEASIBLE_EVERYWHERE = 0.10        # the pool-redesign regime of the stratified rule
NEAR_BAND = 0.02                    # "near the threshold" for counting newly admitted borderline candidates


def _bundle(product):
    import cex_model.app_support as A
    from cex_model.bayes.compare import filter_experiments
    prod, drop = {"HLXSYN": ("HLXSYN", ["DT"])}.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    return bundle


def support_counts(product, ops):
    """How many of a candidate's four operating coordinates lie inside the product's observed range.

    The deployment domain of the article is defined on LOADING alone, because with three to five runs the
    other three coordinates are barely varied; extrapolation in them is reported per candidate rather than
    used to exclude.  A denser pool can therefore admit a candidate that is nominally in-domain and yet
    extrapolates in every other coordinate, which is what this column is here to make visible.
    """
    M = np.asarray(_bundle_op_matrix(_bundle(product)), float)
    lo, hi = M.min(0), M.max(0)
    return ((ops >= lo) & (ops <= hi)).sum(1)


def check_nesting(product, in_dir, seed):
    """The refinement is only interpretable if the smaller pools are prefixes of the larger ones."""
    bundle = _bundle(product)
    pools = {n: np.asarray(candidate_ops_for(bundle, n_candidates=n, seed=seed), float) for n in SIZES}
    for a, b in zip(SIZES, SIZES[1:]):
        if not np.allclose(pools[a], pools[b][:a]):
            raise SystemExit(f"{product}: pool({a}) is not a prefix of pool({b}); refinement is not nested")
    return pools


def meet_probs(gmaps, covs, bias, sd_draws):
    """Deployed predictive P(meet both specifications), averaged over the committed hierarchy draws."""
    return np.array([np.mean([gaussian_meet_prob(g + bias[k], C + sd_draws[k] + C_MEAS, SPEC)
                              for k in range(len(sd_draws))]) for g, C in zip(gmaps, covs)])


def branch_of(p, loading, lo, hi):
    """The stratified rule of the article, read off a pool: which of the three branches this product lands in."""
    inside = (loading >= lo) & (loading <= hi)
    if inside.any() and (p[inside] >= TAU_DEC).any():
        return "confirmatory-run", int(np.flatnonzero(inside)[np.argmax(p[inside])])
    if p.max() <= INFEASIBLE_EVERYWHERE:
        return "pool-redesign", int(np.argmax(p))
    return "validation-run", int(np.argmax(p))


def paired_certificate(gmaps, cov, r_pi, a_star, bias, sd_draws, idx, alpha, n_pool):
    """Proposition 3.8 on this pool, at the family size the pool itself implies."""
    out = {}
    for j in ("S", "F"):
        lf0, lc0, l20 = draw_arms(gmaps[a_star], cov["full"][a_star], cov[j][a_star],
                                  bias, sd_draws, idx, None, 0)
        shift0 = lc0 - lf0
        ok = True
        for a in range(n_pool):
            if a == a_star:
                continue
            lf, lc, l2a = draw_arms(gmaps[a], cov["full"][a], cov[j][a], bias, sd_draws, idx, None, 0)
            X = (lc - lf) - shift0
            ok &= bool(catoni_score(X, -r_pi[a], alpha, (l2a + l20) ** 2) > 0)
        out[j] = bool(ok)
    return out


def run(product, *, in_dir, hier_draws, jac_tag, n_mc, seed, sizes):
    post = Posterior.load(Path(in_dir) / f"{product}_correlated_posterior.npz")
    z = np.load(Path(in_dir) / f"inflation_jacobians_{product}{jac_tag}.npz")
    ops, Gs, gmaps = z["ops"], z["G"], z["g_map"]
    hd = np.load(hier_draws)
    bias, sd_draws = hd[f"b_{product}"], hd["Sd_draws"]
    n = post.n_protein
    lo, hi = DEPLOY[product]
    supp = support_counts(product, ops[:max(sizes)])

    cov = {"full": [], "S": [], "F": []}
    for G in Gs[:max(sizes)]:
        Sr, Gr, ui, vi = rotate_sigma_block(post.cov, G, n)
        cov["full"].append(decision_cov(Gr, Sr))
        cov["S"].append(decision_cov(Gr, schur_cov(Sr, ui, vi)))
        cov["F"].append(variant_d_cov(Sr, Gr, ui, vi))

    rng = np.random.default_rng(seed)
    idx = {"k": rng.integers(0, len(sd_draws), n_mc), "z": rng.standard_normal((n_mc, 2)),
           "u": rng.standard_normal((n_mc, 2))}

    rows = []
    for N in sizes:
        R = {k: np.array([predictive_risk(gmaps[a], cov[k][a], bias, sd_draws) for a in range(N)])
             for k in ("full", "S", "F")}
        order = np.argsort(R["full"])
        a_star = int(order[0])
        delta_act = float(R["full"][order[1]] - R["full"][a_star])
        r_pi = R["full"] - R["full"][a_star]
        p = meet_probs(gmaps[:N], cov["full"][:N], bias, sd_draws)
        loading = ops[:N, 0]
        branch, best = branch_of(p, loading, lo, hi)
        inside = (loading >= lo) & (loading <= hi)
        full = supp[:N] == ops.shape[1]                   # supported in every operating coordinate
        br_supp, best_supp = branch_of(p, np.where(full, loading, np.inf), lo, hi)

        env = {}
        for j in ("S", "F"):
            b_mix = np.array([mixture_bures(cov["full"][a], cov[j][a], sd_draws) for a in range(N)])
            e_an = np.array([eps_analytic(cov["full"][a], cov[j][a]) for a in range(N)])
            eb, ee = np.flatnonzero(r_pi <= b_mix), np.flatnonzero(r_pi <= e_an)
            a_j = int(np.argmin(R[j]))
            env[j] = {"bures_size": int(eb.size), "bures_certified": bool(eb.size == 1 and eb[0] == a_star),
                      "coupling_size": int(ee.size),
                      "coupling_certified": bool(ee.size == 1 and ee[0] == a_star),
                      "preserved": bool(a_j == a_star),
                      # which candidate the compressed law selects, so a preservation failure names it
                      "argmin_compressed": a_j,
                      "loading_argmin_compressed": float(ops[a_j, 0]),
                      "argmin_compressed_is_new": bool(a_j >= sizes[0])}
        alpha = DELTA / (2 * (N - 1))
        paired = paired_certificate(gmaps, cov, r_pi, a_star, bias, sd_draws, idx, alpha, N)
        for j in ("S", "F"):
            env[j]["paired_certified"] = paired[j]

        rows.append({
            "N": N, "a_star": a_star, "loading_a_star": float(loading[a_star]),
            "Delta_act": delta_act, "R_full_min": float(R["full"][a_star]),
            "branch": branch, "best_candidate": best, "loading_best": float(loading[best]),
            "p_best": float(p[best]), "p_max_all": float(p.max()),
            "p_max_deployment": float(p[inside].max()) if inside.any() else None,
            "n_deployment_domain": int(inside.sum()),
            "n_decisive": int((p >= TAU_DEC).sum()),
            "n_decisive_deployment": int((p[inside] >= TAU_DEC).sum()) if inside.any() else 0,
            "n_near_threshold": int((np.abs(p - TAU_DEC) <= NEAR_BAND).sum()),
            "support_a_star": int(supp[a_star]), "support_best": int(supp[best]),
            "branch_support_aware": br_supp, "best_support_aware": best_supp,
            "p_best_support_aware": float(p[best_supp]),
            "loading_best_support_aware": float(loading[best_supp]),
            "n_fully_supported": int(full.sum()),
            "alpha_paired": alpha, "by_reduction": env})
    return {"product": product, "deployment_domain": [lo, hi], "n_mc": n_mc, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=PRODUCTS)
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--hier-draws", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--jac-tag", default="_N48")
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    ap.add_argument("--n-mc", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-nesting-check", action="store_true")
    ap.add_argument("--out", default="results/bayes/pool_refinement.json")
    a = ap.parse_args()

    if not a.skip_nesting_check:
        for p in a.products:
            check_nesting(p, a.in_dir, seed=0)
        print(f"nesting verified: pools {a.sizes} are prefixes of one another on every product")

    res = [run(p, in_dir=a.in_dir, hier_draws=a.hier_draws, jac_tag=a.jac_tag,
               n_mc=a.n_mc, seed=a.seed, sizes=a.sizes) for p in a.products]
    Path(a.out).write_text(json.dumps(res, indent=1))

    print(f"\n{'product':14}{'N':>4}{'branch':>18}{'a*':>4}{'load a*':>9}{'Delta_act':>11}"
          f"{'best':>5}{'p_best':>8}{'dec':>5}{'near':>5}  {'pres S/F':>9} {'B S/F':>7} {'pair S/F':>9}")
    for r in res:
        for row in r["rows"]:
            e = row["by_reduction"]
            yn = lambda k: f"{'y' if e['S'][k] else 'n'}/{'y' if e['F'][k] else 'n'}"
            print(f"{r['product']:14}{row['N']:>4}{row['branch']:>18}{row['a_star']:>4}"
                  f"{row['loading_a_star']:9.1f}{row['Delta_act']:11.3e}{row['best_candidate']:>5}"
                  f"{row['p_best']:8.3f}{row['n_decisive']:>5}{row['n_near_threshold']:>5}  "
                  f"{yn('preserved'):>9} {yn('bures_certified'):>7} {yn('paired_certified'):>9}"
                  f"  supp(best)={row['support_best']}/4  supp-aware={row['branch_support_aware']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
