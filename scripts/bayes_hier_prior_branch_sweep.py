"""Does the engineering branch survive the hierarchy-prior sweep?

bayes_discrepancy_prior_fold_audit.py sweeps the discrepancy hierarchy's priors (bias mean, between-
product covariance, shared discrepancy covariance) and a fold-dependence tempering, and reports the
historical-condition P(meet). This script runs the same sweep and reads, under each variant, what the
article's stratified rule returns on the deployed 24-candidate pool: the specification probability of
every candidate, the best deployable candidate, the pool maximizer, the branch, and the probability
at the candidates the article reports. Everything is post-processing: the Gibbs sampler of
bayes_decision_discrepancy_hier.py on the committed fold residuals, and the canonical decision
Jacobians cached by the fitted-point closure (results/bayes/multistart/{P}_r0_jacobians.npz, the
canonical fit). No solver call.

    python scripts/bayes_hier_prior_branch_sweep.py [--n-iter 20000 --burn 4000 --seed 0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bayes_pool_refinement import DEPLOY, TAU_DEC, branch_of, meet_probs          # noqa: E402
from cex_model.bayes.decision_compression import decision_cov                     # noqa: E402
from cex_model.bayes.posterior import Posterior                                   # noqa: E402

RES = HERE.parent / "results" / "bayes"
_spec = importlib.util.spec_from_file_location("hier", HERE / "bayes_decision_discrepancy_hier.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

PRODUCTS = ['HLXSYN']


def _laws(pid):
    """Canonical decision means and covariances on the deployed pool, historical condition excluded."""
    post = Posterior.load(str(RES / f"{pid}_correlated_posterior.npz"))
    z = np.load(RES / "multistart" / f"{pid}_r0_jacobians.npz")
    n_pool = len(z["ops"]) - 1
    return z["ops"][:n_pool, 0], z["g_map"][:n_pool], [decision_cov(G, post.cov) for G in z["G"][:n_pool]]


def read_branch(pid, loading, gmaps, covs, bias, sd):
    p = meet_probs(gmaps, covs, bias, sd)
    lo, hi = DEPLOY[pid]
    br, best = branch_of(p, loading, lo, hi)
    inside = (loading >= lo) & (loading <= hi)
    return dict(branch=br, best_candidate=int(best), loading_best=float(loading[best]), p_best=float(p[best]),
                p_max_deployment=float(p[inside].max()) if inside.any() else None,
                loading_max_deployment=float(loading[inside][np.argmax(p[inside])]) if inside.any() else None,
                p_max_all=float(p.max()), loading_max_all=float(loading[int(np.argmax(p))]),
                n_decisive=int((p >= TAU_DEC).sum()), p=p.tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-iter", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RES / "hier_prior_branch_sweep.json"))
    a = ap.parse_args()

    # the grid of bayes_discrepancy_prior_fold_audit.py, verbatim
    pf = {pid: H.load_folds(pid, H.SIGMA_MEAS) for pid in PRODUCTS}
    base = dict(nu_d=4.0, s_d=np.array([0.03, 0.05]), nu_b=4.0, s_b=np.array([0.02, 0.03]), v_mu=0.05 ** 2)

    def prior(**kw):
        p = dict(base); p.update(kw); return p

    def temper(folds, design_effect):
        return {p: [dict(r=f["r"], N=f["N"] * design_effect) for f in fl] for p, fl in folds.items()}

    variants = [("baseline", prior(), pf),
                ("Sigma_b_0.5x", prior(s_b=np.array([0.01, 0.015])), pf),
                ("Sigma_b_2x", prior(s_b=np.array([0.04, 0.06])), pf),
                ("mu_b_tight_0.02", prior(v_mu=0.02 ** 2), pf),
                ("mu_b_wide_0.10", prior(v_mu=0.10 ** 2), pf),
                ("Sigma_delta_0.5x", prior(s_d=np.array([0.015, 0.025])), pf),
                ("Sigma_delta_2x", prior(s_d=np.array([0.06, 0.10])), pf),
                ("ICC_0.25", base, temper(pf, 1.5)),
                ("ICC_0.50", base, temper(pf, 2.0)),
                ("ICC_1.00", base, temper(pf, 3.0))]

    laws = {pid: _laws(pid) for pid in PRODUCTS}
    # the deployed draw set, so the baseline variant can be compared with the article's numbers
    hd = np.load(RES / "hier_draws_capB.npz")
    deployed = {pid: read_branch(pid, *laws[pid], hd[f"b_{pid}"], hd["Sd_draws"]) for pid in PRODUCTS}
    for pid in PRODUCTS:
        deployed[pid].pop("p")

    report = dict(n_iter=a.n_iter, burn=a.burn, thin=8, seed=a.seed, decisive_threshold=TAU_DEC,
                  jacobians="results/bayes/multistart/{P}_r0_jacobians.npz (the canonical fit)",
                  deployed_draws=deployed, variants={})
    print(f"{'variant':<18}" + "".join(f"{pid:>16}{'P best':>8}{'P pool':>8}" for pid in PRODUCTS))
    for lab, pr, folds in variants:
        b_draws, Sd = H.gibbs(folds, prior=pr, n_iter=a.n_iter, burn=a.burn, thin=8, seed=a.seed)
        row = {pid: read_branch(pid, *laws[pid], b_draws[pid], Sd) for pid in PRODUCTS}
        report["variants"][lab] = row
        print(f"{lab:<18}" + "".join(f"{row[pid]['branch']:>16}{row[pid]['p_best']:>8.3f}{row[pid]['p_max_all']:>8.3f}"
                                     for pid in PRODUCTS), flush=True)

    branches = {pid: sorted({v[pid]["branch"] for v in report["variants"].values()}) for pid in PRODUCTS}
    best = {pid: sorted({round(v[pid]["loading_best"], 1) for v in report["variants"].values()}) for pid in PRODUCTS}
    report["verdict"] = dict(branches=branches, loading_best=best,
                             branch_invariant=all(len(b) == 1 for b in branches.values()))
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("\nbranches across the sweep:", branches)
    print("selected loadings across the sweep:", best)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
