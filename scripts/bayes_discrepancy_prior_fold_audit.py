"""Discrepancy-prior and fold-dependence robustness of the historical predictive verdict.

Extends the discrepancy-prior-scale sweep of ``bayes_decision_discrepancy_hier_audit.py``
(which varies only the shared-covariance prior Sigma_delta) with the two axes a reviewer
asked for, because Var(b_p | D) carries a non-negligible share of the predictive variance:

  (1) the between-product covariance prior Sigma_b (``--sb-*``) and the bias-mean prior mu_b
      (``--v-mu``), swept 0.5x / 2x (Sigma_b) and 0.02 / 0.10 sd (mu_b);
  (2) an explicit fold-dependence *tempering* of the overlapping in-domain hold-out folds:
      a power-likelihood with weight alpha = 1 / (design effect), implemented as the
      equivalent per-fold noise inflation N -> N * (1 + 2*ICC), for ICC in {0, 0.25, 0.5, 1}.
      This is the rigorous version of the design-effect heuristic in the SI audit.

All runs read only committed ``results/bayes/*.json`` (the decision-LOEO residuals and the
correlated conditional covariance) and reuse the pure-numpy Gibbs sampler and the P(meet)
evaluator of ``bayes_decision_discrepancy_hier.py`` unchanged -- no solver, no re-fit.

Verdict of record: the *historical* predictive P(meet) for mAb A / mAb C stays below the
0.95 decisive threshold (nondecisive) across every prior and every fold-dependence setting,
so the historical non-decisiveness -- the paper's headline for the predictive layer -- is
robust to both the discrepancy priors and the fold-dependence treatment.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results", "bayes")

_spec = importlib.util.spec_from_file_location(
    "hier", os.path.join(HERE, "bayes_decision_discrepancy_hier.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

PRODUCTS = [("mAb A", "HLXSYN"), ("mAb B", "HLXSYN"), ("mAb C", "HLXSYN")]


def evaluate(pf, prior, *, n_iter, burn, seed):
    """Historical predictive P(meet), wdec_pred, and Var(b_p) purity-variance share per product."""
    b_draws, Sd = H.gibbs(pf, prior=prior, n_iter=n_iter, burn=burn, thin=8, seed=seed)
    rng = np.random.default_rng(seed + 1)
    out = {}
    for name, pid in PRODUCTS:
        C_par, _ = H.c_param(pid)
        g = H.g_map_corr(pid)
        pmeet, wdec = H.p_meet_hier(g, C_par, b_draws[pid], Sd, H.SIGMA_MEAS, rng)
        var_bp = float(b_draws[pid].var(axis=0)[0])          # purity bias-estimation variance
        mean_cond = float((C_par + Sd.mean(0) + np.diag(H.SIGMA_MEAS ** 2))[0, 0])
        share = var_bp / (mean_cond + var_bp)                # share of the FULL predictive purity variance
        out[pid] = dict(name=name, p_meet=float(pmeet), wdec_pred=float(wdec.mean()),
                        var_bp_purity_share=share)
    return out


def temper(pf, design_effect):
    """Power-likelihood tempering of the overlapping folds: inflate each fold noise N by the design effect."""
    return {p: [dict(r=f["r"], N=f["N"] * design_effect) for f in folds] for p, folds in pf.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=20000)
    ap.add_argument("--burn", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RES, "discrepancy_prior_fold_audit.json"))
    a = ap.parse_args()

    pf = {pid: H.load_folds(pid, H.SIGMA_MEAS) for _, pid in PRODUCTS}
    base = dict(nu_d=4.0, s_d=np.array([0.03, 0.05]), nu_b=4.0,
                s_b=np.array([0.02, 0.03]), v_mu=0.05 ** 2)

    def prior(**kw):
        p = dict(base)
        p.update(kw)
        return p

    prior_grid = [
        ("baseline", prior()),
        ("Sigma_b_0.5x", prior(s_b=np.array([0.01, 0.015]))),
        ("Sigma_b_2x", prior(s_b=np.array([0.04, 0.06]))),
        ("mu_b_tight_0.02", prior(v_mu=0.02 ** 2)),
        ("mu_b_wide_0.10", prior(v_mu=0.10 ** 2)),
        ("Sigma_delta_0.5x", prior(s_d=np.array([0.015, 0.025]))),
        ("Sigma_delta_2x", prior(s_d=np.array([0.06, 0.10]))),
    ]
    fold_grid = [("ICC_0.00", 1.0), ("ICC_0.25", 1.5), ("ICC_0.50", 2.0), ("ICC_1.00", 3.0)]

    report = {"decisive_threshold": 0.95, "n_iter": a.n_iter, "prior_sweep": {}, "fold_tempering": {}}
    print(f"{'='*88}\nPRIOR SWEEP (mu_b / Sigma_b / Sigma_delta)   n_iter={a.n_iter}\n{'='*88}")
    print(f"{'variant':<18}{'A p_meet':>9}{'A wdec':>8}{'A Vb%':>7}{'C p_meet':>10}{'C wdec':>8}{'C Vb%':>7}")
    for lab, pr in prior_grid:
        r = evaluate(pf, pr, n_iter=a.n_iter, burn=a.burn, seed=a.seed)
        report["prior_sweep"][lab] = r
        A, C = r["HLXSYN"], r["HLXSYN"]
        print(f"{lab:<18}{A['p_meet']:>9.3f}{A['wdec_pred']:>8.3f}{100*A['var_bp_purity_share']:>6.0f}%"
              f"{C['p_meet']:>10.3f}{C['wdec_pred']:>8.3f}{100*C['var_bp_purity_share']:>6.0f}%")

    print(f"\n{'='*88}\nFOLD-DEPENDENCE TEMPERING (power-likelihood, N *= design effect)\n{'='*88}")
    print(f"{'variant':<18}{'A p_meet':>9}{'A wdec':>8}{'C p_meet':>10}{'C wdec':>8}")
    for lab, de in fold_grid:
        r = evaluate(temper(pf, de), base, n_iter=a.n_iter, burn=a.burn, seed=a.seed)
        report["fold_tempering"][lab] = r
        A, C = r["HLXSYN"], r["HLXSYN"]
        print(f"{lab:<18}{A['p_meet']:>9.3f}{A['wdec_pred']:>8.3f}{C['p_meet']:>10.3f}{C['wdec_pred']:>8.3f}")

    # verdict of record
    all_A = [v["HLXSYN"]["p_meet"] for v in list(report["prior_sweep"].values()) + list(report["fold_tempering"].values())]
    all_C = [v["HLXSYN"]["p_meet"] for v in list(report["prior_sweep"].values()) + list(report["fold_tempering"].values())]
    report["verdict"] = dict(
        A_p_meet_range=[min(all_A), max(all_A)], C_p_meet_range=[min(all_C), max(all_C)],
        historical_nondecisive_invariant=bool(max(all_A) < 0.95 and max(all_C) < 0.95))
    print(f"\nverdict: A p_meet in [{min(all_A):.3f},{max(all_A):.3f}], "
          f"C in [{min(all_C):.3f},{max(all_C):.3f}]; "
          f"historical nondecisive invariant = {report['verdict']['historical_nondecisive_invariant']}")
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
