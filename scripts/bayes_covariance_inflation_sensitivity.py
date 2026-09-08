"""Action-level sensitivity of every deployed verdict to the width of the working Gaussian law.

The deployed posterior is a Gauss-Newton local Gaussian law centred at a point that is not stationary, so
its width is an approximation and not a sampled posterior.  The one sampling anchor available -- Hamiltonian
Monte Carlo on mAb A's main conditional slice -- returns a worst-direction decision width of 0.803 where the
deployed law reads 0.543.  This asks the only question that matters for the paper: if the working law were
that much wider, would any of the three products' conclusions change?

The inflation is a scalar on the parameter covariance, Sigma -> kappa^2 Sigma, with kappa the ratio of those
two widths.  Because the decision covariance is G Sigma G^T, that scales the whitened decision width by
exactly kappa at every candidate, so the anchor's discrepancy is reproduced by construction.  It is applied
uniformly to all three products, which is deliberately conservative twice over: the anchor is on one product,
and the decision-active anchor on the same product points the other way (0.501 against 0.543).

Two verdicts are recomputed at each kappa, both at the committed 24-candidate pool:

  * the DEPLOYED predictive read -- P(meet) marginal over the committed hierarchy draws, the best in-domain
    candidate, the emitted action, Pr(move | D), and the tau_dec classification over the pool and over the
    deployment domain.  This is the verdict the article reports.
  * the CONDITIONAL-law compression certificate of main text Corollary 3.5 -- shortfall risks under the full
    and the two compressed posteriors, the action gap, the transport ratios and the action envelopes.

Only the Jacobians need the solver, and only once: every kappa reuses them.  Stage 1 writes them to
``results/bayes/inflation_jacobians_{product}.npz``; stage 2 is pure linear algebra.

    OMP_NUM_THREADS=4 python scripts/bayes_covariance_inflation_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import cex_model.app_support as A
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.decision import decision_jacobian
from cex_model.bayes.decision_compression import (
    bures_w2,
    decision_cov,
    rotate_sigma_block,
    schur_cov,
    shortfall_risk,
    variant_d_cov,
    whiten,
)
from cex_model.bayes.decision_window import (
    candidate_ops_for,
    gaussian_meet_prob,
    worst_dec_from_cov,
)
from cex_model.bayes.posterior import Posterior

_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
PRODUCTS = ['HLXSYN']
SPEC = np.array([0.70, 0.50])
W = np.array([1.0, 1.0])
C_MEAS = np.diag(np.array([0.005, 0.008]) ** 2)
TAU_DEC = 0.95
LOAD_CAP = 35.0
DEPLOY_DOMAIN = (25.0, 35.0)


def _load(product, in_dir):
    post = Posterior.load(Path(in_dir) / f"{product}_correlated_posterior.npz")
    dj = json.loads((Path(in_dir) / f"{product}_decision.json").read_text())
    tol = np.asarray(dj.get("tol", [0.02, 0.05]), float)
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    op = json.loads((Path(in_dir) / f"{product}_decision_window.json").read_text())["decision_op"]
    return post, bundle, tol, [float(x) for x in op]


def stage_jacobians(product, *, in_dir, out_dir, n_steps, n_candidates, seed, tag=""):
    """One decision Jacobian per candidate plus the historical op; the only solver work here."""
    post, bundle, _, hist_op = _load(product, in_dir)
    ops = [list(map(float, o)) for o in candidate_ops_for(bundle, n_candidates=n_candidates, seed=seed)]
    ops.append(hist_op)                              # last row is always the historical condition
    Gs, gs = [], []
    for i, op in enumerate(ops):
        G, g_map, _ = decision_jacobian(bundle, op, post.u_map, n_steps=n_steps, return_extra=True)
        Gs.append(np.atleast_2d(G))
        gs.append(np.asarray(g_map, float))
        print(f"  [{product}] op {i + 1}/{len(ops)} loading={op[0]:.2f}  g={np.round(gs[-1], 4).tolist()}",
              flush=True)
    path = Path(out_dir) / f"inflation_jacobians_{product}{tag}.npz"
    np.savez_compressed(path, ops=np.asarray(ops, float), G=np.asarray(Gs, float),
                        g_map=np.asarray(gs, float), n_steps=n_steps)
    print(f"  wrote {path}")


def _predictive(g_map, C_par, bias, sd):
    """Per-hierarchy-draw P(meet) at one candidate under the deployed predictive law."""
    return np.array([gaussian_meet_prob(g_map + bias[k], C_par + sd[k] + C_MEAS, SPEC)
                     for k in range(len(sd))])


def _breakeven(idx, post, Gs, gmaps, bias, sd, *, lo=1.0, hi=6.0, iters=14):
    """The inflation at which candidate ``idx`` stops clearing tau_dec, by bisection.

    Reported instead of only the verdict at one kappa, because the useful statement is how much room a
    recommendation has: P(meet) is monotone decreasing in the width at a fixed candidate, so the crossing is
    unique.  ``None`` if the candidate does not clear the gate even at kappa = lo.
    """
    n = post.n_protein

    def p_at(kappa):
        Sr, Gr, ui, vi = rotate_sigma_block(float(kappa) ** 2 * post.cov, Gs[idx], n)
        C = decision_cov(Gr, Sr)
        return float(_predictive(gmaps[idx], C, bias, sd).mean())

    if p_at(lo) < TAU_DEC:
        return None
    if p_at(hi) >= TAU_DEC:
        return float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if p_at(mid) >= TAU_DEC else (lo, mid)
    return 0.5 * (lo + hi)


def analyse(product, *, in_dir, kappas, hier_draws):
    post, _, tol, hist_op = _load(product, in_dir)
    z = np.load(Path(in_dir) / f"inflation_jacobians_{product}.npz")
    ops, Gs, gmaps = z["ops"], z["G"], z["g_map"]
    n_pool = len(ops) - 1                            # the last row is the historical op
    hd = np.load(hier_draws)
    bias, sd = hd[f"b_{product}"], hd["Sd_draws"]
    n = post.n_protein
    out = {"product": product, "n_candidates": n_pool, "hier_draws": str(hier_draws),
           "n_hier_draws": int(len(sd)), "decision_op": hist_op, "tol": tol.tolist(),
           "wdec_cond_deployed": None, "kappa_breakeven_recommended": "unset", "by_kappa": {}}

    for kappa in kappas:
        Sigma = float(kappa) ** 2 * post.cov
        laws = []
        for G, g in zip(Gs, gmaps):
            Sr, Gr, ui, vi = rotate_sigma_block(Sigma, G, n)
            laws.append({"g_map": g, "full": decision_cov(Gr, Sr),
                         "S": decision_cov(Gr, schur_cov(Sr, ui, vi)),
                         "D": variant_d_cov(Sr, Gr, ui, vi)})
        pool, hist = laws[:n_pool], laws[n_pool]

        # --- the deployed predictive read -------------------------------------------------------
        pd = [_predictive(d["g_map"], d["full"], bias, sd) for d in pool]
        hist_pd = _predictive(hist["g_map"], hist["full"], bias, sd)
        p_meet = np.array([v.mean() for v in pd])
        p_dec = np.array([float(np.mean(v >= TAU_DEC)) for v in pd])
        load = ops[:n_pool, 0]
        in_dom = load <= LOAD_CAP
        in_dep = (load >= DEPLOY_DOMAIN[0]) & (load <= DEPLOY_DOMAIN[1])
        pick = int(np.flatnonzero(in_dom)[np.argmax(p_meet[in_dom])]) if in_dom.any() else None
        hist_dec = float(np.mean(hist_pd >= TAU_DEC))
        rec = None
        if pick is not None:
            move = float(np.mean((hist_pd < TAU_DEC) & (pd[pick] >= TAU_DEC)))
            action = ("move-operating-point" if move >= 0.5 else
                      ("operate-as-is" if hist_dec >= 0.5 else "prospective-or-abstain"))
            rec = {"idx": pick, "loading": float(load[pick]), "p_meet": float(p_meet[pick]),
                   "p_action_decisive": float(p_dec[pick]), "p_move_given_D": move}
        else:
            action = "redesign-pool" if hist_pd.mean() < 0.5 else "abstain"
        best_dep = int(np.flatnonzero(in_dep)[np.argmax(p_meet[in_dep])]) if in_dep.any() else None

        # --- the conditional-law compression certificate ---------------------------------------
        eye = np.eye(2)
        risks, pm_cond = {}, {}
        for j in ("full", "S", "D"):
            Amat = np.array([[shortfall_risk(d["g_map"] / tol, whiten(d[j], tol), SPEC / tol, eye[q])
                              for q in range(2)] for d in pool])
            risks[j] = Amat @ W
            pm_cond[j] = np.array([gaussian_meet_prob(d["g_map"], d[j], SPEC) for d in pool])
        rf = risks["full"]
        order = np.argsort(rf)
        Delta_act = float(rf[order[1]] - rf[order[0]])
        r_full = rf - rf[order[0]]
        argmin_full = set(np.flatnonzero(r_full <= 0.0).tolist())
        L = float(np.linalg.norm(W))
        tf = {"Delta_act": Delta_act, "argmin_full": int(order[0]),
              "pool_max_regret": float(r_full.max())}
        for j in ("S", "D"):
            bnd = L * np.array([bures_w2(whiten(d["full"], tol), whiten(d[j], tol)) for d in pool])
            env = np.flatnonzero(r_full <= bnd)
            tf[j] = {"eta_bound": float(bnd.max()),
                     "ratio_bound_onesided": float(bnd.max() / Delta_act) if Delta_act > 0 else float("inf"),
                     "envelope_size": int(env.size),
                     "envelope_in_argmin": bool(set(env.tolist()) <= argmin_full),
                     "argmin_compressed": int(np.argmin(risks[j])),
                     "max_Delta": float(max(np.linalg.norm(whiten(d["full"] - d[j], tol), "fro")
                                            for d in pool))}

        rec_wdec = worst_dec_from_cov(hist["full"] + sd.mean(axis=0) + C_MEAS, tol)
        if rec is not None and out.get("kappa_breakeven_recommended", "unset") == "unset":
            out["kappa_breakeven_recommended"] = _breakeven(rec["idx"], post, Gs, gmaps, bias, sd)
        wd_cond = worst_dec_from_cov(hist["full"], tol)
        if kappa == 1.0:
            out["wdec_cond_deployed"] = wd_cond
        out["by_kappa"][f"{kappa:.6f}"] = {
            "kappa": float(kappa),
            "wdec_cond_historical": wd_cond, "wdec_pred_historical": rec_wdec,
            "historical": {"p_meet": float(hist_pd.mean()), "p_action_decisive": hist_dec},
            "recommended_candidate": rec, "action": action,
            "pool_argmax": {"idx": int(np.argmax(p_meet)), "loading": float(load[np.argmax(p_meet)]),
                            "p_meet": float(p_meet.max())},
            "best_in_deployment_domain": (None if best_dep is None else
                                          {"idx": best_dep, "loading": float(load[best_dep]),
                                           "p_meet": float(p_meet[best_dep])}),
            "n_decisive_pool": int(np.sum(p_meet >= TAU_DEC)),
            "n_decisive_deployment_domain": int(np.sum((p_meet >= TAU_DEC) & in_dep)),
            "decisive_idx": np.flatnonzero(p_meet >= TAU_DEC).tolist(),
            "p_meet": p_meet.tolist(), "loading": load.tolist(),
            "conditional": tf,
            "conditional_maxP": {j: float(pm_cond[j].max()) for j in ("full", "S", "D")},
        }
    if out["kappa_breakeven_recommended"] == "unset":
        out["kappa_breakeven_recommended"] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=PRODUCTS)
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--stage", choices=["jacobians", "analyse", "all"], default="all")
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jac-tag", default="",
                    help="filename suffix for the Jacobian cache, so a denser pool can be cached "
                         "without overwriting the deployed 24-candidate one")
    ap.add_argument("--hier-draws", default="results/bayes/hier_draws_capB.npz")
    ap.add_argument("--kappa", type=float, default=None,
                    help="width inflation factor; default is the mAb A anchor ratio "
                         "wdec_nuts / wdec_deployed read from HLXSYN_correlated_nuts.json")
    ap.add_argument("--extra-kappa", type=float, nargs="*", default=[1.2, 1.6, 1.8, 2.0, 2.5, 3.0],
                    help="further inflation factors, so the verdict can be read as a curve rather "
                         "than at one point; 1.0 and the anchor ratio are always included")
    ap.add_argument("--out", default="results/bayes/covariance_inflation_sensitivity.json")
    a = ap.parse_args()

    if a.stage in ("jacobians", "all"):
        for p in a.products:
            stage_jacobians(p, in_dir=a.in_dir, out_dir=a.in_dir, n_steps=a.n_steps,
                            n_candidates=a.n_candidates, seed=a.seed, tag=a.jac_tag)
    if a.stage == "jacobians":
        return

    kappa = a.kappa
    anchor = json.loads((Path(a.in_dir) / "HLXSYN_correlated_nuts.json").read_text())
    if kappa is None:
        kappa = anchor["decision_nuts"]["wdec"] / anchor["deployed_reference"]["wdec_corr"]
    grid = sorted({1.0, float(kappa), *(float(x) for x in a.extra_kappa)})
    res = {"kappa": float(kappa), "kappa_grid": grid,
           "anchor": {"product": "HLXSYN", "slice": anchor.get("sampled"),
                      "wdec_nuts": anchor["decision_nuts"]["wdec"],
                      "wdec_deployed_reference": anchor["deployed_reference"]["wdec_corr"]},
           "tau_dec": TAU_DEC, "spec": SPEC.tolist(), "w": W.tolist(),
           "deployment_domain": list(DEPLOY_DOMAIN), "load_cap": LOAD_CAP,
           "products": {}}
    for p in a.products:
        res["products"][p] = analyse(p, in_dir=a.in_dir, kappas=grid, hier_draws=a.hier_draws)
    Path(a.out).write_text(json.dumps(res, indent=1))

    tags = [f"{k:.6f}" for k in grid]
    print(f"\n=== decision-width inflation; the anchor ratio is kappa = {kappa:.4f} "
          f"(Sigma x {kappa ** 2:.3f}) ===")
    hdr = (f"{'product':14}{'kappa':>7}{'wdec_cond':>11}{'hist P':>9}{'best dep':>10}{'P':>8}"
           f"{'pool max':>10}{'P':>8}{'dec pool':>10}{'action':>24}")
    print(hdr)
    for p in a.products:
        for tag in tags:
            b = res["products"][p]["by_kappa"][tag]
            bd = b["best_in_deployment_domain"]
            print(f"{p:14}{b['kappa']:7.3f}{b['wdec_cond_historical']:11.3f}"
                  f"{b['historical']['p_meet']:9.3f}"
                  f"{(bd['loading'] if bd else float('nan')):10.1f}{(bd['p_meet'] if bd else float('nan')):8.3f}"
                  f"{b['pool_argmax']['loading']:10.1f}{b['pool_argmax']['p_meet']:8.3f}"
                  f"{b['n_decisive_pool']:10d}{b['action']:>24}")
    print(f"\n{'product':14}{'kappa':>7}{'Delta_act':>12}{'argmin':>8}"
          f"{'eta_S/D':>10}{'eta_F/D':>10}{'|C_S|':>7}{'|C_F|':>7}{'maxP cond':>11}")
    for p in a.products:
        for tag in tags:
            b = res["products"][p]["by_kappa"][tag]
            c = b["conditional"]
            print(f"{p:14}{b['kappa']:7.3f}{c['Delta_act']:12.3e}{c['argmin_full']:8d}"
                  f"{c['S']['ratio_bound_onesided']:10.2f}{c['D']['ratio_bound_onesided']:10.2f}"
                  f"{c['S']['envelope_size']:7d}{c['D']['envelope_size']:7d}"
                  f"{b['conditional_maxP']['full']:11.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
