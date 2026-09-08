"""What each fitted point does to the decision, and whether compression survives at each of them.

The six-start audit (bayes_multistart_action_stability.py) found that on mAb A two of the six fitted
points, the two with the lowest objective, select a different in-domain candidate and a different
engineering branch. That leaves three questions open, and this script answers them with the same
laws, risks and rule the article deploys:

  * Is compression stable at every fitted point? Per restart, the full, conditional-mean (S) and
    fixed-block (F) laws are built exactly as bayes_route_b_predictive_action.py and
    bayes_pool_refinement.py build them, and the shortfall minimizer, the gate classification of
    every candidate, the probability maximizer and the engineering branch are read under each.
  * Which fitted point carries the posterior mass, and which parameters moved? Log-mass of the
    Gauss-Newton quadratic model at each fitted point relative to the canonical fit: the objective,
    the Newton-decrement term 0.5 g' Sigma g that a non-stationary stop leaves behind (none of the six
    stops is stationary), and half the log-determinant. Then the prior-whitened displacement split
    into the steric common mode, the steric differential block the article compresses, and the
    retained block, with its projection onto the decision-active channel at the candidates in play.
  * Are the fitted points separate basins or stops along one ridge? The correlated negative log
    posterior evaluated along the straight segment between two fitted points.

The hierarchy draws and the fitted residual covariance are held at the deployed set throughout.

Stages (run in this order):
  jacobians  RUN ON COLAB. One decision Jacobian per candidate (the deployed 24-candidate pool plus
             the historical condition, last) at each restart's fitted point, in the format of the
             deployed inflation_jacobians_{P}.npz. About 25 s per Jacobian.
  gradients  Solver. The objective and its gradient at each restart's fitted point; one forward and
             one backward solve per restart.
  segment    RUN ON COLAB. The objective the restarts minimised, on a straight segment between two
             fitted points. One forward solve per point.
  analyse    Solver-free. Reads the restart posteriors and the caches above.
  mixture    Solver-free. One predictive law mixing the six fitted points, each weighted by its
             quadratic-model mass, and the decision read from it.

    python scripts/bayes_basin_closure.py --stage jacobians --product HLXSYN --restarts 0 1 2 3 4 5
    python scripts/bayes_basin_closure.py --stage gradients --product HLXSYN --restarts 0 1 2 3 4 5
    python scripts/bayes_basin_closure.py --stage segment   --product HLXSYN --from-restart 0 --to-restart 4
    python scripts/bayes_basin_closure.py --stage analyse   --product HLXSYN
    python scripts/bayes_basin_closure.py --stage mixture   --product HLXSYN
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bayes_pool_refinement import DEPLOY, TAU_DEC, _bundle, branch_of, meet_probs          # noqa: E402
from bayes_route_b_predictive_action import eps_analytic, mixture_bures, predictive_risk  # noqa: E402
from bayes_route_b_predictive_action import TOL                                           # noqa: E402
from cex_model.bayes.decision_compression import (                                         # noqa: E402
    decision_cov, rotate_sigma_block, schur_cov, variant_d_cov)
from cex_model.bayes.posterior import Posterior                                            # noqa: E402

RES = Path(__file__).resolve().parent.parent / "results" / "bayes"
MS = RES / "multistart"


def _posterior(product, k):
    return Posterior.load(str(MS / f"{product}_r{k}_posterior.npz"))


def _recorded_objective(ms, k):
    """The objective at the restart's fitted point. map_loss_last is the last line-search trial."""
    r = ms["restarts"][k]
    return float((r.get("optimizer_termination") or {}).get("final_loss", r["map_loss_last"]))


def _pool_ops(product):
    """The deployed scan's pool, then the historical condition, so caches match inflation_jacobians_{P}."""
    scan = json.loads((RES / f"{product}_decision_window_predictive_hier.json").read_text())
    return [list(map(float, r["op"])) for r in scan["rows"]] + [list(map(float, scan["decision_op"]))]


# ------------------------------------------------------------------------------------------ jacobians
def stage_jacobians(product, *, restarts, n_steps, candidates, out_dir):
    from cex_model.bayes.decision import decision_jacobian
    bundle = _bundle(product)
    ops_all = _pool_ops(product)
    sel = list(range(len(ops_all))) if candidates is None else list(candidates)
    for k in restarts:
        post = _posterior(product, k)
        Gs, gs = [], []
        for i in sel:
            op = ops_all[i]
            G, g_map, _ = decision_jacobian(bundle, op, post.u_map, n_steps=n_steps, return_extra=True)
            Gs.append(np.atleast_2d(G))
            gs.append(np.asarray(g_map, float).ravel()[:2])
            print(f"  [{product} r{k}] op {i} loading={op[0]:.2f}  g={np.round(gs[-1], 4).tolist()}", flush=True)
        path = Path(out_dir) / f"{product}_r{k}_jacobians{'' if candidates is None else '_subset'}.npz"
        np.savez_compressed(path, ops=np.asarray([ops_all[i] for i in sel], float), G=np.asarray(Gs, float),
                            g_map=np.asarray(gs, float), n_steps=n_steps, restart=k,
                            candidates=np.asarray(sel, int))
        print(f"  wrote {path}")


# -------------------------------------------------------------------------------------------- segment
def _objective(product, n_steps):
    """The correlated negative log posterior of the multistart, with its shared noise model."""
    import torch
    from bayes_correlated_refit import _blocks_from_targets, _corr_for, unit_block
    from cex_model.bayes import mechanistic_model, physical_prior
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle
    from cex_model.diffsolver.torch_solver import DTYPE

    hyper = json.loads((RES / f"{product}_correlated_multistart.json").read_text())["hyper"]
    bundle = _bundle(product)
    targets = targets_from_bundle(bundle, n_steps)
    n = bundle.components.n_protein
    predict_fn, obs = mechanistic_model(targets, n)
    prior = physical_prior(n)
    blocks = _blocks_from_targets(targets)
    Kinv, idx_t = [], []
    for b in blocks:
        Sig = hyper["sigma2"] * unit_block(_corr_for(hyper["kernel"], b["times"], hyper["ell"]), hyper["rho"])
        Kinv.append(torch.tensor(np.linalg.inv(Sig), dtype=DTYPE))
        idx_t.append(torch.tensor(b["idx"]))

    def _f(u):
        resid = predict_fn(u) - obs
        q = u.new_zeros(())
        for it_, Ki in zip(idx_t, Kinv):
            rb = resid[it_]
            q = q + rb @ (Ki @ rb)
        return 0.5 * q - prior.log_prob(u)

    def neglogpost(u_np):
        with torch.no_grad():
            return float(_f(torch.tensor(np.asarray(u_np, float), dtype=DTYPE)))

    def neglogpost_grad(u_np):
        u = torch.tensor(np.asarray(u_np, float), dtype=DTYPE, requires_grad=True)
        val = _f(u)
        (g,) = torch.autograd.grad(val, u)
        return float(val.detach()), g.detach().cpu().numpy()

    return neglogpost, neglogpost_grad, np.asarray(prior.std, float)


def stage_gradients(product, *, restarts, n_steps, out_dir):
    """Objective and gradient at each fitted point, and the Newton-decrement term the mass needs."""
    ms = json.loads((RES / f"{product}_correlated_multistart.json").read_text())
    _, fg, sp = _objective(product, n_steps)
    for k in restarts:
        post = _posterior(product, k)
        f, g = fg(post.u_map)
        c = float(0.5 * g @ post.cov @ g)
        rec = _recorded_objective(ms, k)
        path = Path(out_dir) / f"{product}_r{k}_gradient.npz"
        np.savez_compressed(path, grad=g, objective=f, recorded_objective=rec, newton_term=c,
                            grad_norm_whitened=float(np.linalg.norm(sp * g)), n_steps=n_steps, restart=k)
        print(f"  [{product} r{k}] objective {f:.6f} (recorded {rec:.6f})  |grad|_w {np.linalg.norm(sp * g):.3f}  "
              f"0.5 g'Sigma g = {c:.4f}", flush=True)
        print(f"  wrote {path}")


def stage_segment(product, *, a, b, n_points, n_steps, out):
    ms = json.loads((RES / f"{product}_correlated_multistart.json").read_text())
    ua, ub = _posterior(product, a).u_map, _posterior(product, b).u_map
    f, _, sp = _objective(product, n_steps)
    ts = np.linspace(0.0, 1.0, n_points)
    loss = []
    for t in ts:
        loss.append(f(ua + t * (ub - ua)))
        print(f"  [{product}] t={t:.3f}  objective={loss[-1]:.4f}", flush=True)
    loss = np.array(loss)
    rec = {k: _recorded_objective(ms, k) for k in (a, b)}
    chord = loss[0] + ts * (loss[-1] - loss[0])
    res = dict(product=product, from_restart=a, to_restart=b, n_steps=n_steps, t=ts.tolist(),
               objective=loss.tolist(), recorded_endpoints=rec,
               endpoint_abs_error=[abs(loss[0] - rec[a]), abs(loss[-1] - rec[b])],
               whitened_length=float(np.linalg.norm((ub - ua) / sp)),
               max_rise_above_start=float((loss - loss[0]).max()),
               max_rise_above_chord=float((loss - chord).max()),
               monotone_decreasing=bool(np.all(np.diff(loss) <= 1e-6)))
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"  endpoints reproduce the recorded objectives to {max(res['endpoint_abs_error']):.4f}; "
          f"max rise above the start {res['max_rise_above_start']:.4f}; "
          f"{'monotone' if res['monotone_decreasing'] else 'not monotone'}")
    print(f"wrote {out}")


# -------------------------------------------------------------------------------------------- analyse
def _laws(post, Gs, n):
    cov = {"full": [], "S": [], "F": []}
    for G in Gs:
        Sr, Gr, ui, vi = rotate_sigma_block(post.cov, G, n)
        cov["full"].append(decision_cov(Gr, Sr))
        cov["S"].append(decision_cov(Gr, schur_cov(Sr, ui, vi)))
        cov["F"].append(variant_d_cov(Sr, Gr, ui, vi))
    return cov


def _active_share(dw, G, sp):
    """Fraction of the whitened displacement lying in the row space of the whitened Jacobian."""
    den = float(np.sum(dw ** 2))
    if den == 0.0:
        return 0.0
    Gw = np.atleast_2d(G) * sp[None, :]
    coef = np.linalg.lstsq(Gw.T, dw, rcond=None)[0]
    return float(np.sum((Gw.T @ coef) ** 2) / den)


def stage_analyse(product, *, restarts, hier_draws, out):
    from cex_model.bayes import physical_prior
    ms = json.loads((RES / f"{product}_correlated_multistart.json").read_text())
    hd = np.load(hier_draws)
    bias, sd = hd[f"b_{product}"], hd["Sd_draws"]
    lo, hi = DEPLOY[product]

    post0 = _posterior(product, 0)
    n = post0.n_protein
    sp = np.asarray(physical_prior(n).std, float)
    z0 = np.load(MS / f"{product}_r0_jacobians.npz")
    ops = z0["ops"]
    pool = np.asarray(_pool_ops(product), float)
    assert ops.shape == pool.shape and np.allclose(ops, pool), "restart 0 cache is not the deployed 24+1 pool"
    n_pool = len(ops) - 1
    loading = ops[:n_pool, 0]

    def _mass_terms(k, post):
        g = MS / f"{product}_r{k}_gradient.npz"
        if not g.exists():
            raise SystemExit(f"{g} absent; run --stage gradients first (the mass needs the Newton term)")
        zg = np.load(g)
        return _recorded_objective(ms, k), float(zg["newton_term"]), float(np.linalg.slogdet(post.cov)[1])

    nlp0, c0, logdet0 = _mass_terms(0, post0)
    # the sigma block shares one prior scale, so the fixed rotation commutes with the whitening
    assert np.allclose(sp[3 * n:4 * n], sp[3 * n]), "sigma block prior scales differ"

    rows = []
    for k in restarts:
        post = _posterior(product, k)
        z = np.load(MS / f"{product}_r{k}_jacobians.npz")
        assert z["ops"].shape == ops.shape and np.allclose(z["ops"], ops), \
            f"restart {k} was cached on a different pool"
        Gs, gmaps = z["G"], z["g_map"]
        cov = _laws(post, Gs[:n_pool], n)
        R = {j: np.array([predictive_risk(gmaps[a], cov[j][a], bias, sd) for a in range(n_pool)])
             for j in cov}
        P = {j: meet_probs(gmaps[:n_pool], cov[j][:n_pool], bias, sd) for j in cov}
        a_star = int(np.argmin(R["full"]))
        r_pi = R["full"] - R["full"][a_star]
        law = {}
        for j in cov:
            br, best = branch_of(P[j], loading, lo, hi)
            inside = (loading >= lo) & (loading <= hi)
            law[j] = dict(a_star=int(np.argmin(R[j])), loading_a_star=float(loading[np.argmin(R[j])]),
                          branch=br, best=best, loading_best=float(loading[best]), p_best=float(P[j][best]),
                          p_max_deployment=float(P[j][inside].max()) if inside.any() else None,
                          n_decisive=int((P[j] >= TAU_DEC).sum()),
                          classifications_changed=int(((P[j] >= TAU_DEC) != (P["full"] >= TAU_DEC)).sum()),
                          minimizer_preserved=bool(np.argmin(R[j]) == a_star),
                          maximizer_preserved=bool(np.argmax(P[j]) == np.argmax(P["full"])),
                          branch_preserved=bool(br == law["full"]["branch"]) if j != "full" else True)
            if j != "full":
                b_mix = np.array([mixture_bures(cov["full"][a], cov[j][a], sd) for a in range(n_pool)])
                e_an = np.array([eps_analytic(cov["full"][a], cov[j][a]) for a in range(n_pool)])
                eb, ee = np.flatnonzero(r_pi <= b_mix), np.flatnonzero(r_pi <= e_an)
                law[j]["bures_certified"] = bool(eb.size == 1 and eb[0] == a_star)
                law[j]["coupling_certified"] = bool(ee.size == 1 and ee[0] == a_star)
        nlp, c, logdet = _mass_terms(k, post)
        dw = (post.u_map - post0.u_map) / sp
        _, dwr, ui, vi = rotate_sigma_block(np.eye(len(dw)), dw[None, :], n)
        dwr = dwr[0]
        common, diff = np.array([3 * n]), vi
        retained = np.array([i for i in range(len(dw)) if i not in set(np.r_[common, diff].tolist())])
        tot = float(np.sum(dw ** 2))
        share = {name: (float(np.sum(dwr[idx] ** 2) / tot) if tot > 0 else 0.0)
                 for name, idx in (("steric_common", common), ("steric_differential", diff), ("retained", retained))}
        top = np.argsort(-np.abs(dw))[:4]
        focus = sorted({law[j]["a_star"] for j in law} | {law[j]["best"] for j in law}) + [n_pool]
        channel = {}
        for a in focus:
            pred = z0["G"][a] @ (post.u_map - post0.u_map) / TOL
            actual = (gmaps[a] - z0["g_map"][a]) / TOL
            channel[str(a)] = dict(loading=float(ops[a, 0]), active_share=_active_share(dw, z0["G"][a], sp),
                                   first_order_shift_tol=pred.tolist(), actual_shift_tol=actual.tolist())
        rows.append(dict(restart=k, objective=nlp, objective_minus_canonical=nlp - nlp0,
                         newton_term=c, newton_decrement=float(np.sqrt(2.0 * c)),
                         quadratic_model_log_mass_relative=-(nlp - nlp0) + (c - c0) + 0.5 * (logdet - logdet0),
                         termination=ms["restarts"][k].get("optimizer_termination"),
                         whitened_displacement=float(np.sqrt(tot)), displacement_share=share,
                         largest_coordinates=[(int(i), float(dw[i])) for i in top],
                         decision_channel=channel, law=law))
        f = law["full"]
        print(f"  r{k}: obj {nlp:8.4f} ({nlp - nlp0:+.3f})  Newton term {c:.3f}  "
              f"log-mass {rows[-1]['quadratic_model_log_mass_relative']:+.3f}  "
              f"|dw| {np.sqrt(tot):.3f} (steric diff {share['steric_differential']:.2f})  "
              f"full: a*={f['loading_a_star']:.1f} {f['branch']} P={f['p_best']:.4f}  "
              f"S/F minimizer {'y' if law['S']['minimizer_preserved'] else 'N'}/{'y' if law['F']['minimizer_preserved'] else 'N'}  "
              f"classifications changed {law['S']['classifications_changed']}/{law['F']['classifications_changed']}  "
              f"branch {'y' if law['S']['branch_preserved'] else 'N'}/{'y' if law['F']['branch_preserved'] else 'N'}")

    res = dict(product=product, deployment_domain=[lo, hi], n_candidates=n_pool, hier_draws=str(hier_draws),
               hierarchy="held at the deployed draws for every restart", rows=rows,
               compression_preserved_everywhere=all(r["law"][j][key] for r in rows for j in ("S", "F")
                                                    for key in ("minimizer_preserved", "branch_preserved"))
               and all(r["law"][j]["classifications_changed"] == 0 for r in rows for j in ("S", "F")),
               full_law_branches=sorted({r["law"]["full"]["branch"] for r in rows}),
               full_law_minimizers=sorted({r["law"]["full"]["loading_a_star"] for r in rows}))
    seg = RES / f"{product}_basin_segment.json"
    if seg.exists():
        res["segment"] = json.loads(seg.read_text())
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"  compression preserved at every fitted point: {res['compression_preserved_everywhere']}; "
          f"full-law branches {res['full_law_branches']}; minimizers {res['full_law_minimizers']}")
    print(f"wrote {out}")


# -------------------------------------------------------------------------------------------- mixture
def stage_mixture(product, *, restarts, hier_draws, out):
    """The six fitted points as one law: each restart's full predictive law weighted by the mass of its
    Gauss-Newton quadratic model, and the shortfall minimizer, gate probabilities and branch read from
    the mixture. A robustness read on the canonical branch, with no new fit and no solver call."""
    hd = np.load(hier_draws)
    bias, sd = hd[f"b_{product}"], hd["Sd_draws"]
    lo, hi = DEPLOY[product]
    post0 = _posterior(product, 0)
    n = post0.n_protein
    z0 = np.load(MS / f"{product}_r0_jacobians.npz")
    ops = z0["ops"]
    n_pool = len(ops) - 1
    loading = ops[:n_pool, 0]
    ms = json.loads((RES / f"{product}_correlated_multistart.json").read_text())

    logmass, P, R = [], [], []
    for k in restarts:
        post = _posterior(product, k)
        z = np.load(MS / f"{product}_r{k}_jacobians.npz")
        assert z["ops"].shape == ops.shape and np.allclose(z["ops"], ops)
        zg = np.load(MS / f"{product}_r{k}_gradient.npz")
        logmass.append(-_recorded_objective(ms, k) + float(zg["newton_term"])
                       + 0.5 * float(np.linalg.slogdet(post.cov)[1]))
        cov = [decision_cov(G, post.cov) for G in z["G"][:n_pool]]
        P.append(meet_probs(z["g_map"][:n_pool], cov, bias, sd))
        R.append(np.array([predictive_risk(z["g_map"][a], cov[a], bias, sd) for a in range(n_pool)]))
        print(f"  [{product}] restart {k} scored", flush=True)
    logmass = np.array(logmass)
    w = np.exp(logmass - logmass.max()); w /= w.sum()
    P, R = np.array(P), np.array(R)
    p_mix, r_mix = w @ P, w @ R
    br, best = branch_of(p_mix, loading, lo, hi)
    a_star = int(np.argmin(r_mix))
    inside = (loading >= lo) & (loading <= hi)
    focus = sorted({a_star, best} | {int(np.argmin(R[i])) for i in range(len(restarts))}
                   | {int(np.argmax(np.where(inside, P[i], -1.0))) for i in range(len(restarts))})
    res = dict(product=product, restarts=list(restarts), weights=w.tolist(),
               log_mass_relative=(logmass - logmass[0]).tolist(),
               branch=br, best=best, loading_best=float(loading[best]), p_best=float(p_mix[best]),
               p_max_deployment=float(p_mix[inside].max()) if inside.any() else None,
               a_star=a_star, loading_a_star=float(loading[a_star]),
               candidates={str(a): dict(loading=float(loading[a]), p_mix=float(p_mix[a]),
                                        p_by_restart=P[:, a].tolist(), r_mix=float(r_mix[a]))
                           for a in focus},
               hier_draws=str(hier_draws))
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"  weights {np.round(w, 3).tolist()}")
    for a in focus:
        print(f"  candidate {a} ({loading[a]:.1f} g/L): P_mix={p_mix[a]:.4f}  by restart {np.round(P[:, a], 4).tolist()}")
    print(f"  mixture: minimizer {loading[a_star]:.1f} g/L, best {loading[best]:.1f} g/L with P={p_mix[best]:.4f}, branch {br}")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["jacobians", "gradients", "segment", "analyse", "mixture"])
    ap.add_argument("--product", default="HLXSYN")
    ap.add_argument("--restarts", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--candidates", type=int, nargs="+", default=None,
                    help="jacobians: cache only these candidate indices (a smoke test); default all")
    ap.add_argument("--out-dir", default=str(MS), help="jacobians, gradients: where the caches go")
    ap.add_argument("--from-restart", type=int, default=0)
    ap.add_argument("--to-restart", type=int, default=4)
    ap.add_argument("--n-points", type=int, default=11)
    ap.add_argument("--hier-draws", default=str(RES / "hier_draws_capB.npz"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.stage == "jacobians":
        stage_jacobians(a.product, restarts=a.restarts, n_steps=a.n_steps, candidates=a.candidates,
                        out_dir=a.out_dir)
    elif a.stage == "gradients":
        stage_gradients(a.product, restarts=a.restarts, n_steps=a.n_steps, out_dir=a.out_dir)
    elif a.stage == "segment":
        stage_segment(a.product, a=a.from_restart, b=a.to_restart, n_points=a.n_points, n_steps=a.n_steps,
                      out=a.out or str(RES / f"{a.product}_basin_segment.json"))
    elif a.stage == "analyse":
        stage_analyse(a.product, restarts=a.restarts, hier_draws=a.hier_draws,
                      out=a.out or str(RES / f"{a.product}_basin_closure.json"))
    else:
        stage_mixture(a.product, restarts=a.restarts, hier_draws=a.hier_draws,
                      out=a.out or str(RES / f"{a.product}_basin_mixture.json"))


if __name__ == "__main__":
    main()
