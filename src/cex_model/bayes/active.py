"""Adaptive experiment design with a stopping rule -- a data-adaptive upper bound.

The parameter-by-parameter (PbP / Yamamoto) method guarantees identifiability
with a fixed prescribed experiment set (~5-6 designed LGEs).  We don't prescribe;
instead we keep adding the single most informative experiment (BOED) until the
posterior says every parameter is identified -- giving a DATA-ADAPTIVE bound:
"from your current data, here are the next N experiments to reach identifiability".

Validated synthetically (ground-truth theta known): the loop terminates, and the
experiments it picks recover the PbP prescription (vary gradient slope to break
keq<->nu; raise loading to constrain sigma) -- information theory rediscovering
the mechanistic design.
"""

from __future__ import annotations

import types

import numpy as np
import torch

from cex_model.bayes.design import (
    OP_FEATURES,
    candidate_pool_for,
    candidate_predict_fn,
    expected_info_gain,
    recommend_experiments_bayes,
)
from cex_model.bayes.identifiability import correlation_pairs, shrinkage
from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, mechanistic_model, unpack_u
from cex_model.bayes.posterior import laplace_posterior, map_fit
from cex_model.bayes.prior import components_to_u
from cex_model.diffsolver.calibrate_diff import ExperimentTarget, _grouped_curve
from cex_model.diffsolver.torch_solver import DTYPE, TorchSimulator
from cex_model.gradients import build_fitting_inlet

__all__ = ["identifiability_report", "simulate_target", "run_adaptive_design"]


def identifiability_report(posterior, prior, *, tau: float = 0.5, rho: float = 0.95) -> dict:
    """Is every parameter identified?

    Identified iff the WORST-direction posterior std is below ``tau`` x the prior
    std (prior-whitened) -- i.e. the data shrank EVERY direction, not merely every
    marginal.  High pairwise correlation is reported but does NOT block
    identification: a tight-but-correlated ridge is identified (small in every
    direction); only a WIDE ridge is not (a prior-dominated direction remains).
    ``rho`` just flags the correlation in the report (keq<->nu stay correlated
    even when tightly pinned -- that is the ridge's shape, not its size).
    """
    sh = shrinkage(posterior, prior)
    unmet = sorted((n for n, r in sh.items() if r >= tau), key=lambda n: -sh[n])
    max_corr = max((abs(d["corr"]) for d in correlation_pairs(posterior, k=10)), default=0.0)
    # prior-whitened posterior covariance M_ij = cov_ij / (prior_std_i * prior_std_j);
    # its largest eigenvalue is the worst-direction (posterior var / prior var) ratio.
    d = 1.0 / prior.std
    M = posterior.cov * d[:, None] * d[None, :]
    worst_dir = float(np.sqrt(max(float(np.linalg.eigvalsh(M).max()), 0.0)))
    H = np.linalg.inv(posterior.cov)
    return {"met": bool(worst_dir < tau), "worst_dir_shrinkage": worst_dir, "unmet_params": unmet,
            "max_corr": float(max_corr), "highly_correlated": bool(max_corr >= rho), "shrinkage": sh,
            "min_eig_precision": float(np.linalg.eigvalsh(H).min()),
            "logdet_precision": float(np.linalg.slogdet(H)[1])}


def _sim_for_op(bundle, op, n_steps: int) -> TorchSimulator:
    e0 = bundle.experiments[0]
    loading, gstart, gend, ecv = (float(x) for x in op)
    inlet = build_fitting_inlet(
        buffer_a=e0.buffer_a, buffer_b=e0.buffer_b, gradient_start_pct=gstart, gradient_end_pct=gend,
        elution_cv=ecv, rt_min=bundle.column.rt, load_amount_g_l=loading,
        component_fractions_pct=bundle.components.fraction_array())
    return TorchSimulator(bundle.column, bundle.components, inlet, loading, bundle.correction, n_steps=n_steps)


def simulate_target(bundle, op, theta_true_u, *, n_steps: int, n_points: int,
                    sigma_obs: float = AKTA_NOISE_FLOOR_G_L, seed: int = 0) -> ExperimentTarget:
    """A synthetic noisy ExperimentTarget at OP ``op`` under ground-truth ``theta_true_u``."""
    n_protein, groups = bundle.components.n_protein, bundle.observation_groups
    sim = _sim_for_op(bundle, op, n_steps)
    u = torch.tensor(np.asarray(theta_true_u, float), dtype=DTYPE)
    with torch.no_grad():
        curve = sim.elution_curve(*unpack_u(u, n_protein), differentiable=False)
    times = torch.linspace(float(curve[0, 0]), float(curve[-1, 0]), n_points, dtype=DTYPE)
    clean = _grouped_curve(curve, times, groups, n_protein)
    noisy = clean + torch.tensor(sigma_obs * np.random.default_rng(seed).standard_normal(tuple(clean.shape)),
                                 dtype=DTYPE)
    return ExperimentTarget(sim=sim, times_s=times, values=noisy, groups=groups)


def _op_record(op, e0):
    return types.SimpleNamespace(loading_g_l=float(op[0]), gradient_start_pct=float(op[1]),
                                 gradient_end_pct=float(op[2]), elution_cv=float(op[3]),
                                 buffer_a=e0.buffer_a, buffer_b=e0.buffer_b)


SELECT_OPS = ("boed", "random", "spacefill", "sensitivity", "boed_noprior", "decision_oed")


def _maximin_pick(pool, existing_ops):
    """Space-filling: the candidate whose nearest existing OP is farthest (range-scaled)."""
    pool = np.asarray(pool, float)
    scale = np.ptp(pool, axis=0)
    scale[scale == 0] = 1.0
    if not existing_ops:
        return pool[0]
    ex = np.asarray(existing_ops, float) / scale
    d = np.linalg.norm(pool[:, None, :] / scale - ex[None, :, :], axis=2).min(axis=1)
    return pool[int(np.argmax(d))]


def _select_next_op(select_op, post, db, prior, *, n_candidates, n_steps, n_points, sigma_obs,
                    seed, rng, decision_cfg=None):
    """Pick the next OP under a design strategy, all drawing from the SAME Sobol pool.

    ``boed`` is handled by the caller via :func:`recommend_experiments_bayes` (unchanged);
    the baselines here let the design-baseline comparison run every arm on one ground truth.
    ``decision_oed`` scores by decision-quantity (pooled purity/yield) variance reduction
    (goal-oriented / L-optimal), reading the spec/tolerance from ``decision_cfg``.
    """
    pool = candidate_pool_for(db, n_candidates=n_candidates, seed=seed)
    if select_op == "random":
        return pool[int(rng.integers(len(pool)))]
    if select_op == "spacefill":
        existing = [[e.loading_g_l, e.gradient_start_pct, e.gradient_end_pct, e.elution_cv]
                    for e in db.experiments]
        return _maximin_pick(pool, existing)
    # information / sensitivity strategies need per-candidate Jacobians at the MAP
    H = np.linalg.inv(np.asarray(post.cov, float))
    if select_op == "boed_noprior":
        w, V = np.linalg.eigh(H - np.asarray(prior.precision(), float))
        H = V @ np.diag(np.clip(w, 1e-9, None)) @ V.T      # data-only Fisher, PSD-clipped
    u_map = np.asarray(post.u_map, float)
    G_dec, dec_tol = None, None
    if select_op == "decision_oed":
        from cex_model.bayes.decision import decision_jacobian
        cfg = dict(decision_cfg or {})
        dec_tol = cfg.pop("tol", (0.02, 0.05))
        cfg.pop("tau_dec", None)
        dop = [getattr(db.experiments[0], f) for f in OP_FEATURES]
        G_dec = decision_jacobian(db, dop, u_map, n_steps=n_steps, **cfg)
    best_op, best_score = pool[0], -np.inf
    for op in pool:
        J = torch.autograd.functional.jacobian(
            candidate_predict_fn(db, op, n_steps=n_steps, n_points=n_points, u_map=u_map),
            torch.tensor(u_map, dtype=DTYPE)).detach().numpy()
        if select_op == "sensitivity":
            score = float(np.linalg.norm(J))
        elif select_op == "decision_oed":
            from cex_model.bayes.decision import decision_oed_score
            score = decision_oed_score(H, J, G_dec, sigma_obs, tol=dec_tol)
        else:
            score = expected_info_gain(H, J, sigma_obs)
        if score > best_score:
            best_score, best_op = score, np.asarray(op, float)
    return best_op


def run_adaptive_design(bundle, targets0, prior, theta_true_u, *, tau: float = 0.5, rho: float = 0.95,
                        select_op: str = "boed", stop_on: str = "param", decision_cfg: dict | None = None,
                        max_experiments: int = 6, n_steps_fit: int = 120,
                        n_steps_design: int = 120, n_points: int = 12, n_candidates: int = 24,
                        sigma_obs: float = AKTA_NOISE_FLOOR_G_L, map_iters: int = 150, lr: float = 0.05,
                        seed: int = 0, progress: bool = False) -> dict:
    """Add ``select_op``-chosen experiments until identified (or budget hit). Synthetic ground truth.

    ``select_op`` (default ``"boed"``) picks the next experiment: ``boed`` = theta-space
    Bayesian D-optimal (the method); ``random`` / ``spacefill`` / ``sensitivity`` /
    ``boed_noprior`` are the design baselines; ``decision_oed`` targets decision-quantity
    variance.  All draw from the same Sobol pool.  ``stop_on`` (default ``"param"``) is the
    stopping criterion: ``"param"`` = ``identifiability_report`` worst-direction (byte-identical
    to the original); ``"decision"`` = ``decision_report`` worst-direction (pooled purity/yield
    vs tolerance, configured by ``decision_cfg``); ``"both"`` = both met; ``"posterior_action"`` =
    stop when the posterior meet-probability makes the ACTION decisive (meets or level-misses spec),
    i.e. the posterior-action gate of the four-state recommendation, not when ``worst_dec < tau``
    (paper §4 -- avoids over-collecting when the posterior mean is far from the action boundary).
    The data-adaptive upper bound is ``result["n_added"]``.
    """
    if select_op not in SELECT_OPS:
        raise ValueError(f"select_op must be one of {SELECT_OPS}, got {select_op!r}")
    if stop_on not in ("param", "decision", "both", "posterior_action"):
        raise ValueError(
            f"stop_on must be 'param', 'decision', 'both', or 'posterior_action', got {stop_on!r}")
    # `spec` (for the posterior-action p_meet) is consumed only by the stop-check's decision_report; pop it
    # out of decision_cfg so it never reaches the design-selection path (decision_jacobian rejects it).
    decision_cfg = dict(decision_cfg or {})
    _dec_spec = decision_cfg.pop("spec", None)
    # own, mutable view so the caller's bundle is not modified as we add experiments
    db = types.SimpleNamespace(components=bundle.components, column=bundle.column,
                               correction=bundle.correction, observation_groups=bundle.observation_groups,
                               product_id=getattr(bundle, "product_id", "?"),
                               experiments=list(bundle.experiments))
    targets = list(targets0)
    n_protein = bundle.components.n_protein
    u_init = components_to_u(bundle.components)
    history, post = [], None
    rng = np.random.default_rng(seed)
    for step in range(max_experiments + 1):
        predict_fn, obs = mechanistic_model(targets, n_protein)
        u_init, _ = map_fit(u_init, predict_fn, obs, prior, sigma_obs=sigma_obs, iters=map_iters, lr=lr)
        post = laplace_posterior(u_init, predict_fn, obs, prior, template=bundle.components, sigma_obs=sigma_obs)
        rep = identifiability_report(post, prior, tau=tau, rho=rho)
        entry = {"n_added": step, "met": rep["met"], "n_unmet": len(rep["unmet_params"]),
                 "worst_dir_shrinkage": rep["worst_dir_shrinkage"], "max_corr": rep["max_corr"],
                 "min_eig_precision": rep["min_eig_precision"]}
        dec = None
        if stop_on in ("decision", "both", "posterior_action"):
            from cex_model.bayes.decision import decision_report, posterior_action_decisive
            dop = [getattr(db.experiments[0], f) for f in OP_FEATURES]
            dec_cfg = dict(decision_cfg)
            if stop_on == "posterior_action":
                dec_cfg["spec"] = _dec_spec if _dec_spec is not None else (0.70, 0.50)  # illustrative pooling spec
            dec = decision_report(post, db, dop, n_steps=n_steps_design, **dec_cfg)
            entry["worst_dec"], entry["met_dec"] = dec["worst_dec"], dec["met"]
            if "p_meet" in dec:
                entry["p_meet"] = dec["p_meet"]
                entry["action_decisive"] = posterior_action_decisive(dec["p_meet"])
        history.append(entry)
        if progress:
            msg = (f"  step {step}: met={rep['met']} worst_dir_shrinkage={rep['worst_dir_shrinkage']:.3f} "
                   f"(tau={tau}) unmet={len(rep['unmet_params'])} max_corr={rep['max_corr']:.3f} "
                   f"min_eig={rep['min_eig_precision']:.2e}")
            if dec is not None:
                msg += f" worst_dec={dec['worst_dec']:.3f} met_dec={dec['met']}"
            print(msg)
        done = (rep["met"] if stop_on == "param"
                else posterior_action_decisive(dec.get("p_meet")) if stop_on == "posterior_action"
                else dec["met"] if stop_on == "decision"
                else (rep["met"] and dec["met"]))
        if done or step == max_experiments:
            break
        if select_op == "boed":
            rec = recommend_experiments_bayes(post, db, k=1, n_candidates=n_candidates,
                                              n_steps=n_steps_design, n_points=n_points,
                                              sigma_obs=sigma_obs, seed=seed + step)
            op = [rec["recommendations"][0][f] for f in OP_FEATURES]
        else:
            op = _select_next_op(select_op, post, db, prior, n_candidates=n_candidates,
                                 n_steps=n_steps_design, n_points=n_points, sigma_obs=sigma_obs,
                                 seed=seed + step, rng=rng, decision_cfg=decision_cfg)
        targets.append(simulate_target(db, op, theta_true_u, n_steps=n_steps_fit, n_points=n_points,
                                       sigma_obs=sigma_obs, seed=seed + 1000 + step))
        db.experiments.append(_op_record(op, db.experiments[0]))
    added_ops = [{f: getattr(e, f) for f in OP_FEATURES} for e in db.experiments[len(bundle.experiments):]]
    result = {"n_added": history[-1]["n_added"], "met": history[-1]["met"], "history": history,
              "posterior": post, "added_ops": added_ops, "stop_on": stop_on}
    if stop_on in ("decision", "both", "posterior_action"):
        result["met_dec"] = bool(history[-1].get("met_dec", False))
    if stop_on == "posterior_action":
        result["action_decisive"] = bool(history[-1].get("action_decisive", False))
        result["p_meet"] = history[-1].get("p_meet")
    return result
