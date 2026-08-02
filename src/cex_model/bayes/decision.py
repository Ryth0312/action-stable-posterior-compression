"""Decision-relevant identifiability: project posterior uncertainty from parameter
space into PROCESS-QUANTITY space (pooled-window purity + yield), and score
experiments by decision-quantity variance reduction (goal-oriented / c-/L-optimal OED).

This is the decision-space analogue of ``identifiability_report`` (``bayes.active``):
instead of asking "is theta determined?" -- the worst PRIOR-whitened direction of the
posterior covariance ``Sigma`` -- it asks "is the predicted pooled purity / yield
determined to within a decision tolerance?" -- the worst TOLERANCE-whitened direction
of the decision covariance ``C = G Sigma Gᵀ``, where ``G = dg/du`` is the Jacobian of
the process quantities ``g(theta)`` (pooled main-group purity + window recovery).

Reuses (does not reimplement):
  * the differentiable fixed-window collection objective
    (:mod:`cex_model.diffsolver.collection_objective`) for ``g`` and ``dg/du``, with the
    collection window held FIXED at the MAP curve (the block-coordinate contract);
  * the per-OP ``TorchSimulator`` builder ``active._sim_for_op``;
  * ``app_support.group_indices`` (acid/main/basic protein columns);
  * the numpy ``collection.optimize_collection_window`` for the Monte-Carlo cross-check;
  * the ``H``/``J``/``sigma`` convention of ``design.expected_info_gain`` for the OED score.
"""

from __future__ import annotations

import numpy as np
import torch

from cex_model.app_support import group_indices
from cex_model.bayes.active import _sim_for_op
from cex_model.bayes.likelihood import unpack_u
from cex_model.bayes.prior import physical_u_bounds
from cex_model.collection import optimize_collection_window
from cex_model.diffsolver.collection_objective import (
    differentiable_fixed_window_objective,
    select_window_for_gradient,
)
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = [
    "DECISION_NAMES",
    "decision_jacobian",
    "decision_covariance",
    "decision_report",
    "decision_covariance_mc",
    "decision_oed_score",
    "posterior_action_decisive",
]

# The decision vector g(theta): pooled-window main-group purity and recovery (yield),
# both fractions in [0, 1] over the collection window.
DECISION_NAMES: tuple[str, ...] = ("pool_purity", "pool_yield")

# Project-canonical purity spec (optimization.ProcessOptimizationConfig defaults).
_ACID_MAX, _MAIN_MIN, _BASIC_MAX = 0.20, 0.70, 0.10

# numpy>=2.0 renamed trapz -> trapezoid; support both (matches predictive.py).
_trapz = getattr(np, "trapezoid", None) or np.trapz


def decision_forward(bundle, op, u_map, *, n_steps: int = 120, acid_max: float = _ACID_MAX,
                     main_min: float = _MAIN_MIN, basic_max: float = _BASIC_MAX, grid: int = 40,
                     inocap_scale: float = 1.0):
    """Build a differentiable ``g_fn(u) -> [pool_purity, pool_yield]`` at one decision OP.

    The collection window is selected ONCE on the MAP curve (numpy
    ``select_window_for_gradient``) and held FIXED by physical time inside the
    differentiable objective, so ``dg/du`` flows through the window integral but not the
    (discrete) window choice -- the block-coordinate contract of
    :mod:`cex_model.diffsolver.collection_objective`.  Returns ``(g_fn, window, group_idx)``.

    ``inocap_scale`` (default 1.0 = no-op) multiplies the column ionic capacity ``Λ₀``
    (``sim.inocap``) -- a clean knob on the SMA shared-capacity term that varies the
    per-molecule loading fraction ``ε = γQ/Λ̄`` WITHOUT changing the keq/ν selectivity, used by
    the capacity-sweep test of the σ decision-null theorem (``bayes.loading_sweep``).
    """
    n = bundle.components.n_protein
    grp = group_indices(bundle.components)
    sim = _sim_for_op(bundle, op, n_steps)
    if inocap_scale != 1.0:
        sim.inocap = sim.inocap * inocap_scale
    u_map_t = torch.tensor(np.asarray(u_map, float), dtype=DTYPE)
    with torch.no_grad():
        curve_map = sim.elution_curve(*unpack_u(u_map_t, n), differentiable=False)
    sel = select_window_for_gradient(
        curve_map.numpy(), grid=grid, acid_idx=grp["acid"], main_idx=grp["main"],
        basic_idx=grp["basic"], acid_max=acid_max, main_min=main_min, basic_max=basic_max)

    def g_fn(u: torch.Tensor) -> torch.Tensor:
        curve = sim.elution_curve(*unpack_u(u, n), differentiable=True)
        r = differentiable_fixed_window_objective(
            curve, start_time_s=sel.start_time_s, end_time_s=sel.end_time_s,
            acid_idx=grp["acid"], main_idx=grp["main"], basic_idx=grp["basic"],
            acid_max=acid_max, main_min=main_min, basic_max=basic_max, min_total=sel.min_total)
        return torch.stack([r.main_fraction, r.recovery])

    return g_fn, sel, grp


def decision_jacobian(bundle, op, u_map, *, n_steps: int = 120, return_extra: bool = False, **spec):
    """``G = dg/du`` (k x dim) at the MAP via reverse-mode autograd.

    Reverse mode only -- forward mode is wrong through the implicit-diff solver (same
    constraint as ``posterior._ggn_hessian`` / ``design``).  ``spec`` forwards the spec
    thresholds / ``grid`` / ``inocap_scale`` (the Λ₀ capacity knob) to :func:`decision_forward`.
    With ``return_extra=True`` also returns ``(g_map, window)``.
    """
    g_fn, sel, _ = decision_forward(bundle, op, u_map, n_steps=n_steps, **spec)
    u = torch.tensor(np.asarray(u_map, float), dtype=DTYPE)
    G = torch.autograd.functional.jacobian(g_fn, u).detach().numpy()
    if return_extra:
        with torch.no_grad():
            g_map = g_fn(u).detach().numpy()
        return G, g_map, sel
    return G


def decision_covariance(G: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Linearized decision covariance ``C = G Sigma Gᵀ`` (k x k); ``Sigma = posterior.cov``."""
    G = np.asarray(G, float)
    return G @ np.asarray(cov, float) @ G.T


# Posterior-action decision thresholds -- mirror of decision_window.P_MEET_HI / P_MEET_LO (pre-registered;
# duplicated here so active.py's stop_on='posterior_action' and decision_report need not import decision_window).
P_MEET_HI: float = 0.90
P_MEET_LO: float = 0.10


def posterior_action_decisive(p_meet, *, p_meet_hi: float = P_MEET_HI, p_meet_lo: float = P_MEET_LO) -> bool:
    """True when the posterior meet-probability makes the ACTION decisive -- either meets spec
    (``p_meet >= p_meet_hi``) or misses it at the level (``p_meet <= p_meet_lo``) -- i.e. NOT ambiguous.

    The stop condition for ``run_adaptive_design(stop_on='posterior_action')`` (paper §4): the loop keeps
    collecting data only while the optimal action is genuinely undecided, not merely while ``worst_dec >= 1``.
    """
    if p_meet is None:
        return False
    p = float(p_meet)
    if not np.isfinite(p):
        return False
    return not (p_meet_lo < p < p_meet_hi)


def _gaussian_meet_prob(g_map, C, spec) -> float:
    """``P(purity >= spec0 AND yield >= spec1)`` for ``g ~ N(g_map, C)`` (bivariate-normal upper orthant);
    the ``decision.py`` companion of ``decision_window.gaussian_meet_prob`` (self-contained, no cross-import)."""
    from scipy.stats import multivariate_normal

    g_map = np.asarray(g_map, float)
    C = np.asarray(C, float)
    C = C + 1e-12 * np.eye(C.shape[0])
    spec = np.asarray(spec, float)
    return float(multivariate_normal(mean=-g_map, cov=C, allow_singular=True).cdf(-spec))


def _report_from_covariance(C, g_map, *, tol, tau_dec: float = 1.0, window=None, op=None, spec=None) -> dict:
    """Assemble the decision report from a decision covariance ``C`` (solver-free).

    Tolerance-whitened ``C~ = T⁻¹ C T⁻¹`` with ``T = diag(tol)``;
    ``worst_dec = sqrt(lambda_max(C~))``; identified iff ``worst_dec < tau_dec``.  This
    is the decision-space mirror of ``identifiability_report``'s prior-whitened worst
    direction -- g(theta) whitened by the DECISION tolerance instead of theta by the prior.
    When ``spec`` is given, also reports ``p_meet = P(g >= spec | data)`` (for the posterior-action gate).
    """
    C = np.asarray(C, float)
    tol = np.asarray(tol, float)
    dinv = 1.0 / tol
    Ctil = C * dinv[:, None] * dinv[None, :]
    worst_dec = float(np.sqrt(max(float(np.linalg.eigvalsh(Ctil).max()), 0.0)))
    std = np.sqrt(np.clip(np.diag(C), 0.0, None))
    names = list(DECISION_NAMES)
    shr = {names[i]: float(std[i] / tol[i]) for i in range(len(names))}
    unmet = sorted((nm for nm, r in shr.items() if r >= 1.0), key=lambda nm: -shr[nm])
    out = {
        "met": bool(worst_dec < tau_dec),
        "worst_dec": worst_dec,
        "decision_std": {names[i]: float(std[i]) for i in range(len(names))},
        "decision_shrinkage": shr,
        "unmet_quantities": unmet,
        "g_map": {names[i]: float(g_map[i]) for i in range(len(names))},
        "C": C.tolist(),
        "tol": [float(t) for t in tol],
        "names": names,
    }
    if spec is not None:
        out["p_meet"] = _gaussian_meet_prob(g_map, C, spec)
    if window is not None:
        out["window_s"] = [float(window.start_time_s), float(window.end_time_s)]
        out["window_mode"] = window.mode
    if op is not None:
        out["op"] = [float(x) for x in op]
    return out


def decision_report(posterior, bundle, op, *, tol=(0.02, 0.05), tau_dec: float = 1.0, spec=None,
                    n_steps: int = 120, acid_max: float = _ACID_MAX, main_min: float = _MAIN_MIN,
                    basic_max: float = _BASIC_MAX, grid: int = 40) -> dict:
    """Is the decision quantity ``g`` (pooled purity + yield) determined to within ``tol``?

    Returns ``met`` / ``worst_dec`` / per-quantity ``decision_std`` & ``decision_shrinkage``
    (= std/tol) / ``unmet_quantities`` / ``g_map`` / window -- field shape deliberately
    matching ``identifiability_report`` so the stopping rule and the paper table treat the
    parameter- and decision-space criteria uniformly.  When ``spec`` (purity_min, yield_min) is given,
    also returns ``p_meet`` (for the posterior-action stopping rule).
    """
    G, g_map, sel = decision_jacobian(
        bundle, op, posterior.u_map, n_steps=n_steps, return_extra=True,
        acid_max=acid_max, main_min=main_min, basic_max=basic_max, grid=grid)
    C = decision_covariance(G, posterior.cov)
    return _report_from_covariance(C, g_map, tol=tol, tau_dec=tau_dec, window=sel, op=op, spec=spec)


def _pool_quantities(curve: np.ndarray, start_idx: int, end_idx: int, main_idx) -> tuple[float, float]:
    """Pooled main-group purity and recovery on a fixed INDEX window (numpy mirror of the
    differentiable objective's ``main_fraction`` / ``recovery``)."""
    curve = np.asarray(curve, float)
    conc = np.clip(curve[:, 2:], 0.0, None)
    t = curve[:, 0]
    collected = _trapz(conc[start_idx:end_idx + 1], t[start_idx:end_idx + 1], axis=0)
    total = float(collected.sum())
    if total <= 0.0:
        return 0.0, 0.0
    mi = list(main_idx) if main_idx else []
    purity = float(collected[mi].sum()) / total if mi else 0.0
    whole = float(_trapz(conc.sum(axis=1), t))
    recovery = total / whole if whole > 0.0 else 0.0
    return purity, recovery


def decision_covariance_mc(posterior, bundle, op, *, n_samples: int = 200, seed: int = 0,
                           n_steps: int = 120, reselect: bool = False, acid_max: float = _ACID_MAX,
                           main_min: float = _MAIN_MIN, basic_max: float = _BASIC_MAX,
                           grid: int = 40, return_gs: bool = False) -> dict:
    """Sample-covariance of ``g`` over posterior draws (the nonlinear cross-check of ``C``).

    For each draw ``u ~ posterior``: simulate the curve at ``op`` on the (fast) differentiable
    solver (detached) and evaluate ``g`` via the numpy collection metrics -- on the FIXED MAP
    window (``reselect=False``, the estimand matched to the linearized ``C``) or a per-draw
    re-optimized window (``reselect=True``, the free-window variance, a secondary number).
    Returns ``{C_mc, mean_g, std_g, n_used}`` (plus the raw per-draw ``gs`` (n_used x 2) when
    ``return_gs=True`` -- the posterior-predictive decision pushforward used by
    :mod:`cex_model.bayes.decision_window` for P(meet-spec) / expected regret).
    """
    n = bundle.components.n_protein
    grp = group_indices(bundle.components)
    sim = _sim_for_op(bundle, op, n_steps)
    us = posterior.samples(n_samples, seed=seed)
    # clip to physical support (sloppy products can sample nu<0 etc., which makes the
    # SMA term blow up and pollutes the MC covariance with NaNs); see prior.physical_u_bounds
    lo, hi = physical_u_bounds(n)
    us = np.clip(us, lo, hi)

    # Fixed window: indices on the (theta-independent) time grid at the MAP curve.
    u_map_t = torch.tensor(np.asarray(posterior.u_map, float), dtype=DTYPE)
    with torch.no_grad():
        curve_map = sim.elution_curve(*unpack_u(u_map_t, n), differentiable=False).numpy()
    sel = select_window_for_gradient(
        curve_map, grid=grid, acid_idx=grp["acid"], main_idx=grp["main"], basic_idx=grp["basic"],
        acid_max=acid_max, main_min=main_min, basic_max=basic_max)

    gs = []
    for u in us:
        ut = torch.tensor(np.asarray(u, float), dtype=DTYPE)
        with torch.no_grad():
            curve = sim.elution_curve(*unpack_u(ut, n), differentiable=False).numpy()
        if reselect:
            w = optimize_collection_window(
                curve, grid=grid, acid_idx=grp["acid"], main_idx=grp["main"], basic_idx=grp["basic"],
                acid_max=acid_max, main_min=main_min, basic_max=basic_max).best
            if not w.feasible:
                continue
            s_idx, e_idx = w.start_index, w.end_index
        else:
            s_idx, e_idx = sel.start_idx, sel.end_idx
        gs.append(list(_pool_quantities(curve, s_idx, e_idx, grp["main"])))

    gs = np.asarray(gs, float)
    names = list(DECISION_NAMES)
    if gs.shape[0] < 2:
        nan2 = np.full((2, 2), np.nan)
        out = {"C_mc": nan2.tolist(), "mean_g": {nm: float("nan") for nm in names},
               "std_g": {nm: float("nan") for nm in names}, "n_used": int(gs.shape[0])}
        if return_gs:
            out["gs"] = gs.tolist()
        return out
    C_mc = np.cov(gs.T)
    out = {"C_mc": np.asarray(C_mc, float).tolist(),
           "mean_g": {names[i]: float(gs[:, i].mean()) for i in range(len(names))},
           "std_g": {names[i]: float(gs[:, i].std(ddof=1)) for i in range(len(names))},
           "n_used": int(gs.shape[0])}
    if return_gs:
        out["gs"] = gs.tolist()
    return out


def decision_oed_score(H: np.ndarray, J_c: np.ndarray, G: np.ndarray, sigma_obs: float,
                       tol=None) -> float:
    """Decision-variance reduction from a candidate experiment ``c`` (goal-oriented OED).

    ``Delta = sum_q w_q [ g_qᵀ H⁻¹ g_q - g_qᵀ (H + J_cᵀ J_c / sigma²)⁻¹ g_q ]`` -- the drop
    in (tolerance-weighted) decision variance, where ``g_q`` is row q of ``G``.
    ``w_q = 1/tol_q²`` when ``tol`` is given (L-optimality on the spec-relative quantity),
    else 1.  Always finite: ``H`` carries the PD prior precision, so ``H`` and
    ``H + J_cᵀ J_c / sigma²`` are PD -- the feature vs the classical c-optimal singularity.
    Same ``H`` / ``J_c`` convention as ``design.expected_info_gain``.
    """
    H = np.asarray(H, float)
    G = np.asarray(G, float)
    J_c = np.asarray(J_c, float)
    M = (J_c.T @ J_c) / (sigma_obs**2)
    A0 = np.linalg.inv(H)
    A1 = np.linalg.inv(H + M)
    w = np.ones(G.shape[0]) if tol is None else 1.0 / (np.asarray(tol, float) ** 2)
    total = 0.0
    for q in range(G.shape[0]):
        g = G[q]
        total += float(w[q]) * float(g @ A0 @ g - g @ A1 @ g)
    return float(total)
