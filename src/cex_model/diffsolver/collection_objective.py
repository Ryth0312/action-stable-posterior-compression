"""Differentiable fixed-window collection objective for gradient-based Phase-2 (D2.2, approach a).

The forward pass selects the collection window with the existing (non-differentiable)
``collection.optimize_collection_window`` / ``yield_cal`` business logic -- including its
degenerate-window guards (``min_total`` / ``min_collect_fraction``). The backward pass then holds
that window FIXED (its indices are constants) and differentiates only the window-integral
yield / recovery / purity through the torch curve w.r.t. the operating conditions. This is the
block-coordinate / straight-through approximation: "hold the optimal window fixed for this local
step, optimise the operating conditions, then re-select the window next iteration" (the optimiser
re-selects every few steps; see D2.3). The window is parameterised by INDICES (simplest, stable);
the op-dependent time grid still flows through the trapezoid spacing.

Two things a gradient optimiser needs that a DE does not:
* INFEASIBLE regions must still yield a usable gradient. When no window meets the purity spec (e.g.
  HLXSYN's genuinely-infeasible strict spec), ``select_window_for_gradient`` returns the
  LEAST-VIOLATING window, so the penalty keeps pushing back toward feasibility instead of going flat.
* Constraints use a SQUARED-HINGE penalty (``relu(violation)**2``), not a log barrier: log barriers
  explode on infeasible starts, which Phase-2 routinely has; the hinge is finite and differentiable
  on both sides of the boundary.

The reported optimum is still ODE-verified downstream (D2.3 keeps the lossless contract); this module
only shapes the SEARCH gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.integrate import trapezoid

from cex_model.collection import yield_cal
from cex_model.sma import MOL_TO_G_L


@dataclass
class GradientWindowSelection:
    """The fixed window the backward pass differentiates through, plus the guards/metadata the
    objective must reuse so forward selection and backward penalty stay consistent."""

    start_idx: int
    end_idx: int
    mode: str                       # "feasible_best" | "least_violation" | "empty"
    min_total: float                # min collected mass (= min_collect_fraction * curve_total)
    curve_total: float
    violation: float | None = None  # total purity violation (0 if feasible, None if empty)
    start_time_s: float = 0.0       # window bounds in PHYSICAL time (for the fixed-time objective)
    end_time_s: float = 0.0


def select_window_for_gradient(
    curve_np: np.ndarray, *, grid: int = 50, min_gap: int = 4, acid_idx, main_idx, basic_idx,
    acid_max: float, main_min: float, basic_max: float, min_collect_fraction: float = 0.05,
) -> GradientWindowSelection:
    """Pick the (fixed) window the backward pass differentiates through.

    Returns a :class:`GradientWindowSelection`; its ``mode`` is:
      * ``"feasible_best"`` -- the highest-yield purity-feasible window (the normal Phase-2 choice);
      * ``"least_violation"`` -- no feasible window exists; the window with the smallest total purity
        violation (tie-break: higher yield), so the gradient still points back toward feasibility;
      * ``"empty"`` -- the curve carries ~no protein (degenerate); the whole curve, large min-mass penalty.
    Mirrors ``optimize_collection_window``'s grid + ``yield_cal`` (same degenerate-mass guard). Pass
    the returned ``min_total`` into :func:`differentiable_fixed_window_objective` so the backward
    penalty uses the SAME min-mass guard as this forward selection.
    """
    curve = np.array(curve_np, dtype=float)
    curve[:, 2:] = np.maximum(curve[:, 2:], 0.0)
    n = curve.shape[0]
    curve_total = float(trapezoid(curve[:, 2:].sum(axis=1), curve[:, 0]))
    min_total = max(min_collect_fraction, 0.0) * curve_total
    ai = list(acid_idx) if acid_idx else [0]
    mi = list(main_idx) if main_idx else []
    bi = list(basic_idx) if basic_idx else []
    cal = np.floor(np.linspace(0, n - 1, grid)).astype(int)

    best_feas: tuple[int, int] | None = None
    best_feas_y = -1.0
    lv: tuple[int, int] | None = None
    lv_v, lv_y = float("inf"), -1.0
    for i in range(grid):
        for j in range(grid):
            if j <= i + min_gap:
                continue
            y, feas, p = yield_cal(curve, int(cal[i]), int(cal[j]), acid_max=acid_max,
                                   main_min=main_min, basic_max=basic_max, acid_idx=ai,
                                   main_idx=mi, basic_idx=bi, min_total=min_total)
            total = y * 60.0 / MOL_TO_G_L
            if total < min_total or total <= 0.0:
                continue  # ignore degenerate near-zero-mass windows (their 0/0 purity is meaningless)
            if feas and y > best_feas_y:
                best_feas, best_feas_y = (int(cal[i]), int(cal[j])), y
            a = float(np.sum(p[ai]))
            v = max(0.0, a - acid_max)
            if mi:
                v += max(0.0, main_min - float(np.sum(p[mi])))
            if bi:
                v += max(0.0, float(np.sum(p[bi])) - basic_max)
            if v < lv_v or (v == lv_v and y > lv_y):
                lv, lv_v, lv_y = (int(cal[i]), int(cal[j])), v, y

    def _ts(s_idx, e_idx):
        return float(curve[s_idx, 0]), float(curve[e_idx, 0])
    if best_feas is not None:
        return GradientWindowSelection(best_feas[0], best_feas[1], "feasible_best", min_total,
                                       curve_total, 0.0, *_ts(*best_feas))
    if lv is not None:
        return GradientWindowSelection(lv[0], lv[1], "least_violation", min_total, curve_total, lv_v, *_ts(*lv))
    return GradientWindowSelection(0, n - 1, "empty", min_total, curve_total, None, *_ts(0, n - 1))


@dataclass
class FixedWindowObjectiveResult:
    """Differentiable window metrics + the scalar ``loss`` (minimise) for the optimiser."""

    loss: torch.Tensor
    score: torch.Tensor          # what's maximised (recovery, or normalised yield)
    yield_g: torch.Tensor        # absolute collected protein (MATLAB units), for reporting
    recovery: torch.Tensor       # window mass / whole-curve mass in [0, 1]
    total: torch.Tensor
    acid_fraction: torch.Tensor
    main_fraction: torch.Tensor
    basic_fraction: torch.Tensor
    penalty: torch.Tensor
    penalties: dict  # {"acid","main","basic","min_total"} breakdown -- which constraint is binding


def differentiable_fixed_window_objective(
    curve_t: torch.Tensor, start_idx: int | None = None, end_idx: int | None = None, *,
    acid_idx, main_idx, basic_idx, acid_max: float, main_min: float, basic_max: float,
    min_total: float = 0.0, min_yield: float = 0.0, objective: str = "recovery", penalty_weight: float = 10.0,
    yield_scale: float = 1.0, yield_tiebreak: float = 0.0,
    start_time_s: float | None = None, end_time_s: float | None = None, eps: float = 1e-12,
) -> FixedWindowObjectiveResult:
    """Torch yield/recovery/purity over the FIXED window ``[start_idx, end_idx]`` + squared-hinge
    penalty; ``loss = -score + penalty_weight * penalty``. Mirrors ``yield_cal`` for the metrics
    (clamps tiny negative concentrations like ``optimize_collection_window``).

    ``objective``: ``"recovery"`` (dimensionless, default) or ``"yield"`` (``yield_g / yield_scale``).
    For ``"recovery"`` with a free loading, recovery is maximised by "collect everything" at low
    loading (recovery -> 1, tiny absolute yield); ``yield_tiebreak`` (alpha) adds a small productivity
    term ``score = recovery + alpha * yield_g / yield_scale`` so the gradient prefers the higher-yield
    point among purity-feasible near-max-recovery operating conditions (alpha 0 = pure recovery).

    Window: pass ``start_idx``/``end_idx`` (fixed-INDEX), or ``start_time_s``/``end_time_s`` (fixed-TIME
    -- indices recovered via ``searchsorted`` on the DETACHED time axis, so as the operating conditions
    stretch the grid the window tracks the same PHYSICAL interval and its bounds don't backprop; the
    D2.2.1 stability fix for the index-drift the fixed-index window suffered under op changes).
    """
    if start_time_s is not None and end_time_s is not None:
        td = curve_t[:, 0].detach().contiguous()  # contiguous boundary -> no searchsorted perf warning
        ntp = td.shape[0]
        start_idx = int(torch.searchsorted(td, torch.as_tensor(float(start_time_s), dtype=td.dtype)).clamp(0, ntp - 1))
        end_idx = int(torch.searchsorted(td, torch.as_tensor(float(end_time_s), dtype=td.dtype)).clamp(0, ntp - 1))
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, ntp - 1)
    seg = curve_t[start_idx:end_idx + 1]
    t = seg[:, 0]
    conc = torch.clamp_min(seg[:, 2:], 0.0)
    collected = torch.trapezoid(conc, t, dim=0)            # (npr,)
    total = collected.sum()
    p = collected / total.clamp_min(eps)
    ai = list(acid_idx) if acid_idx else [0]
    z = total.new_zeros(())
    a = p[ai].sum()
    m = p[list(main_idx)].sum() if main_idx else z
    b = p[list(basic_idx)].sum() if basic_idx else z

    yield_g = total / 60.0 * MOL_TO_G_L
    whole = torch.trapezoid(torch.clamp_min(curve_t[:, 2:], 0.0).sum(dim=1), curve_t[:, 0])
    recovery = total / whole.clamp_min(eps)

    pa = torch.relu(a - acid_max) ** 2
    pm = torch.relu(main_min - m) ** 2 if main_idx else z
    pb = torch.relu(b - basic_max) ** 2 if basic_idx else z
    pt = (torch.relu(min_total - total) ** 2) / (min_total ** 2 + eps) if min_total > 0 else z
    # productivity floor (#4): squared-hinge on the ABSOLUTE collected yield, so the optimiser leaves the
    # free-loading recovery=1 degenerate (collect-everything at low loading -> tiny yield) and the flat
    # objective regains a gradient. min_yield = min_yield_fraction * yield_scale (the caller sets it).
    py = (torch.relu(min_yield - yield_g) ** 2) / (min_yield ** 2 + eps) if min_yield > 0 else z
    penalties = {"acid": pa, "main": pm, "basic": pb, "min_total": pt, "min_yield": py}
    penalty = pa + pm + pb + pt + py

    norm_yield = yield_g / yield_scale
    score = (recovery + yield_tiebreak * norm_yield) if objective == "recovery" else norm_yield
    loss = -score + penalty_weight * penalty
    return FixedWindowObjectiveResult(loss, score, yield_g, recovery, total, a, m, b, penalty, penalties)
