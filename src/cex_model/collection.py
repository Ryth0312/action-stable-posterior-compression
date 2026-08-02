"""Collection window optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from cex_model.sma import MOL_TO_G_L


@dataclass
class CollectionWindow:
    """Optimal or candidate collection interval."""

    start_index: int
    end_index: int
    start_time_s: float
    end_time_s: float
    yield_g: float
    purities: np.ndarray
    acid_fraction: float
    main_fraction: float
    basic_fraction: float
    feasible: bool


@dataclass
class CollectionOptimizationResult:
    """Full collection window search results."""

    best: CollectionWindow
    yield_grid: np.ndarray
    feasible_windows: list[CollectionWindow]


def yield_cal(
    curve: np.ndarray,
    start_index: int,
    end_index: int,
    *,
    acid_max: float = 0.10,
    main_min: float = 0.0,
    basic_max: float = 1.0,
    acid_idx: list[int] | None = None,
    main_idx: list[int] | None = None,
    basic_idx: list[int] | None = None,
    rt_min: float = 5.0,
    min_total: float = 0.0,
) -> tuple[float, bool, np.ndarray]:
    """Calculate yield and purity for a collection window.

    Extends MATLAB ``yield_cal.m`` with per-group purity constraints. Component
    indices (into the protein columns, 0-based) assign each peak to a group;
    feasibility requires ``acid<=acid_max`` and (if those groups are given)
    ``main>=main_min`` and ``basic<=basic_max``. Defaults (acid_idx=[0], no main/
    basic groups) reproduce the original acid-only behavior. ``min_total`` rejects
    DEGENERATE windows that collect a negligible mass (the integrated protein is
    below ``min_total``): their purity *ratios* can pass while they collect ~nothing
    (e.g. an empty tail window), which would otherwise be reported as a 0-yield
    "feasible" optimum and flatten the optimizer's yield objective.

    Parameters
    ----------
    curve : [time, salt, protein...] — proteins in g/L if already scaled
    """
    segment = curve[start_index : end_index + 1]
    t = segment[:, 0]
    conc = segment[:, 2:]
    collected = trapezoid(conc, t, axis=0)
    total = np.sum(collected)
    if total <= 0:
        return 0.0, False, np.zeros(conc.shape[1])
    p = collected / total
    y = total / 60.0 * MOL_TO_G_L  # MATLAB yield scaling

    ai = [0] if acid_idx is None else list(acid_idx)
    mi = list(main_idx) if main_idx is not None else []
    bi = list(basic_idx) if basic_idx is not None else []
    acid_frac = float(np.sum(p[ai])) if ai else 0.0
    feasible = bool(total >= min_total) and acid_frac <= acid_max
    if mi:
        feasible = feasible and float(np.sum(p[mi])) >= main_min
    if bi:
        feasible = feasible and float(np.sum(p[bi])) <= basic_max
    return float(y), feasible, p


def optimize_collection_window(
    curve: np.ndarray,
    *,
    grid: int = 50,
    min_gap: int = 4,
    acid_max: float = 0.10,
    main_min: float = 0.0,
    basic_max: float = 1.0,
    acid_idx: list[int] | None = None,
    main_idx: list[int] | None = None,
    basic_idx: list[int] | None = None,
    rt_min: float = 5.0,
    min_collect_fraction: float = 0.05,
) -> CollectionOptimizationResult:
    """Exhaustive search for the highest-yield collection window meeting purity.

    Extends MATLAB ``searchdigitaltwinwindow.m`` with per-group purity limits
    (see :func:`yield_cal`). ``acid_idx``/``main_idx``/``basic_idx`` map protein
    columns to groups; pass them from the component types. ``min_collect_fraction``
    requires a feasible window to collect at least that fraction of the curve's TOTAL
    eluted protein, which rejects degenerate near-zero-yield windows (e.g. an empty
    tail whose purity ratios pass but which collect ~nothing) — those would otherwise
    be reported as a 0-yield "feasible" optimum and flatten the optimizer's objective.
    """
    # Clip tiny negative concentrations (a stiff-solver "smooth" non-negativity
    # leaves small negatives) so integrated purity fractions stay physical
    # (otherwise a near-zero basic tail can integrate to a negative fraction).
    curve = np.array(curve, dtype=float)
    curve[:, 2:] = np.maximum(curve[:, 2:], 0.0)

    # A feasible window must collect >= min_collect_fraction of the whole curve's protein.
    curve_total = float(trapezoid(curve[:, 2:].sum(axis=1), curve[:, 0]))
    min_total = max(min_collect_fraction, 0.0) * curve_total

    ai = [0] if acid_idx is None else list(acid_idx)
    mi = list(main_idx) if main_idx is not None else []
    bi_ = list(basic_idx) if basic_idx is not None else []

    def _groups(p: np.ndarray) -> tuple[float, float, float]:
        a = float(np.sum(p[ai])) if ai and len(p) else 0.0
        m = float(np.sum(p[mi])) if mi else (float(p[1]) if len(p) > 1 else 0.0)
        b = float(np.sum(p[bi_])) if bi_ else (float(np.sum(p[2:])) if len(p) > 2 else 0.0)
        return a, m, b

    cal_index = np.floor(np.linspace(0, curve.shape[0] - 1, grid)).astype(int)
    yield_grid = np.zeros((grid, grid))
    windows: list[CollectionWindow] = []

    for i in range(grid):
        for j in range(grid):
            if j > i + min_gap:
                y, flag, p = yield_cal(
                    curve, cal_index[i], cal_index[j], acid_max=acid_max,
                    main_min=main_min, basic_max=basic_max,
                    acid_idx=ai, main_idx=mi, basic_idx=bi_, rt_min=rt_min,
                    min_total=min_total,
                )
                if flag:
                    yield_grid[i, j] = y
                    a, m, b = _groups(p)
                    windows.append(
                        CollectionWindow(
                            start_index=int(cal_index[i]),
                            end_index=int(cal_index[j]),
                            start_time_s=float(curve[cal_index[i], 0]),
                            end_time_s=float(curve[cal_index[j], 0]),
                            yield_g=y,
                            purities=p,
                            acid_fraction=a,
                            main_fraction=m,
                            basic_fraction=b,
                            feasible=True,
                        )
                    )

    if not windows:
        return CollectionOptimizationResult(
            best=CollectionWindow(0, 0, 0, 0, 0, np.array([]), 0, 0, 0, False),
            yield_grid=yield_grid,
            feasible_windows=[],
        )

    best_idx = int(np.argmax(yield_grid))
    bi, bj = np.unravel_index(best_idx, yield_grid.shape)
    _, _, p_best = yield_cal(
        curve, cal_index[bi], cal_index[bj], acid_max=acid_max, main_min=main_min,
        basic_max=basic_max, acid_idx=ai, main_idx=mi, basic_idx=bi_, min_total=min_total,
    )
    a, m, b = _groups(p_best)
    best = CollectionWindow(
        start_index=int(cal_index[bi]),
        end_index=int(cal_index[bj]),
        start_time_s=float(curve[cal_index[bi], 0]),
        end_time_s=float(curve[cal_index[bj], 0]),
        yield_g=float(yield_grid[bi, bj]),
        purities=p_best,
        acid_fraction=a,
        main_fraction=m,
        basic_fraction=b,
        feasible=True,
    )
    return CollectionOptimizationResult(
        best=best, yield_grid=yield_grid, feasible_windows=windows
    )


def _group_fractions(
    p: np.ndarray, acid_idx, main_idx, basic_idx
) -> tuple[float, float, float]:
    """acid / main / basic purity fractions from a per-component purity vector."""
    ai = [0] if acid_idx is None else list(acid_idx)
    mi = list(main_idx) if main_idx is not None else []
    bi = list(basic_idx) if basic_idx is not None else []
    a = float(np.sum(p[ai])) if ai and len(p) else 0.0
    m = float(np.sum(p[mi])) if mi else (float(p[1]) if len(p) > 1 else 0.0)
    b = float(np.sum(p[bi])) if bi else (float(np.sum(p[2:])) if len(p) > 2 else 0.0)
    return a, m, b


def apply_concentration_window(
    curve: np.ndarray,
    c_start: float,
    c_end: float,
    *,
    acid_max: float = 0.10,
    main_min: float = 0.0,
    basic_max: float = 1.0,
    acid_idx: list[int] | None = None,
    main_idx: list[int] | None = None,
    basic_idx: list[int] | None = None,
    rt_min: float = 5.0,
) -> CollectionWindow:
    """Collection window from FIXED total-protein concentration triggers.

    Start collecting when the rising total-protein signal first crosses
    ``c_start`` (g/L) and stop when the falling signal drops below ``c_end`` --
    the production "fixed UV-trigger" window, as opposed to a per-batch optimal
    ``[index]`` window. Yield/purity and feasibility under the given limits are
    then evaluated on that window by :func:`yield_cal`. If the peak never reaches
    the triggers, an infeasible (empty) window is returned.
    """
    curve = np.array(curve, dtype=float)
    curve[:, 2:] = np.maximum(curve[:, 2:], 0.0)
    n_prot = curve.shape[1] - 2
    c_tot = curve[:, 2:].sum(axis=1)
    n = len(c_tot)
    pk = int(np.argmax(c_tot))
    if n == 0 or c_tot[pk] <= 0.0 or c_tot[pk] < max(c_start, c_end):
        return CollectionWindow(0, 0, 0.0, 0.0, 0.0, np.zeros(n_prot), 0.0, 0.0, 0.0, False)

    rising = np.where(c_tot[: pk + 1] >= c_start)[0]
    i_start = int(rising[0]) if rising.size else pk
    falling = np.where(c_tot[pk:] >= c_end)[0]
    i_end = int(pk + falling[-1]) if falling.size else pk
    if i_end <= i_start:
        i_end = min(i_start + 1, n - 1)

    y, feasible, p = yield_cal(
        curve, i_start, i_end, acid_max=acid_max, main_min=main_min, basic_max=basic_max,
        acid_idx=acid_idx, main_idx=main_idx, basic_idx=basic_idx, rt_min=rt_min,
    )
    a, m, b = _group_fractions(p, acid_idx, main_idx, basic_idx)
    return CollectionWindow(
        start_index=i_start, end_index=i_end,
        start_time_s=float(curve[i_start, 0]), end_time_s=float(curve[i_end, 0]),
        yield_g=float(y), purities=p, acid_fraction=a, main_fraction=m, basic_fraction=b,
        feasible=bool(feasible),
    )


def fixed_window_thresholds(
    start_concs, end_concs, quantile: float = 0.9
) -> tuple[float, float]:
    """Conservative fixed total-protein triggers from per-batch optimal-window edges.

    ``start_concs``/``end_concs`` are the total-protein concentrations at each
    batch's *optimal* collection start/end. A high ``quantile`` (1.0 = MATLAB's
    ``max``) yields triggers that start collection later and stop it earlier than
    most batches' optima -- conservative on purity at a yield cost; lower the
    quantile to trade robustness for yield.
    """
    s = np.asarray(start_concs, dtype=float)
    e = np.asarray(end_concs, dtype=float)
    if s.size == 0 or e.size == 0:
        return 0.0, 0.0
    return float(np.quantile(s, quantile)), float(np.quantile(e, quantile))
