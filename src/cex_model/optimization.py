"""Gradient and multi-objective optimization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution

from cex_model.column import ColumnParameters
from cex_model.collection import optimize_collection_window, yield_cal
from cex_model.components import ComponentSet, ComponentType
from cex_model.corrections import LoadingCorrection
from cex_model.gradients import build_fitting_inlet
from cex_model.simulator import ChromatographySimulator


@dataclass
class GradientOptimizationConfig:
    """Settings for gradient optimization."""

    bounds_start: tuple[float, float] = (5.0, 30.0)
    bounds_end: tuple[float, float] = (50.0, 100.0)
    bounds_cv: tuple[float, float] | None = None
    optimize_start: bool = True
    optimize_end: bool = True
    optimize_cv: bool = False
    random_seed: int = 42
    de_popsize: int = 15
    de_maxiter: int = 100
    acid_max: float = 0.10
    # >1 / -1 parallelize the DE population (lossless; forced to 1 with a surrogate
    # backend, which is unpicklable and already fast). Default 1 = serial.
    workers: int = 1


@dataclass
class GradientOptimizationResult:
    """Output from gradient optimization."""

    gradient_start_pct: float
    gradient_end_pct: float
    elution_cv: float
    mean_yield: float
    objective: float
    details: list[dict] = field(default_factory=list)


def combined_objective(
    sim: ChromatographySimulator,
    test_conditions: Sequence[dict],
    gradient: tuple[float, float, float],
    buffer_a: float,
    buffer_b: float,
    elution_cv: float,
    acid_max: float = 0.10,
    n_time_points: int | None = None,
) -> float:
    """Mean normalized yield across test conditions (MATLAB ``combined_opt.m``)."""
    g_start, g_end = gradient[0], gradient[1]
    cv = gradient[2] if len(gradient) > 2 else elution_cv
    # Surrogate backends are trained for a fixed grid; honor it (else physics default).
    n_tp = n_time_points if n_time_points is not None else 2000
    yields = []
    for cond in test_conditions:
        fractions = np.array(cond["fractions_pct"])
        comps = ComponentSet.from_parameter_table(
            np.vstack(
                [
                    fractions,
                    sim.components.to_parameter_table()[1:5, : len(fractions)],
                ]
            ),
            # Preserve component names so a surrogate backend's name validation
            # passes (from_parameter_table would otherwise default to C1..Cn).
            names=sim.components.names[: len(fractions)],
        )
        sim.components = comps
        result = sim.run_forward_case(
            buffer_a=buffer_a,
            buffer_b=buffer_b,
            gradient_start_pct=g_start,
            gradient_end_pct=g_end,
            elution_cv=cv,
            load_amount_g_l=cond["load_amount_g_l"],
            n_time_points=n_tp,
        )
        curve = result.elution_curve()
        opt = optimize_collection_window(curve, acid_max=acid_max)
        norm_yield = opt.best.yield_g / sim.column.rt / cond["load_amount_g_l"]
        yields.append(norm_yield)
    return -float(np.mean(yields))


class _GradientObjective:
    """Top-level (picklable) DE objective for :func:`optimize_gradient`."""

    def __init__(self, sim, test_conditions, buffer_a, buffer_b, elution_cv,
                 acid_max, n_tp, opt_start, opt_end, opt_cv):
        self.sim = sim
        self.test_conditions = test_conditions
        self.buffer_a, self.buffer_b = buffer_a, buffer_b
        self.elution_cv, self.acid_max, self.n_tp = elution_cv, acid_max, n_tp
        self.opt_start, self.opt_end, self.opt_cv = opt_start, opt_end, opt_cv

    def __call__(self, x) -> float:
        idx = 0
        g_start = float(x[idx]) if self.opt_start else 25.0
        idx += int(self.opt_start)
        g_end = float(x[idx]) if self.opt_end else 71.0
        idx += int(self.opt_end)
        cv = float(x[idx]) if self.opt_cv else self.elution_cv
        return combined_objective(
            self.sim, self.test_conditions, (g_start, g_end, cv), self.buffer_a,
            self.buffer_b, cv, acid_max=self.acid_max, n_time_points=self.n_tp,
        )


def optimize_gradient(
    column: ColumnParameters,
    components: ComponentSet,
    test_conditions: Sequence[dict],
    *,
    buffer_a: float,
    buffer_b: float,
    elution_cv: float = 15.0,
    config: GradientOptimizationConfig | None = None,
    correction: LoadingCorrection | None = None,
    method: str = "RK23",
    backend=None,
) -> GradientOptimizationResult:
    """Optimize gradient start/end (and optionally length) via differential evolution.

    Matches MATLAB ``gradient_opt_V1_HLXSYN.m`` GA workflow.

    Pass a trained ``backend`` (e.g. ``TorchSurrogateBackend``) to evaluate the
    objective with the ANN surrogate instead of the mechanistic ODE — the DE
    sweep then costs milliseconds per curve instead of tens of seconds. The
    surrogate must be trained on the ``forward`` workflow over the gradient
    ranges being optimized (see generate_surrogate_dataset.py --design-space).
    """
    cfg = config or GradientOptimizationConfig()
    if backend is not None:
        n_tp = int(getattr(backend, "metadata", {}).get("n_time_points", 1000))
        sim = ChromatographySimulator(
            column=column, components=components, n_time_points=n_tp, backend=backend
        )
    else:
        n_tp = None  # physics default (2000) inside combined_objective
        sim_kwargs = dict(column=column, components=components, method=method)
        if correction is not None:
            sim_kwargs["correction"] = correction
        sim = ChromatographySimulator(**sim_kwargs)

    bounds = []
    x0 = []
    if cfg.optimize_start:
        bounds.append(cfg.bounds_start)
        x0.append(25.0)
    if cfg.optimize_end:
        bounds.append(cfg.bounds_end)
        x0.append(71.0)
    if cfg.optimize_cv and cfg.bounds_cv:
        bounds.append(cfg.bounds_cv)
        x0.append(elution_cv)

    opt_cv = bool(cfg.optimize_cv and cfg.bounds_cv)
    obj = _GradientObjective(
        sim, test_conditions, buffer_a, buffer_b, elution_cv, cfg.acid_max, n_tp,
        cfg.optimize_start, cfg.optimize_end, opt_cv,
    )
    # Surrogate backends are unpicklable (and already ms/curve) -> keep serial.
    workers = 1 if backend is not None else cfg.workers
    updating = "deferred" if workers != 1 else "immediate"
    result = differential_evolution(
        obj, bounds=bounds, seed=cfg.random_seed, popsize=cfg.de_popsize,
        maxiter=cfg.de_maxiter, workers=workers, updating=updating,
    )

    x = result.x
    idx = 0
    g_start = float(x[idx]) if cfg.optimize_start else 25.0
    idx += int(cfg.optimize_start)
    g_end = float(x[idx]) if cfg.optimize_end else 71.0
    idx += int(cfg.optimize_end)
    cv = float(x[idx]) if cfg.optimize_cv and cfg.bounds_cv else elution_cv

    return GradientOptimizationResult(
        gradient_start_pct=g_start,
        gradient_end_pct=g_end,
        elution_cv=cv,
        mean_yield=-result.fun,
        objective=result.fun,
        details=[{"success": result.success, "nfev": result.nfev}],
    )


# --------------------------------------------------------------------------- #
# General process optimization: any subset of {loading, flow, feed length,
# gradient start/end/length} optimized within user bounds, maximizing the
# highest-yield collection window that meets acid/main/basic purity limits.
# --------------------------------------------------------------------------- #
PROCESS_VARS: tuple[str, ...] = (
    "loading_g_l", "flow_rate", "feed_cv",
    "gradient_start_pct", "gradient_end_pct", "gradient_cv",
)


@dataclass
class ProcessOptimizationConfig:
    """Decision-variable bounds + purity limits for :func:`optimize_process`.

    ``bounds`` lists the variables to OPTIMIZE, each ``name -> (lo, hi)`` (the
    user-set feasible domain, e.g. ``{"loading_g_l": (25, 45)}``). Any variable
    not in ``bounds`` is held at ``fixed[name]`` or a default. Purity is enforced
    on the collected pool by component-type group (acid/main/basic).
    """

    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)
    acid_max: float = 0.20
    main_min: float = 0.70
    basic_max: float = 0.10
    hold_cv: float = 4.0
    n_time_points: int = 2000
    grid: int = 50
    de_popsize: int = 15
    de_maxiter: int = 60
    random_seed: int = 42
    # >1 (or -1 = all cores) evaluates the differential-evolution population in
    # parallel. Lossless given the seed (forces updating='deferred', which yields
    # the same trajectory for any worker count). Default 1 = serial, unchanged.
    workers: int = 1
    # With a surrogate backend, ODE-verify this many of the best DE candidates (not
    # just the single best) and return the best ODE-FEASIBLE one. The surrogate's
    # few-% purity error can make its top point ODE-infeasible near a narrow
    # feasibility boundary while feasible points sit just below it in the population.
    # ODE solves are cheap (~0.2 s), so ~12 adds ~2 s. Ignored when backend is None.
    verify_top_k: int = 12
    # What to MAXIMIZE over operating conditions: "yield" = absolute collected protein (yield_g, the
    # original behavior), or "recovery" = total-protein recovery fraction = window yield / whole-curve
    # eluted yield (∈[0,1]). They DIVERGE when loading/feed_cv vary (more load -> more absolute yield but
    # often lower recovery); "recovery" matches the MATLAB "max 收率%" at a given loading.
    objective: str = "yield"
    # recovery objective only: among ODE-feasible candidates, select by recovery + this *
    # yield_g/max_feasible_yield instead of pure recovery, so the productive (higher-yield) point wins
    # the free-loading recovery plateau. 0 (default) = pure recovery. Matches the gradient path's
    # tie-break so --compare-ode is a fair comparison (same selection rule).
    recovery_yield_tiebreak: float = 0.0
    # #4 productivity floor (absolute yield_g): in the lossless verify, drop purity-feasible candidates
    # below this (fall back to the highest-yield feasible if none clear it), excluding the free-loading
    # recovery=1 degenerate. The gradient CLI sets it to frac * screened-max-yield so gradient and pure-ODE
    # share the SAME absolute floor (fair). 0 (default) = off.
    min_yield: float = 0.0


@dataclass
class ProcessOptimizationResult:
    """Optimal inputs and the resulting purity-feasible collection window."""

    inputs: dict[str, float]
    yield_g: float
    acid_fraction: float
    main_fraction: float
    basic_fraction: float
    collect_start_s: float
    collect_end_s: float
    feasible: bool
    nfev: int
    recovery: float = float("nan")  # window mass / whole-curve eluted mass (set by ode_verify_*)


_PROCESS_DEFAULTS = {
    "feed_cv": 1.0, "gradient_start_pct": 25.0, "gradient_end_pct": 71.0, "gradient_cv": 15.0,
}


class _ProcessObjective:
    """Top-level (picklable) DE objective for :func:`optimize_process`.

    A module-level callable rather than a closure so ``differential_evolution``
    can ship it to worker processes (``workers``). Each call builds a fresh
    simulator from the candidate inputs (purely functional, no shared state).
    """

    def __init__(self, *, column, components, buffer_a, buffer_b, fractions, opt_vars,
                 fixed, defaults, acid_idx, main_idx, basic_idx, acid_max, main_min,
                 basic_max, hold_cv, n_time_points, grid, correction, method,
                 residual_corrector=None, backend=None, objective="yield"):
        self.column = column
        self.components = components
        self.buffer_a = buffer_a
        self.buffer_b = buffer_b
        self.fractions = fractions
        self.opt_vars = opt_vars
        self.fixed = fixed
        self.defaults = defaults
        self.acid_idx, self.main_idx, self.basic_idx = acid_idx, main_idx, basic_idx
        self.acid_max, self.main_min, self.basic_max = acid_max, main_min, basic_max
        self.hold_cv, self.n_time_points, self.grid = hold_cv, n_time_points, grid
        self.correction, self.method = correction, method
        self.residual_corrector = residual_corrector  # optional ML total-residual layer
        self.backend = backend  # optional surrogate backend (fast forward); None = real ODE
        self.objective = objective  # "yield" (absolute) or "recovery" (window/whole-curve fraction)

    def resolve(self, x: np.ndarray) -> dict[str, float]:
        vals = {v: float(x[k]) for k, v in enumerate(self.opt_vars)}
        for v in PROCESS_VARS:
            if v in vals:
                continue
            if v in self.fixed:
                vals[v] = float(self.fixed[v])
            elif v in self.defaults:
                vals[v] = float(self.defaults[v])
            else:
                raise ValueError(f"{v} must be optimized (in bounds) or set in fixed")
        return vals

    def _forward(self, vals: dict[str, float]):
        """RK23 forward → (curve, whole_curve_yield), the SPEC-INDEPENDENT part of evaluate. Returns
        (None, 0.0) if the solve fails. Factored out so a curve can be collected under MANY purity specs at
        the cost of one ODE solve (the D3.2 multi-spec dataset)."""
        col = self.column
        if abs(vals["flow_rate"] - self.column.flow_rate) > 1e-12:
            col = replace(
                self.column, flow_rate=vals["flow_rate"],
                superficial_velocity=self.column.superficial_velocity
                * vals["flow_rate"] / self.column.flow_rate,
            )
        inlet = build_fitting_inlet(
            buffer_a=self.buffer_a, buffer_b=self.buffer_b,
            gradient_start_pct=vals["gradient_start_pct"], gradient_end_pct=vals["gradient_end_pct"],
            elution_cv=vals["gradient_cv"], rt_min=col.rt, load_amount_g_l=vals["loading_g_l"],
            component_fractions_pct=self.fractions, feed_cv=vals["feed_cv"], hold_cv=self.hold_cv,
        )
        sim_kwargs = dict(column=col, components=self.components, method=self.method)
        if self.correction is not None:
            sim_kwargs["correction"] = self.correction
        # Surrogate backend (ms/curve) for the DE search; the backend's fixed grid wins.
        n_tp = self.n_time_points
        if self.backend is not None:
            n_tp = int(getattr(self.backend, "metadata", {}).get("n_time_points", self.n_time_points))
            sim_kwargs["backend"] = self.backend
            sim_kwargs["n_time_points"] = n_tp
        sim = ChromatographySimulator(**sim_kwargs)
        try:
            result = sim.simulate(inlet, vals["loading_g_l"], n_time_points=n_tp, t_start=0.0)
        except RuntimeError:
            return None, 0.0
        curve = result.elution_curve()
        if self.residual_corrector is not None:
            # ML total-residual layer (gated): scales the total so yield + the total-protein
            # collection-window become ML-driven; out-of-envelope -> gate 0 -> unchanged.
            curve, _ = self.residual_corrector.correct_curve(
                curve, [vals["loading_g_l"], vals["gradient_cv"],
                        vals["gradient_start_pct"], vals["gradient_end_pct"]])
        # Whole-curve eluted total (clipped, same yield_cal formula as the window) for the recovery%
        # objective; recovery = window yield / whole yield is then a clean fraction in [0, 1].
        clipped = np.array(curve, dtype=float)
        clipped[:, 2:] = np.maximum(clipped[:, 2:], 0.0)
        whole_yield = float(yield_cal(clipped, 0, clipped.shape[0] - 1, acid_max=1.0, basic_max=1.0)[0])
        return curve, whole_yield

    def evaluate(self, vals: dict[str, float]):
        curve, whole_yield = self._forward(vals)
        if curve is None:
            return None, 0.0
        opt = optimize_collection_window(
            curve, grid=self.grid, acid_max=self.acid_max, main_min=self.main_min,
            basic_max=self.basic_max, acid_idx=self.acid_idx, main_idx=self.main_idx,
            basic_idx=self.basic_idx,
        )
        return opt, whole_yield

    def score(self, opt, whole_yield) -> float:
        """Value to MAXIMIZE: recovery fraction (window/whole) or absolute yield_g; 0 if infeasible."""
        if opt is None or not opt.best.feasible:
            return 0.0
        if self.objective == "recovery":
            return opt.best.yield_g / whole_yield if whole_yield > 1e-12 else 0.0
        return opt.best.yield_g

    def __call__(self, x: np.ndarray) -> float:
        opt, whole = self.evaluate(self.resolve(x))
        return -self.score(opt, whole)


def verify_process_candidates(obj, candidates, *, opt_vars, top_k, recovery_yield_tiebreak=0.0, min_yield=0.0):
    """ODE-verify candidate operating points and return the best ODE-FEASIBLE one.

    ``obj`` must evaluate on the real ODE (caller sets ``obj.backend = None``). Shared by the
    DE/surrogate path (verify the DE's top-K) and the gradient-refine path (verify the refined
    candidates) so the lossless re-evaluation is identical across paths. De-duplicates by the rounded
    opt-vars, evaluates up to ``top_k`` distinct candidates, keeps the first as the infeasible
    fallback. Returns ``(best_opt, best_vals, best_whole)`` -- a CollectionOptimizationResult, the
    operating-condition dict, and the winner's whole-curve eluted yield (for the recovery report).

    ``recovery_yield_tiebreak`` (alpha, recovery objective only): among purity-feasible candidates,
    select by ``recovery + alpha * yield_g / max_feasible_yield`` instead of pure recovery, so the
    higher-yield (productive) point wins on the free-loading recovery plateau. 0 (default) = pure
    ``obj.score``, first-wins on ties -- unchanged for the DE/surrogate path.

    ``min_yield`` (#4 productivity floor, absolute yield_g units): purity-feasible candidates below it are
    dropped from selection (kept only as a fallback if NONE clear the floor -> the highest-yield feasible),
    so the free-loading recovery=1 degenerate is excluded on BOTH the gradient and pure-ODE paths at the
    SAME absolute floor. 0 (default) = off.
    """
    feasible = []  # (opt, vals, whole, score, yield_g)
    fallback = None
    seen: set = set()
    for x in candidates:
        vals = obj.resolve(np.asarray(x, dtype=float))
        key = tuple(round(vals[v], 6) for v in opt_vars)
        if key in seen:
            continue
        seen.add(key)
        opt, whole = obj.evaluate(vals)
        if fallback is None:  # the first (best-proposed) candidate -> infeasible fallback
            fallback = (opt, vals, whole)
        if opt is not None and opt.best.feasible:
            feasible.append((opt, vals, whole, obj.score(opt, whole), float(opt.best.yield_g)))
        if len(seen) >= top_k:
            break
    if not feasible:
        return fallback if fallback is not None else (None, None, None)
    # #4 productivity floor: prefer purity-feasible candidates whose ABSOLUTE yield clears the floor (drops
    # the free-loading recovery=1 degenerate on both paths); if NONE clear it, fall back to the highest-
    # yield feasible (least floor violation). min_yield=0 -> pool = feasible (unchanged). f[4] = yield_g.
    pool = [f for f in feasible if f[4] >= min_yield] if min_yield > 0.0 else feasible
    if not pool:
        best = max(feasible, key=lambda f: f[4])  # all sub-floor -> closest to the floor (max yield)
    elif recovery_yield_tiebreak > 0.0 and getattr(obj, "objective", "yield") == "recovery":
        ymax = max((f[4] for f in pool), default=1.0) or 1.0
        best = max(pool, key=lambda f: f[3] + recovery_yield_tiebreak * f[4] / ymax)
    else:
        best = max(pool, key=lambda f: f[3])  # max obj.score; ties -> first (matches the old loop)
    return best[0], best[1], best[2]


def optimize_process(
    column: ColumnParameters,
    components: ComponentSet,
    *,
    buffer_a: float,
    buffer_b: float,
    fractions_pct: Sequence[float] | None = None,
    config: ProcessOptimizationConfig | None = None,
    correction: LoadingCorrection | None = None,
    method: str = "RK23",
    residual_corrector=None,
    backend=None,
) -> ProcessOptimizationResult:
    """Find the operating inputs that maximize yield subject to purity limits.

    Optimizes any subset of loading / flow rate / feed length / gradient start /
    gradient end / gradient length (those listed in ``config.bounds``) within the
    user-set bounds, via differential evolution over the mechanistic ODE. For each
    candidate it builds the curve, finds the highest-yield collection window that
    satisfies ``acid<=acid_max`` & ``main>=main_min`` & ``basic<=basic_max``
    (groups taken from the component types), and maximizes that window's yield.
    Infeasible candidates (no window meets purity, or the ODE fails) score 0 so
    the search avoids them.

    Pass a trained surrogate ``backend`` (e.g. ``TorchSurrogateBackend``) to run the DE
    on the fast ANN forward (~ms/curve, 200-3600x), then the returned optimum is
    **re-evaluated on the real ODE** (lossless: the surrogate only proposes; the reported
    yield/purity/window are exact mechanistic values). Surrogate runs serial (unpicklable).
    """
    cfg = config or ProcessOptimizationConfig()
    fractions = np.asarray(
        fractions_pct if fractions_pct is not None else components.fraction_array(), dtype=float
    )
    acid_idx = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.ACID]
    main_idx = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.MAIN]
    basic_idx = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.BASIC]
    if not acid_idx:
        acid_idx = [0]

    opt_vars = [v for v in PROCESS_VARS if v in cfg.bounds]
    if not opt_vars:
        raise ValueError("config.bounds is empty; specify at least one variable to optimize")
    defaults = {"flow_rate": column.flow_rate, **_PROCESS_DEFAULTS}

    obj = _ProcessObjective(
        column=column, components=components, buffer_a=buffer_a, buffer_b=buffer_b,
        fractions=fractions, opt_vars=opt_vars, fixed=cfg.fixed, defaults=defaults,
        acid_idx=acid_idx, main_idx=main_idx, basic_idx=basic_idx, acid_max=cfg.acid_max,
        main_min=cfg.main_min, basic_max=cfg.basic_max, hold_cv=cfg.hold_cv,
        n_time_points=cfg.n_time_points, grid=cfg.grid, correction=correction, method=method,
        residual_corrector=residual_corrector, backend=backend, objective=cfg.objective,
    )

    bounds = [tuple(cfg.bounds[v]) for v in opt_vars]
    # A surrogate backend is unpicklable (and already ms/curve) -> keep serial; else
    # workers != 1 uses deferred updating (scipy), which is seed-deterministic and gives
    # the same trajectory for any worker count -> parallelism is lossless.
    workers = 1 if backend is not None else cfg.workers
    updating = "deferred" if workers != 1 else "immediate"
    result = differential_evolution(
        obj, bounds=bounds, seed=cfg.random_seed, popsize=cfg.de_popsize,
        maxiter=cfg.de_maxiter, tol=0.01, polish=False,
        workers=workers, updating=updating,
    )
    # LOSSLESS verify. With a surrogate backend the DE's single best can be ODE-INFEASIBLE near a
    # narrow purity boundary while feasible points exist lower in the population (the surrogate's small
    # purity error flips feasibility). Re-evaluate the top-K candidates on the real ODE and keep the
    # best ODE-FEASIBLE one; every reported number is then the exact mechanistic value. (backend=None ->
    # K=1: result.x is already the ODE optimum, so this reduces to the previous single re-evaluation.)
    obj.backend = None
    candidates = [np.asarray(result.x, dtype=float)]
    # Bring the final DE population into the verify pool when (a) a surrogate proposed result.x -- its
    # top point can be ODE-infeasible near a purity boundary while feasible points sit lower in the
    # population; or (b) a recovery tie-break is active -- result.x is just ONE point on the free-loading
    # recovery plateau, so the productive (higher-yield) point must be RE-SELECTED from the population
    # under the SAME recovery+yield rule the gradient path uses (else --compare-ode is unfair: pure-ODE
    # would "tie-break" over a single candidate, i.e. not at all). The DE objective is still pure recovery
    # here (the objective-level tie-break is deferred); this reranks the converged population, which is
    # enough for a fair baseline against the gradient refiner.
    use_population = backend is not None or cfg.recovery_yield_tiebreak > 0.0
    if use_population:
        pop = np.atleast_2d(np.asarray(getattr(result, "population", []), dtype=float))
        energies = np.asarray(getattr(result, "population_energies", []), dtype=float)
        if pop.ndim == 2 and pop.shape[0] and energies.shape[0] == pop.shape[0]:
            candidates += [pop[i] for i in np.argsort(energies)]  # best DE energy first
    top_k = max(1, cfg.verify_top_k) if use_population else 1
    best_opt, best_vals, best_whole = verify_process_candidates(
        obj, candidates, opt_vars=opt_vars, top_k=top_k,
        recovery_yield_tiebreak=cfg.recovery_yield_tiebreak, min_yield=cfg.min_yield)
    opt = best_opt
    feasible = opt is not None and opt.best.feasible
    win = opt.best if feasible else None
    recovery = (float(win.yield_g / best_whole) if (win is not None and best_whole and best_whole > 1e-12)
                else float("nan"))
    return ProcessOptimizationResult(
        inputs=best_vals,
        yield_g=float(win.yield_g) if win else 0.0,
        acid_fraction=float(win.acid_fraction) if win else float("nan"),
        main_fraction=float(win.main_fraction) if win else float("nan"),
        basic_fraction=float(win.basic_fraction) if win else float("nan"),
        collect_start_s=float(win.start_time_s) if win else float("nan"),
        collect_end_s=float(win.end_time_s) if win else float("nan"),
        feasible=bool(feasible),
        nfev=int(result.nfev),
        recovery=recovery,
    )


def ode_verify_operating_points(
    column: ColumnParameters, components: ComponentSet, *, buffer_a: float, buffer_b: float,
    points, config: ProcessOptimizationConfig | None = None,
    correction: LoadingCorrection | None = None, method: str = "RK23",
    recovery_yield_tiebreak: float = 0.0, min_yield: float = 0.0,
) -> ProcessOptimizationResult:
    """ODE-verify a list of operating-condition dicts (e.g. from the gradient refiner) and return the
    best ODE-FEASIBLE result. Lossless: every reported number is an exact mechanistic value. Each
    ``points`` entry is a full operating dict (PROCESS_VARS names); the variables listed in
    ``config.bounds`` are read from the point, the rest from ``config.fixed`` / defaults. Builds the
    SAME ``_ProcessObjective`` (backend=None) and uses the SAME ``verify_process_candidates`` as the
    DE/surrogate path, so the three search front-ends share one verify semantics.
    """
    cfg = config or ProcessOptimizationConfig()
    fractions = np.asarray(components.fraction_array(), dtype=float)
    acid_idx = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.ACID]
    main_idx = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.MAIN]
    basic_idx = [j for j, c in enumerate(components.components) if c.component_type == ComponentType.BASIC]
    if not acid_idx:
        acid_idx = [0]
    opt_vars = [v for v in PROCESS_VARS if v in cfg.bounds]
    defaults = {"flow_rate": column.flow_rate, **_PROCESS_DEFAULTS}
    obj = _ProcessObjective(
        column=column, components=components, buffer_a=buffer_a, buffer_b=buffer_b, fractions=fractions,
        opt_vars=opt_vars, fixed=cfg.fixed, defaults=defaults, acid_idx=acid_idx, main_idx=main_idx,
        basic_idx=basic_idx, acid_max=cfg.acid_max, main_min=cfg.main_min, basic_max=cfg.basic_max,
        hold_cv=cfg.hold_cv, n_time_points=cfg.n_time_points, grid=cfg.grid, correction=correction,
        method=method, backend=None, objective=cfg.objective)
    candidates = [np.array([float(p[v]) for v in opt_vars], dtype=float) for p in points]
    best_opt, best_vals, best_whole = verify_process_candidates(
        obj, candidates, opt_vars=opt_vars, top_k=len(candidates),
        recovery_yield_tiebreak=recovery_yield_tiebreak, min_yield=min_yield)
    opt = best_opt
    feasible = opt is not None and opt.best.feasible
    win = opt.best if feasible else None
    recovery = (float(win.yield_g / best_whole) if (win is not None and best_whole and best_whole > 1e-12)
                else float("nan"))
    return ProcessOptimizationResult(
        inputs=best_vals,
        yield_g=float(win.yield_g) if win else 0.0,
        acid_fraction=float(win.acid_fraction) if win else float("nan"),
        main_fraction=float(win.main_fraction) if win else float("nan"),
        basic_fraction=float(win.basic_fraction) if win else float("nan"),
        collect_start_s=float(win.start_time_s) if win else float("nan"),
        collect_end_s=float(win.end_time_s) if win else float("nan"),
        feasible=bool(feasible),
        nfev=len(candidates),
        recovery=recovery,
    )
