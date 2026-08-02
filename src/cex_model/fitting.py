"""SMA parameter fitting (direct iteration + global optimization)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution

from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet, SMAComponent
from cex_model.corrections import LoadingCorrection
from cex_model.metrics import (
    delta_tr_height,
    peak_retention_and_height,
    peak_shape_loss,
    retention_penalty,
    shape_correlation_loss,
)
from cex_model.simulator import ChromatographySimulator

logger = logging.getLogger(__name__)

PARAM_NAMES = ("keq", "kkin", "nu", "sigma")


def _make_simulator(
    column: ColumnParameters,
    components: ComponentSet,
    correction: LoadingCorrection | None,
    method: str,
) -> ChromatographySimulator:
    kwargs = dict(column=column, components=components, method=method)
    if correction is not None:
        kwargs["correction"] = correction
    return ChromatographySimulator(**kwargs)


@dataclass
class FittingConfig:
    """Configuration for SMA parameter fitting."""

    allow_delta_tr: float = 100.0
    allow_delta_height: float = 0.1
    nu_factor: float = 1.01
    kkin_factor: float = 1.05
    sim_cycles: int = 2
    max_inner_iter: int = 100
    use_global: bool = False
    de_popsize: int = 15
    de_maxiter: int = 100
    de_polish: bool = True  # final L-BFGS; expensive for stiff ODEs, off for quick demos
    # DE convergence tolerance. The old hard-coded 1e-8 is far tighter than a noisy
    # peak-shape/RMSE objective warrants; callers with a good warm-start can loosen
    # this so DE stops once the population has converged instead of burning the full
    # maxiter budget. Kept at 1e-8 by default for backward compatibility.
    de_tol: float = 1e-8
    # Seed the DE population with the warm-start (incoming components) via scipy's
    # ``x0``. Because DE keeps the best evaluated member, this guarantees the
    # returned objective never exceeds the warm-start's, so a warm-started global
    # fit cannot do worse than its starting point on the optimized metric.
    de_seed_x0: bool = True
    random_seed: int = 42
    # >1 / -1 parallelize the global-DE population (lossless; forces deferred
    # updating). Default 1 = serial. Only affects the global (use_global) path.
    workers: int = 1
    # Objective for global fitting: "rmse" preserves MATLAB-style total RMSE;
    # "peak_shape" optimizes retention/height/area/width while reports still use RMSE;
    # "shape_corr" optimizes the offset-tolerant Pearson shape + log-height (see
    # metrics.shape_correlation_loss) -- tolerates a small time offset instead of
    # penalising it, so a right-shape candidate is not driven unphysical by a residual
    # pump-delay / feed-timing / grid jitter offset.
    loss_mode: str = "rmse"
    # "shape_corr" knobs (only used when loss_mode == "shape_corr"; defaults are inert):
    #   shape_offset_shared: one tau per experiment (True) vs per-peak (False)
    #   shape_max_shift_s / shape_n_shift: time-offset search grid
    #   shape_height_weight: weight of the log-height term added to (1 - Pearson)
    shape_offset_shared: bool = True
    shape_max_shift_s: float = 180.0
    shape_n_shift: int = 61
    shape_height_weight: float = 1.0
    # Physical-prior / anchoring regularizers added to the global-DE objective
    # (all default 0 -> no change to existing callers). They break the wrong
    # basin the basics fell into (PROJECT_STATE §4.4/§7.0): the optimizer drove
    # the basics to nu below the main's and drifted the dominant peak far from its
    # experimental retention to lower the aggregate loss.
    #   * nu_monotonic_weight: soft penalty on any nu that is *smaller* than an
    #     earlier-eluting component's nu (components are assumed ordered by
    #     elution: acid -> main -> basic). Tolerates the small within-group
    #     inversions seen in real fits while strongly excluding basics-nu < main.
    #   * retention_weight + retention_tol_min: hinged squared per-peak retention
    #     error (see metrics.retention_penalty), anchoring each peak to its
    #     experimental position outside a tolerance window.
    nu_monotonic_weight: float = 0.0
    retention_weight: float = 0.0
    retention_tol_min: float = 0.0
    # Joint xlsx+AKTA recalibration ("C"): when >0, add a per-experiment RMSE of the
    # simulated TOTAL vs that experiment's continuous AKTA total-UV curve (peak-normalised,
    # gradient-start aligned) to the objective, so freeing nu/keq/sigma can satisfy the real
    # total shape as well as the sparse xlsx points. Requires each exp dict to carry an
    # "akta_trace" (cex_model.akta.AktaTrace); experiments without one are unaffected.
    akta_total_weight: float = 0.0
    fixed_params: dict[str, list[bool]] = field(default_factory=dict)


@dataclass
class FittingResult:
    """Output from parameter fitting."""

    components: ComponentSet
    rmse_per_peak: np.ndarray
    rmse_total: float
    history: list[dict]


def _determine_matrix(
    delta: np.ndarray, allow: np.ndarray, correlation: float
) -> tuple[np.ndarray, bool]:
    """Build adjustment matrix (MATLAB ``cal_determin_matrix.m``)."""
    dm = np.zeros_like(delta)
    for j in range(len(delta)):
        if delta[j] < -allow[j]:
            dm[j] = 1.0
        elif delta[j] > allow[j]:
            dm[j] = -1.0
    converged = np.all(dm == 0)
    return dm * correlation, converged


def _fit_nu_iteration(
    sim: ChromatographySimulator,
    exp_case: dict,
    table: np.ndarray,
    com_num: int,
    tr_exp: np.ndarray,
    height_exp: np.ndarray,
    cfg: FittingConfig,
) -> np.ndarray:
    nu_4 = table[3, :com_num].copy()
    nu_factor = cfg.nu_factor
    save_delta_tr: list[np.ndarray] = []

    for i in range(cfg.max_inner_iter):
        table[3, :com_num] = nu_4
        comps = ComponentSet.from_parameter_table(table[:, :com_num])
        sim.components = comps
        result, mse = sim.run_fitting_case(**exp_case)
        curve = result.elution_curve()
        d_tr, _ = delta_tr_height(curve, tr_exp, height_exp)
        dm, flag = _determine_matrix(d_tr, np.full(com_num, cfg.allow_delta_tr), 1.0)
        if flag:
            break
        save_delta_tr.append(d_tr)
        if i > 4:
            for j in range(com_num):
                if (
                    (save_delta_tr[-1][j] - save_delta_tr[-2][j])
                    * (save_delta_tr[-1][j] - save_delta_tr[-3][j])
                    * (save_delta_tr[-1][j] - save_delta_tr[-4][j])
                    == 0
                    and abs(d_tr[j]) > cfg.allow_delta_tr
                ):
                    nu_factor = (nu_factor - 1) / 2 + 1
                    break
        if nu_factor < 1.00004:
            break
        nu_4 = nu_4 * (nu_factor**dm)
    return table


def _fit_kkin_iteration(
    sim: ChromatographySimulator,
    exp_case: dict,
    table: np.ndarray,
    com_num: int,
    tr_exp: np.ndarray,
    height_exp: np.ndarray,
    cfg: FittingConfig,
) -> np.ndarray:
    kkin_4 = table[2, :com_num].copy()
    kkin_factor = cfg.kkin_factor

    for i in range(cfg.max_inner_iter):
        table[2, :com_num] = kkin_4
        comps = ComponentSet.from_parameter_table(table[:, :com_num])
        sim.components = comps
        result, mse = sim.run_fitting_case(**exp_case)
        curve = result.elution_curve()
        _, d_ht = delta_tr_height(curve, tr_exp, height_exp)
        dm, flag = _determine_matrix(
            d_ht, np.full(com_num, cfg.allow_delta_height), -1.0
        )
        if flag:
            break
        kkin_4 = kkin_4 * (kkin_factor**dm)
    return table


def fit_sma_parameters_direct(
    column: ColumnParameters,
    components: ComponentSet,
    experiment: dict,
    config: FittingConfig | None = None,
    *,
    correction: LoadingCorrection | None = None,
    method: str = "RK23",
) -> FittingResult:
    """Direct nu/kkin iterative fitting (MATLAB ``pushbutton9`` workflow)."""
    cfg = config or FittingConfig()
    sim = _make_simulator(column, components, correction, method)
    table = components.to_parameter_table()
    com_num = components.n_protein
    exp_data = experiment["data"]
    tr_exp, ht_exp = peak_retention_and_height(
        np.column_stack([exp_data[:, 0], np.zeros(len(exp_data)), exp_data[:, 1:]])
    )
    tr_exp, ht_exp = (
        np.array([exp_data[np.argmax(exp_data[:, j]), 0] for j in range(1, com_num + 1)]),
        np.array([np.max(exp_data[:, j]) for j in range(1, com_num + 1)]),
    )

    exp_case = {
        "buffer_a": experiment["buffer_a"],
        "buffer_b": experiment["buffer_b"],
        "gradient_start_pct": experiment["gradient_start_pct"],
        "gradient_end_pct": experiment["gradient_end_pct"],
        "elution_cv": experiment["elution_cv"],
        "load_amount_g_l": experiment["load_amount_g_l"],
        "experimental_data": exp_data,
        "observation_groups": experiment.get("observation_groups"),
    }

    mse_record: list[np.ndarray] = []
    para_record: list[np.ndarray] = []

    for cycle in range(cfg.sim_cycles):
        table = _fit_nu_iteration(
            sim, exp_case, table, com_num, tr_exp, ht_exp, cfg
        )
        table = _fit_kkin_iteration(
            sim, exp_case, table, com_num, tr_exp, ht_exp, cfg
        )
        comps = ComponentSet.from_parameter_table(table[:, :com_num])
        sim.components = comps
        _, mse = sim.run_fitting_case(**exp_case)
        mse_record.append(mse)
        para_record.append(table.copy())

    best = table.copy()
    if mse_record:
        for i in range(com_num):
            idx = int(np.argmin([m[i] for m in mse_record]))
            best[:, i] = para_record[idx][:, i]

    final_comps = ComponentSet.from_parameter_table(best[:, :com_num])
    sim.components = final_comps
    _, final_mse = sim.run_fitting_case(**exp_case)

    return FittingResult(
        components=final_comps,
        rmse_per_peak=final_mse[:-1],
        rmse_total=float(final_mse[-1]),
        history=[{"rmse": m.tolist()} for m in mse_record],
    )


def _pack_params(table: np.ndarray, com_num: int) -> np.ndarray:
    return table[1:5, :com_num].ravel()


def _unpack_params(x: np.ndarray, com_num: int, fractions: np.ndarray) -> ComponentSet:
    table = np.zeros((5, com_num))
    table[0, :] = fractions
    table[1:5, :] = x.reshape(4, com_num)
    return ComponentSet.from_parameter_table(table)


def _akta_total_rmse(curve: np.ndarray, akta_trace) -> float:
    """Peak-normalised RMSE of the simulated TOTAL vs an AKTA total-UV trace.

    Gradient-start aligned (so it tracks the real elution position) and divided by the AKTA
    peak height, giving a scale-free, load-invariant term comparable to ``peak_shape_loss``.
    """
    from cex_model.akta_compare import gradient_aligned_grid

    _, a, m = gradient_aligned_grid(akta_trace, curve[:, 0] / 60.0, curve[:, 2:].sum(axis=1), curve[:, 1])
    denom = max(float(np.max(a)), 1e-6)
    return float(np.sqrt(np.mean((a - m) ** 2))) / denom


class _FitObjective:
    """Top-level (picklable) global-DE objective for :func:`fit_sma_parameters`."""

    def __init__(self, sim, com_num, fractions, experiments, loss_mode: str = "rmse",
                 *, nu_monotonic_weight: float = 0.0, retention_weight: float = 0.0,
                 retention_tol_min: float = 0.0, akta_total_weight: float = 0.0,
                 shape_offset_shared: bool = True, shape_max_shift_s: float = 180.0,
                 shape_n_shift: int = 61, shape_height_weight: float = 1.0):
        self.sim = sim
        self.com_num = com_num
        self.fractions = fractions
        self.experiments = experiments
        self.loss_mode = loss_mode
        self.nu_monotonic_weight = nu_monotonic_weight
        self.retention_weight = retention_weight
        self.retention_tol_min = retention_tol_min
        self.akta_total_weight = akta_total_weight
        self.shape_offset_shared = shape_offset_shared
        self.shape_max_shift_s = shape_max_shift_s
        self.shape_n_shift = shape_n_shift
        self.shape_height_weight = shape_height_weight

    def __call__(self, x: np.ndarray) -> float:
        self.sim.components = _unpack_params(x, self.com_num, self.fractions)
        total = 0.0
        for exp in self.experiments:
            try:
                result, mse = self.sim.run_fitting_case(
                    buffer_a=exp["buffer_a"], buffer_b=exp["buffer_b"],
                    gradient_start_pct=exp["gradient_start_pct"],
                    gradient_end_pct=exp["gradient_end_pct"], elution_cv=exp["elution_cv"],
                    load_amount_g_l=exp["load_amount_g_l"], experimental_data=exp["data"],
                    observation_groups=exp.get("observation_groups"),
                )
            except RuntimeError:
                return 1e6  # extreme trial params can make the stiff ODE fail
            groups = exp.get("observation_groups")
            akta = exp.get("akta_trace") if self.akta_total_weight > 0.0 else None
            if (self.loss_mode in ("peak_shape", "shape_corr")
                    or self.retention_weight > 0.0 or akta is not None):
                curve = result.elution_curve()
            if self.loss_mode == "peak_shape":
                total += peak_shape_loss(curve, exp["data"], groups)
            elif self.loss_mode == "shape_corr":
                total += shape_correlation_loss(
                    curve, exp["data"], groups, shared_offset=self.shape_offset_shared,
                    max_shift_s=self.shape_max_shift_s, n_shift=self.shape_n_shift,
                    height_weight=self.shape_height_weight)[0]
            else:
                total += float(mse[-1])
            if self.retention_weight > 0.0:
                total += self.retention_weight * retention_penalty(
                    curve, exp["data"], groups, tol_min=self.retention_tol_min)
            if akta is not None:
                total += self.akta_total_weight * _akta_total_rmse(curve, akta)
        if self.nu_monotonic_weight > 0.0:
            # nu packed contiguously after keq, kkin (see _pack_params); penalize any
            # decrease along the elution-ordered component index.
            nu = x[2 * self.com_num:3 * self.com_num]
            drop = np.minimum(np.diff(nu), 0.0)
            total += self.nu_monotonic_weight * float(np.sum(drop * drop))
        return total


def fit_sma_parameters(
    column: ColumnParameters,
    components: ComponentSet,
    experiments: Sequence[dict],
    config: FittingConfig | None = None,
    bounds_scale: tuple[float, float] = (0.1, 10.0),
    *,
    correction: LoadingCorrection | None = None,
    method: str = "RK23",
    bounds_override: Sequence[tuple[float, float]] | None = None,
    progress: Callable[[float], None] | None = None,
) -> FittingResult:
    """Fit SMA parameters across one or more experiments.

    Uses differential evolution (MATLAB particleswarm/ga equivalent) when
    ``config.use_global`` is True, otherwise direct iterative fitting for
    single experiments.

    ``bounds_override`` supplies explicit per-parameter ``(lo, hi)`` boxes for
    the global optimizer, in the packed order ``[keq.., kkin.., nu.., sigma..]``
    (see :func:`_pack_params`). It is used to warm-start the search inside a
    tight box around a neural-network prediction instead of the default
    ``x0 * bounds_scale``. Ignored by the single-experiment direct path.
    """
    cfg = config or FittingConfig()
    if not cfg.use_global and len(experiments) == 1:
        groups = experiments[0].get("observation_groups")
        one_to_one = not groups or (len(groups) == components.n_protein and
                                    all(group == [idx] for idx, group in enumerate(groups)))
        if one_to_one:
            return fit_sma_parameters_direct(
                column, components, experiments[0], cfg, correction=correction, method=method
            )
        # Direct nu/kkin iteration assumes one experimental column per mechanistic
        # component. Fall through to the global objective for observation mappings.

    sim = _make_simulator(column, components, correction, method)
    com_num = components.n_protein
    table = components.to_parameter_table()
    x0 = _pack_params(table, com_num)
    if bounds_override is not None:
        if len(bounds_override) != len(x0):
            raise ValueError(
                f"bounds_override length {len(bounds_override)} != {len(x0)} params"
            )
        bounds = [(float(lo), float(hi)) for lo, hi in bounds_override]
    else:
        lo, hi = bounds_scale
        bounds = [(v * lo, v * hi) for v in x0]

    objective = _FitObjective(
        sim, com_num, components.fraction_array(), list(experiments), cfg.loss_mode,
        nu_monotonic_weight=cfg.nu_monotonic_weight,
        retention_weight=cfg.retention_weight,
        retention_tol_min=cfg.retention_tol_min,
        akta_total_weight=cfg.akta_total_weight,
        shape_offset_shared=cfg.shape_offset_shared,
        shape_max_shift_s=cfg.shape_max_shift_s,
        shape_n_shift=cfg.shape_n_shift,
        shape_height_weight=cfg.shape_height_weight,
    )
    updating = "deferred" if cfg.workers != 1 else "immediate"

    # Warm-start seeding: scipy requires x0 to lie within ``bounds``, so clip the
    # incoming parameters into the search box. ``start_obj`` is the objective at
    # that seed, on the SAME basis as ``result.fun`` (same _FitObjective), letting
    # the caller gate acceptance on the optimized metric instead of a different one.
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])
    x0c = np.clip(np.asarray(x0, dtype=float), lo_arr, hi_arr)
    de_extra: dict = {}
    start_obj = float("inf")
    if cfg.de_seed_x0:
        de_extra["x0"] = x0c
        start_obj = float(objective(x0c))

    if progress is not None:
        # scipy calls the callback once per generation in the main process (any
        # worker count), so it can drive a UI progress bar without changing the run.
        _gen = {"n": 0}

        def _cb(*_args, **_kwargs):
            _gen["n"] += 1
            progress(min(_gen["n"] / max(cfg.de_maxiter, 1), 1.0))
            return None

        de_extra["callback"] = _cb

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=cfg.random_seed,
        popsize=cfg.de_popsize,
        maxiter=cfg.de_maxiter,
        tol=cfg.de_tol,
        polish=cfg.de_polish,
        workers=cfg.workers,
        updating=updating,
        **de_extra,
    )
    final_comps = _unpack_params(result.x, com_num, components.fraction_array())
    sim.components = final_comps
    mses = []
    for exp in experiments:
        _, mse = sim.run_fitting_case(
            buffer_a=exp["buffer_a"],
            buffer_b=exp["buffer_b"],
            gradient_start_pct=exp["gradient_start_pct"],
            gradient_end_pct=exp["gradient_end_pct"],
            elution_cv=exp["elution_cv"],
            load_amount_g_l=exp["load_amount_g_l"],
            experimental_data=exp["data"],
            observation_groups=exp.get("observation_groups"),
        )
        mses.append(mse)

    avg_mse = np.mean(np.vstack(mses), axis=0)
    return FittingResult(
        components=final_comps,
        rmse_per_peak=avg_mse[:-1],
        rmse_total=float(avg_mse[-1]),
        history=[
            {
                "objective": float(result.fun),
                "start_objective": start_obj,
                "success": bool(result.success),
                "nfev": int(getattr(result, "nfev", 0)),
                "nit": int(getattr(result, "nit", 0)),
            }
        ],
    )
