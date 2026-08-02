"""Gradient-based Phase-2 refinement of operating conditions (D2.3, standalone proposer).

Replaces the inner ODE-DE search with differentiable physics: Sobol-screen a handful of operating
points, then Adam-refine the best M through the differentiable BDF solver (D2.1) against the
fixed-window collection objective (D2.2). This is a PROPOSER ONLY -- it returns candidate operating
points; the caller ODE-verifies them with the shared ``optimization.verify_process_candidates`` so
the lossless contract holds (reported yield/purity/window are exact mechanistic values).

flow is held fixed (its residence time is baked into the solver grid). No ML residual correction is
applied -- physics-only search (the residual layer over-extrapolates on new operating conditions).
The window is re-selected every ``reselect_every`` steps (the fixed-window approximation, D2.2 (a)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import qmc

from cex_model.diffsolver.collection_objective import (
    differentiable_fixed_window_objective,
    select_window_for_gradient,
)
from cex_model.diffsolver.operating import bind_operating_conditions
from cex_model.diffsolver.torch_solver import DTYPE, TorchSimulator, params_from_components
from cex_model.gradients import build_fitting_inlet
from cex_model.optimization import _PROCESS_DEFAULTS

# operating variables the gradient path can optimise (flow is fixed -> excluded). PROCESS_VARS names.
GRAD_OPT_VARS = ("loading_g_l", "gradient_start_pct", "gradient_end_pct", "gradient_cv", "feed_cv")
# PROCESS_VARS name -> bind_operating_conditions kwarg (only loading_g_l differs)
_BIND_NAME = {"loading_g_l": "loading", "gradient_start_pct": "gradient_start_pct",
              "gradient_end_pct": "gradient_end_pct", "gradient_cv": "gradient_cv", "feed_cv": "feed_cv"}
_GRAD_SPAN = 5.0  # require gradient_end_pct > gradient_start_pct + this for the Sobol seeds


@dataclass
class GradientRefineConfig:
    n_sobol: int = 64
    top_m: int = 8
    adam_steps: int = 40
    lr: float = 0.01                # conservative; the fixed-window objective can be choppy
    reselect_every: int = 5
    objective: str = "recovery"     # "recovery" (default; production: max recovery s.t. purity) or "yield"
    penalty_weight: float = 10.0
    yield_tiebreak_alpha: float = 0.05  # recovery mode: small productivity term that breaks the free-
                                        # loading "collect everything at low loading -> recovery=1" plateau
    yield_scale: float = 0.0            # 0 = auto (max screened yield_g) so the yield term sits at O(1)
    min_yield_fraction: float = 0.0     # #4 productivity floor: squared-hinge penalty on yield_g < frac *
                                        # yield_scale, de-degenerating the free-loading recovery=1 plateau
    checkpoint: bool = True         # gradient checkpointing in the solver (memory-safe at large n_steps)
    window_grid: int = 40
    n_steps: int = 900
    random_seed: int = 0


@dataclass
class GradientRefineResult:
    candidates: list[dict]      # refined + screened-fallback operating dicts (full, incl. fixed), refined first
    history: list[dict]         # per refined start: {start_loss, final_loss, recovery, yield_g, window_mode, ...}
    screened: list[dict]        # all Sobol points + their screen loss (before/after reference)
    candidate_meta: list[dict]  # parallel to `candidates`: {source: refined|screened_fallback, diff_loss, metrics}
    min_yield: float = 0.0      # #4 absolute productivity floor used (= min_yield_fraction * yield_scale); 0 = off


def _make_sim(column, components, buffer_a, buffer_b, bounds, fixed, hold_cv, n_steps, correction):
    """One TorchSimulator (column constants + shapes); bind_operating_conditions overwrites the
    op-dependent grid/inlet/gamma per evaluation, so the nominal inlet here is just for construction."""
    def nominal(v, default):
        if v in bounds:
            return 0.5 * (bounds[v][0] + bounds[v][1])
        return float(fixed.get(v, _PROCESS_DEFAULTS.get(v, default)))
    load = nominal("loading_g_l", 20.0)
    inlet = build_fitting_inlet(
        buffer_a=buffer_a, buffer_b=buffer_b, gradient_start_pct=nominal("gradient_start_pct", 25.0),
        gradient_end_pct=nominal("gradient_end_pct", 71.0), elution_cv=nominal("gradient_cv", 15.0),
        rt_min=column.rt, load_amount_g_l=load, component_fractions_pct=components.fraction_array(),
        feed_cv=nominal("feed_cv", 4.0), hold_cv=hold_cv)
    return TorchSimulator(column, components, inlet, load, correction, n_steps=n_steps)


def _full_point(opt_vals: dict, fixed: dict, column) -> dict:
    """Full operating-condition dict (5 phase vars + flow) from the optimised subset + fixed/defaults."""
    d = {}
    for v in GRAD_OPT_VARS:
        if v in opt_vals:
            d[v] = float(opt_vals[v])
        elif v in fixed:
            d[v] = float(fixed[v])
        elif v in _PROCESS_DEFAULTS:
            d[v] = float(_PROCESS_DEFAULTS[v])
        else:
            raise ValueError(f"{v} must be in bounds or fixed")
    d["flow_rate"] = float(fixed.get("flow_rate", column.flow_rate))
    return d


def gradient_refine_phase2(
    column, components, *, buffer_a, buffer_b, bounds, fixed=None, hold_cv=4.0,
    acid_idx, main_idx, basic_idx, acid_max, main_min, basic_max,
    correction=None, config=None,
) -> GradientRefineResult:
    """Sobol-screen + Adam-refine operating conditions through the differentiable solver.

    ``bounds`` / ``fixed`` use PROCESS_VARS names (same as ``ProcessOptimizationConfig``); flow_rate
    must be fixed (gradient path holds it constant). Returns refined candidate operating points
    (best differentiable-objective first) for the caller to ODE-verify.
    """
    cfg = config or GradientRefineConfig()
    fixed = dict(fixed or {})
    if "flow_rate" in bounds:
        raise ValueError("gradient refine holds flow fixed (rt is baked into the solver grid); "
                         "put flow_rate in `fixed`, not `bounds`.")
    opt_vars = [v for v in GRAD_OPT_VARS if v in bounds]
    if not opt_vars:
        raise ValueError("no gradient-optimisable variables in bounds")
    lo = torch.tensor([bounds[v][0] for v in opt_vars], dtype=DTYPE)
    hi = torch.tensor([bounds[v][1] for v in opt_vars], dtype=DTYPE)

    sim = _make_sim(column, components, buffer_a, buffer_b, bounds, fixed, hold_cv, cfg.n_steps, correction)
    fractions = components.fraction_array()
    keq, kkin, nu, sigma = params_from_components(components)  # SMA params FIXED
    sel_kw = dict(grid=cfg.window_grid, acid_idx=acid_idx, main_idx=main_idx, basic_idx=basic_idx,
                  acid_max=acid_max, main_min=main_min, basic_max=basic_max)
    obj_kw = dict(acid_idx=acid_idx, main_idx=main_idx, basic_idx=basic_idx, acid_max=acid_max,
                  main_min=main_min, basic_max=basic_max, objective=cfg.objective,
                  penalty_weight=cfg.penalty_weight, yield_tiebreak=cfg.yield_tiebreak_alpha)

    def bind_kwargs(opt_vals: dict) -> dict:
        out = {}
        for v in GRAD_OPT_VARS:
            if v in opt_vals:
                out[_BIND_NAME[v]] = opt_vals[v]
            elif v in fixed:
                out[_BIND_NAME[v]] = float(fixed[v])
            elif v in _PROCESS_DEFAULTS:
                out[_BIND_NAME[v]] = float(_PROCESS_DEFAULTS[v])
            else:
                raise ValueError(f"{v} must be in bounds or fixed")
        return out

    def forward(opt_vals: dict, differentiable: bool):
        bind_operating_conditions(sim, **bind_kwargs(opt_vals), buffer_a=buffer_a, buffer_b=buffer_b,
                                  fractions=fractions, hold_cv=hold_cv, correction=correction)
        return sim.integrate(keq, kkin, nu, sigma, differentiable=differentiable,
                             checkpoint=cfg.checkpoint and differentiable)

    def obj_res(curve, sel, yscale, min_yield=0.0):  # fixed-TIME window (stable as op stretches grid; D2.2.1)
        return differentiable_fixed_window_objective(
            curve, start_time_s=sel.start_time_s, end_time_s=sel.end_time_s,
            min_total=sel.min_total, yield_scale=yscale, min_yield=min_yield, **obj_kw)

    # 1) Sobol screen (forward only). Keep seeds with a real gradient span if both pct vars are optimised.
    eng = qmc.Sobol(d=len(opt_vars), scramble=True, seed=cfg.random_seed)
    pts = qmc.scale(eng.random(cfg.n_sobol * 2), lo.numpy(), hi.numpy())
    if "gradient_start_pct" in opt_vars and "gradient_end_pct" in opt_vars:
        gs, ge = opt_vars.index("gradient_start_pct"), opt_vars.index("gradient_end_pct")
        pts = pts[pts[:, ge] > pts[:, gs] + _GRAD_SPAN]
    pts = pts[:cfg.n_sobol]
    raw = []
    for row in pts:
        ov = {v: float(row[i]) for i, v in enumerate(opt_vars)}
        with torch.no_grad():
            curve = forward(ov, False)
            sel = select_window_for_gradient(curve.numpy(), **sel_kw)
            r = obj_res(curve, sel, 1.0)  # provisional yscale; yield_g/recovery are raw, independent of it
        raw.append({"ov": ov, "mode": sel.mode, "yield_g": float(r.yield_g), "recovery": float(r.recovery),
                    "penalty": float(r.penalty), "main": float(r.main_fraction),
                    "acid": float(r.acid_fraction), "basic": float(r.basic_fraction)})

    # auto yield-scale so the productivity term (recovery mode) / the yield objective sits at O(1).
    yscale = cfg.yield_scale
    if yscale <= 0.0:
        yscale = max((r["yield_g"] for r in raw), default=1.0) or 1.0
    min_yield = max(cfg.min_yield_fraction, 0.0) * yscale  # #4 absolute productivity floor (0 = off)

    def loss_of(r):
        ny = r["yield_g"] / yscale
        score = (r["recovery"] + cfg.yield_tiebreak_alpha * ny) if cfg.objective == "recovery" else ny
        # mirror the differentiable objective's squared-hinge floor so the SCREEN ranks floor-aware too
        floor_pen = ((max(0.0, min_yield - r["yield_g"]) ** 2) / (min_yield ** 2)) if min_yield > 0 else 0.0
        return -score + cfg.penalty_weight * (r["penalty"] + floor_pen)
    for r in raw:
        r["loss"] = loss_of(r)
    raw.sort(key=lambda r: r["loss"])
    starts = raw[:cfg.top_m]

    # 2) Adam-refine each start (unconstrained u -> sigmoid into [lo, hi]; re-select window every K steps)
    history, refined = [], []
    for st in starts:
        x0 = torch.tensor([st["ov"][v] for v in opt_vars], dtype=DTYPE)
        frac0 = ((x0 - lo) / (hi - lo)).clamp(1e-4, 1.0 - 1e-4)
        u = torch.log(frac0 / (1.0 - frac0)).clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=cfg.lr)
        sel = None
        for step in range(cfg.adam_steps):
            opt.zero_grad()
            x = lo + (hi - lo) * torch.sigmoid(u)
            ov = {v: x[i] for i, v in enumerate(opt_vars)}
            curve = forward(ov, True)
            if sel is None or step % cfg.reselect_every == 0:
                sel = select_window_for_gradient(curve.detach().numpy(), **sel_kw)
            obj_res(curve, sel, yscale, min_yield).loss.backward()
            opt.step()
        with torch.no_grad():
            xf = lo + (hi - lo) * torch.sigmoid(u)
            ovf = {v: float(xf[i]) for i, v in enumerate(opt_vars)}
            cf = forward(ovf, False)
            self_sel = select_window_for_gradient(cf.numpy(), **sel_kw)
            rf = obj_res(cf, self_sel, yscale, min_yield)
        full = _full_point(ovf, fixed, column)
        hist = {"start_loss": st["loss"], "final_loss": float(rf.loss), "window_mode": self_sel.mode,
                "start_recovery": st["recovery"], "recovery": float(rf.recovery),
                "start_yield_g": st["yield_g"], "yield_g": float(rf.yield_g),
                "main": float(rf.main_fraction), "acid": float(rf.acid_fraction),
                "basic": float(rf.basic_fraction),
                "penalties": {k: float(v) for k, v in rf.penalties.items()},
                "window_s": [self_sel.start_time_s, self_sel.end_time_s],
                "loading_g_l": full["loading_g_l"]}
        history.append(hist)
        refined.append((float(rf.loss), ovf, hist))

    refined.sort(key=lambda t: t[0])
    # candidate pool = refined + the un-refined top-M screened points, so the ODE-verify can fall back
    # to the screen if Adam worsened a start -> gradient refine can only help, never beat Sobol screening.
    # candidate_meta[i] (parallel to candidates) tags each point's source + metrics so the caller can
    # report whether the ODE winner came from gradient REFINE or just the Sobol SCREEN (D2.3 provenance).
    candidates, candidate_meta = [], []
    for loss, ov, hist in refined:
        candidates.append(_full_point(ov, fixed, column))
        candidate_meta.append({"source": "refined", "diff_loss": loss, "metrics": hist})
    for st in starts:
        candidates.append(_full_point(st["ov"], fixed, column))
        candidate_meta.append({"source": "screened_fallback", "diff_loss": st["loss"],
                               "metrics": {"recovery": st["recovery"], "yield_g": st["yield_g"],
                                           "main": st["main"], "acid": st["acid"], "basic": st["basic"],
                                           "mode": st["mode"]}})
    screened = [{"loss": r["loss"], "mode": r["mode"], "yield_g": r["yield_g"], "recovery": r["recovery"],
                 **r["ov"]} for r in raw]
    return GradientRefineResult(candidates=candidates, history=history, screened=screened,
                                candidate_meta=candidate_meta, min_yield=min_yield)
