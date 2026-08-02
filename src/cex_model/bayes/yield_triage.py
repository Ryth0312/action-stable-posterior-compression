"""HLXSYN pooled-yield triage: is the under-covering delta-method interval rescuable, or is MC the
honest fallback? (Task 3.)

At the historical HLXSYN operating point the raw delta-method (Gaussian) interval for pooled *yield*
under-covers the nonlinear posterior pushforward (empirical coverage well below nominal at large N).
This module diagnoses *why* and decides one of three honest verdicts, reusing the existing MC pushforward
and the logit transformed interval (``bayes.coverage_mc_band``):

  * ``linear_ok``      -- the linear interval already covers within one DKW half-band of nominal;
  * ``logit_rescued``  -- the [0,1]-respecting logit-delta interval reaches nominal AND materially beats
                          linear, verified on an INDEPENDENT-seed MC set (so the rescue is not the
                          same-seed MC quantile interval, which is nominal by construction);
  * ``MC_fallback``    -- neither analytic interval covers; report the MC quantile interval and flag the
                          delta-method as not-to-be-trusted for this QoI.

It also emits the supporting evidence: (i) the nonlinear MC pushforward, (ii) linear vs logit intervals
and their coverages, (iii) a 1-D operating-condition line scan (P(meet)/worst_dec/yield interval along
loading, exposing the move-operating-point lever), and (iv) window-boundary diagnostics that explain a
coverage collapse (a pooling window whose edge cuts a peak makes yield a nonlinear function of theta).

Honest framing (Plan review): a larger N does NOT change the interval's true coverage -- it only tightens
the DKW band so the empirical estimate can be *adjudicated* against nominal. For a mid-range yield with a
narrow interval the logit link is numerically ~a no-op, so a large delta_q curvature ⇒ MC_fallback is the
expected, correct result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

from cex_model.app_support import group_indices
from cex_model.bayes.active import _sim_for_op
from cex_model.bayes.coverage_mc_band import (
    _interval_from_mu_std,
    coverage_with_dkw,
    sample_nonlinear_qoi,
)
from cex_model.bayes.decision import decision_covariance, decision_jacobian
from cex_model.bayes.decision_window import gaussian_meet_prob, worst_dec_from_cov, DEFAULT_SPEC
from cex_model.bayes.likelihood import unpack_u
from cex_model.bayes.loading_sweep import _load_product_setup
from cex_model.diffsolver.collection_objective import select_window_for_gradient
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = [
    "transformed_interval",
    "window_boundary_diagnostics",
    "line_scan",
    "triage_report",
]

_QOI_INDEX = {"purity": 0, "yield": 1}
_trapz = getattr(np, "trapezoid", None) or np.trapz
_AXES = {"loading": 0, "gradient_start": 1, "gradient_end": 2, "elution_cv": 3}


def transformed_interval(mu: float, std: float, alpha: float = 0.05, link: str = "logit") -> tuple[float, float]:
    """Delta-method ``(1-alpha)`` interval via ``bayes.coverage_mc_band._interval_from_mu_std`` -- the
    single transformed-interval primitive shared with the coverage report (``link="logit"`` respects
    [0,1] and is asymmetric; guards ``p`` off {0,1} and a degenerate ``std=0``)."""
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return _interval_from_mu_std(float(mu), float(std), z, link)


def _map_curve(bundle, op, u_map, n_steps: int):
    n = bundle.components.n_protein
    sim = _sim_for_op(bundle, op, n_steps)
    ut = torch.tensor(np.asarray(u_map, float), dtype=DTYPE)
    with torch.no_grad():
        curve = sim.elution_curve(*unpack_u(ut, n), differentiable=False).numpy()
    return curve, group_indices(bundle.components)


def window_boundary_diagnostics(bundle, op, u_map, n_steps: int = 120, *, grid: int = 40,
                                acid_max: float = 0.20, main_min: float = 0.70, basic_max: float = 0.10) -> dict:
    """Diagnose whether the MAP-selected pooling window sits on a BOUNDARY that would make the pooled
    quantity a nonlinear function of theta (and so make the delta-method interval under-cover): a window
    edge cutting through signal, a large main-peak fraction lost outside the window, a window pinned to the
    time-grid edge, or a near-degenerate collected total."""
    curve, grp = _map_curve(bundle, op, u_map, n_steps)
    sel = select_window_for_gradient(curve, grid=grid, acid_idx=grp["acid"], main_idx=grp["main"],
                                     basic_idx=grp["basic"], acid_max=acid_max, main_min=main_min, basic_max=basic_max)
    t = curve[:, 0]
    conc = np.clip(curve[:, 2:], 0.0, None)
    total_curve = conc.sum(axis=1)
    ntime = curve.shape[0]
    s, e = int(sel.start_idx), int(sel.end_idx)
    peak = float(total_curve[s:e + 1].max()) if e >= s else 0.0
    edge = max(float(total_curve[s]), float(total_curve[e])) if peak > 0 else 0.0
    main = list(grp["main"])
    main_curve = conc[:, main].sum(axis=1) if main else np.zeros_like(t)
    main_in = float(_trapz(main_curve[s:e + 1], t[s:e + 1]))
    main_all = float(_trapz(main_curve, t))
    total_in = float(_trapz(total_curve[s:e + 1], t[s:e + 1]))
    return {
        "window_idx": [s, e],
        "window_s": [float(sel.start_time_s), float(sel.end_time_s)],
        "touches_grid_edge": bool(s <= 0 or e >= ntime - 1),
        "edge_signal_fraction": float(edge / peak) if peak > 0 else float("nan"),
        "main_mass_fraction_in_window": float(main_in / main_all) if main_all > 0 else float("nan"),
        "total_collected": total_in,
        "degenerate": bool(total_in <= 1e-8),
    }


def line_scan(product: str, *, qoi: str = "yield", axis: str = "loading", lo: float = 12.0, hi: float = 70.0,
              n: int = 10, spec=DEFAULT_SPEC, alpha: float = 0.05, n_steps: int = 120,
              in_dir: str = "results/bayes") -> list[dict]:
    """Scan one operating-condition axis (default loading 12->70 g/L at the historical gradient), reporting
    per point the pooled ``g_map``, ``worst_dec``, ``P(meet-spec)``, the linear+logit yield intervals, and
    the window-boundary flags -- exposing where the yield spec becomes meetable (the move-operating-point
    lever) and where the window boundary corrupts the interval."""
    if axis not in _AXES:
        raise ValueError(f"axis must be one of {list(_AXES)}, got {axis!r}")
    idx_q = _QOI_INDEX[qoi]
    post, base_op, tol, _n, _prior, bundle = _load_product_setup(product, in_dir)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ax = _AXES[axis]
    rows = []
    for val in np.linspace(lo, hi, n):
        op = list(base_op)
        op[ax] = float(val)
        try:
            G, g_map, _ = decision_jacobian(bundle, op, post.u_map, n_steps=n_steps, return_extra=True)
            C = decision_covariance(G, post.cov)
            std_q = float(np.sqrt(max(C[idx_q, idx_q], 0.0)))
            bnd = window_boundary_diagnostics(bundle, op, post.u_map, n_steps)
            rows.append({
                axis: float(val),
                "g_map_purity": float(g_map[0]), "g_map_yield": float(g_map[1]),
                "worst_dec": worst_dec_from_cov(C, tol),
                "p_meet": gaussian_meet_prob(g_map, C, spec),
                f"{qoi}_linear_interval": list(transformed_interval(g_map[idx_q], std_q, alpha, "linear")),
                f"{qoi}_logit_interval": list(transformed_interval(g_map[idx_q], std_q, alpha, "logit")),
                "edge_signal_fraction": bnd["edge_signal_fraction"],
                "main_mass_fraction_in_window": bnd["main_mass_fraction_in_window"],
                "degenerate": bnd["degenerate"],
            })
        except Exception as exc:   # a blown-up OP (e.g. overload) must not stop the scan
            rows.append({axis: float(val), "error": str(exc)})
    return rows


@dataclass
class YieldTriageConfig:
    product: str = "HLXSYN"
    qoi: str = "yield"
    draws: int = 5000
    seed: int = 0
    nominal_alpha: float = 0.05
    dkw_eta: float = 0.05
    n_steps: int = 120
    spec: tuple = DEFAULT_SPEC
    scan_axis: str = "loading"
    scan_lo: float = 12.0
    scan_hi: float = 70.0
    scan_n: int = 10
    in_dir: str = "results/bayes"
    out_path: str = "results/bayes/HLXSYN_yield_triage.json"


def _mc_quantile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    return (float(np.quantile(samples, alpha / 2.0)), float(np.quantile(samples, 1.0 - alpha / 2.0)))


def triage_report(config: YieldTriageConfig | None = None, **overrides) -> dict:
    """Run the full HLXSYN yield triage and write ``{product}_yield_triage.json``.

    Decision (three honest verdicts): ``linear_ok`` if the linear interval covers within one DKW half-band;
    else ``logit_rescued`` iff the logit interval reaches nominal AND materially beats linear, verified on an
    INDEPENDENT-seed MC set; else ``MC_fallback`` (report the MC quantile interval; the same-seed MC quantile
    is nominal by construction, so only the independent-seed check can promote to rescued)."""
    config = config or YieldTriageConfig()
    for k, v in overrides.items():
        setattr(config, k, v)
    p, qoi, alpha = config.product, config.qoi, config.nominal_alpha
    nominal = 1.0 - alpha
    idx_q = _QOI_INDEX[qoi]

    post, base_op, tol, _n, _prior, bundle = _load_product_setup(p, config.in_dir)
    G, g_map, _ = decision_jacobian(bundle, base_op, post.u_map, n_steps=config.n_steps, return_extra=True)
    C = decision_covariance(G, post.cov)
    mu = float(g_map[idx_q]); std = float(np.sqrt(max(C[idx_q, idx_q], 0.0)))
    lin = transformed_interval(mu, std, alpha, "linear")
    logit = transformed_interval(mu, std, alpha, "logit")

    # nonlinear MC pushforward: two INDEPENDENT seeds (analytic intervals adjudicated on A; MC quantile
    # interval from A is honestly re-checked on B, since its same-seed coverage is nominal by construction).
    samples_a = sample_nonlinear_qoi(p, qoi, config.draws, n_steps=config.n_steps, in_dir=config.in_dir, seed=config.seed)
    samples_b = sample_nonlinear_qoi(p, qoi, config.draws, n_steps=config.n_steps, in_dir=config.in_dir, seed=config.seed + 1)
    cov_lin = coverage_with_dkw(samples_a, lin, config.dkw_eta)
    cov_logit = coverage_with_dkw(samples_a, logit, config.dkw_eta)
    dkw_half = float(cov_lin["dkw_eps"])
    mc_int = _mc_quantile_interval(samples_a, alpha) if samples_a.size else (float("nan"), float("nan"))
    cov_mc_independent = coverage_with_dkw(samples_b, mc_int, config.dkw_eta)

    def _within(c):
        return np.isfinite(c) and abs(c - nominal) <= dkw_half

    lin_c, logit_c = cov_lin["empirical_coverage"], cov_logit["empirical_coverage"]
    if _within(lin_c):
        verdict = "linear_ok"
    elif _within(logit_c) and (logit_c - lin_c) > dkw_half:
        verdict = "logit_rescued"
    else:
        verdict = "MC_fallback"

    boundary = window_boundary_diagnostics(bundle, base_op, post.u_map, config.n_steps)
    scan = line_scan(p, qoi=qoi, axis=config.scan_axis, lo=config.scan_lo, hi=config.scan_hi, n=config.scan_n,
                     spec=tuple(config.spec), alpha=alpha, n_steps=config.n_steps, in_dir=config.in_dir)

    report = {
        "product": p, "qoi": qoi, "decision_op": [float(x) for x in base_op], "n_steps": config.n_steps,
        "draws": config.draws, "seed": config.seed, "nominal_coverage": nominal, "dkw_eps": dkw_half,
        "verdict": verdict,
        "g_map": {"pool_purity": float(g_map[0]), "pool_yield": float(g_map[1])},
        "delta_std": std,
        "linear_interval": [float(lin[0]), float(lin[1])], "linear_coverage": lin_c,
        "logit_interval": [float(logit[0]), float(logit[1])], "logit_coverage": logit_c,
        "mc_quantile_interval": [float(mc_int[0]), float(mc_int[1])],
        "mc_quantile_coverage_independent_seed": cov_mc_independent["empirical_coverage"],
        "n_draws_used": int(cov_lin["n"]),
        "window_boundary": boundary,
        "line_scan": {"axis": config.scan_axis, "rows": scan},
        "note": ("logit link corrects only boundary/link skew; for a mid-range yield with a narrow interval "
                 "it is ~a no-op, so a large delta-method curvature ⇒ MC_fallback is the expected result. "
                 "Larger N only tightens the DKW band; it does not change the interval's true coverage."),
        "claim_level": "coverage_diagnostic_not_certificate",
    }
    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
