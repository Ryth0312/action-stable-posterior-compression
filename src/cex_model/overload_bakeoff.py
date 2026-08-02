"""Overload-isotherm bake-off: a forward-sensitivity screen of candidate
one-parameter, coverage-dependent SMA overload corrections.

Motivation. The decision-level hold-out (paper Section 6.6) flags a high-loading
model-adequacy failure on one product: the dilute-calibrated SMA mistimes the
main peak by ~+275 s at 40 g/L, so the recommendation procedure abstains
out-of-domain rather than act. A natural question is whether a simple
loading-dependent isotherm term would repair the extrapolation. This module
screens the candidate one-parameter forms on a synthetic bundle by *forward
sensitivity* (no fit): for each form and coefficient it measures the main-peak
retention shift versus base SMA at high occupancy (out-of-domain) and low
occupancy (in-domain).

The screen is a structural argument, not a fit to the real product: gradient
retention is governed by the characteristic charge nu (the salt-dependence
exponent), a global equilibrium property, so a coverage-dependent correction
must either (a) be strong but perturb *all* occupancies -- breaking the in-domain
fits -- or (b) be localized to high occupancy (theta^2 / hinge) and then far too
weak, or it destabilizes the stiff exponent. Affinity (keq) is not a
gradient-retention lever (retention ~ keq^(1/nu)); the steric factor sigma is the
wrong lever. The upshot: no one-parameter form is simultaneously strong,
localized, and numerically stable, so the flagged region is a *structural*
inadequacy -- a fuller overload isotherm is future work, and abstaining is the
right in-scope action.

All math is the RK23/numpy reference path (``cex_model.sma.tran_ode_rhs``); the
overload term is added per node as a function of the occupancy
``theta = (gamma.Q).(nu+sigma)/Lambda0``. ``overload_rhs`` at ``coeff == 0`` (or
``form == "base"``) is bit-identical to the base RHS, which the test suite guards
(and which also guards against drift from ``sma.tran_ode_rhs``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cex_model.simulator as _simmod
from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet, ComponentType, SMAComponent
from cex_model.sma import _apply_nonneg, tran_ode_rhs
from cex_model.simulator import ChromatographySimulator

# hinge parameters (softplus onset near the in-domain occupancy ceiling)
THETA0_DEFAULT = 0.45
WID_DEFAULT = 0.05

# form key -> human-readable label (as plotted in the paper figure)
FORMS: dict[str, str] = {
    "base": "base SMA",
    "sigma": r"sigma_eff = sigma*(1+c*theta)",
    "nu": r"nu_eff = nu*(1-c*theta)",
    "keqexp": r"keq_eff = keq*exp(-c*theta)",
    "nu2": r"nu_eff = nu*(1-c*theta^2)",
    "keq2": r"keq_eff = keq*(1-c*theta^2)",
    "hkeq": r"keq_eff = keq*exp(-c*softplus(theta-theta0))",
    "hnu": r"nu_eff = nu*(1-c*softplus(theta-theta0))",
}


def overload_rhs(
    t: float,
    y: np.ndarray,
    state,
    *,
    form: str = "base",
    coeff: float = 0.0,
    theta0: float = THETA0_DEFAULT,
    wid: float = WID_DEFAULT,
) -> np.ndarray:
    """EDM+SMA RHS with a coverage-dependent overload term.

    Mirrors :func:`cex_model.sma.tran_ode_rhs` and adds one of the candidate
    overload corrections. At ``coeff == 0`` or ``form == "base"`` it returns the
    base RHS unchanged (short-circuit), so it reduces exactly to base SMA.
    """
    if form == "base" or coeff == 0.0:
        return tran_ode_rhs(t, y, state)

    gs = state.column.grid_size
    nc = state.components.n_total
    Y = _apply_nonneg(y.reshape(gs, nc * 2 - 1, order="F"), state)
    C = Y[:, :nc]
    Q_protein = Y[:, nc:]
    col = state.column
    N = state.edm.N
    T = state.edm.T if state.edm.T is not None else N @ state.edm.A_C
    vb = col.velocity / col.epsbed
    c1 = (1.0 - col.epsbed) / col.epsbed
    gamma = state.gamma if state.gamma is not None else state.correction.gamma(state.loading_g_l, nc)
    nusig_p = state.nusig_protein if state.nusig_protein is not None else (state.nu + state.sigma)[1:]

    dC = T @ C
    dC += np.outer(N[:, 0], vb * (state.inlet.inlet_concentrations(t) - C[0, :]))

    Qeff = gamma[1:] * Q_protein
    base_sma = col.inocap - Qeff @ nusig_p
    theta = np.clip((col.inocap - base_sma) / col.inocap, 0.0, None)  # per-node occupancy

    nu_p = state.nu[1:]
    keq_p = state.keq[1:]
    kkin_p = state.kkin[1:]
    gamma_p = gamma[1:]
    sigma_p = state.sigma[1:]
    csalt = C[:, 0]

    sma_sum = np.maximum(base_sma, 1e-12)
    des = gamma_p * Q_protein * (csalt[:, None] ** nu_p[None, :])
    if form == "sigma":            # sigma_eff = sigma*(1+coeff*theta) -> Lambda smaller
        sma_sum = np.maximum(base_sma - coeff * theta * (Qeff @ sigma_p), 1e-12)
        ads = keq_p * (sma_sum[:, None] ** nu_p[None, :]) * C[:, 1:]
    elif form == "nu":             # nu_eff = nu*(1-coeff*theta) in the exponents
        nu_eff = np.clip(nu_p[None, :] * (1.0 - coeff * theta[:, None]), 0.1, None)
        ads = keq_p * (sma_sum[:, None] ** nu_eff) * C[:, 1:]
        des = gamma_p * Q_protein * (csalt[:, None] ** nu_eff)
    elif form == "keqexp":         # keq_eff = keq*exp(-coeff*theta)  (Mollerup-lite activity)
        ads = keq_p * np.exp(-coeff * theta[:, None]) * (sma_sum[:, None] ** nu_p[None, :]) * C[:, 1:]
    elif form == "nu2":            # nu_eff = nu*(1-coeff*theta^2)  (theta^2-localized)
        nu_eff = np.clip(nu_p[None, :] * (1.0 - coeff * theta[:, None] ** 2), 0.1, None)
        ads = keq_p * (sma_sum[:, None] ** nu_eff) * C[:, 1:]
        des = gamma_p * Q_protein * (csalt[:, None] ** nu_eff)
    elif form == "keq2":           # keq_eff = keq*(1-coeff*theta^2)
        fac = np.clip(1.0 - coeff * theta[:, None] ** 2, 0.05, None)
        ads = keq_p * fac * (sma_sum[:, None] ** nu_p[None, :]) * C[:, 1:]
    elif form == "hkeq":           # hinge affinity: keq*exp(-coeff*softplus((theta-theta0)/wid)*wid)
        excess = np.logaddexp(0.0, (theta[:, None] - theta0) / wid) * wid
        ads = keq_p * np.exp(-coeff * excess) * (sma_sum[:, None] ** nu_p[None, :]) * C[:, 1:]
    elif form == "hnu":            # hinge nu
        excess = np.logaddexp(0.0, (theta[:, None] - theta0) / wid) * wid
        nu_eff = np.clip(nu_p[None, :] * (1.0 - coeff * excess), 0.1, None)
        ads = keq_p * (sma_sum[:, None] ** nu_eff) * C[:, 1:]
        des = gamma_p * Q_protein * (csalt[:, None] ** nu_eff)
    else:
        raise ValueError(f"unknown overload form: {form!r}")

    dQ = (ads - des) / kkin_p
    dC[:, 1:] -= c1 * dQ
    return np.column_stack([dC, dQ]).ravel(order="F")


def default_bundle() -> tuple[ColumnParameters, ComponentSet]:
    """Synthetic 3-component CEX bundle (main 70% feed) used for the screen.

    Loadings of ~10 / ~25 g/L give peak occupancy ~25% / ~59%, bracketing the
    26-48% occupancy of the real study products (in-domain) and the overload
    regime (out-of-domain).
    """
    col = ColumnParameters(grid_size=21)
    comps = ComponentSet(components=[
        SMAComponent("acid", 7.2, 0.030, 38.0, 3.0e-5, 18, ComponentType.ACID),
        SMAComponent("main", 8.4, 0.036, 42.0, 2.6e-5, 70, ComponentType.MAIN),
        SMAComponent("basic", 9.1, 0.040, 45.0, 2.2e-5, 12, ComponentType.BASIC),
    ])
    return col, comps


@dataclass
class _RunResult:
    retention_s: float
    theta_max: float
    stable: bool


def _forward(sim: ChromatographySimulator, loading: float, form: str, coeff: float,
             theta0: float, wid: float, main_idx: int) -> _RunResult:
    """One forward run with the overload RHS swapped in (restored in finally)."""
    orig = _simmod.tran_ode_rhs
    _simmod.tran_ode_rhs = lambda t, y, st: overload_rhs(
        t, y, st, form=form, coeff=coeff, theta0=theta0, wid=wid)
    try:
        res = sim.run_forward_case(
            buffer_a=0.02, buffer_b=0.20, gradient_start_pct=20, gradient_end_pct=95,
            elution_cv=25, load_amount_g_l=loading, n_time_points=1000)
    except Exception:
        return _RunResult(np.nan, np.nan, False)
    finally:
        _simmod.tran_ode_rhs = orig
    cv = res.elution_curve()
    c = np.clip(cv[:, 2 + main_idx], 0, None)
    tR = float((cv[:, 0] * c).sum() / c.sum()) if c.sum() > 0 else np.nan
    # peak occupancy from the raw solution
    gs = sim.column.grid_size
    nc = sim.components.n_total
    gamma = sim.correction.gamma(loading, nc)
    nusig = (sim.components.nu_array() + sim.components.sigma_array())[1:]
    yT = res.raw_solution.y.T
    th = 0.0
    for row in yT:
        Qp = np.maximum(row.reshape(gs, nc * 2 - 1, order="F")[:, nc:], 0.0)
        th = max(th, float(((gamma[1:] * Qp) @ nusig / sim.column.inocap).max()))
    return _RunResult(tR, th, np.isfinite(tR))


def run_bakeoff(
    *,
    high_loading: float = 25.0,
    low_loading: float = 10.0,
    high_coeffs: dict[str, list[float]] | None = None,
    low_coeffs: dict[str, list[float]] | None = None,
    theta0: float = THETA0_DEFAULT,
    wid: float = WID_DEFAULT,
    main_idx: int = 1,
) -> dict:
    """Run the full screen and return a committed-artifact-ready dict.

    For each form and coefficient, reports the main-peak retention shift versus
    base SMA (seconds and %) at the high (out-of-domain) and low (in-domain)
    loadings, plus per-loading peak occupancy and a stability flag.
    """
    high_coeffs = high_coeffs or {
        "nu": [0.3, 0.5, 1.0, 2.0], "keqexp": [0.3, 0.5, 1.0], "sigma": [0.3, 0.5, 1.0],
        "nu2": [1.0, 2.0], "keq2": [1.0, 2.0, 4.0], "hkeq": [2.0, 4.0, 8.0], "hnu": [2.0, 4.0],
    }
    low_coeffs = low_coeffs or {
        "nu": [0.5], "keqexp": [0.5], "sigma": [0.5], "nu2": [2.0],
        "keq2": [2.0], "hkeq": [8.0], "hnu": [8.0],
    }
    col, comps = default_bundle()
    sim = ChromatographySimulator(column=col, components=comps, method="BDF",
                                  nonneg="smooth", rtol=1e-6)
    out: dict = {"theta0": theta0, "wid": wid, "inocap": float(col.inocap),
                 "forms": FORMS, "loadings": {}}
    for tag, loading, grid in [("out_of_domain", high_loading, high_coeffs),
                               ("in_domain", low_loading, low_coeffs)]:
        base = _forward(sim, loading, "base", 0.0, theta0, wid, main_idx)
        rows = {}
        for form, coeffs in grid.items():
            rows[form] = []
            for c in coeffs:
                r = _forward(sim, loading, form, c, theta0, wid, main_idx)
                d = r.retention_s - base.retention_s if r.stable else None
                rows[form].append({
                    "coeff": float(c),
                    "retention_s": None if not r.stable else float(round(r.retention_s, 1)),
                    "delta_s": None if d is None else float(round(d, 1)),
                    "delta_pct": None if d is None else float(round(100 * d / base.retention_s, 2)),
                    "stable": bool(r.stable),
                })
        out["loadings"][tag] = {
            "loading_g_per_l": loading,
            "base_retention_s": round(base.retention_s, 1),
            "peak_occupancy": round(base.theta_max, 3),
            "forms": rows,
        }
    return out
