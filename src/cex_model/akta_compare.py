"""Glue for comparing the mechanistic model against AKTA total-UV curves.

Shared by the residual diagnostic (``scripts/akta_residual_diagnostic.py``) and the
physical refit (``scripts/akta_physical_refit.py``): the mapping from AKTA fitting
files to their calibrated experiment conditions, a thin total-curve simulate wrapper
(explicit column so a refit can pass a modified one), and small alignment helpers.

Kept separate from :mod:`cex_model.akta` (the pure UNICORN loader) because this layer
depends on the simulator / product bundle; ``simulate_elution`` is imported lazily to
avoid pulling the heavy app-support stack just to load a CSV.
"""

from __future__ import annotations

import numpy as np

# AKTA fitting file -> (loading_g_L, gradient_length_CV, gradient_start_pct), matched to
# the calibrated experiment (conditions + xlsx curve) in the product bundle. Loads are
# the ACTUAL/measured values recorded in the configs. Excludes the HLXSYN 等度/isocratic
# outlier; HLXSYN exp5 (DT) has no fitting-AKTA counterpart.
AKTA_FITTING: dict[str, dict[str, tuple[float, float, float]]] = {'HLXSYN': {'20240131-HLXSYN-IEC-25G L-25CV-RT=8 001.csv': (25.0, 25.0, 20.0), '20240201-HLXSYN-IEC-45G L-25CV 001.csv': (45.0, 25.0, 20.0), '20240204-HLXSYN-IEC-35G L-5-85%B-20CV+5CV 001.csv': (35.0, 25.0, 5.0)}}


def simulate_total(column, components, e, correction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one experiment's total protein -> (time_min, total_g_l, salt_mol_l).

    ``column`` is explicit so a physical refit can pass a modified column (e.g. a new
    ``dax`` / ``ionic_capacity``) while everything else stays calibrated.
    """
    from cex_model.app_support import simulate_elution

    cur = simulate_elution(
        column, components, buffer_a=e.buffer_a, buffer_b=e.buffer_b,
        gradient_start_pct=e.gradient_start_pct, gradient_end_pct=e.gradient_end_pct,
        elution_cv=e.elution_cv, loading_g_l=e.loading_g_l, correction=correction,
    )
    return cur[:, 0] / 60.0, cur[:, 2:].sum(axis=1), cur[:, 1]


def find_experiment(bundle, key: tuple[float, float, float]):
    """Bundle experiment matching (loading, elution_cv, gradient_start_pct); None if absent."""
    load, cv, gs = key
    cands = [e for e in bundle.experiments
             if e.elution_cv == cv and e.gradient_start_pct == gs]
    if not cands:
        return None
    return min(cands, key=lambda e: abs(e.loading_g_l - load))


def grad_start_min(t_min: np.ndarray, salt: np.ndarray) -> float:
    """Time (min) the simulated salt gradient starts rising (>1% of its span)."""
    lo, hi = float(salt.min()), float(salt.max())
    if hi - lo <= 0:
        return 0.0
    j = np.where(salt > lo + 0.01 * (hi - lo))[0]
    return float(t_min[j[0]]) if len(j) else 0.0


def scale_components(components, kkin_mult: float = 1.0, sigma_mult: float = 1.0):
    """ComponentSet with all proteins' kkin / sigma scaled (global broadening/overload levers).

    kkin (kinetics) and sigma (steric shielding) are the shared, position-preserving shape
    levers a single total-UV curve can constrain; keq/nu (retention/selectivity) are left
    untouched so the coupled identifiability problem is not reopened.
    """
    from cex_model.fitting import _pack_params, _unpack_params

    table = components.to_parameter_table().copy()  # rows: fraction, keq, kkin, nu, sigma
    n = table.shape[1]
    table[2] *= kkin_mult
    table[4] *= sigma_mult
    return _unpack_params(_pack_params(table, n), n, table[0])


def apex_aligned_grid(tr, m_t: np.ndarray, m_y: np.ndarray,
                      step_min: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample AKTA + model totals onto a common grid relative to the PEAK APEX.

    Apex alignment removes the elution-timing/position offset, isolating peak SHAPE so the
    broadening/overload levers (which cannot move the peak centre) are tested fairly.
    Returns ``(dt_min, akta_g_l, model_g_l)`` over the AKTA in-gradient product region.

    The grid is derived from the AKTA trace ALONE (the model apex only re-centres the
    model curve via interpolation, clamped at its edges), so the returned length is
    independent of the model params -- required for a fixed-length least-squares residual.
    """
    a_ta, _ = tr.product_peak()
    m_ta = float(m_t[int(np.argmax(m_y))])
    lo, hi = tr.gradient_window()
    grid = np.arange(lo - a_ta, hi - a_ta, step_min)
    return grid, np.interp(a_ta + grid, tr.uv_min, tr.uv_g_l), np.interp(m_ta + grid, m_t, m_y)


def gradient_aligned_grid(tr, m_t: np.ndarray, m_y: np.ndarray, m_salt: np.ndarray,
                          step_min: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample AKTA + model totals onto a common grid relative to GRADIENT START.

    Gradient-start (not apex) alignment keeps elution-timing/position error visible, so
    a refit that shifts a peak the wrong way is penalised rather than masked. Returns
    ``(rel_min, akta_g_l, model_g_l)`` over the AKTA in-gradient product region.
    """
    lo, hi = tr.gradient_window()
    a0 = lo
    m0 = grad_start_min(m_t, m_salt)
    rel = np.arange(0.0, hi - a0, step_min)
    akta = np.interp(a0 + rel, tr.uv_min, tr.uv_g_l)
    model = np.interp(m0 + rel, m_t, m_y)
    return rel, akta, model
