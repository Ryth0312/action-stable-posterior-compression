"""Inlet boundary conditions and gradient profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class InletProfile:
    """Piecewise-linear inlet schedule.

    ``segments`` rows: [c_start, c_end, t_start, t_end] for salt (index 0)
    and each protein component.

    Times in seconds.
    """

    segments: np.ndarray  # shape (n_rows, 4)
    n_components: int

    @property
    def duration(self) -> float:
        return float(self.segments[2, 3])

    def inlet_concentration(self, t: float, index: int) -> float:
        """Return inlet concentration at time *t* for species *index* (0=salt).

        Matches MATLAB ``inlet_cal.m``.
        """
        if index == 0:
            for i in range(self.segments.shape[0] - self.n_components + 1):
                c0, c1, ts, te = self.segments[i]
                if ts <= t <= te:
                    if te == ts:
                        return float(c0)
                    return float(
                        c0 + (t - ts) * (c1 - c0) / (te - ts)
                    )
            return 0.0

        comp_row = self.segments.shape[0] - self.n_components + index
        c0, _, ts, te = self.segments[comp_row]
        if ts <= t <= te:
            return float(c0)
        return 0.0

    def inlet_concentrations(self, t: float) -> np.ndarray:
        """All inlet concentrations at time *t* (vectorized over components).

        Identical formulas to :meth:`inlet_concentration` for every species,
        returned as a length-``n_components`` array (salt at index 0). Replaces a
        per-component Python call in the ODE RHS.
        """
        nc = self.n_components
        out = np.zeros(nc)
        # salt (index 0): piecewise-linear over the leading salt segments
        n_salt_seg = self.segments.shape[0] - nc + 1
        for i in range(n_salt_seg):
            c0, c1, ts, te = self.segments[i]
            if ts <= t <= te:
                out[0] = c0 if te == ts else c0 + (t - ts) * (c1 - c0) / (te - ts)
                break
        # proteins (index >=1): constant over their feed row, else 0
        if nc > 1:
            prot = self.segments[self.segments.shape[0] - nc + 1 :]  # (nc-1, 4)
            mask = (prot[:, 2] <= t) & (t <= prot[:, 3])
            out[1:] = np.where(mask, prot[:, 0], 0.0)
        return out


def build_fitting_inlet(
    *,
    buffer_a: float,
    buffer_b: float,
    gradient_start_pct: float,
    gradient_end_pct: float,
    elution_cv: float,
    rt_min: float,
    load_amount_g_l: float,
    component_fractions_pct: np.ndarray,
    feed_cv: float = 10.0,
    hold_cv: float = 3.0,
    mol_to_g_l: float = 150_000.0,
) -> InletProfile:
    """Build inlet profile for parameter fitting (MATLAB ``SMA.m``).

    Parameters
    ----------
    buffer_a, buffer_b : mol/L Na+ at 0% and 100% B
    gradient_start_pct, gradient_end_pct : %B (0-1 scale in MATLAB, 0-100 here)
    elution_cv : gradient length in column volumes
    rt_min : residence time per CV, minutes
    load_amount_g_l : feed loading concentration, g/L
    component_fractions_pct : mass fractions (%), length n_protein
    feed_cv : loading phase length in CV (default 10, matches SMA.m)
    hold_cv : isocratic hold after gradient, CV
    """
    start_frac = gradient_start_pct / 100.0
    end_frac = gradient_end_pct / 100.0
    gradient_start = buffer_a * (1 - start_frac) + buffer_b * start_frac
    gradient_end = buffer_a * (1 - end_frac) + buffer_b * end_frac

    feedtime = rt_min * 60.0 * feed_cv
    grad_start_t = feedtime
    grad_end_t = feedtime + elution_cv * rt_min * 60.0
    sim_end_t = grad_end_t + hold_cv * rt_min * 60.0

    n_protein = len(component_fractions_pct)
    n_comp = n_protein + 1
    n_rows = n_comp + 2
    inlet = np.zeros((n_rows, 4))

    inlet[0] = [1e-4, 1e-4, 0.0, feedtime]
    inlet[1] = [gradient_start, gradient_end, grad_start_t, grad_end_t]
    inlet[2] = [gradient_end, gradient_end, grad_end_t, sim_end_t]

    for j, frac in enumerate(component_fractions_pct):
        row = 3 + j
        conc = frac / mol_to_g_l * load_amount_g_l / 100.0 / 10.0
        inlet[row] = [conc, conc, 0.0, feedtime]

    return InletProfile(segments=inlet, n_components=n_comp)


def build_simulation_inlet(
    *,
    buffer_a: float,
    buffer_b: float,
    gradient_start_pct: float,
    gradient_end_pct: float,
    elution_cv: float,
    rt_min: float,
    load_amount_g_l: float,
    component_fractions_pct: np.ndarray,
    feed_cv: float = 1.0,
    hold_cv: float = 4.0,
    mol_to_g_l: float = 150_000.0,
) -> InletProfile:
    """Build inlet for forward simulation / gradient optimization.

    Matches MATLAB ``generate_curve.m`` (1 CV load, 4 CV hold).
    """
    return build_fitting_inlet(
        buffer_a=buffer_a,
        buffer_b=buffer_b,
        gradient_start_pct=gradient_start_pct,
        gradient_end_pct=gradient_end_pct,
        elution_cv=elution_cv,
        rt_min=rt_min,
        load_amount_g_l=load_amount_g_l,
        component_fractions_pct=component_fractions_pct,
        feed_cv=feed_cv,
        hold_cv=hold_cv,
        mol_to_g_l=mol_to_g_l,
    )
