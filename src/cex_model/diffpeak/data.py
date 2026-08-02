"""Load a product's REAL experimental outlet curves for the differentiable peak model (D4.1).

NO RK23, NO SMA parameters -- just the measured per-(observed-)peak elution traces that ``app_support`` already
parses from the lab files, plus the operating conditions of each experiment. HLXSYN/HLXSYN have MERGED observed
peaks (e.g. AP1+AP2, or a near-inert tail); we model the observed peaks AS-IS (the identifiable unit), not the
underlying SMA components -- this line is about fitting what was actually measured, not reproducing the SMA
component structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import cex_model.app_support as A
from cex_model.components import ComponentType

# Operating-condition features that VARY across a product's experiments (flow + buffers are constant per
# product; feed_cv is not recorded per calibration experiment) -- loading drives peak AREA, the gradient drives
# retention/width. ``loading_g_l`` MUST stay first (the area model reads it directly).
OP_FEATURES: tuple[str, ...] = ("loading_g_l", "gradient_start_pct", "gradient_end_pct", "elution_cv")


@dataclass
class Experiment:
    name: str
    times: np.ndarray            # (T,) elution-relative time [s]
    curves: np.ndarray           # (n_peaks, T) measured concentration per observed peak [g/L]
    op: np.ndarray               # (n_op,) raw operating-condition vector in OP_FEATURES order
    loading: float               # loading_g_l (== op[0]; the area scale)


@dataclass
class PeakData:
    product: str
    n_peaks: int
    peak_types: tuple[str, ...]  # 'acid' / 'main' / 'basic' per observed peak (in elution/column order)
    frac: np.ndarray             # (n_peaks,) KNOWN feed fraction (proportion) of each observed peak
    experiments: list[Experiment]

    @property
    def op_matrix(self) -> np.ndarray:
        return np.stack([e.op for e in self.experiments])  # (n_exp, n_op)

    def op_standardizer(self, idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Mean/std of the OP features over experiments ``idx`` (default all) -- fit on TRAIN only for LOO."""
        M = self.op_matrix if idx is None else self.op_matrix[idx]
        return M.mean(axis=0), M.std(axis=0) + 1e-8


def load_peak_data(product: str) -> PeakData:
    """Real outlet curves + operating conditions for ``product`` (HLXSYN/HLXSYN/HLXSYN), grouped into observed
    peaks. Each observed peak's KNOWN feed fraction = the summed feed fraction of the components mapped to it
    (``observation_groups``); its type is the (shared) component type -> the retention-order weak constraint."""
    b = A.load_product(product)
    comps = b.components.components
    frac_comp = np.asarray(b.components.fraction_array(), dtype=float)
    frac_comp = frac_comp / frac_comp.sum() if frac_comp.sum() > 0 else frac_comp
    types = [c.component_type for c in comps]

    exps: list[Experiment] = []
    groups_ref = None
    for e in b.experiments:
        cur = np.asarray(e.curve, dtype=float)
        times, traces = cur[:, 0], cur[:, 1:].T            # traces: (n_obs, T)
        groups = e.observation_groups or [[j] for j in range(traces.shape[0])]
        groups_ref = groups
        op = np.array([getattr(e, "loading_g_l"), e.gradient_start_pct, e.gradient_end_pct, e.elution_cv],
                      dtype=float)
        exps.append(Experiment(name=e.name, times=times, curves=traces, op=op, loading=float(e.loading_g_l)))

    n_peaks = exps[0].curves.shape[0]
    frac = np.array([float(frac_comp[g].sum()) for g in groups_ref], dtype=float)
    ptypes = tuple(_group_type([types[i] for i in g]) for g in groups_ref)
    return PeakData(product=product, n_peaks=n_peaks, peak_types=ptypes, frac=frac, experiments=exps)


def _group_type(group_types: list[ComponentType]) -> str:
    """Observed-peak type = the type shared by its components (MAIN wins if a group mixes, else the first)."""
    vals = [t.value for t in group_types]
    return "main" if "main" in vals else vals[0]
