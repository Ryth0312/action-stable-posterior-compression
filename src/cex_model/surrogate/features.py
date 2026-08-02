"""Feature and target encoding for ANN surrogate models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet
from cex_model.gradients import InletProfile


@dataclass(frozen=True)
class FeatureSpec:
    """Names and component metadata for a surrogate feature vector."""

    feature_names: list[str]
    component_names: list[str]


def build_feature_vector(
    column: ColumnParameters,
    components: ComponentSet,
    inlet: InletProfile,
    loading_g_l: float,
    *,
    n_time_points: int,
    t_start: float,
    method_code: float = 0.0,
) -> tuple[np.ndarray, FeatureSpec]:
    """Encode simulator inputs into a fixed-length numeric feature vector."""
    feature_names: list[str] = []
    values: list[float] = []

    def add(name: str, value: float) -> None:
        feature_names.append(name)
        values.append(float(value))

    add("column_volume", column.column_volume)
    add("column_length", column.column_length)
    add("flow_rate", column.flow_rate)
    add("total_porosity", column.total_porosity)
    add("superficial_velocity", column.superficial_velocity)
    add("dax", column.dax)
    add("ionic_capacity", column.ionic_capacity)
    add("grid_size", column.grid_size)
    add("rt_min", column.rt)
    add("loading_g_l", loading_g_l)
    add("n_time_points", n_time_points)
    add("t_start", t_start)
    add("method_code", method_code)
    add("inlet_duration", inlet.duration)

    for i, row_name in enumerate(["feed", "gradient", "hold"]):
        c0, c1, ts, te = inlet.segments[i]
        add(f"{row_name}_salt_start", c0)
        add(f"{row_name}_salt_end", c1)
        add(f"{row_name}_time_start", ts)
        add(f"{row_name}_time_end", te)

    for idx, comp in enumerate(components.components, start=1):
        add(f"component_{idx}_fraction", comp.fraction)
        add(f"component_{idx}_nu", comp.nu)
        add(f"component_{idx}_keq", comp.keq)
        add(f"component_{idx}_kkin", comp.kkin)
        add(f"component_{idx}_sigma", comp.sigma)

    return (
        np.asarray(values, dtype=np.float32),
        FeatureSpec(feature_names=feature_names, component_names=components.names),
    )


def curve_to_target(
    curve: np.ndarray, output_time_s: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an elution curve into ANN target values on a fixed time grid.

    Parameters
    ----------
    curve:
        Array with columns ``[time_s, salt_mol_L, proteins_g_L...]``.
    output_time_s:
        Optional target time grid. If omitted, the curve's own grid is used.

    Returns
    -------
    output_time_s, target:
        Time grid and target matrix with columns ``[salt, proteins...]``.
    """
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] < 3:
        raise ValueError(
            "curve must be 2D with columns [time_s, salt_mol_L, protein...]"
        )
    if output_time_s is None:
        output_time_s = curve[:, 0].astype(np.float32)
    else:
        output_time_s = np.asarray(output_time_s, dtype=np.float32)

    target = np.column_stack(
        [
            np.interp(output_time_s, curve[:, 0], curve[:, col])
            for col in range(1, curve.shape[1])
        ]
    )
    return output_time_s.astype(np.float32), target.astype(np.float32)


def target_is_physical(target: np.ndarray, loading_g_l: float) -> bool:
    """Reject a blown-up ODE solve (an extreme-param corner where the stiff EDM+SMA BDF diverges).

    Such solves return FINITE-but-huge values (1e3..1e6 g/L), not an exception, so they slip past
    the generator's RuntimeError skip and poison the dataset (the surrogate then trains on garbage).
    Real elution curves are O(1-20) g/L, bounded by the feed; this caps the protein channels (col 1..)
    at ``max(1e3, 50*loading)`` and requires finiteness. Salt (col 0, mol/L) is excluded from the cap.
    """
    target = np.asarray(target, dtype=float)
    if not np.all(np.isfinite(target)):
        return False
    prot = target[:, 1:] if target.ndim == 2 and target.shape[1] > 1 else target
    cap = max(1e3, 50.0 * max(float(loading_g_l), 1.0))
    return float(np.max(np.abs(prot))) <= cap

