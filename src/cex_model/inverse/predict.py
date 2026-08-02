"""Inference wrapper for the inverse parameter-estimation network."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet, ComponentType
from cex_model.gradients import build_fitting_inlet
from cex_model.inverse.encoding import vector_to_params_table
from cex_model.inverse.model import load_inverse_artifact
from cex_model.surrogate.features import build_feature_vector


def resample_proteins(curve: np.ndarray, output_time_s: np.ndarray, n_protein: int) -> np.ndarray:
    """Interpolate a curve onto the encoder grid -> (n_time, n_protein) g/L.

    ``curve`` columns are ``[time_s, *channels]``.  The trailing ``n_protein``
    channels are taken as proteins (a leading salt channel, if present, is
    ignored), matching the encoder's proteins-only convention.
    """
    curve = np.asarray(curve, dtype=float)
    t = curve[:, 0]
    proteins = curve[:, -n_protein:]
    grid = np.asarray(output_time_s, dtype=float)
    out = np.column_stack(
        [np.interp(grid, t, proteins[:, j]) for j in range(n_protein)]
    )
    return out


class InversePredictor:
    """Load an inverse artifact and estimate SMA parameters from curves."""

    def __init__(self, artifact: dict):
        self.model = artifact["model"]
        self.encoder = artifact["encoder_obj"]
        self.y_mean = np.asarray(artifact["y_mean"], dtype=float)
        self.y_std = np.asarray(artifact["y_std"], dtype=float)
        self.param_names = list(artifact["param_names"])
        self.component_names = list(artifact["component_names"])
        self.fractions = np.asarray(artifact["fractions"], dtype=float)
        self.metadata = artifact.get("metadata", {})

    @classmethod
    def from_artifact(cls, path: str | Path, *, map_location: str = "cpu") -> "InversePredictor":
        return cls(load_inverse_artifact(path, map_location=map_location))

    # -- core (transformed param-space) ------------------------------------ #
    def predict_vectors(self, proteins: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Batch predict transformed (log-keq/kkin) packed param vectors."""
        x = self.encoder.encode(proteins, features)
        with torch.no_grad():
            y_norm = self.model(torch.from_numpy(x.astype(np.float32))).cpu().numpy()
        return y_norm * self.y_std + self.y_mean

    def vector_to_components(
        self, vec: np.ndarray, *, types: Sequence[ComponentType] | None = None
    ) -> ComponentSet:
        table = vector_to_params_table(vec, self.fractions)
        return ComponentSet.from_parameter_table(table, names=self.component_names, types=types)

    # -- single real experiment -------------------------------------------- #
    def _feature_vector(
        self,
        column: ColumnParameters,
        base_components: ComponentSet,
        *,
        buffer_a: float,
        buffer_b: float,
        gradient_start_pct: float,
        gradient_end_pct: float,
        elution_cv: float,
        load_amount_g_l: float,
    ) -> np.ndarray:
        inlet = build_fitting_inlet(
            buffer_a=buffer_a,
            buffer_b=buffer_b,
            gradient_start_pct=gradient_start_pct,
            gradient_end_pct=gradient_end_pct,
            elution_cv=elution_cv,
            rt_min=column.rt,
            load_amount_g_l=load_amount_g_l,
            component_fractions_pct=base_components.fraction_array(),
        )
        feat, _ = build_feature_vector(
            column,
            base_components,
            inlet,
            load_amount_g_l,
            n_time_points=int(self.metadata.get("n_time_points", 1000)),
            t_start=float(self.metadata.get("t_start", 0.0)),
            method_code=0.0,
        )
        return feat

    def predict_from_curve(
        self,
        curve: np.ndarray,
        column: ColumnParameters,
        base_components: ComponentSet,
        *,
        buffer_a: float,
        buffer_b: float,
        gradient_start_pct: float,
        gradient_end_pct: float,
        elution_cv: float,
        load_amount_g_l: float,
        types: Sequence[ComponentType] | None = None,
    ) -> tuple[ComponentSet, np.ndarray]:
        """Estimate parameters from one experimental curve.

        Returns the predicted ``ComponentSet`` and the transformed param vector
        (the latter is what multi-experiment aggregation operates on).
        """
        feat = self._feature_vector(
            column,
            base_components,
            buffer_a=buffer_a,
            buffer_b=buffer_b,
            gradient_start_pct=gradient_start_pct,
            gradient_end_pct=gradient_end_pct,
            elution_cv=elution_cv,
            load_amount_g_l=load_amount_g_l,
        )
        proteins = resample_proteins(curve, self.encoder.output_time_s, self.encoder.n_protein)
        vec = self.predict_vectors(proteins[None, ...], feat[None, :])[0]
        if types is None:
            types = [c.component_type for c in base_components.components]
        return self.vector_to_components(vec, types=types), vec

    @staticmethod
    def aggregate_vectors(vectors: Sequence[np.ndarray], method: str = "median") -> np.ndarray:
        """Late-fusion of per-experiment transformed param vectors -> one vector.

        Aggregating in the transformed (log) space is the geometric mean for
        keq/kkin, which is the natural average for multiplicative quantities.
        """
        arr = np.vstack([np.asarray(v, dtype=float) for v in vectors])
        if method == "mean":
            return arr.mean(axis=0)
        return np.median(arr, axis=0)
