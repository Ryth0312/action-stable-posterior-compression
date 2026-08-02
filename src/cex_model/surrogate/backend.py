"""PyTorch surrogate simulation backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cex_model.sma import MOL_TO_G_L
from cex_model.simulator import SimulationResult
from cex_model.surrogate.features import build_feature_vector
from cex_model.surrogate.model import load_artifact


class TorchSurrogateBackend:
    """ANN backend that returns ``SimulationResult`` from a saved PyTorch model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        clip_nonnegative: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.artifact = load_artifact(self.model_path, map_location=map_location)
        self.model = self.artifact["model"]
        self.clip_nonnegative = clip_nonnegative

    @property
    def metadata(self) -> dict:
        return dict(self.artifact.get("metadata", {}))

    def _validate_request(self, simulator, n_time_points: int) -> None:
        metadata = self.metadata
        expected_components = metadata.get("component_names")
        if expected_components and list(expected_components) != simulator.components.names:
            raise ValueError(
                "Surrogate component names do not match simulator components. "
                f"Expected {expected_components}, got {simulator.components.names}"
            )
        expected_n_tp = metadata.get("n_time_points")
        if expected_n_tp is not None and int(expected_n_tp) != int(n_time_points):
            raise ValueError(
                "Surrogate was trained for a different n_time_points value. "
                f"Expected {expected_n_tp}, got {n_time_points}"
            )

    def simulate(
        self,
        *,
        simulator,
        inlet,
        loading_g_l: float,
        n_time_points: int | None = None,
        method: str | None = None,
        t_start: float = 0.0,
    ) -> SimulationResult:
        """Predict a fixed-grid elution curve and wrap it as ``SimulationResult``."""
        n_tp = n_time_points or simulator.n_time_points
        self._validate_request(simulator, n_tp)

        workflow = self.metadata.get("workflow", "fitting")
        feature, spec = build_feature_vector(
            simulator.column,
            simulator.components,
            inlet,
            loading_g_l,
            n_time_points=n_tp,
            t_start=t_start,
            method_code=0.0 if workflow == "fitting" else 1.0,
        )
        if feature.shape[0] != int(self.artifact["input_dim"]):
            raise ValueError(
                "Surrogate feature dimension mismatch. "
                f"Expected {self.artifact['input_dim']}, got {feature.shape[0]}"
            )

        x = torch.from_numpy(feature).float()
        x = (x - self.artifact["x_mean"]) / self.artifact["x_std"]
        # Zero the features that were CONSTANT in training (x_std sitting at the 1e-8 clamp floor):
        # they carry no signal in a per-product surrogate (fixed column + SMA params), and their float32
        # storage noise (~1e-6) otherwise divides by ~1e-8 into HUGE (+-100s) values that (a) the model
        # would spuriously fit and (b) shift by platform-dependent float rounding -> fragile inference.
        # train_surrogate.py / run_surrogate_kfold.py apply the IDENTICAL zeroing, so the model is trained
        # with these = 0 and never depends on them. The threshold MUST match the training scripts (1e-6).
        x = torch.where(self.artifact["x_std"] > 1e-6, x, torch.zeros_like(x))
        with torch.no_grad():
            y = self.model(x.unsqueeze(0)).squeeze(0)
        if self.artifact.get("target_transform", "direct") == "pca":
            pca_mean = self.artifact.get("pca_mean")
            pca_components = self.artifact.get("pca_components")
            if pca_mean is None or pca_components is None:
                raise ValueError("PCA surrogate artifact is missing PCA parameters")
            y = y @ pca_components + pca_mean
        y = y * self.artifact["y_std"] + self.artifact["y_mean"]
        target = y.cpu().numpy().reshape(tuple(self.artifact["target_shape"]))
        if not np.all(np.isfinite(target)):
            raise ValueError("Surrogate prediction contains NaN or Inf values")

        salt = target[:, 0].astype(float)
        proteins_g_l = target[:, 1:].astype(float)
        if proteins_g_l.shape[1] != simulator.components.n_protein:
            raise ValueError(
                "Surrogate protein output count does not match simulator components. "
                f"Expected {simulator.components.n_protein}, got {proteins_g_l.shape[1]}"
            )
        if self.clip_nonnegative:
            salt = np.maximum(salt, 0.0)
            proteins_g_l = np.maximum(proteins_g_l, 0.0)

        return SimulationResult(
            time_s=self.artifact["output_time_s"].cpu().numpy().astype(float),
            salt_mol_l=salt,
            protein_mol_l=proteins_g_l / MOL_TO_G_L,
            feed_end_index=0,
            raw_solution={"backend": "torch_surrogate", "model_path": str(self.model_path)},
        )
