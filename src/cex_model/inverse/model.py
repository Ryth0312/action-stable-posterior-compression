"""Artifact (de)serialization for the inverse parameter-estimation network.

The network itself is the same small MLP used by the forward surrogate
(:class:`cex_model.surrogate.model.SurrogateMLP`); only the input/output meaning
differs.  The artifact bundles the frozen :class:`InverseEncoder` (curve+
condition encoding) and the parameter-space normalization so a saved model is
fully self-describing for prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cex_model.inverse.encoding import InverseEncoder
from cex_model.surrogate.model import SurrogateMLP


def _np(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def save_inverse_artifact(
    path: str | Path,
    *,
    model: SurrogateMLP,
    encoder: InverseEncoder,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    param_names: list[str],
    component_names: list[str],
    fractions: np.ndarray,
    hidden_dims: tuple[int, ...],
    metadata: dict[str, Any],
) -> None:
    """Persist the inverse model with its encoder and param normalization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "inverse",
            "state_dict": model.state_dict(),
            "input_dim": int(encoder.x_mean.size),
            "output_dim": int(_np(y_mean).size),
            "hidden_dims": tuple(int(v) for v in hidden_dims),
            # parameter-space (transformed) normalization
            "y_mean": _np(y_mean),
            "y_std": _np(y_std),
            "param_names": list(param_names),
            "component_names": list(component_names),
            "fractions": _np(fractions),
            # encoder (frozen on the training split)
            "encoder": {
                "output_time_s": _np(encoder.output_time_s),
                "n_protein": int(encoder.n_protein),
                "cond_index": np.asarray(encoder.cond_index, dtype=np.int64),
                "chan_mean": _np(encoder.chan_mean),
                "chan_std": _np(encoder.chan_std),
                "pca_mean": _np(encoder.pca_mean),
                "pca_components": _np(encoder.pca_components),
                "x_mean": _np(encoder.x_mean),
                "x_std": _np(encoder.x_std),
                "use_peak_features": bool(encoder.use_peak_features),
            },
            "metadata": metadata,
        },
        path,
    )


def load_inverse_artifact(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict:
    """Load an inverse artifact; returns dict with ``model`` and ``encoder``."""
    artifact = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = SurrogateMLP(
        int(artifact["input_dim"]),
        int(artifact["output_dim"]),
        tuple(artifact["hidden_dims"]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    enc = artifact["encoder"]
    artifact["model"] = model
    artifact["encoder_obj"] = InverseEncoder(
        output_time_s=enc["output_time_s"],
        n_protein=int(enc["n_protein"]),
        cond_index=np.asarray(enc["cond_index"], dtype=int),
        chan_mean=enc["chan_mean"],
        chan_std=enc["chan_std"],
        pca_mean=enc["pca_mean"],
        pca_components=enc["pca_components"],
        x_mean=enc["x_mean"],
        x_std=enc["x_std"],
        use_peak_features=bool(enc["use_peak_features"]),
    )
    return artifact
