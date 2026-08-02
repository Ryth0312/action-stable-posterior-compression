"""PyTorch models and artifact helpers for ANN surrogates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class SurrogateMLP(nn.Module):
    """Small multilayer perceptron for fixed-grid chromatogram prediction."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (128, 128)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def save_artifact(
    path: str | Path,
    *,
    model: SurrogateMLP,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    output_time_s: torch.Tensor,
    target_shape: tuple[int, int],
    metadata: dict[str, Any],
    hidden_dims: tuple[int, ...],
    target_transform: str = "direct",
    pca_mean: torch.Tensor | None = None,
    pca_components: torch.Tensor | None = None,
) -> None:
    """Save a model artifact with normalization and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_mean.numel()),
            "output_dim": int(next(reversed(model.network)).out_features),
            "target_output_dim": int(y_mean.numel()),
            "hidden_dims": tuple(int(v) for v in hidden_dims),
            "x_mean": x_mean.detach().cpu(),
            "x_std": x_std.detach().cpu(),
            "y_mean": y_mean.detach().cpu(),
            "y_std": y_std.detach().cpu(),
            "output_time_s": output_time_s.detach().cpu(),
            "target_shape": tuple(int(v) for v in target_shape),
            "metadata": metadata,
            "target_transform": target_transform,
            "pca_mean": pca_mean.detach().cpu() if pca_mean is not None else None,
            "pca_components": (
                pca_components.detach().cpu() if pca_components is not None else None
            ),
        },
        path,
    )


def load_artifact(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    """Load a saved surrogate artifact and instantiate its MLP."""
    artifact = torch.load(Path(path), map_location=map_location, weights_only=False)
    model = SurrogateMLP(
        int(artifact["input_dim"]),
        int(artifact["output_dim"]),
        tuple(artifact["hidden_dims"]),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    artifact["model"] = model
    return artifact
