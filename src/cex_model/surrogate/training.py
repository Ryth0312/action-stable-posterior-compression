"""Reusable single-split train + score for the ANN surrogate (shared by k-fold & LOPO).

Lifted verbatim (recipe-preserving) from ``scripts/run_surrogate_kfold.py``'s per-fold body so
both random k-fold and leave-one-product-out (cross-product) use the *same* training procedure:
feature/target normalization + PCA fit on the TRAIN POOL ONLY (no leakage), optional
concentration-weighted loss, multi-restart Adam with an inner-validation early stop, then the
held-out split scored in physical units (g/L). torch-only (surrogate extra).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cex_model.surrogate.model import SurrogateMLP


@dataclass
class TrainConfig:
    """Training recipe knobs (mirror run_surrogate_kfold.py CLI defaults)."""

    hidden_dims: tuple[int, ...] = (128, 128)
    epochs: int = 200
    batch_size: int = 16
    lr: float = 1e-3
    patience: int = 30
    min_delta: float = 0.0
    inner_val_fraction: float = 0.15
    restarts: int = 1
    lr_schedule: str = "none"  # "none" | "plateau"
    target_normalization: str = "channel"  # "channel" | "pointwise" | "global"
    target_transform: str = "pca"  # "pca" | "direct"
    pca_components: int = 32
    loss_weighting: str = "none"  # "none" | "concentration"
    loss_alpha: float = 3.0
    seed: int = 1
    clamp_output: bool = False  # clip predictions to [0, margin*train_channel_max] (physical bound)
    clamp_margin: float = 1.5   # so out-of-distribution LOPO extrapolation can't explode to 1e6


def _fit_normalization(target_tensor, x, stats_idx, mode, n_time):
    """Feature/target normalization fit on ``stats_idx`` only (train pool)."""
    xs = x[stats_idx]
    x_mean = xs.mean(dim=0)
    x_std = xs.std(dim=0).clamp_min(1e-8)
    ts = target_tensor[stats_idx]
    if mode == "pointwise":
        y_train = ts.reshape(ts.shape[0], -1)
        y_mean = y_train.mean(dim=0)
        y_std = y_train.std(dim=0).clamp_min(1e-8)
    elif mode == "global":
        y_mean = torch.full((ts.shape[1] * ts.shape[2],), float(ts.mean()))
        y_std = torch.full_like(y_mean, float(ts.std().clamp_min(1e-8)))
    else:  # channel (default)
        channel_mean = ts.mean(dim=(0, 1))
        channel_std = ts.std(dim=(0, 1)).clamp_min(1e-8)
        y_mean = channel_mean.repeat(n_time)
        y_std = channel_std.repeat(n_time)
    return x_mean, x_std, y_mean, y_std


def train_and_score(
    features: np.ndarray,
    targets: np.ndarray,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: TrainConfig,
    *,
    output_columns: list[str] | None = None,
    split_tag: int = 0,
) -> dict:
    """Train on ``train_pool_idx`` (inner-val early stop + restarts) and score ``test_idx``.

    ``features`` (n, F), ``targets`` (n, T, C). Returns a report dict with overall/per-channel/
    protein RMSE (physical units), max abs error, best epoch and the per-restart inner-val losses.
    Identical recipe to one k-fold fold; the only difference here is WHO is in train vs test.
    """
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    target_tensor = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    n, n_time, n_channels = target_tensor.shape
    train_pool_np = np.asarray(train_pool_idx, dtype=np.int64)
    test_idx_t = torch.from_numpy(np.asarray(test_idx, dtype=np.int64))
    train_pool = torch.from_numpy(train_pool_np.copy())
    if output_columns is None:
        output_columns = [f"ch{i}" for i in range(n_channels)]

    n_inner_val_size = max(1, int(len(train_pool_np) * cfg.inner_val_fraction))
    n_inner_train_size = len(train_pool_np) - n_inner_val_size

    # --- Normalization + PCA fit on the TRAIN POOL ONLY (no leakage) ---
    x_mean, x_std, y_mean, y_std = _fit_normalization(
        target_tensor, x, train_pool, cfg.target_normalization, n_time)
    x_norm = (x - x_mean) / x_std
    y_full = target_tensor.reshape(n, -1)
    y_norm_full = (y_full - y_mean) / y_std

    if cfg.target_transform == "pca":
        pool_norm_full = y_norm_full[train_pool]
        pca_mean = pool_norm_full.mean(dim=0)
        _, _, vh = torch.linalg.svd(pool_norm_full - pca_mean, full_matrices=False)
        n_comp = min(cfg.pca_components, vh.shape[0])
        pca_components = vh[:n_comp].contiguous()
        y_norm = (y_norm_full - pca_mean) @ pca_components.T
    else:
        pca_mean = pca_components = None
        y_norm = y_norm_full

    if cfg.loss_weighting == "concentration" and n_channels > 1:
        channel_max = target_tensor[train_pool].amax(dim=(0, 1)).clamp_min(1e-8)
        weights_matrix = torch.ones_like(target_tensor)
        protein_rel = (target_tensor[:, :, 1:] / channel_max[1:]).clamp_min(0.0)
        weights_matrix[:, :, 1:] = 1.0 + cfg.loss_alpha * protein_rel
        loss_weights_full = weights_matrix.reshape(n, -1)
    else:
        loss_weights_full = torch.ones_like(y_norm_full)

    def reconstruct_full(pred):
        return pred @ pca_components + pca_mean if cfg.target_transform == "pca" else pred

    loss_fn = nn.MSELoss()

    def weighted_loss(pred, target, full_target, weights):
        if cfg.loss_weighting == "none" and cfg.target_transform == "direct":
            return loss_fn(pred, target)
        return torch.mean((reconstruct_full(pred) - full_target) ** 2 * weights)

    def _train_attempt(attempt_seed: int):
        split = np.random.default_rng(attempt_seed).permutation(train_pool_np)
        inner_val = torch.from_numpy(split[:n_inner_val_size].copy())
        inner_train = torch.from_numpy(split[n_inner_val_size:].copy())
        train_ds = TensorDataset(x_norm[inner_train], y_norm[inner_train],
                                 y_norm_full[inner_train], loss_weights_full[inner_train])
        torch.manual_seed(attempt_seed)
        model = SurrogateMLP(x.shape[1], y_norm.shape[1], cfg.hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(cfg.patience // 4, 5))
            if cfg.lr_schedule == "plateau" else None)
        gen = torch.Generator().manual_seed(attempt_seed)
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
        best_score, best_epoch, stale = float("inf"), 0, 0
        best_state = copy.deepcopy(model.state_dict())
        for epoch in range(cfg.epochs):
            model.train()
            for xb, yb, yb_full, wb in loader:
                optimizer.zero_grad()
                weighted_loss(model(xb), yb, yb_full, wb).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(weighted_loss(model(x_norm[inner_val]), y_norm[inner_val],
                                               y_norm_full[inner_val],
                                               loss_weights_full[inner_val]).item())
            if scheduler is not None:
                scheduler.step(val_loss)
            if val_loss < best_score - cfg.min_delta:
                best_score, best_epoch, stale = val_loss, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if cfg.patience > 0 and stale >= cfg.patience:
                break
        return best_state, best_score, best_epoch

    attempts, best_state, best_inner_val, best_epoch = [], None, float("inf"), 0
    for r in range(max(cfg.restarts, 1)):
        attempt_seed = cfg.seed + split_tag * 1000 + r
        state, score, ep = _train_attempt(attempt_seed)
        attempts.append({"seed": attempt_seed, "inner_val_loss": score, "best_epoch": ep})
        if score < best_inner_val:
            best_inner_val, best_state, best_epoch = score, state, ep

    model = SurrogateMLP(x.shape[1], y_norm.shape[1], cfg.hidden_dims)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_full = reconstruct_full(model(x_norm[test_idx_t]))
        pred_phys = (pred_full * y_std + y_mean).reshape(-1, n_time, n_channels)
        if cfg.clamp_output:
            # Concentrations are non-negative and bounded by ~the feed; clip to
            # [0, margin*train_channel_max] so an out-of-distribution held-out product (LOPO)
            # can't extrapolate to 1e6 g/L and drown the metric. No effect on in-range predictions.
            cap = cfg.clamp_margin * target_tensor[train_pool].amax(dim=(0, 1))
            pred_phys = torch.minimum(pred_phys.clamp(min=0.0), cap.view(1, 1, -1))
    err = (pred_phys - target_tensor[test_idx_t]).numpy()

    per_channel = [float(np.sqrt(np.mean(err[:, :, c] ** 2))) for c in range(n_channels)]
    protein_rmse = (float(np.sqrt(np.mean(np.square(per_channel[1:]))))
                    if n_channels > 1 else None)
    return {
        "n_train": int(n_inner_train_size), "n_inner_val": int(n_inner_val_size),
        "n_test": int(len(test_idx_t)), "best_epoch": best_epoch,
        "rmse": float(np.sqrt(np.mean(err**2))), "protein_rmse": protein_rmse,
        "max_abs_diff": float(np.max(np.abs(err))) if err.size else 0.0,
        "per_channel_rmse": per_channel, "restarts": attempts,
        "chosen_inner_val_loss": best_inner_val,
    }
