"""Pool several PRODUCTS' design-space surrogate datasets into one cross-product dataset.

The physics is one shared EDM+SMA operator; products (HLXSYN/03/04) are just parameter instances,
and the per-component SMA params already live in the feature vector. Since each product's
``output_time_s`` is ``linspace(0, t_end, T)``, target index k == fraction k/(T-1) of that run's
elution — the curves are aligned in NORMALISED time, so we resample onto a common grid and pool,
recording a per-sample ``product_ids`` array for leave-one-product-out. numpy-only (no torch).
"""

from __future__ import annotations

import numpy as np


def resample_to_normalized_grid(targets: np.ndarray, n_grid: int) -> np.ndarray:
    """Resample ``(n, T, C)`` onto ``n_grid`` points over the normalised [0,1] curve-fraction axis.

    Identity (a copy) when ``T == n_grid`` — the common case when every product was generated with
    the same ``--output-points``.
    """
    targets = np.asarray(targets, dtype=np.float32)
    n, t, c = targets.shape
    if t == n_grid:
        return targets.copy()
    own = np.linspace(0.0, 1.0, t)
    common = np.linspace(0.0, 1.0, n_grid)
    out = np.empty((n, n_grid, c), dtype=np.float32)
    for ch in range(c):
        out[:, :, ch] = np.apply_along_axis(
            lambda row: np.interp(common, own, row), 1, targets[:, :, ch])
    return out


def pool_products(
    arrays: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    *,
    n_grid: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pool ``[(features, targets), ...]`` from several products into one cross-product set.

    Requires identical feature dim and channel count across products (same #components; variable-N
    products need the Step-2 set-equivariant model). Returns
    ``(features, targets, product_ids, n_grid)`` where ``targets`` is on the common normalised grid.
    """
    if len(arrays) != len(labels):
        raise ValueError(f"labels count ({len(labels)}) != datasets count ({len(arrays)})")
    feat_dims = {f.shape[1] for f, _ in arrays}
    chans = {t.shape[2] for _, t in arrays}
    if len(chans) != 1:
        raise ValueError(f"datasets have different channel counts {chans}; the cross-product probe "
                         "needs the same #components (Step-2 set-equivariant model handles variable N)")
    if len(feat_dims) != 1:
        raise ValueError(f"datasets have different feature dims {feat_dims}; expected identical "
                         "positional layout for same-component-count products")
    grid = n_grid or min(t.shape[1] for _, t in arrays)
    feats, targs, pids = [], [], []
    for label, (f, t) in zip(labels, arrays):
        feats.append(np.asarray(f, dtype=np.float32))
        targs.append(resample_to_normalized_grid(t, grid))
        pids.append(np.full(f.shape[0], label, dtype=object))
    return (np.concatenate(feats, axis=0), np.concatenate(targs, axis=0),
            np.concatenate(pids, axis=0).astype(str), int(grid))
