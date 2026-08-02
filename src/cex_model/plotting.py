"""Visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_chromatogram(
    curve: np.ndarray,
    exp_data: np.ndarray | None = None,
    *,
    title: str = "Simulated vs Experimental Chromatogram",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot outlet chromatogram with optional experimental overlay."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    t = curve[:, 0]
    n_prot = curve.shape[1] - 2
    for j in range(n_prot):
        ax1.plot(t, curve[:, 2 + j], label=f"Sim C{j+1}")
    if exp_data is not None:
        for j in range(exp_data.shape[1] - 1):
            ax1.plot(
                exp_data[:, 0],
                exp_data[:, 1 + j],
                ".",
                markersize=8,
                label=f"Exp C{j+1}",
            )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Protein Conc. (g/L)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, curve[:, 1], "--", color="gray", label="Salt")
    ax2.set_ylabel("Ion strength (mol/L)")
    ax1.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_residuals(
    curve: np.ndarray,
    exp_data: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(exp_data.shape[1] - 1):
        idx = np.array(
            [np.where(curve[:, 0] > exp_data[k, 0])[0][0] for k in range(len(exp_data))]
        )
        resid = curve[idx, 2 + j] - exp_data[:, 1 + j]
        ax.plot(exp_data[:, 0], resid, ".-", label=f"C{j+1}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual (g/L)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_collection_yield_grid(
    yield_grid: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(yield_grid, origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, label="Yield")
    ax.set_xlabel("End grid index")
    ax.set_ylabel("Start grid index")
    ax.set_title("Collection window yield grid")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_yamamoto_regression(
    cv: np.ndarray,
    cs: np.ndarray,
    fit_line: np.ndarray | None = None,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cv, cs, "o", label="Mid-gradient salt")
    if fit_line is not None:
        ax.plot(cv, fit_line, "-", label="Regression")
    ax.set_xlabel("Gradient length (CV)")
    ax.set_ylabel("Salt at midpoint (mol/L)")
    ax.legend()
    ax.set_title("Yamamoto regression")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
