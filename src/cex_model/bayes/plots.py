"""Basic figures for the Bayesian calibration: posterior-predictive bands and
the identifiability bar chart.  Headless (Agg) so it runs on Colab.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cex_model.inverse.encoding import log_mask  # noqa: E402

__all__ = ["plot_posterior_predictive", "plot_identifiability", "plot_design", "plot_corner"]

# Double-blind label map for figure TITLES (gated by env BLIND_LABELS=1; default OFF ->
# original output; filenames are unaffected).
import os  # noqa: E402

_ANON = {"HLXSYN": "mAb E", "HLXSYN": "mAb D",
         "HLXSYN": "mAb A", "HLXSYN": "mAb B", "HLXSYN": "mAb C"}


def _blind(s: str) -> str:
    if os.environ.get("BLIND_LABELS") != "1":
        return s
    for k, v in _ANON.items():
        s = s.replace(k, v)
    return s


def plot_posterior_predictive(result: dict, out_dir, *, product: str | None = None) -> list[Path]:
    """One PNG per experiment: real points + posterior-predictive mean & band per observed peak."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    product = product or result.get("product", "product")
    level = result.get("level", 0.9)
    paths = []
    # experiment index for titles/filenames keeps them ASCII (names may be CJK).
    for i, e in enumerate(result["per_experiment"]):
        b = e["band"]
        t, lo, hi, mean, real = b["t"] / 60.0, b["lo"], b["hi"], b["mean"], b["real"]
        n_obs = mean.shape[1]
        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = plt.get_cmap("tab10")
        for k in range(n_obs):
            c = cmap(k % 10)
            ax.fill_between(t, lo[:, k], hi[:, k], color=c, alpha=0.2)
            ax.plot(t, mean[:, k], color=c, lw=1.5, label=f"obs{k + 1} pred")
            ax.scatter(t, real[:, k], color=c, s=10, alpha=0.7)
        ax.set_xlabel("time (min)")
        ax.set_ylabel("concentration (g/L)")
        ax.set_title(_blind(f"{product} exp{i + 1}: posterior-predictive ({int(level * 100)}% band) "
                     f"vs real | coverage={e['coverage']:.2f}"))
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = out_dir / f"{product}_exp{i + 1}_predictive.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_identifiability(shrinkage: dict[str, float], out_path, *, title: str = "identifiability") -> Path:
    """Bar chart of posterior_std/prior_std per parameter (pinned < 0.5 highlighted)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(shrinkage.keys())
    vals = np.array([shrinkage[n] for n in names])
    colors = ["tab:green" if v < 0.5 else "tab:gray" for v in vals]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(names)), 4))
    ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.axhline(1.0, color="tab:red", ls=":", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("posterior_std / prior_std")
    ax.set_title(f"{title} (green = data-pinned < 0.5; red = prior-dominated ~ 1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_design(op_matrix, recommendations: list[dict], out_path, *, product: str = "product") -> Path:
    """2D OP scatter: existing experiments (blue) vs recommended next experiments (red stars)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat = ("loading_g_l", "gradient_start_pct", "gradient_end_pct", "elution_cv")
    idx = {n: i for i, n in enumerate(feat)}
    op_matrix = np.asarray(op_matrix, float)
    rec = np.array([[r[f] for f in feat] for r in recommendations]) if recommendations else np.zeros((0, 4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (fx, fy) in zip(axes, [("loading_g_l", "gradient_end_pct"), ("gradient_start_pct", "elution_cv")]):
        ax.scatter(op_matrix[:, idx[fx]], op_matrix[:, idx[fy]], c="tab:blue", s=40, label="existing")
        if len(rec):
            ax.scatter(rec[:, idx[fx]], rec[:, idx[fy]], c="tab:red", marker="*", s=140, label="recommended")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.legend(fontsize=8)
    fig.suptitle(_blind(f"{product}: BOED recommended next experiments"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _default_corner_names(posterior) -> list[str]:
    """The 4 params of the most strongly (keq<->nu) correlated component."""
    from cex_model.bayes.identifiability import correlation_pairs
    suffix = correlation_pairs(posterior, k=1)[0]["params"][0].split("_", 1)[1]  # e.g. "c2"
    return [n for p in ("keq", "kkin", "nu", "sigma") if (n := f"{p}_{suffix}") in posterior.names]


def plot_corner(posterior, out_path, *, names=None, n_samples: int = 2000, seed: int = 0,
                title: str | None = None) -> Path:
    """Posterior corner (pairwise marginals) in u-space -- shows the keq<->nu ridge + sigma width.

    Defaults to the 4 SMA params of the most keq<->nu-correlated component; pass
    ``names`` (e.g. ``["keq_c2","nu_c2"]``) to override.  keq/kkin axes are log10
    (their native inference space), so the structural degeneracy is a straight ridge.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = names or _default_corner_names(posterior)
    idx = [posterior.names.index(n) for n in names]
    S = posterior.samples(n_samples, seed=seed)[:, idx]
    is_log = log_mask(posterior.n_protein)[idx]
    labels = [f"log10 {n}" if is_log[i] else n for i, n in enumerate(names)]
    k = len(names)
    fig, ax = plt.subplots(k, k, figsize=(2.2 * k, 2.2 * k), squeeze=False)
    for i in range(k):
        for j in range(k):
            a = ax[i][j]
            if j > i:
                a.axis("off")
                continue
            if i == j:
                a.hist(S[:, i], bins=40, color="tab:blue", alpha=0.8)
                a.set_yticks([])
            else:
                a.scatter(S[:, j], S[:, i], s=3, alpha=0.15, color="tab:blue", edgecolors="none")
                a.set_title(f"r={np.corrcoef(S[:, j], S[:, i])[0, 1]:+.2f}", fontsize=7, pad=1)
            a.set_xlabel(labels[j], fontsize=8) if i == k - 1 else a.set_xticklabels([])
            (a.set_ylabel(labels[i], fontsize=8) if j == 0 and i > 0 else a.set_yticklabels([]))
    fig.suptitle(title or f"{getattr(posterior, 'engine', '')} posterior corner")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
