#!/usr/bin/env python3
"""One de-identified chromatogram panel per product, for the article's scientific-setting figure.

The panel is what a reader needs to see once: the eluate of a single run, cut into fractions, with the
acidic, main and basic charge variants separated, the fitted model through the assayed points, and the
collection window that turns a chromatogram into a purity and a yield.

Everything that carries process information is normalised away, because the recipe is proprietary while
the shape is not:

  * the abscissa is the fraction of the salt gradient elapsed, so the buffer molarities, the gradient
    endpoints in %B and the elution volume in column volumes all cancel;
  * the ordinate is divided by the run's own main-species peak, so absolute concentration and the load
    cancel;
  * species are named acidic / main / basic, never by their assay column;
  * products are named mAb A, B, C, and the run is identified only by its index.

What survives is peak count and order, the separation the gradient achieves, how well the fit tracks the
points, and where the window sits -- which is the whole content of the figure.

The panel is drawn at the correlated-residual refit, the posterior every deployed read in the article uses,
so the pooled purity and yield in the title are the ones the rest of the article reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Product -> blinded label. The run drawn is the one performed at the product's decision operating
# condition, found by matching it below, so the window and the pooled numbers in the title belong to the
# curve on the panel and not to some other run.
PRODUCTS = [("HLXSYN", "mAb A"), ("HLXSYN", "mAb B"), ("HLXSYN", "mAb C")]

# Measured columns -> the three families a reader is told about. Model components are grouped onto the
# assayed columns upstream, so this maps observation groups, not model species.
FAMILY = {0: "acidic", 1: "main"}          # anything after index 1 is a basic variant
COLOUR = {"acidic": "tab:orange", "main": "tab:blue", "basic": "tab:green"}


def _family(k: int, n: int) -> str:
    return FAMILY.get(k, "basic")


def _run_at_decision_op(experiments, decision_op) -> int:
    """Index of the run performed at the decision operating condition."""
    hits = [i for i, e in enumerate(experiments)
            if [e.loading_g_l, e.gradient_start_pct, e.gradient_end_pct, e.elution_cv] == list(decision_op)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one run at {decision_op}, found {hits}")
    return hits[0]


def load_panel(pid: str):
    """Observed fractions and the fitted curve for one run, in normalised coordinates."""
    import torch

    import cex_model.app_support as A
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.posterior import Posterior
    from cex_model.diffsolver.calibrate_diff import targets_from_bundle

    prod, drop = ("HLXSYN", ["DT"]) if pid == "HLXSYN" else (pid, [])
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)

    dec_op = json.loads((ROOT / "results" / "bayes" / f"{pid}_decision_window.json").read_text())["decision_op"]
    run_idx = _run_at_decision_op(bundle.experiments, dec_op)

    from cex_model.bayes.likelihood import unpack_u
    from cex_model.diffsolver.torch_solver import DTYPE

    # The DEPLOYED posterior, so the curve, the window and the title agree with every other read in the
    # article. The independent-residual fit is a different law and reports different pooled numbers.
    post = Posterior.load(ROOT / "results" / "bayes" / f"{pid}_correlated_posterior.npz")
    n_prot = post.n_protein
    targets = targets_from_bundle(bundle, n_steps=300)
    tg = targets[run_idx]

    u = torch.tensor(np.asarray(post.u_map, float), dtype=DTYPE)
    with torch.no_grad():
        full = tg.sim.elution_curve(*unpack_u(u, n_prot), differentiable=False)

    full = np.asarray(full.detach().cpu(), float)
    t_model = full[:, 0]
    # Group the model species onto the assayed columns exactly as the likelihood does.
    groups = tg.groups if tg.groups is not None else [[j] for j in range(n_prot)]
    y_model = np.column_stack([full[:, 2 + np.asarray(g)].sum(1) for g in groups])
    t_obs = np.asarray(tg.times_s, float)
    y_obs = np.asarray(tg.values.detach().cpu() if hasattr(tg.values, "detach") else tg.values, float)

    # Abscissa: fraction of the salt gradient elapsed. Both endpoints come from the run's own inlet
    # schedule, so the buffer molarities, the %B endpoints and the elution volume all cancel.
    exp = bundle.experiments[run_idx]
    rt = float(exp.rt_min) * 60.0
    g0, g1 = 0.0, float(exp.elution_cv) * rt

    def norm_t(t):
        return (np.asarray(t, float) - g0) / (g1 - g0)

    # Ordinate: the run's own main-species peak.
    scale = float(np.nanmax(y_model[:, 1])) or 1.0

    dec = json.loads((ROOT / "results" / "bayes" / f"{pid}_correlated.json").read_text())["decision_report"]
    w0, w1 = dec["window_s"]
    g = dec["g_map"]
    return (norm_t(t_model), y_model / scale, norm_t(t_obs), y_obs / scale,
            (norm_t(w0), norm_t(w1)), y_obs.shape[1],
            float(g["pool_purity"]), float(g["pool_yield"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "fig_chromatograms"))
    a = ap.parse_args()

    # Authored at 11 in and set at \textwidth (about 5 in), so sizes here are 2.2x the printed ones.
    plt.rcParams.update({"font.size": 12.5, "axes.labelsize": 12.5, "xtick.labelsize": 12.5,
                         "ytick.labelsize": 12.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), sharey=True)
    for ax, (pid, label) in zip(axes, PRODUCTS):
        tm, ym, to, yo, (w0, w1), k, pur, yld = load_panel(pid)
        ax.axvspan(w0, w1, color="0.85", zorder=0)
        seen = set()
        for j in range(k):
            fam = _family(j, k)
            c = COLOUR[fam]
            ax.plot(tm, ym[:, j], color=c, lw=1.3,
                    label=fam if fam not in seen else None, zorder=2)
            ax.scatter(to, yo[:, j], color=c, s=11, zorder=3, edgecolor="white", linewidth=0.3)
            seen.add(fam)
        ax.set_title(f"{label}   pooled purity {pur:.2f}, yield {yld:.2f}", fontsize=12)
        ax.set_xlabel("fraction of salt gradient elapsed")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("concentration\n(main peak = 1)")
    axes[0].legend(fontsize=10.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200)
    print(f"wrote {a.out}.pdf and .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
