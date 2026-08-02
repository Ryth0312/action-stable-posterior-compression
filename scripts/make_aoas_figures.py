"""Regenerate the two AOAS article figures from committed result JSON (no solver, no torch).

fig_r2_triage    <- results/bayes/r2_paired_coupling_*.json      (step4/step4b)
fig_calibration  <- results/bayes/r6_fullpipeline_calibration_{iso,aniso}.json  (step6)

Usage:  python scripts/make_aoas_figures.py [--out-dir docs]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Authored at the imsart text-block width so \includegraphics[width=\textwidth] does not shrink the type.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "figure.titlesize": 9})
TEXT_IN = 5.0

LABEL = {"HLXSYN": "mAb A", "HLXSYN": "mAb B", "HLXSYN": "mAb C"}
P_HI, P_LO = 0.95, 0.05
ARMS = [("conditional", "conditional (no discrepancy layer)", "tab:blue"),
        ("hier-est", "hier-est (deployed, estimated from 9 folds)", "tab:red"),
        ("oracle", "oracle (truth known)", "tab:green")]
MIN_N = 5      # reliability bins below this count are plotted by no one: the Wilson interval spans (0,1)


def _rows(path):
    """r6 artifacts are either a bare row list (pre-metadata runs) or {meta, rows}."""
    d = json.loads(Path(path).read_text())
    return d["rows"] if isinstance(d, dict) else d

def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_r2(src: Path, out: Path):
    rows = json.loads(src.read_text())[::-1]        # top-to-bottom mAb A, B, C, matching the tables
    fig, ax = plt.subplots(figsize=(TEXT_IN, 2.1))
    for i, r in enumerate(rows):
        lo, hi = r["certified_interval"]
        p = r["P_lin_untruncated"]
        col = "tab:green" if r["classification_safe"] else "tab:red"
        # mAb B's interval is 0.7% of the axis; a bar that thin disappears entirely under a 5pt dot, and
        # the bar colour is what carries the verdict. Taller bar, smaller dot, so the colour stays visible.
        ax.barh(i, hi - lo, left=lo, height=0.34, color=col, alpha=0.85,
                edgecolor=col, linewidth=0.9)
        ax.plot([p], [i], "o", color="black", ms=3.2, mec="white", mew=0.6, zorder=3)
        ax.annotate(f"b={r['certified_bound']:.3f}", (hi, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=col)
    for thr in (P_LO, P_HI):
        ax.axvline(thr, ls=":", color="0.35", lw=1)
        ax.annotate(f"{thr:g}", (thr, len(rows) - 0.52), ha="center", fontsize=7, color="0.45")
    ax.set_yticks(range(len(rows)), [LABEL[r["product"]] for r in rows])
    ax.set_xlim(-0.06, 1.22)
    ax.set_ylim(-0.55, len(rows) - 0.3)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel(r"$P(\mathrm{meet})$ under the conditional law")
    # No in-figure title: IMS style carries explanatory text in the caption, and the caption already states
    # the green/red rule. Repeating it inside the graphic only forces the type smaller.
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def fig_cal(src_iso: Path, src_aniso: Path, out: Path, product: str = "HLXSYN"):
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.9), sharey=True)
    for ax, src, title in ((axes[0], src_iso, "isotropic discrepancy"),
                           (axes[1], src_aniso, "anisotropic (6:1, rotated)")):
        rows = {r["arm"]: r for r in _rows(src) if r["product"] == product}
        ax.plot([0, 1], [0, 1], ls="--", color="0.7", lw=1)
        ax.axvline(P_HI, ls=":", color="0.4", lw=1)
        for arm, lab, col in ARMS:
            rel = [b for b in rows[arm]["reliability"] if b["n"] >= MIN_N]
            x = [b["p_mean"] for b in rel]
            y = [b["y_freq"] for b in rel]
            err = np.array([_wilson(round(b["y_freq"] * b["n"]), b["n"]) for b in rel]).T
            ax.errorbar(x, y, yerr=np.abs(err - np.array(y)), fmt="o-", ms=3, lw=1.0,
                        capsize=1.5, color=col, label=lab)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(r"predicted $\hat p$")
        ax.set_title(title)
    axes[0].set_ylabel("realised frequency")
    # Same as fig_r2: the caption already gives the product, the replicate count and the interval type, so a
    # suptitle would only duplicate it. The count is still asserted here against the artifact.
    n = next(r["n"] for r in _rows(src_iso) if r["product"] == product)
    assert n == 300, f"caption says 300 replicates; artifact says {n}"
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=200)
    plt.close(fig)


# Run-supported loading range per product, from that product's own historical runs, and the adequacy cap.
# The DEPLOYMENT domain is the intersection: deployment is only authorised where the product has runs AND
# hold-out adequacy has been demonstrated, which fails above LOAD_CAP. Shading the raw run range instead would
# paint mAb A's 35-45 g/L as deployable, which is exactly the region the hold-out check fails.
RUN_RANGE = {"HLXSYN": (25.0, 45.0), "HLXSYN": (8.77, 26.31), "HLXSYN": (25.0, 40.0)}
LOAD_CAP = 35.0
P_DEC = 0.95
# Safe-experiment loading domain, from diffpeak.design.safe_bounds: the observed range widened by
# max(half its span, a quarter of the global limits) and clamped to those limits. This is the box the Sobol
# candidate pool is actually drawn from, so shading it -- rather than the pool's own extent plus a pad -- is
# what makes the grey band a stated domain instead of an artefact of where the points happened to land.
_LOAD_LIMITS = (5.0, 50.0)


def _safe_range(lo, hi):
    half = max(0.5 * (hi - lo), 0.25 * (_LOAD_LIMITS[1] - _LOAD_LIMITS[0]))
    return max(_LOAD_LIMITS[0], lo - half), min(_LOAD_LIMITS[1], hi + half)


def fig_window(res: Path, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.8), sharey=True)
    # Nonlinear empirical-convolution reads, from the _capB runs: those integrate the predictive layer
    # over the same hierarchy draws as the screening scan, so the plotted gap is a decision-law
    # difference and nothing else. The pre-capB files used the primary draw set and are not comparable
    # with the scan -- on mAb A that draw-set effect reaches 0.017, as large as the law effect itself.
    # mAb A's pool was scanned in full; on mAb C the solver is far more expensive, so only a screened
    # subset was run and the below-range candidate comes from the 2-op file. Both mAb C files are
    # needed, or the one candidate that clears the screening threshold appears without its counterpart.
    emp = {}
    for prod in ("HLXSYN", "HLXSYN"):
        f = res / f"empirical_convolution_window_{prod}_capB.json"
        if f.exists():
            emp[prod] = [(r["loading"], r["phat_meet"]) for r in json.loads(f.read_text())["rows"]]
    f2 = res / "empirical_convolution_HLXSYN_capB.json"
    if f2.exists():
        have = {round(x, 2) for x, _ in emp.get("HLXSYN", [])}
        for o in json.loads(f2.read_text())["ops"].values():
            if round(o["op"][0], 2) not in have:
                emp.setdefault("HLXSYN", []).append((o["op"][0], o["p_meet_empirical"]))
    for ax, prod in zip(axes, ["HLXSYN", "HLXSYN", "HLXSYN"]):
        d = json.loads((res / f"{prod}_decision_window_predictive_hier.json").read_text())
        rows = sorted(d["rows"], key=lambda r: r["loading"])
        lo, hi = RUN_RANGE[prod]
        dep_hi = min(hi, LOAD_CAP)
        safe_lo, safe_hi = _safe_range(lo, hi)
        xlo, xhi = safe_lo - 1.5, safe_hi + 1.5

        ax.axvspan(safe_lo, safe_hi, color="0.93", zorder=0)               # safe-experiment domain
        ax.axvspan(lo, dep_hi, color="#cfe3f5", zorder=0)                  # deployment domain
        if hi > LOAD_CAP:                                                  # run-supported but not adequate
            ax.axvspan(LOAD_CAP, hi, color="#f3ded0", zorder=0)
            ax.axvline(LOAD_CAP, color="0.45", lw=0.8, ls="-", zorder=1)
        ax.axhline(P_DEC, ls=":", color="0.30", lw=1, zorder=1)
        if prod == "HLXSYN":
            ax.annotate("0.95", (xlo + 0.6, P_DEC), xytext=(0, 2), textcoords="offset points",
                        fontsize=6.5, color="0.30")

        dep = [r for r in rows if lo <= r["loading"] <= dep_hi]
        oth = [r for r in rows if not (lo <= r["loading"] <= dep_hi)]
        ax.plot([r["loading"] for r in oth], [r["p_meet"] for r in oth], "o", ms=3,
                mfc="white", mec="0.45", mew=0.7, zorder=2)
        ax.plot([r["loading"] for r in dep], [r["p_meet"] for r in dep], "o", ms=3.5,
                color="tab:blue", zorder=3)
        ax.plot([d["decision_op"][0]], [d["historical"]["p_meet"]], "s", ms=5,
                color="black", zorder=5)
        if dep:
            b = max(dep, key=lambda r: r["p_meet"])
            ax.plot([b["loading"]], [b["p_meet"]], "*", ms=11, color="tab:green",
                    mec="0.2", mew=0.4, zorder=6)
        bs = max(rows, key=lambda r: r["p_meet"])
        if not dep or bs["loading"] != max(dep, key=lambda r: r["p_meet"])["loading"]:
            ax.plot([bs["loading"]], [bs["p_meet"]], "^", ms=6, color="tab:orange",
                    mec="0.2", mew=0.4, zorder=6)
        by_load = {round(r["loading"], 2): r["p_meet"] for r in rows}
        for x, y in emp.get(prod, []):
            scr = by_load.get(round(x, 2))
            if scr is not None:                      # tie each nonlinear read to its screening value
                ax.plot([x, x], [min(scr, y), max(scr, y)], "-", color="tab:red", lw=0.7,
                        alpha=0.55, zorder=1.5)
            # Under the screening markers: where the two laws agree the circle sits inside the open diamond,
            # which is the whole point on mAb A and would be hidden if the diamonds were drawn on top.
            ax.plot([x], [y], "D", ms=4.5, mfc="none", mec="tab:red", mew=1.0, zorder=1.6)
        if prod == "HLXSYN":                   # everything sits on zero; give the reader the scale
            ax.annotate(r"max $1{\times}10^{-5}$", (0.5, 0.16), xycoords="axes fraction",
                        ha="center", fontsize=6.5, color="0.25")
        ax.set_xlim(xlo, xhi)
        ax.set_title(LABEL[prod], fontsize=9)
        ax.set_xlabel("loading (g/L)")
    axes[0].set_ylim(-0.04, 1.06)
    axes[0].set_ylabel(r"$P(\mathrm{meet})$")
    # The circles carry every screening number in the deployment table, so they need legend entries of their
    # own; without them the two marks that hold all the data are the only unlabelled things on the page.
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=3.5, color="tab:blue", label="screen, in deployment domain"),
        plt.Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec="0.45", label="screen, outside it"),
        plt.Line2D([], [], ls="none", marker="D", ms=4, mfc="none", mec="tab:red", label="nonlinear read"),
        plt.Line2D([], [], ls="none", marker="s", ms=5, color="black", label="historical"),
        plt.Line2D([], [], ls="none", marker="*", ms=10, color="tab:green", label="best deployable"),
        plt.Line2D([], [], ls="none", marker="^", ms=6, color="tab:orange", label="best screen, not deployable"),
        Patch(color="#cfe3f5", label="deployment domain"),
        Patch(color="#f3ded0", label="adequacy fails"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/bayes")
    ap.add_argument("--out-dir", default="docs")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out_dir)
    fig_r2(Path(sorted(glob.glob(str(res / "r2_paired_coupling_*.json")))[0]), out / "fig_r2_triage")
    fig_cal(res / "r6_fullpipeline_calibration_iso.json",
            res / "r6_fullpipeline_calibration_aniso.json", out / "fig_calibration")
    fig_window(res, out / "fig_window")
    print(f"wrote {out}/fig_r2_triage, fig_calibration and fig_window (.pdf and .png)")


if __name__ == "__main__":
    main()
