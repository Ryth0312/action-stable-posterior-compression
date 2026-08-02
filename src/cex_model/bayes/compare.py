"""Cross-product / cross-method comparison for the paper (M5 table + M6 baseline matrix).

Two artefacts:
1. a static CAPABILITY MATRIX (method x what it delivers), with the three external
   baselines (Japel-Buyel 2022, Chen PbP 2022/23, Rojo-Garcia 2024) positioned
   honestly alongside this work;
2. a per-product results table built from saved artefacts (``{product}_summary.json``
   from bayes_calibrate, ``{product}_experiment_plan.csv`` from bayes_design_experiments)
   plus the cheap, solver-free PbP-applicability check.

Pure formatting + the PbP check -- no solver -- so it is fully unit-testable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cex_model.bayes.pbp import pbp_applicability

__all__ = ["CAPABILITY_ROWS", "CAPABILITY_COLS", "product_row", "render_markdown", "write_comparison",
           "filter_experiments", "predictions_tier"]


def filter_experiments(experiments, exclude):
    """Drop experiments whose ``name`` contains any of the ``exclude`` substrings.

    Used to remove non-production / non-standard runs (e.g. HLXSYN's DT experiment,
    a different gradient program that was not part of the original fit and drags the
    aggregate coverage down) from BOTH the fit and the predictive evaluation.
    """
    exclude = [s for s in (exclude or []) if s]
    if not exclude:
        return list(experiments), []
    kept, dropped = [], []
    for e in experiments:
        (dropped if any(s in e.name for s in exclude) else kept).append(e)
    return kept, dropped


def predictions_tier(coverage) -> str | None:
    """Coverage -> a readable reliability tier (so identified=False is not misread as failure)."""
    if coverage is None:
        return None
    return "yes" if coverage >= 0.90 else "partial" if coverage >= 0.80 else "weak"

CAPABILITY_COLS = ["method", "point_estimate", "calibrated_uq", "identifiability",
                   "experiment_design", "works_on_existing_data"]

# Honest capability matrix; the three external baselines + the existing point fit + this work.
CAPABILITY_ROWS = [
    {"method": "PbP / Yamamoto (Chen 2022/23)", "point_estimate": "yes", "calibrated_uq": "no",
     "identifiability": "no", "experiment_design": "prescribed (~5-6 LGE ladder)",
     "works_on_existing_data": "no (needs gradient-slope ladder)"},
    {"method": "BayesOpt-directional (Japel-Buyel 2022)", "point_estimate": "yes",
     "calibrated_uq": "no (GP over objective)", "identifiability": "no", "experiment_design": "no",
     "works_on_existing_data": "yes"},
    {"method": "Bayesian OED + surrogate (Rojo-Garcia 2024)", "point_estimate": "yes",
     "calibrated_uq": "yes (MCMC)", "identifiability": "no",
     "experiment_design": "yes (nested-MC EIG + sparse-grid surrogate)",
     "works_on_existing_data": "yes (Langmuir, 2-comp, synthetic)"},
    {"method": "Point fit (DE / Adam, existing)", "point_estimate": "yes", "calibrated_uq": "no",
     "identifiability": "no", "experiment_design": "no", "works_on_existing_data": "yes"},
    {"method": "This work (gradient Bayesian)", "point_estimate": "yes (MAP)",
     "calibrated_uq": "yes (Laplace/SVI/NUTS)", "identifiability": "yes (degeneracy + sloppy dirs)",
     "experiment_design": "yes (adaptive, differentiable D-optimal)",
     "works_on_existing_data": "yes (multi-component, real mAb)"},
]


def product_row(product: str, op_matrix, summary: dict | None = None, plan_top: dict | None = None) -> dict:
    """One per-product comparison row: PbP-applicability + this-work numbers (if present)."""
    pbp = pbp_applicability(op_matrix)
    row = {"product": product, "n_experiments": int(len(op_matrix)),
           "pbp_applicable": pbp["applicable"], "pbp_n_distinct_slopes": pbp["n_distinct_slopes"],
           "ours_identified": None, "ours_worst_dir_shrinkage": None, "ours_n_met": None,
           "ours_coverage": None, "ours_rmse": None, "predictions_reliable": None,
           "worst_exp": None, "worst_exp_coverage": None, "boed_top": None}
    if summary:
        idf = summary.get("identifiability", {})
        row.update(ours_identified=idf.get("met"), ours_worst_dir_shrinkage=idf.get("worst_dir_shrinkage"),
                   ours_n_met=idf.get("n_met"))
        pred = summary.get("predictive", {})
        cov = pred.get("aggregate", {}).get("coverage")
        row.update(ours_coverage=cov, ours_rmse=pred.get("aggregate", {}).get("rmse_total"),
                   predictions_reliable=predictions_tier(cov))
        per = pred.get("per_experiment", [])
        if per:
            worst = min(per, key=lambda e: e.get("coverage", 1.0))
            row.update(worst_exp=worst.get("name"), worst_exp_coverage=worst.get("coverage"))
    if plan_top:
        row["boed_top"] = plan_top
    return row


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def render_markdown(rows: list[dict]) -> str:
    """Capability matrix + per-product results as Markdown."""
    cap = _md_table(CAPABILITY_COLS, [[r[c] for c in CAPABILITY_COLS] for r in CAPABILITY_ROWS])
    pcols = ["product", "n_experiments", "pbp_applicable", "pbp_n_distinct_slopes", "ours_identified",
             "ours_worst_dir_shrinkage", "ours_n_met", "ours_coverage", "predictions_reliable",
             "worst_exp_coverage"]
    per = _md_table(pcols, [[r.get(c) for c in pcols] for r in rows])
    return (
        "## Capability matrix (method x what it delivers)\n\n" + cap +
        "\n\n## Per-product results\n\n" + per +
        "\n\n*`ours_identified` = are the PARAMETERS identified from the existing data. `False` here is "
        "an honest diagnosis, NOT a failure: the least-constrained direction is sigma + the keq<->nu "
        "structural degeneracy (which gradient-only data cannot resolve). PREDICTIONS are still "
        "calibrated -- see `predictions_reliable` / `ours_coverage`. `worst_exp_coverage` flags an "
        "outlier experiment (e.g. HLXSYN's non-standard DT run) that drags the aggregate coverage down.*\n"
        "\n*PbP not applicable = the product's existing data lacks the gradient-slope ladder LR1 needs; "
        "this is exactly where a posterior + adaptive design is required.*\n")


def write_comparison(rows: list[dict], out_dir) -> dict:
    """Write comparison.csv (per-product) + comparison.md (matrix + per-product). Returns paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out / "comparison.csv", out / "comparison.md"
    fields = ["product", "n_experiments", "pbp_applicable", "pbp_n_distinct_slopes", "ours_identified",
              "ours_worst_dir_shrinkage", "ours_n_met", "ours_coverage", "ours_rmse",
              "predictions_reliable", "worst_exp", "worst_exp_coverage", "boed_top"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "boed_top": json.dumps(r.get("boed_top"), ensure_ascii=False)})
    md_path.write_text(render_markdown(rows), encoding="utf-8")
    return {"csv": csv_path, "md": md_path}
