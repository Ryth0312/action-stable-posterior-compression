#!/usr/bin/env python3
"""Write ``MANIFEST.csv``: the command, flags and checksum behind every committed artifact.

Two targets, because the public archive may not name a product:

``--target referee``  one row per file in ``results/bayes``, with the producing command, the flags
                      whose defaults would produce a *different* file, the paper object it feeds,
                      and a sha256 of the shipped bytes. Files with no recorded producer are
                      emitted with an empty ``script`` and a "provenance not recorded" note rather than
                      omitted,
                      so the gaps in the record are visible instead of invisible.

``--target public``   the synthetic-twin pipeline only. No product identifiers, no checksums of
                      product artifacts, because none ship.

The stage numbering (S1--S13) matches the run order in Supplement B.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIELDS = ["stage", "tier", "paper_object", "script", "flags", "output", "sha256", "note"]

# (output glob, stage, tier, script, flags-that-must-be-passed, paper object)
# ``flags`` lists only what a default invocation would get WRONG, plus the arguments that select
# the product or the output path. A blank paper_object means the artifact is an input to a later
# stage rather than something the article reads directly.
RULES: list[tuple[str, str, str, str, str, str]] = [
    (r"^(?!.*correlated).*_posterior\.npz$", "S1", "T2", "bayes_calibrate.py",
     "--product {P} --n-steps 300 --init config --map-iters 150 --engine laplace", ""),
    (r"^.*_summary\.json$", "S1", "T2", "bayes_calibrate.py",
     "--product {P} --n-steps 300 --init config --map-iters 150 --engine laplace", ""),
    (r"^.*_decision\.json$", "S2", "T2", "bayes_decision.py",
     "--product {P} --n-steps 300 --map-iters 150 --mc-samples 200", ""),
    (r"^.*_validation\.json$", "S3", "T2", "bayes_loeo.py",
     "--product {P} --n-steps 300 --loeo --decision", "hold-out coverage, article Section 5.1"),
    (r"^.*_correlated_multistart\.json$", "S4", "T2", "bayes_correlated_refit.py",
     "--product HLXSYN --multistart 6 --multistart-scale 0.1 --n-steps 300 --map-iters 120 "
     "--optimizer lbfgs --rho-max 0.9",
     "multistart audit, article Section 5.1 and Supplement B"),
    (r"^.*_correlated_loeo\.json$", "S4", "T2", "bayes_correlated_refit.py",
     "--product {P} --decision-loeo --kernel ou --n-steps 300 --map-iters 300 --optimizer lbfgs "
     "[--rho-max 0.9 for HLXSYN and HLXSYN, omitted for HLXSYN]", ""),
    (r"^.*_correlated_posterior\.npz$|^.*_correlated\.json$", "S4", "T2",
     "bayes_correlated_refit.py",
     "--product {P} --kernel ou --n-steps 300 --map-iters 120 --optimizer lbfgs "
     "[--reparam --reparam-polish 6 --reparam-damping 0.7 for HLXSYN; "
     "--rho-max 0.9 for HLXSYN and HLXSYN]",
     "correlated refit; the posterior every deployment read is taken at"),
    (r"^.*_correlated_nuts.*$", "S5", "T2", "bayes_correlated_nuts.py",
     "--product HLXSYN --kernel ou --rho-max 0.9 --n-steps 300; --n-steps-lik must be repeated on "
     "the --pool-only step",
     "Hamiltonian Monte Carlo anchor, article Section 5.1"),
    (r"^solver_agnostic.*$", "S6", "T2", "bayes_solver_agnostic.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN --fd-rel 1e-2",
     "independent-integrator agreement, article Section 5.1"),
    (r"^timing\.json$", "S7", "T2", "bayes_timing.py",
     "--product HLXSYN --n-steps 150 300 --map-iters 120", ""),
    (r"^.*_prior_sensitivity\.json$", "S8", "T2", "bayes_prior_sensitivity.py",
     "--product {P} --n-steps 300 --map-iters 150 --scales 1.0 --shifts -0.5 0.0 0.5", ""),
    (r"^.*_sigma_ablation\.json$", "S8", "T2", "bayes_sigma_ablation.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN --no-synthetic --n-steps 300 "
     "--map-iters 150 --mc", ""),
    (r"^.*_spectral\.json$|^sigma_direction.*$", "S8", "T2", "bayes_sigma_direction_audit.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN", ""),
    (r"^restricted_sigma_gain.*$", "S8", "T2", "bayes_restricted_sigma_gain.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN SYN2 --no-synthetic --n-steps 120 "
     "--stride 1", ""),
    (r"^restricted_fisher.*$", "S8", "T2", "bayes_restricted_fisher_certificate.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN SYN2 --no-synthetic --n-steps 120 "
     "--stride 1", ""),
    (r"^decision_gain_certificate\.json$", "S8", "T2", "bayes_decision_gain.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN --n-steps 100 120", ""),
    (r"^c_param_correlated\.json$|^decision_discrepancy(_correlated|_ou_loeo)?\.json$", "S9", "T0",
     "bayes_decision_discrepancy.py",
     "--c-param-json results/bayes/c_param_correlated.json --label correlated-all3",
     "decision-level discrepancy layer, article Section 5.1"),
    (r"^decision_discrepancy_hier.*\.json$|^hier_draws.*\.npz$", "S9", "T0",
     "bayes_decision_discrepancy_hier.py",
     "--seed 0 --dump-draws results/bayes/hier_draws_capB.npz; do NOT pass --v-mu; the default "
     "invocation IS the capped (capB) set and the _capB suffix is a filename choice, not a flag",
     "hierarchy draws every deployment read integrates over"),
    (r"^hier_audit\.txt$", "S9", "T0", "bayes_decision_discrepancy_hier_audit.py",
     "no arguments; captured stdout", ""),
    (r"^discrepancy_prior_fold_audit\.json$", "S9", "T0", "bayes_discrepancy_prior_fold_audit.py",
     "--n-iter 20000 --burn 4000 --seed 0", ""),
    (r"^predictive_closure\.json$", "S9", "T0", "bayes_predictive_closure.py",
     "no arguments", "article Table 5, the wdec columns"),
    (r"^.*_decision_window_predictive_hier\.json$", "S10", "T2", "bayes_decision_window.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --n-samples 200 "
     "--seed 0 --hier-draws results/bayes/hier_draws_capB.npz --loading-cap 35.0 "
     "--sigma-meas 0.005 0.008",
     "article Figure 3 and Table 5; the deployed scan"),
    (r"^.*_decision_window\.json$", "S10", "T2", "bayes_decision_window.py",
     "--products HLXSYN --n-steps 300 --n-samples 200 --n-candidates 24 --seed 0 "
     "--gate posterior_action  (the default gate is 'determinability' and does NOT reproduce this)",
     "experiment-level value, article Table 1"),
    (r"^hier_scan/.*$", "S10", "T2", "bayes_decision_window.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --n-samples 200 "
     "--seed 0 --predictive --disc-json results/bayes/decision_discrepancy_correlated.json "
     "--out-dir results/bayes/hier_scan  (the out-dir is read back by "
     "bayes_empirical_convolution.py)", ""),
    (r"^pmeet_alignment.*$", "S11", "T2", "step2_pmeet_alignment.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --seed 0 "
     "--spec 0.70 0.50 --w 1.0 1.0 --posterior correlated_posterior  (the default posterior is "
     "the independent-residual one and also sets the filename tag)",
     "article Table 3"),
    (r"^meet_margin_certificate\.json$", "S11", "T2", "step2d_meet_margin_certificate.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --seed 0 "
     "--posterior correlated_posterior --spec 0.70 0.50 --n-mc 200000 --n-paired 400000 "
     "--delta 0.05 --n-samples 200 --sigma-meas 0.005 0.008 --n-tube-draws 200 "
     "--hier-draws results/bayes/hier_draws_capB.npz",
     "article Table 4"),
    (r"^empirical_convolution_window_.*$", "S11", "T2", "bayes_empirical_convolution.py",
     "--product {P} --empirical-window --n-candidates 24 --n-steps 300 --n-inner 40000 "
     "--load-cap 35.0 --seed 0 (--n-theta 1000 for HLXSYN, 500 for HLXSYN with "
     "--only-loadings 28.07 29.77)",
     "article Figure 3 and Table 5, the nonlinear reads"),
    (r"^empirical_convolution_(?!window).*$", "S11", "T2", "bayes_empirical_convolution.py",
     "--product {P} --n-steps 300 --n-theta 2000 --n-inner 40000 --seed 0", ""),
    (r"^nonlinear_meet_certificate.*$", "S11", "T2", "step2e_nonlinear_meet_certificate.py",
     "--product HLXSYN --n-steps 300 --n-candidates 24 --seed 0 --spec 0.70 0.50 "
     "--sigma-meas 0.005 0.008 --delta 0.05 --posterior correlated_posterior "
     "--hier-draws results/bayes/hier_draws_capB.npz --laws predictive --domain 25.0 35.0 "
     "--extra-loadings 19.65",
     "Supplement A Table 1; article Section 5.2, the nonlinear closure"),
    (r"^r2_paired_coupling.*$", "S12", "T2", "step4_r2_paired_coupling.py",
     "--products HLXSYN HLXSYN HLXSYN --n-draws 4000 --seed 0 --spec 0.70 0.50; then "
     "step4b_r2_reverdict.py --in 'results/bayes/r2_paired_coupling_*.json' --spec 0.70 0.50, "
     "which rewrites the same file in place",
     "article Figure 1; Supplement A Table 3"),
    (r"^r5b_decision_sbc.*$", "S12", "T2", "step5_decision_sbc.py",
     "--product {P} --layer ou --engine frozen --n-datasets 400 --n-post 200 "
     "--scale-sweep 0.5 1.0 --disc-wdec 1.271 --floor-wdec 1.271 --prior-cov full --pool-size 6 "
     "(defaults are n-datasets 100 and disc-wdec 0.0, neither of which reproduces this)",
     "Supplement A Table 4"),
    (r"^decision_compression_real_ladder\.json$", "S13", "T0", "step2b_real_ladder.py",
     "no arguments", ""),
    (r"^voi_nullity.*prior_whitened.*$", "S13", "T0", "step3b_voi_prior_whitened.py",
     "no arguments; cwd-independent", "article Table 1; Supplement A Table 2"),
    (r"^voi_nullity.*$", "S13", "T0", "step3_voi_nullity.py",
     "no arguments; must be run from cex_model_python/", ""),
    (r"^r6_fullpipeline_calibration_iso\.json$", "S13", "T0", "step6_fullpipeline_calibration.py",
     "--n-replicates 300 --shape iso --disc-wdec 1.271 --bias-wdec 0.6 --n-post 200 "
     "--pool-size 8 --gibbs-iter 3000 --gibbs-burn 600 --gibbs-thin 6 --seed 0",
     "article Figure 2; Supplement A Tables 5 and 6"),
    (r"^r6_fullpipeline_calibration_aniso.*\.json$", "S13", "T0",
     "step6_fullpipeline_calibration.py",
     "--n-replicates 300 --shape aniso --aniso-ratio 6.0 --disc-wdec 1.271 --bias-wdec 0.6 "
     "--n-post 200 --pool-size 8 --gibbs-iter 3000 --gibbs-burn 600 --gibbs-thin 6 --seed 0 "
     "(add --fold-icc 0.5 and --out ..._aniso_icc.json for the icc variant)",
     "article Figure 2; Supplement A Tables 5 and 6"),
    (r"^synthetic_twin_truth\.json$", "S0", "T1", "make_synthetic_twin.py",
     "no arguments", "the twin's ground truth"),
]

# Files that are not article objects. Classified rather than force-mapped, so that the
# "provenance not recorded" note keeps its meaning.
CLASSIFY: list[tuple[str, str, str]] = [
    (r"(^|/)gnl_.*\.npz$", "scripts/step2e_nonlinear_meet_certificate.py",
     "solver-pushforward cache written under --cache-dir; deleting it costs solver time, not "
     "correctness"),
    (r"(^|/)gs_.*\.npz$", "scripts/bayes_empirical_convolution.py",
     "solver-pushforward cache written under --gs-cache; deleting it costs solver time, not "
     "correctness"),
    (r"\.png$", "", "diagnostic figure written by the producing script; not an article object"),
    (r"_experiment_plan\.csv$|_real_design\.json$|_design\.json$|active_design",
     "", "design helper; not an article object"),
    (r"^SYN[0-9]*_|^CMC_class_level\.json$",
     "", "synthetic probe used while developing the model class; not an article object"),
]


# The synthetic-twin path, which is what the public archive can run end to end.
TWIN_STAGES: list[tuple[str, str, str, str, str]] = [
    ("S0", "T1", "make_synthetic_twin.py", "no arguments",
     "configs/*_hlxsyn.yaml, data/synthetic_twin/*.csv, results/bayes/synthetic_twin_truth.json"),
    ("S1", "T1", "bayes_calibrate.py",
     "--product HLXSYN --n-steps 300 --init config --map-iters 150 --engine laplace --predict "
     "--predict-samples 100 --out-dir results/bayes",
     "results/bayes/HLXSYN_posterior.npz, HLXSYN_summary.json"),
    ("S2", "T1", "bayes_decision.py",
     "--product HLXSYN --n-steps 300 --map-iters 150 --mc-samples 200",
     "results/bayes/HLXSYN_decision.json"),
    ("S3", "T1", "bayes_loeo.py",
     "--product HLXSYN --n-steps 300 --loeo --decision",
     "results/bayes/HLXSYN_validation.json"),
    ("S4", "T1", "bayes_correlated_refit.py",
     "--product HLXSYN --kernel ou --n-steps 300 --map-iters 120 --optimizer lbfgs --rho-max 0.9",
     "results/bayes/HLXSYN_correlated.json, HLXSYN_correlated_posterior.npz"),
    ("S10", "T1", "bayes_decision_window.py",
     "--products HLXSYN --n-steps 300 --n-samples 200 --n-candidates 24 --seed 0 "
     "--gate posterior_action",
     "results/bayes/HLXSYN_decision_window.json"),
]


# Every object the article and Supplement A report. Generation fails if any of these ends up with
# no row, so the manifest cannot silently stop covering a table.
PAPER_OBJECTS = [
    "article Table 1", "article Table 3", "article Table 4", "article Table 5",
    "article Figure 1", "article Figure 2", "article Figure 3",
    "Supplement A Table 1", "Supplement A Table 2", "Supplement A Table 3",
    "Supplement A Table 4", "Supplement A Tables 5 and 6",
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _product_of(name: str) -> str:
    m = re.match(r"^(LAB[0-9A-Za-z]+?)(?:_exclDT)?_", name)
    if not m:
        return ""
    return name[:name.index("_", len(m.group(1)) - 1)] if "_exclDT_" in name else m.group(1)


def _match(rel: str):
    for pat, stage, tier, script, flags, obj in RULES:
        if re.search(pat, rel):
            return stage, tier, script, flags, obj
    return None


def referee_rows(res: Path) -> list[dict]:
    rows = []
    files = [p for p in sorted(res.rglob("*")) if p.is_file()]
    for p in files:
        rel = str(p.relative_to(res))
        hit = _match(rel)
        if hit is None:
            script, note = "", "provenance not recorded"
            for pat, sc, nt in CLASSIFY:
                if re.search(pat, rel):
                    script, note = sc, nt
                    break
            rows.append(dict(stage="", tier="", paper_object="", script=script, flags="",
                             output=f"results/bayes/{rel}", sha256=_sha256(p), note=note))
            continue
        stage, tier, script, flags, obj = hit
        prod = _product_of(Path(rel).name)
        if "_exclDT" in rel:
            prod = "HLXSYN"
        rows.append(dict(stage=stage, tier=tier, paper_object=obj,
                         script=f"scripts/{script}", flags=flags.replace("{P}", prod or "{P}"),
                         output=f"results/bayes/{rel}", sha256=_sha256(p), note=""))
    rows.sort(key=lambda r: (r["stage"] or "ZZ", r["output"]))
    return rows


def public_rows() -> list[dict]:
    return [dict(stage=s, tier=t, paper_object="synthetic twin (no article number)",
                 script=f"scripts/{sc}", flags=fl, output=out, sha256="", note="run by `make synthetic`")
            for s, t, sc, fl, out in TWIN_STAGES]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("public", "referee"), required=True)
    ap.add_argument("--results", default=str(ROOT / "results" / "bayes"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.target == "referee":
        res = Path(a.results)
        if not res.is_dir():
            print(f"no such directory: {res}", file=sys.stderr)
            return 2
        rows = referee_rows(res)
    else:
        rows = public_rows()

    if a.target == "referee":
        covered = " | ".join(r["paper_object"] for r in rows)
        missing = [o for o in PAPER_OBJECTS if o not in covered]
        if missing:
            print("manifest does not cover: " + "; ".join(missing), file=sys.stderr)
            return 3

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    gap = sum(1 for r in rows if r["note"] == "provenance not recorded")
    obj = sum(1 for r in rows if r["paper_object"])
    print(f"wrote {out}  ({len(rows)} rows, {obj} bound to an article object, "
          f"{gap} with no recorded provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
