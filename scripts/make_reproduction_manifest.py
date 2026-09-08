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

The stage numbering (S0--S13) refines the seven-stage run order in Supplement B, which gives the key.
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
    (r"^(?!.*correlated)(?!multistart/).*_posterior\.npz$", "S1", "T2", "bayes_calibrate.py",
     "--product {P} --n-steps 300 --init config --map-iters 150 --engine laplace", ""),
    # the four cross-product summaries the S1 rule used to swallow; each has its own producer
    (r"^decision_reduction_summary\.json$", "S1", "T2", "bayes_decision_reduction.py",
     "defaults", ""),
    (r"^entropy_metric_summary\.json$", "S1", "T2", "bayes_entropy_metric.py", "defaults", ""),
    (r"^mass_residual_metric_summary\.json$", "S1", "T2", "bayes_mass_residual_metric.py",
     "defaults", ""),
    (r"^sigma_ablation_summary\.json$", "S1", "T2", "bayes_sigma_ablation.py", "defaults", ""),
    (r"^LAB[0-9A-Za-z_]*_summary\.json$", "S1", "T2", "bayes_calibrate.py",
     "--product {P} --n-steps 300 --init config --map-iters 150 --engine laplace", ""),
    (r"^.*_decision\.json$", "S2", "T2", "bayes_decision.py",
     "--product {P} --n-steps 300 --map-iters 150 --mc-samples 200", ""),
    (r"^LAB[0-9A-Za-z_]*_validation\.json$", "S3", "T2", "bayes_loeo.py",
     "--product {P} --n-steps 300 --loeo --decision", "hold-out coverage, article Section 4.2"),
    (r"^multistart/.*_r[0-9]+_posterior\.npz$", "S4", "T2", "bayes_correlated_refit.py",
     "--product {P} --multistart 6 --multistart-scale 0.1 --n-steps 300 --map-iters 600 "
     "--optimizer adaptive --chunk 150 --plateau-tol 0.05 --rho-max 0.9 --seed 0", ""),
    (r"^multistart/.*_r[0-9]+_jacobians\.npz$", "S4", "T2", "bayes_basin_closure.py",
     "--stage jacobians --product {P} --restarts 0 1 2 3 4 5", ""),
    (r"^multistart/.*_r[0-9]+_gradient\.npz$", "S4", "T2", "bayes_basin_closure.py",
     "--stage gradients --product {P} --restarts 0 1 2 3 4 5", ""),
    (r"^LAB[0-9A-Za-z_]*_basin_segment.*\.json$", "S4", "T2", "bayes_basin_closure.py",
     "--stage segment --product {P} --from-restart 0 --to-restart 4",
     "Supplement B, the fitted-point closure"),
    (r"^LAB[0-9A-Za-z_]*_basin_closure\.json$", "S4", "T0", "bayes_basin_closure.py",
     "--stage analyse --product {P}",
     "article Section 4.1 and Supplement B, the fitted-point closure"),
    (r"^LAB[0-9A-Za-z_]*_basin_mixture\.json$", "S4", "T0", "bayes_basin_closure.py",
     "--stage mixture --product {P}",
     "mass-weighted mixture of fitted points; retained in the archive, not reported in the article"),
    (r"^multistart_action_stability\.json$", "S4", "T2", "bayes_multistart_action_stability.py",
     "no arguments; reads results/bayes/multistart/*_posterior.npz and the deployed scans",
     "article Table 2 and Section 4.1; the recommendation re-derived at each restart"),
    (r"^.*_correlated_multistart\.json$", "S4", "T2", "bayes_correlated_refit.py",
     "--product {P} --multistart 6 --multistart-scale 0.1 --n-steps 300 --map-iters 600 "
     "--optimizer adaptive --chunk 150 --plateau-tol 0.05 --rho-max 0.9 --seed 0",
     "multistart audit, article Section 4.1 and Supplement B"),
    (r"^.*_correlated_loeo\.json$", "S4", "T2", "bayes_correlated_refit.py",
     "--product {P} --decision-loeo --kernel ou --n-steps 300 --map-iters 600 "
     "--optimizer adaptive --chunk 150 --plateau-tol 0.05 --rho-max 0.9", ""),
    (r"^(?!decision_discrepancy).*_correlated_posterior\.npz$"
     r"|^(?!decision_discrepancy).*_correlated\.json$", "S4", "T2",
     "bayes_correlated_refit.py",
     "--product {P} --kernel ou --n-steps 300 --map-iters 600 --optimizer adaptive --chunk 150 "
     "--plateau-tol 0.05 --rho-max 0.9",
     "correlated refit; the posterior every deployment read is taken at"),
    (r"^.*_correlated_nuts.*$", "S5", "T2", "bayes_correlated_nuts.py",
     "--product HLXSYN --kernel ou --rho-max 0.9 --n-steps 300; --n-steps-lik must be repeated on "
     "the --pool-only step",
     "Hamiltonian Monte Carlo anchor; retained in the archive, not reported in the article"),
    (r"^solver_agnostic.*$", "S6", "T2", "bayes_solver_agnostic.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN --fd-rel 1e-2",
     "independent-integrator agreement, article Section 4.1"),
    (r"^timing\.json$", "S7", "T2", "bayes_timing.py",
     "--product HLXSYN --n-steps 150 300 --map-iters 120", ""),
    (r"^.*_prior_sensitivity\.json$", "S8", "T2", "bayes_prior_sensitivity.py",
     "--product {P} --n-steps 300 --map-iters 150 --scales 1.0 --shifts -0.5 0.0 0.5", ""),
    (r"^.*_sigma_ablation\.json$", "S8", "T2", "bayes_sigma_ablation.py",
     "--products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN --no-synthetic --n-steps 300 "
     "--map-iters 150 --mc", ""),
    (r"^sigma_direction_audit_corr\.json$", "S8", "T2", "bayes_sigma_direction_audit.py",
     "--products HLXSYN HLXSYN HLXSYN --posterior correlated_posterior  (the deployed capped refit; "
     "the default is the independent-residual fit and also sets the filename tag)",
     "Supplement A, the common-mode shrinkage range"),
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
     "--c-param-json results/bayes/c_param_correlated.json --label correlated-all3 "
     "(for decision_discrepancy_ou_loeo.json add --folds-json results/bayes/{product}_correlated_loeo.json "
     "--out results/bayes/decision_discrepancy_ou_loeo.json)",
     "decision-level discrepancy layer, article Section 3.1"),
    (r"^decision_discrepancy_hier.*\.json$|^hier_draws.*\.npz$", "S9", "T0",
     "bayes_decision_discrepancy_hier.py",
     "--seed 0 --dump-draws results/bayes/hier_draws_capB.npz; do NOT pass --v-mu; the default "
     "invocation IS the capped (capB) set and the _capB suffix is a filename choice, not a flag",
     "hierarchy draws every deployment read integrates over"),
    (r"^hier_audit\.txt$", "S9", "T0", "bayes_decision_discrepancy_hier_audit.py",
     "no arguments; captured stdout", ""),
    (r"^discrepancy_prior_fold_audit\.json$", "S9", "T0", "bayes_discrepancy_prior_fold_audit.py",
     "--n-iter 20000 --burn 4000 --seed 0", ""),
    (r"^hier_prior_branch_sweep\.json$", "S9", "T0", "bayes_hier_prior_branch_sweep.py",
     "defaults", "Supplement B, the hierarchy-prior branch sweep"),
    (r"^predictive_closure\.json$", "S9", "T0", "bayes_predictive_closure.py",
     "no arguments", "article Section 5.1, the decision-width values"),
    (r"^.*_decision_window_predictive_hier\.json$", "S10", "T2", "bayes_decision_window.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --n-samples 200 "
     "--seed 0 --hier-draws results/bayes/hier_draws_capB.npz --loading-cap 35.0 "
     "--sigma-meas 0.005 0.008",
     "article Figure 2 and article Table 3; the deployed scan"),
    (r"^.*_decision_window\.json$", "S10", "T2", "bayes_decision_window.py",
     "--products HLXSYN --n-steps 300 --n-samples 200 --n-candidates 24 --seed 0 "
     "--gate posterior_action  (the default gate is 'determinability' and does NOT reproduce this)",
     "article Figure 1; the chromatogram panels"),
    (r"^hier_scan/.*$", "S10", "T2", "bayes_decision_window.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --n-samples 200 "
     "--seed 0 --predictive --disc-json results/bayes/decision_discrepancy_correlated.json "
     "--out-dir results/bayes/hier_scan", ""),
    (r"^pmeet_alignment.*$", "S11", "T2", "step2_pmeet_alignment.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --seed 0 "
     "--spec 0.70 0.50 --w 1.0 1.0 --posterior correlated_posterior  (the default posterior is "
     "the independent-residual one and also sets the filename tag)",
     "Supplement A, the conditional-law compression table"),
    (r"^meet_margin_certificate\.json$", "S11", "T2", "step2d_meet_margin_certificate.py",
     "--products HLXSYN HLXSYN HLXSYN --n-steps 300 --n-candidates 24 --seed 0 "
     "--posterior correlated_posterior --spec 0.70 0.50 --n-mc 200000 --n-paired 400000 "
     "--delta 0.05 --n-samples 200 --sigma-meas 0.005 0.008 --n-tube-draws 200 "
     "--hier-draws results/bayes/hier_draws_capB.npz",
     "Supplement A Table 1; the deployed threshold certificate"),
    (r"^route_b_predictive_action\.json$", "S11", "T0", "bayes_route_b_predictive_action.py",
     "--n-mc 200000  (no solver: reads the cached decision Jacobians)",
     "article Table 4; Supplement A, the deployed-law action certificate"),
    (r"^pairwise_regret_shift\.json$", "S11", "T0", "bayes_pairwise_regret_shift.py",
     "--ladder  (no solver: reads the cached decision Jacobians and the Route B risks)",
     "article Table 4; Supplement A, the paired compression-regret certificate"),
    (r"^certificate_coverage_study\.json$", "S11", "T0", "bayes_certificate_coverage_study.py",
     "--n-rep 1000 --n-mc 100000 --n-pilot 20000  (synthetic; no solver, no product data)",
     "Supplement A, the coverage audit"),
    (r"^certificate_coverage_study_n4e5\.json$", "S11", "T0", "bayes_certificate_coverage_study.py",
     "--n-rep 250 --n-mc 400000 --n-pilot 20000 --seed 7 "
     "--out results/bayes/certificate_coverage_study_n4e5.json  (the large-sample rung of the same audit)",
     "Supplement A, the coverage audit"),
    (r"^inflation_jacobians_.*_N48\.npz$", "S10", "T2", "bayes_covariance_inflation_sensitivity.py",
     "--stage jacobians --n-steps 300 --n-candidates 48 --seed 0 --jac-tag _N48  "
     "(the nested pool; its first 24 rows reproduce the deployed cache bit for bit)",
     "Supplement A, the pool-refinement study"),
    (r"^inflation_jacobians_.*\.npz$", "S10", "T2", "bayes_covariance_inflation_sensitivity.py",
     "--stage jacobians --n-steps 300 --n-candidates 24 --seed 0",
     "cached decision Jacobians shared by the width-inflation and deployed-law action runs"),
    (r"^pool_refinement\.json$", "S11", "T0", "bayes_pool_refinement.py",
     "(defaults; no solver, reads the _N48 Jacobian cache)",
     "article Section 5.2; Supplement A, the pool-refinement study"),
    (r"^candidate_linearisation_screen\.json$", "S11", "T2", "bayes_candidate_linearisation_screen.py",
     "--jac-tag _N48 --indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 45 "
     "--n-dir 3 --z 2.0", "article Section 4.3; Supplement A, the candidate-wise linearization screen"),
    (r"^covariance_inflation_sensitivity\.json$", "S11", "T0",
     "bayes_covariance_inflation_sensitivity.py", "--stage analyse",
     "Supplement B, the covariance-scale sensitivity analysis"),
    (r"^residual_autocorrelation\.json$", "S4", "T2", "bayes_residual_autocorrelation.py",
     "(defaults; lag-1 autocorrelation of the reference fit's within-run residuals, before and after "
     "whitening by each product's fitted OU-plus-nugget covariance)",
     "article Section 4.2; Supplement B, the correlated-residual refit"),
    (r"^draw_set_comparison\.json$", "S9", "T1", "bayes_draw_set_comparison.py",
     "(defaults; recomputes the predictive width and P(meet) at each historical condition under both "
     "committed draw sets)",
     "Supplement B, the canonical hierarchy draws"),
    (r"^predictive_flip_certificate\.json$", "S12", "T1",
     "bayes_predictive_flip_certificate.py",
     "(defaults; assembles the two directional disagreement counts with the support-gap bridge and "
     "re-derives them from the committed paired draws)",
     "article Section 4.3; Supplement A, the candidate-level certificate at the deployed law"),
    (r"^paired_draws_[A-Za-z0-9_]+_op[0-9.]+\.npz$", "S12", "T2", "step4_r2_paired_coupling.py",
     "--save-paired on the pool-candidate certificate run below; the paired nonlinear and linearised "
     "whitened decision draws, retained so any predictive-layer question is post-processing",
     "article Section 4.3; Supplement A, the candidate-level certificate at the deployed law"),
    (r"^predictive_linearisation_certificate\.json$", "S12", "T1",
     "bayes_predictive_linearisation_certificate.py",
     "(defaults; reuses the stored paired tail rate, which the shared additive predictive draw leaves "
     "invariant, and re-measures only the boundary tube and the support-gap bridge)",
     "article Section 4.3; Supplement A, the candidate-level certificate at the deployed law"),
    (r"^candidate_certificate_joint_level\.json$", "S12", "T0",
     "bayes_candidate_certificate_joint_level.py",
     "(defaults; arithmetic on the stored Clopper-Pearson counts, no solver)",
     "article Section 4.3; Supplement A, the candidate-level certificate table"),
    (r"^r2_paired_coupling_[A-Za-z0-9_]+_op[0-9.]+\.json$", "S12", "T2", "step4_r2_paired_coupling.py",
     "--products <one product> --posterior correlated_posterior --n-draws 4000 --seed 0 --guard solver "
     "--op <four operating coordinates> --hier-draws results/bayes/hier_draws_capB.npz "
     "--save-paired results/bayes/paired_draws_PRODUCT_op<loading>.npz  (the pool-candidate certificate; "
     "the op tag in the filename is its loading, and the run cannot overwrite the historical-condition "
     "artifact)",
     "article Section 4.3; Supplement A, the candidate-level certificate table"),
    (r"^r2_paired_coupling.*$", "S12", "T2", "step4_r2_paired_coupling.py",
     "--products HLXSYN HLXSYN HLXSYN --n-draws 4000 --seed 0 --spec 0.70 0.50; then "
     "step4b_r2_reverdict.py --in 'results/bayes/r2_paired_coupling_*.json' --spec 0.70 0.50, "
     "which rewrites the same file in place",
     ""),
    (r"^r5b_decision_sbc.*$", "S12", "T2", "step5_decision_sbc.py",
     "--product <from filename> --layer <from filename> --engine <from filename> --n-datasets 400 --n-post 200 "
     "--scale-sweep 0.5 1.0 --disc-wdec 1.271 --floor-wdec 1.271 --prior-cov full --pool-size 6 "
     "(defaults are n-datasets 100 and disc-wdec 0.0, neither of which reproduces this)",
     ""),
    (r"^decision_compression_real_ladder\.json$", "S13", "T0", "step2b_real_ladder.py",
     "no arguments", ""),
    (r"^voi_nullity_prior_whitened_corr\.json$", "S13", "T0", "step3b_voi_prior_whitened.py",
     "--posterior correlated_posterior  (reads the decision Jacobian from the cached "
     "inflation_jacobians_{product}.npz, whose last row is the historical condition)",
     "Supplement A Table 2; the local information-direction diagnostic"),
    (r"^voi_nullity.*prior_whitened.*$", "S13", "T0", "step3b_voi_prior_whitened.py",
     "no arguments; cwd-independent", ""),
    (r"^voi_nullity.*$", "S13", "T0", "step3_voi_nullity.py",
     "no arguments; must be run from cex_model_python/", ""),
    (r"^r6_fullpipeline_calibration_iso\.json$", "S13", "T0", "step6_fullpipeline_calibration.py",
     "--n-replicates 300 --shape iso --disc-wdec 1.271 --bias-wdec 0.6 --n-post 200 "
     "--pool-size 8 --gibbs-iter 3000 --gibbs-burn 600 --gibbs-thin 6 --seed 0",
     "Supplement A Figure 1 and Supplement A Table 3"),
    (r"^r6_fullpipeline_calibration_aniso.*\.json$", "S13", "T0",
     "step6_fullpipeline_calibration.py",
     "--n-replicates 300 --shape aniso --aniso-ratio 6.0 --disc-wdec 1.271 --bias-wdec 0.6 "
     "--n-post 200 --pool-size 8 --gibbs-iter 3000 --gibbs-burn 600 --gibbs-thin 6 --seed 0 "
     "(add --fold-icc 0.5 and --out ..._aniso_icc.json for the icc variant)",
     "Supplement A Figure 1 and Supplement A Table 3"),
    (r"^synthetic_twin_truth\.json$", "S0", "T1", "make_synthetic_twin.py",
     "no arguments", "the twin's ground truth"),
]

# Files that are not article objects. Classified rather than force-mapped, so that the
# "provenance not recorded" note keeps its meaning.
CLASSIFY: list[tuple[str, str, str]] = [
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
     "--product HLXSYN --kernel ou --n-steps 300 --map-iters 150 --optimizer lbfgs --rho-max 0.9 "
     "--c-param-out results/bayes/HLXSYN_c_param_correlated.json  (without the last flag a twin "
     "refit would merge into the shared c_param_correlated.json)",
     "results/bayes/HLXSYN_correlated.json, HLXSYN_correlated_posterior.npz, HLXSYN_c_param_correlated.json"),
    ("S10", "T1", "bayes_decision_window.py",
     "--products HLXSYN --n-steps 300 --n-samples 200 --n-candidates 24 --seed 0 "
     "--gate posterior_action",
     "results/bayes/HLXSYN_decision_window.json"),
]


# Every object the article and Supplement A report. Generation fails if any of these ends up with
# no row, so the manifest cannot silently stop covering a table.
PAPER_OBJECTS = [
    "article Table 2", "article Table 3", "article Table 4",
    "article Figure 1", "article Figure 2",
    "Supplement A Table 1", "Supplement A Table 2", "Supplement A Table 3",
    "Supplement A Figure 1",
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Artifacts that reproduce in distribution rather than to the byte, which Supplement B records.
NOT_BIT_EXACT = [
    (r"^hier_draws\.npz$",
     "read rather than rebuilt: this is the primary draw set, and the flags above rebuild the deployed one"),
]


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
        note = next((n for pat, n in NOT_BIT_EXACT if re.search(pat, rel)), "")
        rows.append(dict(stage=stage, tier=tier, paper_object=obj,
                         script=f"scripts/{script}", flags=flags.replace("{P}", prod or "{P}"),
                         output=f"results/bayes/{rel}", sha256=_sha256(p), note=note))
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
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    gap = sum(1 for r in rows if r["note"] == "provenance not recorded")
    obj = sum(1 for r in rows if r["paper_object"])
    print(f"wrote {out}  ({len(rows)} rows, {obj} bound to an article object, "
          f"{gap} with no recorded provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
