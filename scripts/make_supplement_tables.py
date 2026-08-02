"""Emit the Supplement A result tables as LaTeX, straight from the committed JSON.

Three tables, so that a reviewer never has to open a result file to audit a number:
  s:tab-r2   the joint-region linearisation certificate, per product
  s:tab-r5b  the local decision-calibration study, per product x law
  s:tab-r6a  the hierarchy-in-the-loop study, rank calibration
  s:tab-r6b  the hierarchy-in-the-loop study, probability scores

Usage:  python scripts/make_supplement_tables.py > docs/_supplement_tables.tex
"""
from __future__ import annotations

import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1] / "results/bayes"
LABEL = {"HLXSYN": r"\pA{}", "HLXSYN": r"\pB{}", "HLXSYN": r"\pC{}"}
ARMS = ["conditional", "hier-est", "oracle", "floor-0", "floor-0.5", "floor-1", "floor-1.5", "floor-2"]
ARM_TEX = {"conditional": "conditional", "hier-est": "hier-est", "oracle": "oracle",
           "floor-0": "floor $\\kappa{=}0$", "floor-0.5": "floor $\\kappa{=}\\tfrac12$",
           "floor-1": "floor $\\kappa{=}1$", "floor-1.5": "floor $\\kappa{=}\\tfrac32$",
           "floor-2": "floor $\\kappa{=}2$"}


def _rows(path):
    """r6 artifacts are either a bare row list (pre-metadata runs) or {meta, rows}."""
    d = json.loads(pathlib.Path(path).read_text())
    return d["rows"] if isinstance(d, dict) else d

def _p(x):
    """p-value: fixed decimals down to 0.001, then scientific, so tiny values stay legible."""
    if x != x:
        return "---"
    if x >= 1e-3:
        return f"{x:.3f}"
    if x == 0.0:
        return r"$<10^{-300}$"
    m, e = f"{x:.0e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def _f(x, d=3):
    return "---" if x != x else f"{x:.{d}f}"


def _ci(c, d=2):
    return "---" if not c or c[0] != c[0] else f"[{c[0]:.{d}f},\\,{c[1]:.{d}f}]"


def table_r2():
    rows = json.loads(next(R.glob("r2_paired_coupling_*.json")).read_text())
    pct = "\\%"
    lines = [
        (r"paired posterior draws $N$", lambda r: f"${r['n_draws']}$"),
        (r"nonlinear $P(\text{meet})$", lambda r: f"${r['P_X_meet_nonlinear']:.4f}$"),
        (r"linearised $P(\text{meet})$, clipped law, $N$ draws", lambda r: f"${r['P_Y_meet_linear']:.4f}$"),
        (r"\quad same, $4\times10^{6}$ draws (bridge input)", lambda r: f"${r['P_lin_clipped']:.4f}$"),
        (r"linearised $P(\text{meet})$, untruncated", lambda r: f"${r['P_lin_untruncated']:.4f}$"),
        (r"observed gap", lambda r: f"${r['observed_gap']:.4f}$"),
        (r"draws clipped to the physical box", lambda r: f"${100 * r['clip_fraction_tube']:.0f}{pct}$"),
        (r"grid point $t^\ast$", lambda r: f"${r['t_star']:.2f}$"),
        (r"boundary tube $\hat t_1$", lambda r: f"${r['term1_tube_hat']:.5f}$"),
        (r"\quad Clopper--Pearson upper $t_1^{+}$", lambda r: f"${r['term1_tube_upper']:.5f}$"),
        (r"paired tail $\hat t_2$", lambda r: f"${r['term2_tail_hat']:.5f}$"),
        (r"\quad Clopper--Pearson upper $t_2^{+}$", lambda r: f"${r['term2_tail_upper']:.5f}$"),
        (r"support-gap bridge", lambda r: f"${r['support_gap_upper']:.4f}$"),
        (r"\textbf{certified bound} $b$", lambda r: "$\\mathbf{%.4f}$" % r["certified_bound"]),
        (r"large-sample floor $\hat t_1+\hat t_2$", lambda r: f"${r['large_sample_floor']:.4f}$"),
        (r"finite-sample share of $b$", lambda r: f"${100 * r['finite_sample_share']:.0f}{pct}$"),
        (r"\textbf{certified interval}",
         lambda r: "$[%.3f,%.3f]$" % tuple(r["certified_interval"])),
        (r"classification preserved",
         lambda r: "yes" if r["classification_safe"] else "\\textbf{no}"),
    ]
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{The joint-region linearisation certificate, every reported quantity, transposed so the "
           r"three products compare column-wise. Both coupling terms are taken under $\bm Y$'s own clipped law; "
           r"the tube uses $4\times10^{6}$ pure linear-algebra draws and the paired tail the $N$ solver draws at "
           r"the stored $t^\ast$. Each is bounded by an exact one-sided Clopper--Pearson rate at "
           r"$\alpha=\delta/(2m+2)$ per test, with $\delta=0.05$, $m=50$ grid points and the two extra tests the bridge needs, so the $0.95$ is the level delivered rather than the level budgeted before the bridge was counted; and the "
           r"support-gap bridge carries the certificate from the clipped law to the untruncated "
           r"$P_{\rm lin}$ the pipeline reports, on which the interval is centred; $b$ is the sum of the "
           r"three uppers. The bridge is computed from the $4\times10^{6}$-draw clipped estimate, which is "
           r"why that row is given separately from the $N$-draw one. The "
           r"finite-sample share is $(b-\text{floor})/b$: on \pC{} only a tenth of the width is sampling error, "
           r"so more draws would not rescue the classification.}",
           r"\label{s:tab-r2}", r"\footnotesize",
           r"\begin{tabular}{@{}l" + "c" * len(rows) + r"@{}}", r"\toprule",
           " & " + " & ".join(LABEL[r["product"]] for r in rows) + r" \\", r"\midrule"]
    for name, fn in lines:
        out.append(name + " & " + " & ".join(fn(r) for r in rows) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def table_r5b():
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{Local decision calibration around the fitted posterior geometry, $400$ replicates per "
           r"product, reference scale $s$. ``KS'' are one-sample Kolmogorov--Smirnov $p$-values of the jittered "
           r"simulation-based-calibration ranks against $\mathrm{U}(0,1)$ for the two decision quantities, and "
           r"of the joint two-dimensional Mahalanobis probability integral transform; not rejected at "
           r"$0.05/3=0.017$ (Bonferroni over the three). The remaining columns score $P(\text{meet})$ at a fixed "
           r"action. \pB{} is degenerate throughout: $P(\text{meet})=0$ at every candidate, so only the rank "
           r"half is informative. The $s=1$ block is the regime control: there the conditional law fails even "
           r"with no discrepancy, which is why the calibration claim is read at $s\le\tfrac12$.}",
           r"\label{s:tab-r5b}", r"\footnotesize",
           r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
           r"$s$ & & law & KS purity & KS yield & KS joint & ECE & Brier & tail $n$@hit \\", r"\midrule"]
    blocks = [(0, "$\\tfrac12$"), (1, "$1$")]
    for idx, slab in blocks:
        for i, prod in enumerate(["HLXSYN", "HLXSYN", "HLXSYN"]):
            f = R / f"r5b_decision_sbc_{prod}_ou_frozen.json"
            rec = json.loads(f.read_text())[idx]
            for law in ("conditional", "predictive"):
                sb = {x["label"]: x["ks_pvalue"] for x in rec["sbc_g"][law]}
                pm = rec["pmeet"][law]["fixed"]
                tail = ("---" if pm["tail_hi_n"] == 0
                        else f"${pm['tail_hi_n']}$@${pm['tail_hi_hit_freq']:.3f}$")
                first = slab if (i == 0 and law == "conditional") else ""
                lab = LABEL[prod] if law == "conditional" else ""
                out.append(f"{first} & {lab} & {law} & {_p(sb['pool_purity'])} & {_p(sb['pool_yield'])} & "
                           f"{_p(sb['2d_mahalanobis_pit'])} & {_f(pm['ece'])} & {_f(pm['brier'])} & {tail} \\\\")
            if i < 2:
                out.append(r"\addlinespace[2pt]")
        if idx == 0:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def _r6_body(kind):
    """kind='rank' -> the three KS columns; kind='score' -> ECE, Brier, slope, decisive tail."""
    out = []
    for shape, slab in [("iso", "isotropic"), ("aniso", r"anisotropic ($6{:}1$, rotated)"),
                        ("aniso_icc", "anisotropic, with a shared within-product fold component")]:
        rows = _rows(R / f"r6_fullpipeline_calibration_{shape}.json")
        out.append(r"\midrule")
        out.append(r"\multicolumn{" + ("5" if kind == "rank" else "6") +
                   r"}{@{}l}{\emph{" + slab + r"}} \\")
        for prod in ["HLXSYN", "HLXSYN", "HLXSYN"]:
            sel = {r["arm"]: r for r in rows if r["product"] == prod}
            for j, arm in enumerate(ARMS):
                r = sel.get(arm)
                if r is None:
                    continue
                lab = LABEL[prod] if j == 0 else ""
                if kind == "rank":
                    out.append(f"{lab} & {ARM_TEX[arm]} & {_p(r['sbc_ks_purity'])} & "
                               f"{_p(r['sbc_ks_yield'])} & {_p(r['sbc_ks_pit'])} " + r"\\")
                else:
                    tail = ("---" if r["tail_hi_n"] == 0 else
                            f"${r['tail_hi_n']}$@${r['tail_hi_hit_freq']:.3f}$ {_ci(r.get('tail_hi_ci'))}")
                    slope = ("---" if r["slope"] != r["slope"]
                             else f"${r['slope']:.2f}$ {_ci(r.get('slope_ci'))}")
                    out.append(f"{lab} & {ARM_TEX[arm]} & ${r['ece']:.3f}$ {_ci(r.get('ece_ci'), 3)} & "
                               f"${r['brier']:.3f}$ & {slope} & {tail} " + r"\\")
            out.append(r"\addlinespace[2pt]")
    return out


def _longtable(label, caption, colspec, header, body, size=r"\footnotesize"):
    """A page-breaking table. 72 data rows do not fit a float, and a truncated tabular loses rows silently."""
    ncol = header.count("&") + 1
    return "\n".join([
        r"\begin{center}", size,
        r"\begin{longtable}{" + colspec + "}",
        r"\caption{" + caption + r"}\label{" + label + r"}\\",
        r"\toprule", header + r" \\", r"\midrule", r"\endfirsthead",
        r"\multicolumn{" + str(ncol) + r"}{@{}l}{\emph{\tablename~\thetable, continued}} \\",
        r"\toprule", header + r" \\", r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{" + str(ncol) + r"}{r@{}}{\emph{continued on the next page}} \\",
        r"\endfoot", r"\bottomrule", r"\endlastfoot",
        *body,
        r"\end{longtable}", r"\end{center}"])


def table_r6_rank():
    cap = (r"Hierarchy in the loop: rank calibration. The deployed estimator is re-fitted inside every replicate from "
           r"synthetic folds on the real $2/5/2$ in-domain fold structure; $300$ replicates per cell. Entries are "
           r"one-sample Kolmogorov--Smirnov $p$-values against $\mathrm{U}(0,1)$ for the jittered "
           r"simulation-based-calibration rank of each decision quantity, and for the joint two-dimensional "
           r"Mahalanobis probability integral transform; not rejected at $0.05/3=0.017$. The reading is in the joint "
           r"column: hier-est is not rejected on purity for \pA{} and \pC{} but fails yield and the joint transform "
           r"under mis-orientation, where the oracle passes. The floor rows are mis-scaled by the factor $\kappa$. ")
    return _longtable("s:tab-r6a", cap, r"@{}llccc@{}",
                      r"& arm & KS purity & KS yield & KS joint", _r6_body("rank"))


def table_r6_score():
    cap = (r"Hierarchy in the loop: probability scores for $P(\text{meet})$, same cells as Table~\ref{s:tab-r6a}. ECE "
           r"is expected calibration error over ten equal-width bins, reported with a $1000$-sample bootstrap interval "
           r"and \emph{alongside} Brier because it is bin-dependent and not a proper score---the correctly scaled "
           r"floor $\kappa{=}1$ attains a lower ECE than the oracle on \pC{} while losing on Brier. Slope is the "
           r"logistic recalibration of the outcome on $\mathrm{logit}\,\hat p$ with intercept, calibrated at $1$, "
           r"``---'' where saturation leaves the fit unidentified. The last column counts decisive calls $\hat "
           r"p\ge0.95$ and their realised frequency with a Jeffreys interval: the deployed arm's interval excludes "
           r"$0.95$ on \pC{} under mis-orientation. ")
    spec = (r"@{}ll>{\raggedright\arraybackslash}p{0.20\textwidth}c"
            r">{\raggedright\arraybackslash}p{0.17\textwidth}"
            r">{\raggedright\arraybackslash}p{0.22\textwidth}@{}")
    return _longtable("s:tab-r6b", cap, spec,
                      r"& arm & ECE & Brier & slope & decisive $n$@hit", _r6_body("score"))


if __name__ == "__main__":
    print("%% Generated by scripts/make_supplement_tables.py -- do not edit by hand.")
    print(table_r2(), "\n")
    print(table_r5b(), "\n")
    print(table_r6_rank(), "\n")
    print(table_r6_score())
