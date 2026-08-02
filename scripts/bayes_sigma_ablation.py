"""σ freeze / compress ablation — can σ be fixed or compressed without changing the decision?

Fits the full model (arm A) then re-fits with σ frozen / compressed six ways, separating **fit
equivalence** (χ²/dof, RMSE, predictive residuals) from **decision equivalence** (tolerance-whitened
decision mean shift, decision covariance / worst_dec / P(meet) change, operating recommendation), plus the
computational benefit (runtime, Hessian condition number, posterior rank). Per-arm verdict
PASS_DECISION_EQUIVALENT / WARN_FIT_CHANGED / FAIL_DECISION_CHANGED; ν/keq negative controls confirm power.

    A  full model
    B  σ fixed at full-model MAP
    C  σ prior-only / no likelihood update (zero σ Jacobian columns)
    D  common-mode-only σ nuisance, differential σ frozen (compression on the fixed (1,…,1) basis)
    E  σ fixed at nominal / prior mean, refit
    F  negative controls: ν-fixed and keq-fixed (both @MAP and @nominal)

For real products the committed {product}_posterior.npz is reused for arm A (so arm A byte-matches the
paper), which also makes arms B/C/D and the @MAP controls pure linear algebra (no solver) -- only the
@nominal arms (E, NC_*_nominal) re-fit. Use --n-steps matching the committed fit (production = 300).

    OMP_NUM_THREADS=4 python scripts/bayes_sigma_ablation.py                                  # synthetic only
    OMP_NUM_THREADS=4 python scripts/bayes_sigma_ablation.py --products HLXSYN HLXSYN HLXSYN --n-steps 300
    OMP_NUM_THREADS=4 python scripts/bayes_sigma_ablation.py --products HLXSYN --recommend --mc --n-steps 300
    OMP_NUM_THREADS=4 python scripts/bayes_sigma_ablation.py --no-synthetic --products HLXSYN  # skip SYN3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.sigma_ablation import ARMS, run_product, run_synthetic


def _plot(rec: dict, path: Path) -> None:
    """Bar chart: per-arm decision worst_dec and Hessian condition number, verdict-coloured."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = list(rec["arms"].keys())
    wd = [rec["arms"][n]["decision"]["worst_dec"] for n in names]
    cond = [rec["arms"][n]["benefit"]["hessian_cond"] for n in names]
    color = {"BASELINE": "#555", "PASS_DECISION_EQUIVALENT": "#2ca02c",
             "WARN_FIT_CHANGED": "#ff7f0e", "FAIL_DECISION_CHANGED": "#d62728"}
    cols = [color.get(rec["arms"][n]["verdict"], "#999") for n in names]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].bar(names, wd, color=cols)
    ax[0].axhline(1.0, ls="--", c="k", lw=0.8)
    ax[0].set_ylabel("decision worst_dec")
    ax[0].set_title(f"{rec['product']} — σ freeze/compress ablation (neg-ctrl: {rec['negative_control_verdict']})")
    ax[1].bar(names, cond, color=cols)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Hessian cond number")
    ax[1].tick_params(axis="x", rotation=45)
    for lab in ax[1].get_xticklabels():
        lab.set_ha("right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _summarize(name: str, rec: dict) -> None:
    print(f"=== σ ablation: {name}  (neg-control: {rec['negative_control_verdict']}) ===")
    print(f"    σ arms outcome-equivalent (rec/spec/determinability unchanged): "
          f"{rec.get('sigma_arms_outcome_equivalent', [])}")
    for arm, a in rec["arms"].items():
        d, f, b = a["decision"], a["fit"], a["benefit"]
        oe = "" if arm == "A_full" else (" outcome=OK" if a.get("decision_outcome_equivalent")
                                         else " outcome=CHANGED")
        fr = a.get("reasons", {}).get("fail", [])
        print(f"  {arm:18s} {a['verdict']:24s} worst_dec={d['worst_dec']:.3f} P(meet)={d['p_meet']:.3f} "
              f"chi2/dof={f['chi2_dof']:.2f} RMSE={f['rmse_g_l']:.4f} cond={b['hessian_cond']:.1e} "
              f"rank={b['informed_rank']}/{b['nominal_dim']}{oe}"
              + (f"  fail={fr}" if fr else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=[],
                    help="real products (committed {p}_decision.json); default: SYN3 synthetic only. "
                         "Main cohort: HLXSYN HLXSYN HLXSYN; appendix: HLXSYN HLXSYN")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic SYN3 run")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--n-steps", type=int, default=120, help="[real products] BDF grid for fit + decision")
    ap.add_argument("--syn-n-steps", type=int, default=40, help="[synthetic]")
    ap.add_argument("--map-iters", type=int, default=150)
    ap.add_argument("--arms", nargs="*", default=list(ARMS), help=f"subset of {list(ARMS)}")
    ap.add_argument("--recommend", action="store_true",
                    help="also compute the four-state operating recommendation per arm (Colab-heavy)")
    ap.add_argument("--mc", action="store_true", help="add the Monte-Carlo decision cross-check per arm")
    ap.add_argument("--n-candidates", type=int, default=12, help="candidate OP pool for --recommend")
    ap.add_argument("--no-reuse-posterior", action="store_true",
                    help="[real products] fit arm A fresh instead of loading the committed {p}_posterior.npz")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    if not args.no_synthetic:
        results["SYN3"] = run_synthetic(n_steps=args.syn_n_steps, map_iters=args.map_iters, arms=tuple(args.arms),
                                        recommend=args.recommend, mc=args.mc, n_candidates=args.n_candidates,
                                        verbose=True)

    for p in args.products:
        results[p] = run_product(p, in_dir=args.in_dir, n_steps=args.n_steps, map_iters=args.map_iters,
                                 arms=tuple(args.arms), recommend=args.recommend, mc=args.mc,
                                 n_candidates=args.n_candidates, reuse_posterior=not args.no_reuse_posterior,
                                 verbose=True)

    for name, rec in results.items():
        _summarize(name, rec)
        (out_dir / f"{name}_sigma_ablation.json").write_text(json.dumps(rec, indent=2))
        print(f"    json -> {out_dir / f'{name}_sigma_ablation.json'}")
        if not args.no_plot:
            _plot(rec, out_dir / f"{name}_sigma_ablation.png")

    summary = {
        "products": list(results.keys()),
        "negative_control_verdict": {n: r["negative_control_verdict"] for n, r in results.items()},
        "sigma_arms_pass": {n: r["sigma_arms_pass"] for n, r in results.items()},
        "sigma_arms_outcome_equivalent": {n: r.get("sigma_arms_outcome_equivalent", [])
                                          for n, r in results.items()},
        "verdicts": {n: {arm: a["verdict"] for arm, a in r["arms"].items()} for n, r in results.items()},
    }
    (out_dir / "sigma_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  summary json -> {out_dir / 'sigma_ablation_summary.json'}")


if __name__ == "__main__":
    main()
