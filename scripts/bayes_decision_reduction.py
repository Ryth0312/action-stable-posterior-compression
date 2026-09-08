"""Decision-design invariance under the σ common-mode reduction — does the experiment-design layer move?

§3.11 shows compressing σ to its common-capacity mode (Arm D) preserves the decision *outcome* at the
historical operating point. This asks the §2.12 follow-up: rerun the operating-window map + VoI stopping
rule under Arm D's reduced posterior and check the **design layer** is invariant vs the full model —
the recommended operating window, the four-state recommendation, the decision state, and the decision
covariance (relative-Frobenius < gate, the §3.11 axis). The honest read is *invariance* (a smaller,
better-conditioned model needs no re-planned experiments), not a σ-vs-keq VoI contrast (null on a
determined decision). Reuses ``sigma_ablation``'s block algebra (no solver for the reduction) and the
§2.12 ``decision_window`` deliverable; no new estimator.

    OMP_NUM_THREADS=4 python scripts/bayes_decision_reduction.py                                    # SYN3 only
    OMP_NUM_THREADS=4 python scripts/bayes_decision_reduction.py --products HLXSYN HLXSYN HLXSYN --n-steps 300
    OMP_NUM_THREADS=4 python scripts/bayes_decision_reduction.py --products HLXSYN --reductions D_sigma_commonmode B_sigma_map --n-steps 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.decision_reduction import run_product, run_synthetic


def _summarize(name: str, rec: dict) -> None:
    a0 = rec["arms"]["A_full"]["design"]
    print(f"=== decision-reduction invariance: {name} ===")
    print(f"  A_full              action={a0['action']:20s} state={a0['decision_state']:22s} "
          f"window={_fmt(a0['recommended_op'])} maxVoI={a0['max_voi_worst_dec']:.3f}")
    for arm, a in rec["arms"].items():
        if arm == "A_full":
            continue
        d, inv = a["design"], a["invariance"]
        note = (f"  reasons={inv['reasons']}" if inv["reasons"]
                else "  [window_argmax_flip: non-operative]" if inv.get("window_argmax_flip") else "")
        print(f"  {arm:18s}  action={d['action']:20s} state={d['decision_state']:22s} "
              f"window={_fmt(d['recommended_op'])} relFrob={inv['rel_cov_frobenius']:.3f} -> {inv['verdict']}{note}")


def _fmt(op):
    return "None" if op is None else str([round(float(x), 1) for x in op])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=[],
                    help="real products (committed {p}_decision.json + {p}_posterior.npz); default: SYN3. "
                         "Main cohort: HLXSYN HLXSYN HLXSYN; appendix: HLXSYN HLXSYN")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic SYN3 run")
    ap.add_argument("--reductions", nargs="*", default=["D_sigma_commonmode"],
                    help="at-MAP σ reductions to test vs full model (D_sigma_commonmode [, B_sigma_map])")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--n-steps", type=int, default=120, help="[real products] BDF grid (match committed fit, 300)")
    ap.add_argument("--syn-n-steps", type=int, default=40, help="[synthetic]")
    ap.add_argument("--n-candidates", type=int, default=12, help="candidate OP pool for the operating-window/VoI")
    ap.add_argument("--gate", type=float, default=0.25, help="decision-covariance relative-Frobenius gate")
    ap.add_argument("--no-reuse-posterior", action="store_true",
                    help="[real products] fit arm A fresh instead of the committed {p}_posterior.npz")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    if not args.no_synthetic:
        results["SYN3"] = run_synthetic(n_steps=args.syn_n_steps, reductions=tuple(args.reductions),
                                        n_candidates=args.n_candidates, gate=args.gate, verbose=True)
    for p in args.products:
        results[p] = run_product(p, in_dir=args.in_dir, n_steps=args.n_steps,
                                 reductions=tuple(args.reductions), n_candidates=args.n_candidates,
                                 gate=args.gate, reuse_posterior=not args.no_reuse_posterior, verbose=True)

    for name, rec in results.items():
        _summarize(name, rec)
        (out_dir / f"{name}_decision_reduction.json").write_text(json.dumps(rec, indent=2))
        print(f"    json -> {out_dir / f'{name}_decision_reduction.json'}")

    summary = {
        "products": list(results.keys()),
        "design_invariant": {n: r["design_invariant"] for n, r in results.items()},
        "verdicts": {n: {arm: a["invariance"]["verdict"]
                         for arm, a in r["arms"].items() if arm != "A_full"}
                     for n, r in results.items()},
    }
    (out_dir / "decision_reduction_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  summary json -> {out_dir / 'decision_reduction_summary.json'}")


if __name__ == "__main__":
    main()
