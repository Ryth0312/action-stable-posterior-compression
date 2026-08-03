"""σ-forcing-restricted finite-horizon reachability gain certificate
(docs/decision_null_theorem.md §7-1).

Full-state κ_S bounds sup over ALL perturbation directions and is vacuous. But σ-sensitivity solves
Ṡ_σ=J S_σ+F_σ, S_σ(0)=0, forced only along the low-dim σ-subspace, so the object that matters is the
σ-restricted finite-horizon gain ‖∫Φ(τ,s)F_σ(s)ds‖. This module builds a genuine a-priori over-estimate
(norm-inside integral ∫‖Φ F_σ‖ds) of the measured ‖S_σ(τ)‖ using a hand-written BDF-matched forward
variational propagator (cross-checked vs the solver FD), and reports the bound/measured slack, the
reachability-Gramian L2 gain, and a PASS_RESTRICTED_KAPPA / WARN_USEFUL_DIAGNOSTIC / FAIL_VACUOUS verdict.

    OMP_NUM_THREADS=4 python scripts/bayes_restricted_sigma_gain.py                      # synthetic only
    OMP_NUM_THREADS=4 python scripts/bayes_restricted_sigma_gain.py --products HLXSYN     # + real products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.restricted_sigma_gain import certify, certify_synthetic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--n-steps", type=int, default=120, help="[real products]")
    ap.add_argument("--syn-n-steps", type=int, default=60, help="[synthetic]")
    ap.add_argument("--stride", type=int, default=4, help="source-time subsample for the per-slice forward "
                    "propagation (semi-analytic vs FD stays ~1.0 up to stride~4)")
    ap.add_argument("--products", nargs="*", default=[],
                    help="real products (committed {p}_posterior.npz + {p}_decision.json); default: SYN2 only")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic SYN2 run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if not args.no_synthetic:
        print("=== restricted sigma gain certificate: SYN2 (synthetic) ===")
        results["SYN2"] = certify_synthetic(n_comp=2, nu=4.0, sigma=30.0, loading=8.0,
                                            n_steps=args.syn_n_steps, stride=args.stride)

    for p in args.products:
        print(f"=== restricted sigma gain certificate: {p} ===")
        results[p] = certify(p, in_dir=args.in_dir, n_steps=args.n_steps, stride=args.stride)

    for name, r in results.items():
        rg, xc = r["restricted_gain"], r["cross_check"]
        print(f"  {name:14s} bound/measured={rg['bound_to_measured_ratio']:.3f}  "
              f"semi/FD={xc['semi_analytic_over_fd']:.3f}(ok={xc['propagator_ok']})  "
              f"gramian={rg['gramian_L2_gain']:.2e}  full_state_vacuous={r['euclidean']['full_state_vacuous']}  "
              f"-> {r['decision']}")
        out_path = out_dir / f"restricted_sigma_gain_{name}.json"
        out_path.write_text(json.dumps(r, indent=2))
        print(f"    json -> {out_path}")

    decisions = {name: r["decision"] for name, r in results.items()}
    n_pass = sum(1 for d in decisions.values() if d == "PASS_RESTRICTED_KAPPA")
    n_fail = sum(1 for d in decisions.values() if d == "FAIL_VACUOUS")
    overall = ("PASS_RESTRICTED_KAPPA" if n_pass == len(decisions) and decisions
               else "FAIL_VACUOUS" if n_fail == len(decisions) and decisions
               else "MIXED")
    summary = {"products": list(results.keys()), "decisions": decisions, "overall_decision": overall}
    (out_dir / "restricted_sigma_gain_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  overall decision: {overall}")
    print(f"  summary json -> {out_dir / 'restricted_sigma_gain_summary.json'}")


if __name__ == "__main__":
    main()
