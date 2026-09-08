"""Structured mass--residual metric route to the open a-priori κ_S constant
(docs/decision_null_theorem.md §7-1; the general off-diagonal / time-varying metric family).

Changes coordinates (C,Q)->(M,R) to split the strictly-stable conserved mass mode M=C+c1*Q from the
stiff reaction residual R=Q-Q* (departure from local SMA adsorption equilibrium), pulls back a block
metric diag(W_M, w^2*W_R) with W_R=diag(v/|p|) (symmetrizes the SMA capacity coupling), and reports the
2x2 block-comparison contraction rate rho(t) and the candidate constant kappa_S_MR (time-resolved +
sup-bound), the pullback conditioning chi_P, and the PASS/WARN/FAIL verdict per product.

    OMP_NUM_THREADS=4 python scripts/bayes_mass_residual_metric.py                       # synthetic only
    OMP_NUM_THREADS=4 python scripts/bayes_mass_residual_metric.py --products HLXSYN      # + real products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.mass_residual_metric import certify, certify_synthetic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--n-steps", type=int, default=120, help="[real products]")
    ap.add_argument("--syn-n-steps", type=int, default=60, help="[synthetic]")
    ap.add_argument("--stride", type=int, default=4, help="subsample for the per-step block-comparison "
                    "linear algebra + sigma-forcing autograd Jacobian")
    ap.add_argument("--residual", choices=["equilibrium", "Q"], default="equilibrium",
                    help="residual coordinate: equilibrium departure Q-Q* (time-varying H) or plain Q (H const)")
    ap.add_argument("--h-dot", choices=["fd", "zero"], default="fd",
                    help="headline metric-derivative mode (both computed, reported side-by-side)")
    ap.add_argument("--omega", type=float, default=1.0, help="mass/residual block-metric balance (tunes "
                    "chi_P only; bc and hence rho are omega-independent)")
    ap.add_argument("--products", nargs="*", default=[],
                    help="real products (committed {p}_posterior.npz + {p}_decision.json); default: SYN2 only")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic SYN2 run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if not args.no_synthetic:
        print("=== mass-residual metric certificate: SYN2 (synthetic) ===")
        results["SYN2"] = certify_synthetic(n_comp=2, nu=4.0, sigma=30.0, loading=8.0,
                                            n_steps=args.syn_n_steps, stride=args.stride, omega=args.omega,
                                            residual=args.residual, H_dot=args.h_dot)

    for p in args.products:
        print(f"=== mass-residual metric certificate: {p} ===")
        results[p] = certify(p, in_dir=args.in_dir, n_steps=args.n_steps, stride=args.stride,
                             omega=args.omega, residual=args.residual, H_dot=args.h_dot)

    for name, r in results.items():
        mr = r["mass_residual"]
        print(f"  {name:14s} chi_P={mr['chi_P']:.2e}  rho_worst={mr['rho_worst']:+.3e}  "
              f"int_rho={mr['net_integral_rho']:+.2e}  "
              f"kappa_S_MR={mr['kappa_S_MR']:.2e}  bound/measured={mr['bound_to_measured_ratio']:.2e}  "
              f"-> {r['decision']}")
        out_path = out_dir / f"mass_residual_metric_{name}.json"
        out_path.write_text(json.dumps(r, indent=2))
        print(f"    json -> {out_path}")

    decisions = {name: r["decision"] for name, r in results.items()}
    n_pass = sum(1 for d in decisions.values() if d == "PASS_NONVACUOUS")
    n_fail = sum(1 for d in decisions.values() if d == "FAIL_VACUOUS")
    overall = ("PASS_NONVACUOUS" if n_pass == len(decisions) and decisions
               else "FAIL_VACUOUS" if n_fail == len(decisions) and decisions
               else "MIXED")
    summary = {"products": list(results.keys()), "decisions": decisions, "overall_decision": overall}
    (out_dir / "mass_residual_metric_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  overall decision: {overall}")
    print(f"  summary json -> {out_dir / 'mass_residual_metric_summary.json'}")


if __name__ == "__main__":
    main()
