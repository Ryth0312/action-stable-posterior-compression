"""Thermodynamic entropy metric route for the open a-priori κ_S constant (ENTROPY_METRIC_KAPPA_S_PLAN.md).

Diagnoses whether the SMA adsorption/desorption entropy Hessian metric (which supplies the
off-diagonal capacity coupling missing from the diagonal contraction-metric family already tried in
`bayes/contraction.py`) makes the log-norm diagnostic materially better than the vacuous Euclidean
one (`bayes/groenwall.py`), and reports the candidate a-priori constant `kappa_S_ent` alongside the
zeta/omega/epsilon residual diagnostics (Tasks 1-4 of the plan). Applies the plan's own decision
criterion per product and writes a summary across products.

    OMP_NUM_THREADS=4 python scripts/bayes_entropy_metric.py                      # synthetic only
    OMP_NUM_THREADS=4 python scripts/bayes_entropy_metric.py --products HLXSYN     # + real products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.entropy_metric import certify, certify_synthetic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--n-steps", type=int, default=120, help="[real products]")
    ap.add_argument("--syn-n-steps", type=int, default=60, help="[synthetic]")
    ap.add_argument("--stride", type=int, default=4, help="subsample for the generalized eigenproblem "
                    "+ sigma-forcing autograd Jacobian (mu_ent is the costly part)")
    ap.add_argument("--dot-m", choices=["fd", "zero"], default="fd",
                    help="headline dot(M) mode (both are always computed and reported)")
    ap.add_argument("--products", nargs="*", default=[],
                    help="real products (committed {p}_posterior.npz + {p}_decision.json); "
                    "default: synthetic SYN2 only")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic SYN2 run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if not args.no_synthetic:
        print("=== entropy metric certificate: SYN2 (synthetic) ===")
        results["SYN2"] = certify_synthetic(n_comp=2, nu=4.0, sigma=30.0, loading=8.0,
                                            n_steps=args.syn_n_steps, stride=args.stride, dot_M=args.dot_m)

    for p in args.products:
        print(f"=== entropy metric certificate: {p} ===")
        results[p] = certify(p, in_dir=args.in_dir, n_steps=args.n_steps, stride=args.stride, dot_M=args.dot_m)

    for name, r in results.items():
        e, ent, res = r["euclidean"], r["entropy"], r["residual"]
        print(f"  {name:14s} sup_mu2={e['sup_mu2']:7.3f}  sup_mu_ent={ent['sup_mu_ent']:7.3f}  "
              f"int+mu2={e['int_positive_mu2']:9.1f}  int+mu_ent={ent['int_positive_mu_ent']:9.1f}  "
              f"kappa_S_ent/kappa_eff={ent['bound_to_measured_ratio']:.2e}  "
              f"p95_zeta={res['p95_zeta']:.2f}  p95_omega={res['p95_omega']:.3f}  -> {r['decision']}")
        out_path = out_dir / f"entropy_metric_{name}.json"
        out_path.write_text(json.dumps(r, indent=2))
        print(f"    json -> {out_path}")

    decisions = {name: r["decision"] for name, r in results.items()}
    n_continue = sum(1 for d in decisions.values() if d == "continue_entropy")
    overall = ("continue_entropy" if n_continue == len(decisions) and decisions
               else "fallback_adjoint_gain" if all(d == "fallback_adjoint_gain" for d in decisions.values())
               else "inconclusive")
    summary = {"products": list(results.keys()), "decisions": decisions, "overall_decision": overall}
    (out_dir / "entropy_metric_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  overall decision: {overall}")
    print(f"  summary json -> {out_dir / 'entropy_metric_summary.json'}")


if __name__ == "__main__":
    main()
