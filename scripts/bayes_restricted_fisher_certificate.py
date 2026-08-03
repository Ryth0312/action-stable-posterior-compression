"""σ-restricted finite-horizon OBSERVATION-gain Fisher certificate
(docs/decision_null_theorem.md §7-1) — the observation/Fisher-side twin of the state-side
``bayes_restricted_sigma_gain.py``.

Bounds the σσ Fisher block non-circularly, ``λmax(F_σσ) ≤ (ε·κ_obs,σ)²/σ_obs²``, and converts it to an
a-priori lower bound on the σ-block posterior shrinkage ``worst_dir_σ`` — closing the Fisher-null leg of
Theorem 1's dual with the same discipline as the state gain and the decision-side ratio law.

    OMP_NUM_THREADS=4 python scripts/bayes_restricted_fisher_certificate.py                    # synthetic only
    OMP_NUM_THREADS=4 python scripts/bayes_restricted_fisher_certificate.py --products HLXSYN   # + real products
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.restricted_fisher_certificate import certify, certify_synthetic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--n-steps", type=int, default=120, help="[real products]")
    ap.add_argument("--syn-n-steps", type=int, default=40, help="[synthetic]")
    ap.add_argument("--stride", type=int, default=1, help="source-time subsample for the per-slice forward "
                    "propagation (stride=1 = every step; matches the committed state-gain quality)")
    ap.add_argument("--products", nargs="*", default=[],
                    help="real products (committed {p}_posterior.npz); default: SYN2 only")
    ap.add_argument("--no-synthetic", action="store_true", help="skip the synthetic SYN2 run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if not args.no_synthetic:
        print("=== restricted fisher certificate: SYN2 (synthetic) ===")
        results["SYN2"] = certify_synthetic(n_comp=2, nu=4.0, sigma=30.0, loading=8.0,
                                            n_steps=args.syn_n_steps, stride=args.stride)

    for p in args.products:
        print(f"=== restricted fisher certificate: {p} ===")
        results[p] = certify(p, in_dir=args.in_dir, n_steps=args.n_steps, stride=args.stride)

    for name, r in results.items():
        rf, xc = r["restricted_fisher"], r["cross_check"]
        print(f"  {name:14s} eps={r['epsilon']:.3e}  F_bound={rf['F_sigma_sigma_bound']:.3e}  "
              f"F_exact={rf['F_sigma_sigma_exact']:.3e}  bound/exact={rf['bound_over_exact']:.3f}  "
              f"shrink_bound={rf['posterior_shrinkage_bound']:.3f}<=worst_dir={rf['posterior_worst_dir_exact']:.3f}  "
              f"F_exact/fd={xc['F_exact_over_fd']:.3f}(ok={xc['propagator_ok']})  -> {r['decision']}")
        out_path = out_dir / f"restricted_fisher_certificate_{name}.json"
        out_path.write_text(json.dumps(r, indent=2))
        print(f"    json -> {out_path}")

    decisions = {name: r["decision"] for name, r in results.items()}
    n_pass = sum(1 for d in decisions.values() if d == "PASS_RESTRICTED_FISHER")
    n_fail = sum(1 for d in decisions.values() if d == "FAIL_VACUOUS")
    overall = ("PASS_RESTRICTED_FISHER" if n_pass == len(decisions) and decisions
               else "FAIL_VACUOUS" if n_fail == len(decisions) and decisions
               else "MIXED")
    summary = {"products": list(results.keys()), "decisions": decisions, "overall_decision": overall}
    (out_dir / "restricted_fisher_certificate_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  overall decision: {overall}")
    print(f"  summary json -> {out_dir / 'restricted_fisher_certificate_summary.json'}")


if __name__ == "__main__":
    main()
