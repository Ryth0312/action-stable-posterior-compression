"""Decision-space gain witness for the open a-priori kappa_S gap (kappa closure brief, Deliverable A).

Does NOT claim a full-state a-priori kappa_S certificate. Packages the exact, constant-free
decision-space quantities (B = T^-1 G diag(sigma_prior), the ratio law r = ||B_sigma||/||B_nu|| =
O(eps)) into a reproducible report with genuine mesh-stability (G/eps recomputed by a real solve at
each n_steps, not read once from a committed snapshot).

    python scripts/bayes_decision_gain.py \
      --products HLXSYN HLXSYN HLXSYN HLXSYN HLXSYN \
      --n-steps 80 100 120 160 \
      --out results/bayes/decision_gain_certificate.json
"""

from __future__ import annotations

import argparse

from cex_model.bayes.decision_gain import DecisionGainConfig, run_decision_gain_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=['HLXSYN'])
    ap.add_argument("--main-products", nargs="+", default=['HLXSYN'])
    ap.add_argument("--n-steps", nargs="+", type=int, default=[80, 100, 120, 160])
    ap.add_argument("--spectral-dir", default="results/bayes")
    ap.add_argument("--out", default="results/bayes/decision_gain_certificate.json")
    args = ap.parse_args()

    config = DecisionGainConfig(products=args.products, n_steps=args.n_steps,
                                spectral_dir=args.spectral_dir, out_path=args.out,
                                strict_main_products=tuple(args.main_products))
    print("=== decision-space gain witness (magnitude_witness_not_full_state_kappa_S) ===")
    report = run_decision_gain_report(config)
    for p, v in report["products"].items():
        if v.get("status") == "SKIPPED_MISSING_DATA":
            print(f"  {p:14s} SKIPPED ({v.get('error')})")
            continue
        mesh = v.get("mesh") or {}
        print(f"  {p:14s} n_steps={v['n_steps']:4d}  eps={v['epsilon']*100:6.3f}%  "
              f"||B_sigma||={v['B_sigma_norm']:7.3f}  r/eps={v['r_over_epsilon']:5.2f}  "
              f"sigma-share exact={v['sigma_share_exact']*100:.4f}% pred(r^2)={v['sigma_share_pred_from_r2']*100:.4f}%  "
              f"mesh(Bsigma)={mesh.get('B_sigma_rel_spread', float('nan')):.1%}  -> {v['status']}")
    gs = report["global_summary"]
    print(f"\n  main-product r/eps: mean={gs['main_products_r_over_eps_mean']:.3f} "
          f"cv={gs['main_products_r_over_eps_cv']:.3f}")
    print(f"  full_state_kappa_S_claimed = {gs['full_state_kappa_S_claimed']}")
    print(f"  json -> {args.out}")


if __name__ == "__main__":
    main()
