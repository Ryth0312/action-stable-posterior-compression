"""Solver-agnostic re-check of the decision verdict (Lever 4).

Recompute ``worst_dec`` on the INDEPENDENT RK23 reference (explicit RK, no implicit-diff BDF / IFT /
smooth-clip) by finite differences and compare to the committed differentiable-BDF value — showing the
decision verdict is not an artifact of the in-house differentiable solver.

    OMP_NUM_THREADS=4 python scripts/bayes_solver_agnostic.py --products HLXSYN HLXSYN HLXSYN
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.solver_agnostic import certify_solver_agnostic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=['HLXSYN'])
    ap.add_argument("--in-dir", default="results/bayes")
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--fd-rel", type=float, default=1e-2)
    args = ap.parse_args()

    print("=== solver-agnostic decision check (RK23 explicit-RK truth path vs differentiable BDF) ===")
    print("    headline = g(MAP) + decision Jacobian G (solver-agnostic); worst_dec FD = corroboration")
    res = {}
    for p in args.products:
        try:
            r = certify_solver_agnostic(p, in_dir=args.in_dir, fd_rel=args.fd_rel)
        except Exception as e:  # noqa: BLE001 -- skip a product with no committed decision/posterior
            print(f"  !! {p}: {type(e).__name__}: {e}")
            continue
        res[p] = r
        cos = "/".join(f"{c:.3f}" for c in r["decision_jacobian_row_cosine"])
        print(f"  {p}: g(MAP) RK23={r['g_map_rk23']['pool_purity']:.3f}/{r['g_map_rk23']['pool_yield']:.3f}"
              f" (Δg≤{r['g_max_abs_diff']:.3f})  ‖ΔG‖/‖G‖={r['decision_jacobian_rel_diff']:.1%}"
              f" rowcos={cos}  worst_dec bdf={r['worst_dec_bdf_recomputed']:.3f}"
              f"  rawFD={r['worst_dec_rk23']:.3f}  whitenedFD={r['worst_dec_rk23_whitened']:.3f}"
              f" (reproduces={r['worst_dec_whitened_reproduces']})  -> solver_agnostic={r['decision_solver_agnostic']}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "solver_agnostic_worstdec.json").write_text(json.dumps(res, indent=2))
    print(f"  json -> {out / 'solver_agnostic_worstdec.json'}")


if __name__ == "__main__":
    main()
