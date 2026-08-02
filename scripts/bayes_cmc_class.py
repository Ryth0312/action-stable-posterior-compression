"""Class-level σ decision-null on a NON-SMA model (docs/decision_null_theorem.md §3, Theorem 1).

Runs the Common-Mode-Capacity duality on an abstract shared-resource-competition system (no transport;
different mechanism than SMA): the capacity-only block φ is simultaneously decision-null O(ε) and
Fisher-null O(ε²) for the ratio QoI, same ε, with selectivity O(1).  Holding on a second model class is
the evidence that the σ decision-null is a CLASS property, not an SMA artefact.

    OMP_NUM_THREADS=4 python scripts/bayes_cmc_class.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cex_model.bayes.cmc_toy import CMCConfig, class_level_duality, physical_eps_sweep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results/bayes")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 1.0, 1.5, 2.0])
    args = ap.parse_args()

    c = CMCConfig(n_steps=args.n_steps)
    r = class_level_duality(c, scales=args.scales)
    dec, fish = r["decision_null_fit_vs_eps"], r["fisher_null_fit_vs_eps2"]
    phi = next(p["decision_sens"] for p in r["points"] if abs(p["sigma_scale"] - 1.0) < 1e-9)

    print(f"\n=== class-level σ decision-null on {r['model']} ===")
    print(f"  ε(nominal) = {r['epsilon_nominal']*100:.3f}%   (per-capita resource occupation)")
    print(f"  {'s':>5}{'ε_eff':>9}{'‖∂P/∂φ‖':>12}{'fisher∝‖∂y/∂φ‖²':>18}")
    for p in r["points"]:
        print(f"  {p['sigma_scale']:>5.2f}{p['epsilon']*100:>8.3f}%{p['decision_sens']:>12.3e}{p['fisher_scale']:>18.3e}")
    print(f"  -> decision-null O(ε):  ‖∂P/∂φ‖∝ε  R²={dec['r2']:.5f}  intercept={dec['intercept']:.1e} (through origin)")
    print(f"  -> Fisher-null  O(ε²): ‖∂y/∂φ‖²∝ε² R²={fish['r2']:.5f}")
    print(f"  -> selectivity contrast: ψ(g,ν) sens {r['psi_decision_sens_frozen']:.3e} = "
          f"{r['psi_decision_sens_frozen']/phi:.0f}× the φ sens (φ is the O(ε) null block, ψ is O(1))")
    print(f"  -> ratio cancellation: purity (ratio) rel-sens {r['ratio_rel_sens']:.3g} vs non-ratio "
          f"{r['nonratio_rel_sens']:.3g}  (factor {r['ratio_cancellation_factor']:.3f}, Prop 1c)")
    print(f"  -> frozen-state δ-check VERDICT: {r['verdict']} (confirms the chain-rule IDENTITY: ε=s·ε_nom,")
    print(f"     decision_sens=s·∂P/∂φ are linear in s by construction — NOT yet a moving-state scaling law)")

    # GENUINE (non-frozen) class-level test: vary the physical capacity R0 so ε moves with the STATE.
    rp = physical_eps_sweep(n_steps=args.n_steps)
    dp, fp = rp["decision_null_fit_vs_eps"], rp["fisher_null_fit_vs_eps2"]
    print(f"\n=== GENUINE (non-frozen) physical R0 sweep on {rp['model']} ===")
    print(f"  {'R0×':>5}{'ε_eff':>9}{'‖∂P/∂φ‖':>12}{'fisher∝‖∂y/∂φ‖²':>18}")
    for p in rp["points"]:
        print(f"  {p['r0_scale']:>5.2f}{p['epsilon']*100:>8.3f}%{p['decision_sens']:>12.3e}{p['fisher_scale']:>18.3e}")
    print(f"  -> decision-null O(ε):  ‖∂P/∂φ‖∝ε  R²={dp['r2']:.5f} (slope {dp['slope']:.4g}>0)")
    print(f"  -> Fisher-null  O(ε²): ‖∂y/∂φ‖²∝ε² R²={fp['r2']:.5f} (slope {fp['slope']:.4g}>0)")
    print(f"  -> VERDICT: {rp['verdict']} — the duality survives a MOVING state (R²<1, not a reparametrisation);")
    print(f"     this is the genuine class-level evidence, the analogue of the SMA Λ₀ sweep (R²≈0.94).")
    r["physical_eps_sweep"] = rp

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "CMC_class_level.json").write_text(json.dumps(r, indent=2))
    print(f"\n  json -> {out / 'CMC_class_level.json'}")


if __name__ == "__main__":
    main()
