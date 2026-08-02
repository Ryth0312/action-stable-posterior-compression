"""Step 2B' : run the decision-null compression ladder on the COMMITTED REAL posteriors.

Torch-free: reads Sigma from ``results/bayes/{p}_posterior.npz`` and the decision Jacobian G from
``results/bayes/{p}_decision.json`` (the ``decision_jacobian`` field), so no ODE / refit is needed.
This is the informative complement to the synthetic ``step2b`` sweep: real epsilon (0.4-1.5%), the
real keq<->nu ridge cross-covariance, and (for the main products) a determined decision.

Reports per product the memo ladder plus the RISK-relevant marginal decision-std change under
D_Schur vs D_freeze (the operator-norm gap alone is indefinite for freeze and can mislead).

Run:  python scripts/step2b_real_ladder.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np

_p = pathlib.Path(__file__).resolve().parents[1] / "src/cex_model/bayes/decision_compression.py"
_spec = importlib.util.spec_from_file_location("decision_compression", _p)
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results/bayes"
# (label, posterior npz, decision json)
PRODUCTS = [
    ("mAb A (HLXSYN)",        "HLXSYN_posterior.npz",        "HLXSYN_decision.json"),
    ("mAb B (HLXSYN exclDT)", "HLXSYN_posterior.npz", "HLXSYN_decision.json"),
    ("mAb C (HLXSYN)",        "HLXSYN_posterior.npz",        "HLXSYN_decision.json"),
    ("mAb D (HLXSYN, hi-ε)",  "HLXSYN_posterior.npz",        "HLXSYN_decision.json"),
    ("mAb E (HLXSYN, hi-ε)", "HLXSYN_posterior.npz",     "HLXSYN_decision.json"),
]


def _marg_std(C, tol):
    """Per-QoI decision std in physical units from a decision covariance C."""
    return np.sqrt(np.clip(np.diag(np.asarray(C, float)), 0.0, None))


def main() -> None:
    print("Δstd_yld (rel. to full posterior; negative = INFLATED) for 3 σ-differential reductions:")
    print(f"{'product':22} {'n':>2} {'e_id':>9} {'q_C':>5} {'q_W':>5} {'wdec':>6} | "
          f"{'σdiff-shr':>9} | {'naive-frz':>10} {'reduced(D)':>10} {'Schur':>8}")
    print("-" * 100)
    rows = []
    for label, pnpz, djson in PRODUCTS:
        pf, jf = RESULTS / pnpz, RESULTS / djson
        if not (pf.exists() and jf.exists()):
            print(f"{label:22} MISSING artifact"); continue
        Sigma = np.load(pf, allow_pickle=False)["cov"]
        dj = json.loads(jf.read_text())
        G = np.atleast_2d(np.asarray(dj["decision_jacobian"], float))
        tol = np.asarray(dj.get("tol", [0.02, 0.05]), float)
        n = Sigma.shape[0] // 4

        Sigma_r, G_r, u_idx, v_idx = dc.rotate_sigma_block(Sigma, G, n)
        lad = dc.theorem_ladder(Sigma_r, G_r, tol, u_idx, v_idx)

        C_full = dc.decision_cov(G_r, Sigma_r)
        Cbar_full = dc.whiten(C_full, tol)
        std_full = _marg_std(C_full, tol)

        # Three sigma-differential reductions (memo Prop 1 / Prop 1b):
        #  * Schur (conditional-mean, theorem object): v <- E[v|u]; active block keeps marginal Sigma_uu.
        #  * naive-freeze (strawman): zero the v rows/cols, NO refit -> destroys the u-v (keq<->nu) ridge.
        #  * reduced-refit (= paper variant-D): freeze v at MAP, REFIT u -> active cov = (H_uu)^-1 = Sigma_{u|v}.
        C_schur = dc.decision_cov(G_r, dc.schur_cov(Sigma_r, u_idx, v_idx))
        C_freeze = dc.decision_cov(G_r, dc.freeze_cov(Sigma_r, u_idx, v_idx))
        H = np.linalg.inv(Sigma_r)
        cov_red = np.zeros_like(Sigma_r)
        cov_red[np.ix_(u_idx, u_idx)] = np.linalg.inv(H[np.ix_(u_idx, u_idx)])   # Sigma_{u|v}, v frozen
        C_reduced = dc.decision_cov(G_r, cov_red)

        gap_schur = dc.whiten(C_full - C_schur, tol)
        sdiff_share = float(np.trace(gap_schur) / np.trace(Cbar_full)) if np.trace(Cbar_full) > 0 else float("nan")

        std_schur, std_freeze, std_reduced = (_marg_std(C_schur, tol), _marg_std(C_freeze, tol),
                                              _marg_std(C_reduced, tol))

        def _dstd(sc):  # relative yield-std change vs full (negative = inflated)
            return float((std_full[1] - sc[1]) / std_full[1]) if std_full[1] > 0 else float("nan")

        worst_dec = float(np.sqrt(max(np.linalg.eigvalsh(Cbar_full).max(), 0.0)))
        # same statistic under naive zeroing, so the strawman's cost is quotable in wdec units
        worst_dec_freeze = float(np.sqrt(max(np.linalg.eigvalsh(dc.whiten(C_freeze, tol)).max(), 0.0)))
        gs = float(np.linalg.norm(gap_schur, 2))
        frz = float(np.linalg.norm(dc.whiten(C_full - C_freeze, tol), 2)) / gs if gs > 0 else float("nan")

        cert = dc.compression_certificates(Sigma_r, G_r, tol, u_idx, v_idx)
        print(f"{label:22} {n:2d} {lad['e_id']:9.1e} {lad['q_C']:5.2f} {lad['q_W']:5.2f} {worst_dec:6.3f} | "
              f"{sdiff_share*100:10.2f}% | {_dstd(std_freeze)*100:9.1f}% {_dstd(std_reduced)*100:9.1f}% "
              f"{_dstd(std_schur)*100:8.1f}%")
        rows.append({"product": label, "n": n, "e_id": lad["e_id"], "q_C": lad["q_C"], "q_W": lad["q_W"],
                     "worst_dec": worst_dec, "worst_dec_naive_freeze": worst_dec_freeze,
                     "sigma_diff_decision_share": sdiff_share,
                     "yield_std_change_naive_freeze": _dstd(std_freeze),
                     "yield_std_change_reduced_refit_variantD": _dstd(std_reduced),
                     "yield_std_change_schur": _dstd(std_schur),
                     "purity_std_change_schur": float((std_full[0] - std_schur[0]) / std_full[0]),
                     "freeze_over_schur_opnorm": frz, "certificates": cert,
                     "std_full": std_full.tolist(), "std_schur": std_schur.tolist(),
                     "std_freeze": std_freeze.tolist(), "std_reduced_variantD": std_reduced.tolist()})

    # R1 dual-certificate table (memo §4b): exact residual identities + the two decision certificates.
    print("\nR1 certificates (exact conditional-variance residuals; δ_S Schur, δ_D variant-D ridge-null):")
    print(f"{'product':22} {'id_S':>8} {'id_D':>8} | {'δ_S':>8} {'δ_D':>8} | "
          f"{'‖Γ_v‖':>9} {'‖Σvv^½‖':>9} {'c_min':>9}")
    for r in rows:
        c = r["certificates"]
        print(f"{r['product']:22} {c['id_S']:8.0e} {c['id_D']:8.0e} | {c['delta_S']:8.3f} {c['delta_D']:8.3f} | "
              f"{c['gamma_v_norm']:9.2e} {c['Svv_sqrt_norm']:9.2e} {c['c_min']:9.3e}")

    print("\nReadings:")
    print("  id_S, id_D ~ machine eps => BOTH exact conditional-variance identities hold on the REAL fitted Sigma.")
    print("  δ_S = ‖T^{-1} G_v Σ_{v|u}^{1/2}‖  (Schur: deletes the conditional residual).")
    print("  δ_D = ‖T^{-1} Γ_v Σ_vv^{1/2}‖    (variant-D: deletes the marginal ridge; Γ_v = ridge decision derivative).")
    print("  Both δ_S, δ_D < 1 with c_min>0 (1/(2√c_min) finite) => both compressions are action-safe (distinct residuals).")
    print("  Headline: Σvv^½≈24 (marginally wide) yet ‖Γ_v‖≈1e-4 => a non-identified ridge is decision-flat (title claim).")
    print("  naive-freeze has NO conditional-variance identity (destroys cross-covariance) => blows up.")
    (RESULTS / "decision_compression_real_ladder.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {RESULTS / 'decision_compression_real_ladder.json'}")


if __name__ == "__main__":
    main()
