"""R3a: goal-oriented value-of-information nullity (Woodbury) on the committed real posteriors.

Torch-free. Verifies, on the committed posteriors + decision Jacobians:

  (i)   Woodbury VoI identity   C - C_c = M_c (I + B_cᵀ Σ B_c)⁻¹ M_cᵀ,  M_c = G Σ B_c,
        with B_c = J_cᵀ/σ the candidate experiment's information square-root, to machine precision;
        and that it equals the implemented decision_oed_score trace tr(T⁻¹ (C - C_c) T⁻¹).
  (ii)  EXACT nullity: a candidate whose sensitivities lie in ker(G Σ) leaves C_c = C exactly
        (VoI = 0), no matter how large its Fisher information; a decision-aligned candidate does not.
  (iii) sigma near-nullity: an experiment informing only the sigma-differential block (the run a
        parameter-space D-optimal design most prizes) has ‖T⁻¹ M_c‖ ≈ 0 and VoI ≈ 0.
  (iv)  discrepancy-floor ceiling: an additive floor common to the pre- and pre-posterior laws
        cancels in C - C_c, so worst_dec_pred,c ≥ worst_dec_floor for every theta-only experiment.

This is NOT an EVSI theorem: it is the exact algebra of the local-Gaussian decision-precision VoI
(the object decision.decision_oed_score already computes), read as a stopping-rule nullity.

Run:  python scripts/step3_voi_nullity.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

# load decision_compression.py directly (bypass the torch-polluted cex_model.bayes.__init__)
_dc_path = Path(__file__).resolve().parent.parent / "src/cex_model/bayes/decision_compression.py"
_spec = importlib.util.spec_from_file_location("decision_compression", _dc_path)
_dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dc)

IN_DIR = Path("results/bayes")
PRODUCTS = ["HLXSYN", "HLXSYN", "HLXSYN"]
SIGMA_OBS = 0.02       # matches the project measurement-noise convention (decision_oed_score sigma)
N_PROTEIN = 5          # 20 params = 5 proteins x 4 blocks; sigma block = indices [3n:4n]


def _common_mode_direction(n: int) -> np.ndarray:
    e = np.ones(n) / np.sqrt(n)
    return e


def _sigma_bases(dim: int, n: int):
    """Orthonormal bases (in the ORIGINAL param basis) of the sigma COMMON-mode (dim x 1) and
    sigma-DIFFERENTIAL (dim x (n-1)) subspaces: zero outside the sigma block [3n:4n]."""
    e_c = _common_mode_direction(n)
    Q, _ = np.linalg.qr(np.column_stack([e_c, np.eye(n)]))   # Q[:,0] ~ e_c, Q[:,1:] differential
    if np.dot(Q[:, 0], e_c) < 0:
        Q[:, 0] *= -1.0
    B_common = np.zeros((dim, 1))
    B_common[3 * n:4 * n, :] = Q[:, :1]
    B_diff = np.zeros((dim, n - 1))
    B_diff[3 * n:4 * n, :] = Q[:, 1:]
    return B_common, B_diff


def _load(product):
    post = np.load(IN_DIR / f"{product}_posterior.npz", allow_pickle=True)
    Sigma = np.asarray(post["cov"], float)
    dj = json.loads((IN_DIR / f"{product}_decision.json").read_text())
    G = np.atleast_2d(np.asarray(dj["decision_jacobian"], float))
    tol = np.asarray(dj["tol"], float)
    return Sigma, G, tol


def _preposterior_C(G, Sigma, B_c):
    """Direct (non-Woodbury) pre-posterior decision covariance C_c = G (H + B_c B_cᵀ)⁻¹ Gᵀ."""
    H = np.linalg.inv(Sigma)
    Hc = H + B_c @ B_c.T
    return G @ np.linalg.inv(Hc) @ G.T


def _woodbury_reduction(G, Sigma, B_c):
    """C - C_c via Woodbury: M_c (I + B_cᵀ Σ B_c)⁻¹ M_cᵀ, M_c = G Σ B_c."""
    M_c = G @ Sigma @ B_c
    K = np.linalg.inv(np.eye(B_c.shape[1]) + B_c.T @ Sigma @ B_c)
    return M_c @ K @ M_c.T, M_c


def _oed_trace(G, Sigma, B_c, tol):
    """Implemented decision-variance VoI = tr(W (C - C_c)), W = diag(1/tol²) = T⁻¹·T⁻¹."""
    H = np.linalg.inv(Sigma)
    M = B_c @ B_c.T
    A0 = np.linalg.inv(H)
    A1 = np.linalg.inv(H + M)
    w = 1.0 / (tol ** 2)
    return float(sum(w[q] * (G[q] @ A0 @ G[q] - G[q] @ A1 @ G[q]) for q in range(G.shape[0])))


def _wdec(C, tol):
    Ct = C / tol[:, None] / tol[None, :]
    return float(np.sqrt(max(float(np.linalg.eigvalsh(Ct).max()), 0.0)))


def _candidate(G, Sigma, tol, B_c, C):
    """Full VoI diagnostics for one candidate information square-root B_c."""
    Tinv = np.diag(1.0 / tol)
    C_c = _preposterior_C(G, Sigma, B_c)
    red, M_c = _woodbury_reduction(G, Sigma, B_c)
    SB = Sigma @ B_c                                   # posterior-propagated sensitivity
    dec_frac = (float(np.linalg.norm(G @ SB) ** 2) /
                max(float(np.linalg.norm(SB) ** 2) * float(np.linalg.norm(G, 2) ** 2), 1e-300))
    return {
        "voi_trace": _oed_trace(G, Sigma, B_c, tol),
        "wdec_c": _wdec(C_c, tol),
        "TinvMc_norm": float(np.linalg.norm(Tinv @ M_c)),
        "woodbury_err": float(np.max(np.abs((C - C_c) - red))),
        "max_abs_C_change": float(np.max(np.abs(C - C_c))),
    }


def analyze(product):
    Sigma, G, tol = _load(product)
    dim = Sigma.shape[0]
    C = G @ Sigma @ G.T

    # --- decision-relevant / decision-null subspaces of parameter space (row/kernel of G Σ) ---
    GS = G @ Sigma                                    # (2 x dim); decision-relevant rows
    _, s, Vt = np.linalg.svd(GS, full_matrices=True)
    r = int(np.sum(s > 1e-10 * s.max()))              # decision-relevant rank (<= 2)
    dec_dirs, null_dirs = Vt[:r].T, Vt[r:].T          # col(Σ Gᵀ) and ker(G Σ)
    # sloppiest posterior direction = what a parameter-space D-optimal design most prizes
    evals, evecs = np.linalg.eigh(Sigma)
    top_sigma_dir = evecs[:, -1:]                     # top-Σ-eigenvector (largest posterior variance)
    B_common, B_diff = _sigma_bases(dim, N_PROTEIN)

    scale = 50.0 / SIGMA_OBS                           # a large Fisher amplitude to stress nullity
    cands = {
        "null":       scale * null_dirs[:, : dim - r],   # informs ker(G Σ) -> exact nullity
        "dec":        scale * dec_dirs,                  # informs col(Σ Gᵀ) -> maximal VoI
        "sigma_diff": scale * B_diff,                    # non-identified, ridge-correlated sigma modes
        "sigma_cmn":  scale * B_common,                  # identified, ratio-cancelling sigma common mode
        "d_optimal":  scale * top_sigma_dir,             # sloppiest posterior direction (D-optimal prize)
    }
    w = 1.0 / (tol ** 2)
    out = {"product": product, "dim": dim, "dec_rank": r,
           "cand": {k: _candidate(G, Sigma, tol, B, C) for k, B in cands.items()},
           "wdec_full": _wdec(C, tol),
           # theta-only VoI ceiling: inform ALL of parameter space (C_c -> 0) => VoI -> tr(W C).
           # col(Sigma Gᵀ) is only the maximal EQUAL-RANK decision-aligned experiment, not the global max.
           "theta_only_ceiling_trWC": float(sum(w[q] * C[q, q] for q in range(G.shape[0])))}

    # E-optimal (Delta wdec) zero condition differs from the trace (L-optimal) one: ker(G Sigma) is the
    # STRONG decision-covariance null (any monotone covariance criterion unchanged) and is SUFFICIENT but NOT
    # NECESSARY for V_E=0. Build a candidate whose reduction misses the leading decision eigenspace: M_c ∝ w
    # with w ⟂ T⁻¹ u_1 (u_1 = top eigvec of Cbar=T⁻¹CT⁻¹), so V_E=0 while M_c≠0 and V_L>0.
    Cbar = C / tol[:, None] / tol[None, :]
    ev, U2 = np.linalg.eigh(Cbar)
    u1 = U2[:, -1]
    t1 = u1 / tol                                        # T⁻¹ u_1 (2-vector)
    w = np.array([-t1[1], t1[0]])                        # ⟂ t1 in the 2-D decision space
    GS_pinv = np.linalg.pinv(G @ Sigma)                  # (GΣ)⁺, GΣ row-full-rank ⇒ GΣ(GΣ)⁺ = I_2
    d = GS_pinv @ w
    B_enull = scale * (d / np.linalg.norm(d))[:, None]   # unit direction at the safe amplitude; M_c ∝ w ⟂ T⁻¹u_1
    C_en = _preposterior_C(G, Sigma, B_enull)
    wdec_c_en = _wdec(C_en, tol)
    out["e_null_not_strong_null"] = {
        "Mc_norm": float(np.linalg.norm(G @ Sigma @ B_enull)),   # != 0
        "voi_trace_L": _oed_trace(G, Sigma, B_enull, tol),        # V_L > 0
        "delta_wdec_E": float(out["wdec_full"] - wdec_c_en),      # V_E ~ 0
    }

    # counterexample: VoI is NOT a function of M_c alone. Build B2 = B1 + delta with delta in ker(G Sigma)
    # (so M_c = G Sigma B2 = G Sigma B1 unchanged) but a different self-information Gram B_cᵀΣB_c -> different VoI.
    B1 = scale * dec_dirs[:, :1]
    delta = 3.0 * scale * null_dirs[:, :1]                # G Sigma delta = 0
    B2 = B1 + delta
    out["voi_not_Mc_alone"] = {
        "Mc_diff": float(np.linalg.norm(G @ Sigma @ B1 - G @ Sigma @ B2)),
        "voi_B1": _oed_trace(G, Sigma, B1, tol),
        "voi_B2": _oed_trace(G, Sigma, B2, tol),
    }

    # --- unification with R1: strongly informing (pinning) the sigma-differential block drives the
    #     pre-posterior decision covariance C_c to the R1 variant-D compression covariance C_D, so the
    #     sigma-diff VoI equals the R1 residual tr(W (C - C_D)). Confirms sigma-runs are ACTION-null
    #     (their VoI is the action-null R1 residual), not variance-null. Uses a safe strong scale
    #     (not Fisher->inf, which is numerically ill-conditioned). ---
    w = 1.0 / (tol ** 2)
    Sr, Gr, ui, vi = _dc.rotate_sigma_block(Sigma, G, N_PROTEIN)
    C_D = _dc.variant_d_cov(Sr, Gr, ui, vi)
    voi_R1 = float(sum(w[q] * (C - C_D)[q, q] for q in range(G.shape[0])))     # tr(W (C - C_D))
    voi_sig = out["cand"]["sigma_diff"]["voi_trace"]                           # sigma-diff VoI (strong Fisher)
    C_c_sig = _preposterior_C(G, Sigma, cands["sigma_diff"])
    out["r1_unification"] = {
        "voi_variantD_residual": voi_R1,
        "voi_sigma_diff": voi_sig,
        "match_abs_err": float(abs(voi_R1 - voi_sig)),
        "Cc_vs_CD_fro": float(np.linalg.norm(C_c_sig - C_D)),   # pre-posterior C_c approaches C_D
        "wdec_D": _wdec(C_D, tol),                              # action-null wdec after pinning sigma-diff
    }

    # (iv) discrepancy-floor ceiling: floor common to pre/post cancels in C - C_c
    floor_scale = (1.3 * tol.max()) ** 2               # so wdec_floor ~ the paper's predictive read (>1)
    Sigma_floor = floor_scale * np.eye(2)
    wdec_floor = _wdec(Sigma_floor, tol)
    C_dec_c = _preposterior_C(G, Sigma, cands["dec"])  # best (decision-aligned) experiment
    out["floor"] = {
        "wdec_floor": wdec_floor,
        "wdec_pred_full": _wdec(C + Sigma_floor, tol),
        "wdec_pred_best_c": _wdec(C_dec_c + Sigma_floor, tol),
        "floor_cancels_err": float(np.max(np.abs(((C + Sigma_floor) - (C_dec_c + Sigma_floor)) - (C - C_dec_c)))),
        "ceiling_holds": bool(_wdec(C_dec_c + Sigma_floor, tol) >= wdec_floor - 1e-12),
    }
    return out


def main():
    rows = [analyze(p) for p in PRODUCTS]
    labels = {"null": "ker(GΣ)", "dec": "col(ΣGᵀ)", "sigma_diff": "σ-diff", "sigma_cmn": "σ-common", "d_optimal": "D-opt top-Σ"}
    for a in rows:
        print(f"\n=== {a['product']}  (decision rank r={a['dec_rank']}, wdec_full={a['wdec_full']:.3f}) ===")
        print(f"  {'candidate':14} {'VoI(trace)':>12} {'ΔC_max':>10} {'‖T⁻¹Mc‖':>10} {'wdec_c':>8} {'woodbury_err':>13}")
        for k in ("null", "sigma_cmn", "d_optimal", "sigma_diff", "dec"):
            c = a["cand"][k]
            print(f"  {labels[k]:14} {c['voi_trace']:12.3e} {c['max_abs_C_change']:10.2e} "
                  f"{c['TinvMc_norm']:10.2e} {c['wdec_c']:8.3f} {c['woodbury_err']:13.1e}")
        u_ = a["r1_unification"]
        print(f"  R1 unification: VoI(σ-diff)={u_['voi_sigma_diff']:.4e}  = tr(W(C-C_D))={u_['voi_variantD_residual']:.4e}  "
              f"(err={u_['match_abs_err']:.1e}, ‖C_c-C_D‖={u_['Cc_vs_CD_fro']:.1e})  wdec_D={u_['wdec_D']:.3f}")
        ce_ = a["voi_not_Mc_alone"]
        en_ = a["e_null_not_strong_null"]
        print(f"  θ-only VoI ceiling tr(WC)={a['theta_only_ceiling_trWC']:.3f}  (decision-aligned col(ΣGᵀ) reaches "
              f"{100*a['cand']['dec']['voi_trace']/a['theta_only_ceiling_trWC']:.0f}% of it)")
        print(f"  E-null⊋strong-null: candidate with ‖M_c‖={en_['Mc_norm']:.2e}≠0, V_L(trace)={en_['voi_trace_L']:.3e}>0, "
              f"but Δwdec(E)={en_['delta_wdec_E']:.1e}≈0 (misses leading decision eigenspace)")
        print(f"  VoI≠f(M_c): same M_c (diff={ce_['Mc_diff']:.1e}) gives VoI {ce_['voi_B1']:.3e} vs {ce_['voi_B2']:.3e} "
              f"(self-information Gram matters)")
        f_ = a["floor"]
        print(f"  floor ceiling: wdec_floor={f_['wdec_floor']:.3f}  best-c pred wdec={f_['wdec_pred_best_c']:.3f}  "
              f"ceiling_holds={f_['ceiling_holds']}  cancels_err={f_['floor_cancels_err']:.1e}")
    Path(IN_DIR / "voi_nullity.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {IN_DIR / 'voi_nullity.json'}")


if __name__ == "__main__":
    main()
