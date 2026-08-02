"""Abstract Common-Mode-Capacity (CMC) system — a NON-SMA second instance of the σ decision-null
theorem (docs/decision_null_theorem.md §3, Theorem 1).

Shared-resource competition model (ecology/chemistry flavour, NO transport/convection — a different
mechanism than the SMA-EDM chromatography PDE):

    R(y; φ) = R0 − Σ_j (ν_j + φ_j) β_j y_j           shared capacity (common-mode); φ enters ONLY here
    ẏ_i     = g_i · R(y)^{ν_i} · s(t) − δ_i y_i       consumer i; (g_i, ν_i, δ_i) = selectivity, direct
    P       = ∫_W y_1 / ∫_W Σ_j y_j                   ratio QoI (relative abundance of the target)

This is the CMC class of Theorem 1: the block φ ("interference"/steric analogue) enters the dynamics
only through the shared scalar capacity R, with per-capita occupation ε_j = β_j y_j / R.  We verify, on
this non-SMA model, the duality the theorem predicts — φ is simultaneously
  * decision-null   O(ε)   (‖∂P/∂φ‖ ∝ ε), and
  * Fisher-null     O(ε²)  (‖∂y_obs/∂φ‖² ∝ ε²),
controlled by the SAME ε, with the selectivity block (g, ν) FROZEN.  As in the SMA σ-channel test the
clean isolated check freezes the state at the operating point and scales only the φ-coupling
(reparametrisation φ_eff = φ_nom + s·δ, differentiated at δ=0), so the relations are exact through the
origin.  Holding on a second, structurally-different model is the evidence that the σ decision-null is a
CLASS property, not an SMA artefact.
"""

from __future__ import annotations

import numpy as np
import torch

from cex_model.bayes.loading_sweep import linear_fit
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = ["CMCConfig", "integrate", "decision_and_obs", "epsilon", "class_level_duality",
           "physical_eps_sweep"]


class CMCConfig:
    """Parameters of the shared-resource CMC system (defaults give a clean 2-consumer run)."""

    def __init__(self, *, g=(1.0, 0.8), nu=(3.0, 2.0), delta=(0.5, 0.5), phi=(5.0, 5.0),
                 beta=(0.1, 0.1), R0=1.0, T=10.0, n_steps=100, s_start=0.1, s_end=1.0,
                 window=(0.4, 0.8), target=0):
        self.g = torch.tensor(g, dtype=DTYPE)
        self.nu = torch.tensor(nu, dtype=DTYPE)
        self.delta = torch.tensor(delta, dtype=DTYPE)
        self.phi = torch.tensor(phi, dtype=DTYPE)
        self.beta = torch.tensor(beta, dtype=DTYPE)
        self.R0, self.T, self.n_steps = float(R0), float(T), int(n_steps)
        self.s_start, self.s_end = float(s_start), float(s_end)
        self.window, self.target = window, int(target)
        self.n = len(g)


def _s_of(tt, c):
    return c.s_start + (c.s_end - c.s_start) * (tt / c.T)


def integrate(c: CMCConfig, *, g=None, nu=None, delta=None, phi_eff=None):
    """RK4 fixed-step forward.  ``phi_eff`` (= φ) enters ONLY the shared capacity ``R = R0 −
    Σ(ν+φ)βy``; the selectivity (g, ν, δ) enter directly (ν also in the exponent ``R^ν``).  Params
    default to ``c``'s; pass grad-enabled overrides to differentiate w.r.t. them."""
    g = c.g if g is None else g
    nu = c.nu if nu is None else nu
    delta = c.delta if delta is None else delta
    phi_eff = c.phi if phi_eff is None else phi_eff
    nusig = nu + phi_eff
    nt, dt = c.n_steps, c.T / c.n_steps
    t = torch.linspace(0.0, c.T, nt + 1, dtype=DTYPE)
    y = torch.zeros(c.n, dtype=DTYPE)
    ys = [y]

    def rhs(yy, tt):
        R = (c.R0 - (nusig * c.beta * yy).sum()).clamp(min=1e-6)   # shared capacity; φ only here
        return g * R ** nu * _s_of(tt, c) - delta * yy
    for k in range(nt):
        tk = t[k]
        k1 = rhs(y, tk)
        k2 = rhs(y + 0.5 * dt * k1, tk + 0.5 * dt)
        k3 = rhs(y + 0.5 * dt * k2, tk + 0.5 * dt)
        k4 = rhs(y + dt * k3, tk + dt)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y)
    return t, torch.stack(ys)                                      # (nt+1, n)


def _window_idx(c: CMCConfig):
    a = int(c.window[0] * c.n_steps)
    b = int(c.window[1] * c.n_steps)
    return a, b


def decision_and_obs(c: CMCConfig, *, s: float = 1.0, delta_perturb: torch.Tensor | None = None):
    """Returns ``(purity, target_collected, obs)`` with φ_eff = φ_nom + s·δ in the capacity (δ=0 ⇒ the
    forward is φ_nom for every s ⇒ state frozen; ∂/∂δ = s·∂/∂φ).  ``purity`` = ratio QoI,
    ``target_collected`` = the NON-ratio numerator (for the ratio-cancellation comparison), ``obs`` =
    the flattened trajectory (the Fisher observations)."""
    dp = torch.zeros(c.n, dtype=DTYPE) if delta_perturb is None else delta_perturb
    t, Y = integrate(c, phi_eff=c.phi + s * dp)
    a, b = _window_idx(c)
    win = Y[a:b + 1]
    tw = t[a:b + 1]
    coll = torch.trapezoid(win, tw, dim=0)                        # ∫_W y_i
    total = coll.sum()
    purity = coll[c.target] / total.clamp(min=1e-12)
    return purity, coll[c.target], Y.reshape(-1)


def epsilon(c: CMCConfig) -> float:
    """ε = max_{t,j} β_j y_j / R along the nominal trajectory (per-capita resource occupation)."""
    with torch.no_grad():
        _, Y = integrate(c, phi_eff=c.phi)
        R = (c.R0 - (Y * (c.nu + c.phi) * c.beta).sum(dim=1)).clamp(min=1e-6)   # (nt+1,)
        occ = (c.beta * Y) / R[:, None]
    return float(occ.amax())


def _phi_jac(c: CMCConfig, s: float, which: str):
    """∂(quantity)/∂δ at δ=0  (= s·∂/∂φ, state frozen).  which ∈ {purity, target, obs}.
    The vector ``obs`` uses forward-mode (few inputs, many outputs); it is a plain RK4 (no implicit
    diff), so forward-mode is valid here (unlike the SMA IFT solver)."""
    def f(dp):
        purity, tgt, obs = decision_and_obs(c, s=s, delta_perturb=dp)
        return {"purity": purity, "target": tgt, "obs": obs}[which]
    x = torch.zeros(c.n, dtype=DTYPE)
    if which == "obs":
        return torch.autograd.functional.jacobian(f, x, strategy="forward-mode", vectorize=True).detach().numpy()
    return torch.autograd.functional.jacobian(f, x).detach().numpy()


def class_level_duality(c: CMCConfig | None = None, *, scales=(0.25, 0.5, 1.0, 1.5, 2.0)) -> dict:
    """Verify the Theorem-1 duality on this non-SMA CMC model: decision-null O(ε), Fisher-null O(ε²),
    same ε, selectivity frozen, ratio cancellation.  Each scale is a real autograd pass."""
    c = c or CMCConfig()
    eps_nom = epsilon(c)
    n = c.n
    a, b = _window_idx(c)
    # selectivity (ψ = g, ν) decision sensitivity at nominal — the O(1) CONTRAST to φ's O(ε) (it must
    # be far larger than the φ sensitivity; "frozen across the φ-sweep" is by construction).
    def purity_of(g_nu):
        _, Y = integrate(c, g=g_nu[:n], nu=g_nu[n:], delta=c.delta, phi_eff=c.phi)
        coll = Y[a:b + 1].sum(dim=0)
        return coll[c.target] / coll.sum().clamp(min=1e-12)
    g_nu0 = torch.cat([c.g, c.nu])
    psi_sens = float(np.linalg.norm(torch.autograd.functional.jacobian(purity_of, g_nu0).detach().numpy()))

    pts = []
    for s in sorted({round(float(x), 6) for x in scales}):
        G_dec = _phi_jac(c, s, "purity")                          # ∝ s·∂P/∂φ
        G_tot = _phi_jac(c, s, "target")                          # ∝ s·∂N/∂φ (non-ratio)
        G_obs = _phi_jac(c, s, "obs")                             # ∝ s·∂y_obs/∂φ
        pts.append({"sigma_scale": s, "epsilon": s * eps_nom,
                    "decision_sens": float(np.linalg.norm(G_dec)),
                    "nonratio_sens": float(np.linalg.norm(G_tot)),
                    "fisher_scale": float(np.linalg.norm(G_obs) ** 2)})

    eps = [p["epsilon"] for p in pts]
    dec_fit = linear_fit(eps, [p["decision_sens"] for p in pts])           # decision-null: ∝ ε
    fish_fit = linear_fit([e * e for e in eps], [p["fisher_scale"] for p in pts])   # Fisher-null: ∝ ε²
    # ratio cancellation: purity (ratio) relative sensitivity vs the non-ratio numerator's
    with torch.no_grad():
        P0, N0, _ = decision_and_obs(c)
    nom = next(p for p in pts if abs(p["sigma_scale"] - 1.0) < 1e-9)
    ratio_rel = nom["decision_sens"] / float(P0)
    nonratio_rel = nom["nonratio_sens"] / float(N0)

    strict = bool(dec_fit["r2"] > 0.999 and fish_fit["r2"] > 0.999 and dec_fit["slope"] > 0
                  and abs(dec_fit["intercept"]) < 1e-6 * max(p["decision_sens"] for p in pts))
    return {
        "model": "shared-resource-competition (non-SMA CMC)", "n": n, "epsilon_nominal": eps_nom,
        "psi_decision_sens_frozen": psi_sens,
        "decision_null_fit_vs_eps": dec_fit, "fisher_null_fit_vs_eps2": fish_fit,
        "ratio_rel_sens": ratio_rel, "nonratio_rel_sens": nonratio_rel,
        "ratio_cancellation_factor": (ratio_rel / nonratio_rel) if nonratio_rel > 0 else float("nan"),
        "verdict": "STRICT-CONFIRMED" if strict else "CHECK",
        "points": pts,
    }


# --------------------------------------------------------------------------- genuine (non-frozen) test

# Gentle capacity (R0) multipliers (±~22%): lowering R0 raises the per-capita occupation ε PHYSICALLY
# (the state moves), the analogue of the SMA gentle Λ₀ capacity sweep (``loading_sweep.sweep_capacity``,
# ±15%).  A wider sweep keeps the decision-null O(ε) linear but the obs O(ε²) degrades as the trajectory
# reshapes far from nominal (the same state-reshaping confound the SMA Λ₀ sweep stays gentle to avoid).
DEFAULT_R0_SCALES = (1.22, 1.12, 1.0, 0.9, 0.8)


def _purity_obs_of_phi(c: CMCConfig, phi: torch.Tensor):
    """Forward at the PARAMETER ``φ`` (a real value, not a frozen δ-perturbation): integrates the state at
    ``φ`` and returns ``(purity, obs_flat)`` — for differentiating the decision / observations w.r.t. the
    ACTUAL ``φ`` block (so the trajectory responds to ``φ``, unlike the δ=0 frozen-state check)."""
    t, Y = integrate(c, phi_eff=phi)
    a, b = _window_idx(c)
    coll = torch.trapezoid(Y[a:b + 1], t[a:b + 1], dim=0)
    purity = coll[c.target] / coll.sum().clamp(min=1e-12)
    return purity, Y.reshape(-1)


def _phi_grads_physical(c: CMCConfig) -> tuple[float, float]:
    """``(‖∂P/∂φ‖, ‖∂y_obs/∂φ‖²)`` at the REAL operating state of ``c`` — ``φ`` is NOT frozen, the
    trajectory moves with ``c`` (here, with its ``R0``).  Reverse-mode for the scalar purity, forward-mode
    for the vector obs (few inputs).  This is the genuine derivative the frozen-state δ-check replaces by a
    state-independent ``s·∂/∂φ``."""
    def purity_of(phi):
        return _purity_obs_of_phi(c, phi)[0]

    def obs_of(phi):
        return _purity_obs_of_phi(c, phi)[1]

    g_dec = torch.autograd.functional.jacobian(purity_of, c.phi).detach().numpy()
    g_obs = torch.autograd.functional.jacobian(
        obs_of, c.phi, strategy="forward-mode", vectorize=True).detach().numpy()
    return float(np.linalg.norm(g_dec)), float(np.linalg.norm(g_obs) ** 2)


def physical_eps_sweep(*, g=(1.0, 0.8), nu=(3.0, 2.0), delta=(0.5, 0.5), phi=(5.0, 5.0),
                       beta=(0.1, 0.1), R0=1.0, T=10.0, n_steps=100, window=(0.4, 0.8), target=0,
                       r0_scales=DEFAULT_R0_SCALES, r2_min: float = 0.95, verbose: bool = True) -> dict:
    """GENUINE (non-frozen) class-level ε-scaling test of the CMC duality on the non-SMA model.

    :func:`class_level_duality` freezes the forward state at ``δ=0`` and scales only the φ-coupling, so
    ``decision_sens = s·‖∂P/∂φ‖`` and ``ε = s·ε_nom`` are BOTH linear in ``s`` by the chain-rule identity —
    its ``R²=1.000`` confirms that identity on ANY model, NOT that the O(ε)/O(ε²) duality is a genuine
    scaling LAW of a moving state.  Here we instead vary a PHYSICAL capacity knob ``R0`` (lowering it
    raises occupation), recompute ``ε`` along the actually-changed trajectory, and take ``‖∂P/∂φ‖`` /
    ``‖∂y_obs/∂φ‖²`` by real autograd w.r.t. ``φ`` at each moved state.  If the duality is a real law it is
    ``‖∂P/∂φ‖ ∝ ε`` and ``‖∂y_obs/∂φ‖² ∝ ε²`` with strong (but NOT tautologically exact) linearity — the
    direct analogue of the SMA Λ₀ sweep (R²≈0.94 there), which the frozen-state identity cannot give.
    """
    pts = []
    for sc in sorted({round(float(x), 6) for x in r0_scales}, reverse=True):
        c = CMCConfig(g=g, nu=nu, delta=delta, phi=phi, beta=beta, R0=R0 * sc, T=T, n_steps=n_steps,
                      window=window, target=target)
        eps = epsilon(c)
        dec, fish = _phi_grads_physical(c)
        pts.append({"r0_scale": sc, "R0": float(R0 * sc), "epsilon": eps,
                    "decision_sens": dec, "fisher_scale": fish})
        if verbose:
            print(f"  R0×{sc:5.2f}  ε={eps*100:7.3f}%  ‖∂P/∂φ‖={dec:.4e}  ‖∂y_obs/∂φ‖²={fish:.4e}")

    eps = [p["epsilon"] for p in pts]
    dec_fit = linear_fit(eps, [p["decision_sens"] for p in pts])                       # O(ε)
    fish_fit = linear_fit([e * e for e in eps], [p["fisher_scale"] for p in pts])      # O(ε²)
    genuine = bool(dec_fit["r2"] > r2_min and dec_fit["slope"] > 0
                   and fish_fit["r2"] > r2_min and fish_fit["slope"] > 0)
    return {
        "model": "shared-resource-competition (non-SMA CMC) — PHYSICAL R0 sweep, state NOT frozen",
        "knob": "R0_capacity", "n": len(g), "r0_scales": list(r0_scales), "r2_min": r2_min,
        "decision_null_fit_vs_eps": dec_fit, "fisher_null_fit_vs_eps2": fish_fit,
        "verdict": "GENUINE-CONFIRMED" if genuine else "CHECK",
        "note": ("non-frozen physical sweep: ε varies through a moving state, so R² is NOT constrained to "
                 "1 by construction (contrast class_level_duality's frozen-state chain-rule identity)."),
        "points": pts,
    }
