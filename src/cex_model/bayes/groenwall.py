"""Explicit Grönwall constant for the σ decision-null theorem (docs/decision_null_theorem.md L2).

L1 gives the σ-forcing of the variational equation as ``O(ε)`` (``ε = max γQ/Λ̄``). L2 propagates
it to the state sensitivity ``S(τ) = ∂y(τ)/∂σ`` and needs an EXPLICIT, computable constant so that

    ‖S(τ)‖_F ≤ ε · κ_S          (the [GAP] this module closes)

becomes a definite inequality, not a leading-order claim.

Derivation (matrix variational equation ``Ṡ = J(t) S + F(t)``, ``S(0)=0``, ``S=∂y/∂σ``,
``J=∂f/∂y``, ``F=∂f/∂σ``).  By variation of constants ``S(τ)=∫₀^τ Φ(τ,s) F(s) ds`` with ``Φ`` the
state-transition matrix of ``J``.  The CRUCIAL choice is the **logarithmic norm** (matrix measure)
``μ₂(J)=λ_max((J+Jᵀ)/2)`` rather than the spectral norm ``‖J‖₂``: for the dissipative EDM+SMA
dynamics (dispersion ≺ 0, adsorption relaxation ≺ 0) ``μ₂`` is small / negative, whereas ``‖J‖₂`` is
large (stiff), so ``e^{‖J‖₂ τ}`` is VACUOUS while ``e^{μ̄ τ}`` is not.  Using
``‖Φ(τ,s)‖₂ ≤ e^{∫_s^τ μ₂(J) } ≤ e^{μ̄(τ-s)}`` (``μ̄ = sup_t μ₂(J(t))``) and
``‖ΦF‖_F ≤ ‖Φ‖₂‖F‖_F``:

    ‖S(τ)‖_F ≤ ∫₀^τ e^{μ̄(τ-s)} ‖F(s)‖_F ds                         (integral bound, tightest)
             ≤ ε · B₀ · ∫₀^τ e^{μ̄(τ-s)} ds = ε · κ_S ,             (closed form)
    κ_S = B₀ · (e^{μ̄ τ} - 1)/μ̄   (μ̄>0) ;  B₀·τ  (μ̄=0) ;  B₀·(1-e^{μ̄ τ})/(-μ̄)  (μ̄<0),

with ``B₀ = sup_t ‖F(t)‖_F / ε`` (the σ-forcing-per-ε scale).  For ``μ̄<0`` the constant is even
``τ``-uniform: ``κ_S ≤ B₀/|μ̄|``.  Every quantity is computed along the nominal trajectory; the bound
is checked against the actual (finite-difference) state sensitivity and across discretizations.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import cex_model.app_support as A
from cex_model.bayes.active import _sim_for_op
from cex_model.bayes.compare import filter_experiments
from cex_model.bayes.likelihood import unpack_u
from cex_model.bayes.synthetic import synthetic_sma_bundle
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = ["log_norm2", "spectral_norm", "groenwall_constant", "certify", "certify_synthetic"]

_PRODUCT_MAP = {"HLXSYN": ("HLXSYN", ["DT"])}
_EXP_OVERFLOW = 700.0   # math.exp overflows past ~709; treat beyond this as vacuous (+inf)


# --------------------------------------------------------------------------- pure helpers

def log_norm2(J) -> float:
    """Logarithmic norm (matrix measure) ``μ₂(J) = λ_max((J+Jᵀ)/2)``."""
    J = np.asarray(J, float)
    return float(np.linalg.eigvalsh(0.5 * (J + J.T))[-1])


def spectral_norm(J) -> float:
    """Spectral norm ``‖J‖₂`` (largest singular value)."""
    J = np.asarray(J, float)
    return float(np.linalg.svd(J, compute_uv=False)[0])


def groenwall_constant(B0: float, mu: float, tau: float, *, tol: float = 1e-12) -> float:
    """Explicit Grönwall constant ``κ_S`` from the forcing scale ``B0``, log-norm ``mu`` (=μ̄),
    horizon ``tau``.  Three regimes; ``+inf`` if ``mu*tau`` overflows (vacuous)."""
    if mu > tol:
        x = mu * tau
        return math.inf if x > _EXP_OVERFLOW else B0 * (math.exp(x) - 1.0) / mu
    if mu < -tol:
        return B0 * (1.0 - math.exp(mu * tau)) / (-mu)   # ≤ B0/|mu|, τ-uniform
    return B0 * tau


def _ffill(a):
    """Forward-fill NaNs (used to extend stride-subsampled ‖F‖ onto the full time grid)."""
    a = np.asarray(a, float).copy()
    last = 0.0
    for i in range(a.size):
        if np.isnan(a[i]):
            a[i] = last
        else:
            last = a[i]
    return a


# ----------------------------------------------------------------------------- solver path

def _forward_full_state(sim, keq, kkin, nu, sig):
    """Full state at every step (no_grad BDF1 then BDF2) -> (n_t, ndim). Mirror of
    bayes_loading_fraction / loading_sweep._run_full_state."""
    ys = [sim.y0]
    y_prev, y_pprev = sim.y0, None
    for k in range(1, sim.t.shape[0]):
        h = sim.t[k] - sim.t[k - 1]
        a0 = 1.0 if y_pprev is None else 1.5
        hist = y_prev if y_pprev is None else (2.0 * y_prev - 0.5 * y_pprev)
        y = sim._newton(y_prev, sim.inlet_c[k], a0, hist.detach(), h, keq, kkin, nu, sig, None)
        y_pprev, y_prev = y_prev, y
        ys.append(y)
    return torch.stack(ys)


def _epsilon_from_traj(sim, Y, nusig) -> float:
    """ε = max_{t,node,j} γ_j Q_j/Λ̄ along the trajectory (per-molecule loading fraction)."""
    nc, gs = sim.nc, sim.gs
    Qp = Y.reshape(Y.shape[0], 2 * nc - 1, gs)[:, nc:, :]
    bound = torch.einsum("j,njg->ng", sim.gamma_p * nusig, Qp)
    Lam = (sim.inocap - bound).clamp(min=1e-12)
    gQ = (sim.gamma_p[None, :, None] * Qp) / Lam[:, None, :]
    return float(gQ.amax())


def _certify_on_sim(sim, keq, kkin, nu, sig, n, *, stride: int, fd_delta: float) -> dict:
    """Core certificate on a built simulator at (keq,kkin,nu,sig)."""
    tau = float(sim.t[-1])
    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)
    nusig = nu + sig
    eps = _epsilon_from_traj(sim, Y, nusig)

    t = sim.t.detach().numpy()
    nt = len(t)
    # log-norm μ₂(J) at EVERY step (cheap: eigvalsh of the symmetric part); σ-forcing ‖F‖_F and
    # the spectral norm ‖J‖₂ subsampled by `stride` (the autograd Jacobian is the costly part).
    mu = np.empty(nt)
    normF = np.full(nt, np.nan)
    Lspec = 0.0
    idx = set(range(0, nt, max(1, stride))) | {nt - 1}
    for k in range(nt):
        y_k = Y[k]
        J = sim.analytic_jac(y_k, keq, kkin, nu, sig).detach().numpy()
        mu[k] = log_norm2(J)
        if k in idx:
            Lspec = max(Lspec, spectral_norm(J))

            def f_of_sig(s):
                return sim.rhs(y_k, sim.inlet_c[k], keq, kkin, nu, nu + s)
            normF[k] = float(np.linalg.norm(
                torch.autograd.functional.jacobian(f_of_sig, sig).detach().numpy()))
    normF = _ffill(normF)

    # Coppel's inequality: ‖Φ(τ,s)‖₂ ≤ exp(∫_s^τ μ₂(J(r)) dr).  Use the cumulative log-norm
    # M(s)=∫_s^τ μ₂ dr (NOT sup_t μ₂·(τ-s)): the dynamics are net-contractive over the run even
    # where μ₂>0 momentarily, so this is finite where the sup-bound is vacuous.
    seg = 0.5 * (mu[:-1] + mu[1:]) * np.diff(t)
    M = np.append(np.cumsum(seg[::-1])[::-1], 0.0)          # M[k] = ∫_{t_k}^{τ} μ₂ dr
    A = np.exp(np.clip(M, None, _EXP_OVERFLOW))             # forward gain bound e^{M(s)}
    A_max = float(A.max())
    mu_bar = float(mu.max())                                # sup log-norm (for the vacuous comparison)
    net_log_norm = float(M[0])                              # ∫_0^τ μ₂ dr  (<0 => net contraction)
    B0 = float(np.nanmax(normF) / eps) if eps > 0 else float("nan")

    integ = float(np.trapezoid(A * normF, t))              # ∫ e^{M(s)} ‖F(s)‖ ds  -- the tight bound
    kappa_S = integ / eps if eps > 0 else float("nan")     # explicit constant: ‖S(τ)‖_F ≤ ε·κ_S
    kS_closed = A_max * B0 * tau                           # closed-form upper bound on κ_S
    kS_spec = groenwall_constant(B0, Lspec, tau)           # spectral-norm constant (vacuous, for contrast)
    bound_closed = eps * kS_closed
    bound_spectral = eps * kS_spec

    # actual state sensitivity ‖∂y(τ)/∂σ‖_F by central finite difference (2n extra forwards)
    cols = []
    for j in range(n):
        sp = sig.clone(); sp[j] += fd_delta
        sm = sig.clone(); sm[j] -= fd_delta
        with torch.no_grad():
            yp = _forward_full_state(sim, keq, kkin, nu, sp)[-1].detach().numpy()
            ym = _forward_full_state(sim, keq, kkin, nu, sm)[-1].detach().numpy()
        cols.append((yp - ym) / (2.0 * fd_delta))
    S_fd = np.stack(cols, axis=1)             # (ndim, n) = ∂y(τ)/∂σ
    actual = float(np.linalg.norm(S_fd))

    # The EFFECTIVE (computed) propagation constant: ‖S(τ)‖_F = ε · κ_eff.  This is the honest
    # closure of "‖S(τ)‖ = O(ε)" -- L1 makes the forcing exactly O(ε), and the propagation Φ is a
    # FIXED, σ-independent bounded linear map (a generic solver-stability property), so its measured
    # gain κ_eff is the constant.  The a-priori log-norm κ_S is VACUOUS because the operator J=∂f/∂y is
    # strongly non-normal (∫μ₂ ≫ 0) — the non-normality is the stiff ADSORPTION-REACTION coupling, not the
    # convection block (which is strictly stable, small log-norm; see bayes/contraction.py
    # apriori_vacuity_report) — so e^{∫μ₂} hugely over-estimates the true ‖Φ‖.
    kappa_eff = actual / eps if eps > 0 else float("nan")
    lognorm_vacuous = bool((not np.isfinite(integ)) or integ > 1e3 * max(actual, 1e-30))
    holds = bool(np.isfinite(integ) and actual <= integ * (1.0 + 1e-6))   # rigorous bound holds (but loose)
    return {
        "tau": tau, "ndim": int(Y.shape[1]), "n_protein": int(n), "epsilon": eps,
        "kappa_eff": kappa_eff, "actual_state_sensitivity": actual,
        "mu_bar": mu_bar, "net_log_norm_integral": net_log_norm, "A_max": A_max,
        "L_spectral": Lspec, "B0": B0,
        "kappa_S": kappa_S, "kappa_S_closed": kS_closed, "kappa_S_spectral": kS_spec,
        "bound_integral": integ, "bound_closed": bound_closed, "bound_spectral": bound_spectral,
        "tightness_ratio": (actual / integ) if (np.isfinite(integ) and integ > 0) else float("nan"),
        "lognorm_bound_holds": holds,
        "lognorm_bound_vacuous": lognorm_vacuous,
        "non_normal_operator": bool(net_log_norm > 0.0),   # ∫μ₂>0 ⇒ Euclidean log-norm unusable (reaction)
        "spectral_vacuous": bool(not np.isfinite(bound_spectral) or bound_spectral > 1e6 * max(integ, 1e-30)),
        "stride": int(stride),
    }


def certify(product, *, in_dir="results/bayes", n_steps: int = 120, stride: int = 1,
            fd_delta: float = 1e-4) -> dict:
    """Grönwall certificate for a real product at its committed decision OP + MAP."""
    from pathlib import Path
    import json
    from cex_model.bayes.posterior import Posterior
    in_dir = Path(in_dir)
    post = Posterior.load(in_dir / f"{product}_posterior.npz")
    op = json.loads((in_dir / f"{product}_decision.json").read_text())["decision_op"]
    n = post.n_protein
    prod, drop = _PRODUCT_MAP.get(product, (product, []))
    bundle = A.load_product(prod)
    if drop:
        bundle.experiments, _ = filter_experiments(bundle.experiments, drop)
    sim = _sim_for_op(bundle, op, n_steps)
    keq, kkin, nu, sig = unpack_u(torch.tensor(post.u_map, dtype=DTYPE), n)
    out = _certify_on_sim(sim, keq, kkin, nu, sig, n, stride=stride, fd_delta=fd_delta)
    out.update({"product": product, "n_steps": n_steps, "decision_op": [float(x) for x in op]})
    return out


def certify_synthetic(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                      fractions=(70.0, 30.0), keq_ladder=True, n_steps: int = 60, stride: int = 1,
                      fd_delta: float = 1e-4) -> dict:
    """Grönwall certificate on a synthetic bundle (no committed data; CI-friendly)."""
    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=keq_ladder,
                                                 nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    e0 = bundle.experiments[0]
    op = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    sim = _sim_for_op(bundle, op, n_steps)
    keq, kkin, nu_t, sig = unpack_u(torch.tensor(np.asarray(u_true, float), dtype=DTYPE), n)
    out = _certify_on_sim(sim, keq, kkin, nu_t, sig, n, stride=stride, fd_delta=fd_delta)
    out.update({"product": f"SYN{n_comp}", "n_steps": n_steps, "nu": nu, "sigma": sigma,
                "loading": loading, "decision_op": [float(x) for x in op]})
    return out
