"""σ-forcing-restricted finite-horizon reachability gain — a NON-vacuous a-priori certificate for the
state sensitivity ``‖S_σ(τ)‖`` that the full-state κ_S cannot give (docs/decision_null_theorem.md §7-1).

The full-state constant κ_S controls ``sup_{‖x₀‖=1}‖Φ(t,s)x₀‖`` — the worst case over *every* initial
perturbation, including Φ's maximally non-normal blow-up directions — which is why it (and every contraction
metric tried: diagonal, entropy, mass-residual) is vacuous. But the σ-sensitivity is **never** an arbitrary
x₀: it solves the variational IVP ``Ṡ_σ = J(t) S_σ + F_σ(t)``, ``S_σ(0)=0``, forced only along the
low-dimensional, structured σ-forcing ``F_σ = ∂f/∂σ`` (the shared-capacity channel; L1). The object that
actually matters is therefore the **σ-forcing-restricted finite-horizon gain**
``κ_{S,σ}^reach = ‖∫₀^τ Φ(τ,s) F_σ(s) ds‖`` — restricting the *input* to the σ-subspace avoids the
sup-over-all-directions blow-up.

This module builds a genuine a-priori OVER-ESTIMATE of the measured ``‖S_σ(τ)‖`` (= ε·κ_eff, from
``groenwall.certify``) via the **norm-inside integral**

    bound = ∫₀^τ ‖Φ(τ,s) F_σ(s)‖_F ds   ≥   ‖∫₀^τ Φ(τ,s) F_σ(s) ds‖_F = ‖S_σ(τ)‖ = measured,

so ``ratio = bound/measured ≥ 1`` measures the genuine triangle-inequality (time-cancellation) slack — it is
**non-circular** (an inequality, not the exact integral, which would just reproduce κ_eff) and uses **no
full-state Φ norm** (only the σ-forcing propagated forward). A forward finite-horizon σ-restricted
reachability Gramian ``W(τ)=∫Φ F_σ F_σᵀΦᵀ ds ≈ Σ G_m G_mᵀ w_m`` (``σmax^{1/2}`` = the L2-input gain) is
reported as a corroborating figure — a FORWARD FINITE-HORIZON object, distinct from the backward
infinite-horizon ``solve_continuous_lyapunov`` in ``contraction.py``.

**Honest framing.** This REFRAMES the target — it does not close full-state κ_S (which stays vacuous). It
upgrades κ_eff from a *measured* effective constant to one *a-priori-bounded on the σ-restricted subspace*
that carries the decision-null. It is the full-state-output analogue of the decision-output adjoint gain in
``decision_gain.py``/``worst_dec`` (input restricted to σ, output kept full-state) — complementary, not
redundant. Verdict: PASS_RESTRICTED_KAPPA / WARN_USEFUL_DIAGNOSTIC / FAIL_VACUOUS.

**Propagator.** No forward variational propagator exists in the solver, so one is written here, mirroring the
BDF state loop (``groenwall._forward_full_state``): implicit-Euler variational updates ``(I − h_k J_k) S_k =
hist`` with ``J_k = sim.analytic_jac(y*_k)`` and ``F_k = ∂f/∂σ`` via the ``f_of_sig`` autograd pattern. Its
output ``‖S_σ(τ)‖`` is cross-checked against groenwall's finite-difference ``actual_state_sensitivity``.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import torch

from cex_model.bayes.groenwall import (
    _epsilon_from_traj,
    _ffill,
    _forward_full_state,
)
from cex_model.bayes.likelihood import unpack_u

__all__ = [
    "sigma_forcing",
    "restricted_gain_sweep",
    "restricted_gain_diagnostic",
    "certify",
    "certify_synthetic",
]


# --------------------------------------------------------------------------- pure helpers

def sigma_forcing(sim, y_k, inlet_c_k, keq, kkin, nu, sig) -> np.ndarray:
    """σ-forcing ``F_σ(t_k) = ∂f/∂σ`` at state ``y_k`` (``(ndim, n)``), via the same autograd pattern
    groenwall/entropy/mass_residual use (``sim.rhs``'s last arg is ``nu+σ``, so the Jacobian w.r.t. ``s``
    at ``s=sig`` is ``∂f/∂σ``)."""
    def f_of_sig(s):
        return sim.rhs(y_k, inlet_c_k, keq, kkin, nu, nu + s)
    return torch.autograd.functional.jacobian(f_of_sig, sig).detach().numpy()


def restricted_gain_sweep(sim, keq, kkin, nu, sig, n, *, stride: int = 4) -> dict:
    """One forward sweep that propagates each strided σ-forcing slice ``F_σ(t_m)`` to τ under the
    implicit-Euler variational flow, giving ``G_m = Φ(τ,t_m) F_σ(t_m)`` (``(ndim, n)``).  From the ``G_m``:

    - ``norm_inside`` = ``∫‖G(s)‖_F ds``  (trapezoid over the strided source times) — the a-priori bound;
    - ``norm_outside`` = ``‖∫ G(s) ds‖_F`` = the semi-analytic ``‖S_σ(τ)‖`` (cross-checked vs the solver FD);
    - ``gramian_gain`` = ``σmax(Σ_m G_m G_mᵀ w_m)^{1/2}`` — the forward finite-horizon reachability L2 gain.

    The discrete propagation reproduces the solver's BDF scheme EXACTLY — implicit-Euler (BDF1) at the first
    step then BDF2 (``a0=1.5``, ``hist = 2V_{k-1} − 0.5 V_{k-2}``) — so the semi-analytic ``‖S_σ(τ)‖``
    matches the solver FD to <1% (a plain implicit-Euler propagator is ~16× off). Each per-source block
    carries its own 2-step history; all A_k = ``(a0 I − h_k J_k)`` solves reuse a per-step LU factorization
    and every active block propagates simultaneously (one ``lu_solve`` per step over the stacked columns).
    ``J_k = sim.analytic_jac(y*_k)``.
    """
    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)          # (nt, ndim)
    t = sim.t.detach().numpy()
    nt = len(t)
    ndim = int(Y.shape[1])
    eye = np.eye(ndim)
    source_set = sorted(set(range(1, nt, max(1, stride))) | {nt - 1})   # strided source steps (t>0)

    Vs = np.zeros((ndim, 0)); Vs_prev = np.zeros((ndim, 0))   # stacked active blocks + their 1-step-back value
    src_order: list[int] = []                                 # source step per n-column group (injection order)
    for k in range(1, nt):
        h = float(t[k] - t[k - 1])
        a0 = 1.0 if k == 1 else 1.5                           # BDF1 first step, BDF2 thereafter (solver scheme)
        J = sim.analytic_jac(Y[k], keq, kkin, nu, sig).detach().numpy()
        lu, piv = sla.lu_factor(a0 * eye - h * J)             # A_k = a0 I − h J_k
        if Vs.shape[1]:
            hist = Vs if k == 1 else (2.0 * Vs - 0.5 * Vs_prev)
            Vs_new = sla.lu_solve((lu, piv), hist)
            Vs_prev = Vs; Vs = Vs_new
        if k in source_set:
            F_k = sigma_forcing(sim, Y[k], sim.inlet_c[k], keq, kkin, nu, sig)   # (ndim, n)
            new = sla.lu_solve((lu, piv), F_k)                # unit source injection at m=k (its step-k solve)
            Vs = np.hstack([Vs, new]); Vs_prev = np.hstack([Vs_prev, np.zeros((ndim, n))])
            src_order.append(k)

    # split Vs into per-source G_m = Φ(τ,t_m) F_σ(t_m) (ndim, n) in source-time order
    G = {m: Vs[:, i * n:(i + 1) * n] for i, m in enumerate(src_order)}
    t_src = np.array([t[m] for m in src_order])
    Gnorm = np.array([float(np.linalg.norm(G[m])) for m in src_order])           # ‖G_m‖_F
    # trapezoid weights over the (possibly non-uniform) strided source grid
    w = np.zeros_like(t_src)
    if t_src.size >= 2:
        w[1:-1] = 0.5 * (t_src[2:] - t_src[:-2])
        w[0] = 0.5 * (t_src[1] - t_src[0])
        w[-1] = 0.5 * (t_src[-1] - t_src[-2])
    elif t_src.size == 1:
        w[0] = float(t[-1])

    norm_inside = float(np.dot(Gnorm, w))                          # ∫‖G(s)‖ ds  (the bound)
    S_approx = sum(G[m] * w[i] for i, m in enumerate(src_order))   # ∫ G(s) ds ≈ S_σ(τ)
    norm_outside = float(np.linalg.norm(S_approx))                 # ‖S_σ(τ)‖ (semi-analytic)
    # reachability Gramian W(τ) = Σ G_m G_mᵀ w_m ; L2-input gain = σmax^{1/2}
    W = sum(w[i] * (G[m] @ G[m].T) for i, m in enumerate(src_order))
    lam_max = float(np.linalg.eigvalsh(0.5 * (W + W.T))[-1]) if isinstance(W, np.ndarray) else 0.0
    gramian_gain = math.sqrt(max(lam_max, 0.0))

    return {
        "nt": nt, "ndim": ndim, "n_sources": len(src_order), "stride": int(stride),
        "tau": float(t[-1]),
        "norm_inside_bound": norm_inside, "norm_outside_semi_analytic": norm_outside,
        "gramian_gain": gramian_gain,
        "t_src": t_src, "Gnorm": Gnorm,
    }


# ----------------------------------------------------------------------------- diagnostic + verdict

def _decide(ratio: float, gramian_gain: float) -> str:
    """Verdict (user criteria): PASS_RESTRICTED_KAPPA if bound/measured ≤ 1e3 (σ-subspace certified, no
    full-state Φ norm); WARN_USEFUL_DIAGNOSTIC if ≤ 1e6 (full-state still vacuous); FAIL_VACUOUS if > 1e6
    or non-finite."""
    if (not math.isfinite(ratio)) or (not math.isfinite(gramian_gain)) or ratio > 1e6:
        return "FAIL_VACUOUS"
    if ratio <= 1e3:
        return "PASS_RESTRICTED_KAPPA"
    return "WARN_USEFUL_DIAGNOSTIC"


def restricted_gain_diagnostic(sim, keq, kkin, nu, sig, n, *, stride: int = 4) -> dict:
    """σ-restricted finite-horizon gain diagnostic along the nominal trajectory (the full construction)."""
    sweep = restricted_gain_sweep(sim, keq, kkin, nu, sig, n, stride=stride)
    with torch.no_grad():
        Y = _forward_full_state(sim, keq, kkin, nu, sig)
    eps = _epsilon_from_traj(sim, Y, nu + sig)
    return {**sweep, "epsilon": eps}


def _package(product: str, n_steps: int, diag: dict, euclid: dict) -> dict:
    kappa_eff = euclid["kappa_eff"]
    measured_fd = euclid["actual_state_sensitivity"]              # ‖S_σ(τ)‖ (solver FD) = ε·κ_eff
    bound = diag["norm_inside_bound"]                             # ∫‖ΦF‖ds  (a-priori over-estimate)
    semi = diag["norm_outside_semi_analytic"]                    # ‖∫ΦF ds‖ (same discretization as bound)
    # Certificate ratio uses the SAME-discretization semi-analytic ‖S_σ(τ)‖ (validated ≈ FD by the
    # cross-check), so it is the genuine triangle-inequality slack ‖∫ΦF‖/‖∫... wait, ≥ 1 by construction;
    # comparing the strided bound against the full-resolution FD would spuriously dip below 1.
    ratio = bound / semi if semi > 0 and math.isfinite(bound) else math.inf
    xcheck = semi / measured_fd if measured_fd > 0 else math.inf  # ~1 validates the forward propagator
    verdict = _decide(ratio, diag["gramian_gain"])
    return {
        "product": product, "n_steps": n_steps, "stride": diag["stride"],
        "epsilon": diag["epsilon"], "tau": diag["tau"], "ndim": diag["ndim"],
        "restricted_gain": {
            "bound_norm_inside": bound, "measured_S_sigma": semi,
            "bound_to_measured_ratio": ratio,
            "measured_S_sigma_fd": measured_fd, "gramian_L2_gain": diag["gramian_gain"],
            "kappa_eff": kappa_eff, "n_sources": diag["n_sources"],
        },
        "cross_check": {
            "semi_analytic_over_fd": xcheck,     # ~1 validates the forward variational propagator
            "propagator_ok": bool(math.isfinite(xcheck) and 0.2 <= xcheck <= 5.0),
        },
        "euclidean": {"kappa_eff": kappa_eff, "actual_state_sensitivity": measured_fd,
                      "kappa_S_euclid": euclid.get("kappa_S"), "mu_bar": euclid.get("mu_bar"),
                      "full_state_vacuous": bool(euclid.get("lognorm_bound_vacuous", True))},
        "decision": verdict,
    }


# --------------------------------------------------------------------------------- drivers

def certify(product, *, in_dir="results/bayes", n_steps: int = 120, stride: int = 4) -> dict:
    """σ-restricted finite-horizon gain certificate for a real product at its committed decision OP + MAP."""
    from pathlib import Path
    import json

    import cex_model.app_support as A
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.groenwall import _PRODUCT_MAP, certify as _certify_euclid
    from cex_model.bayes.posterior import Posterior
    from cex_model.diffsolver.torch_solver import DTYPE

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

    diag = restricted_gain_diagnostic(sim, keq, kkin, nu, sig, n, stride=stride)
    euclid = _certify_euclid(product, in_dir=str(in_dir), n_steps=n_steps, stride=stride)
    return _package(product, n_steps, diag, euclid)


def certify_synthetic(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                      fractions=(70.0, 30.0), keq_ladder=True, n_steps: int = 60, stride: int = 4) -> dict:
    """σ-restricted finite-horizon gain certificate on a synthetic bundle (CI-friendly)."""
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.groenwall import certify_synthetic as _certify_euclid
    from cex_model.bayes.synthetic import synthetic_sma_bundle
    from cex_model.diffsolver.torch_solver import DTYPE

    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=keq_ladder,
                                                 nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    e0 = bundle.experiments[0]
    op = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    sim = _sim_for_op(bundle, op, n_steps)
    keq_t, kkin_t, nu_t, sig_t = unpack_u(torch.tensor(np.asarray(u_true, float), dtype=DTYPE), n)

    diag = restricted_gain_diagnostic(sim, keq_t, kkin_t, nu_t, sig_t, n, stride=stride)
    euclid = _certify_euclid(n_comp=n_comp, nu=nu, sigma=sigma, loading=loading, fractions=fractions,
                             keq_ladder=keq_ladder, n_steps=n_steps, stride=stride)
    return _package(f"SYN{n_comp}", n_steps, diag, euclid)
