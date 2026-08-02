"""Contraction-metric route to a NON-VACUOUS a-priori Grönwall constant (docs/decision_null_theorem.md
§7, item 1) — and the honest finding that, for SMA, the contraction-metric *family tried* is vacuous.

The Euclidean log-norm Grönwall constant for the σ decision-null propagation is vacuous on SMA because
the discretized operator ``J(t)=∂f/∂y`` is **strongly non-normal** — but the non-normality is the stiff
**adsorption-reaction** coupling (``μ₂(reaction)≫μ₂(transport)≈0``; ``apriori_vacuity_report``), NOT the
convection–dispersion transport block, which is strictly stable (``α(Tliq)<0``, ≈−0.02 to −0.12, small log-norm).
A contraction metric ``M ≻ 0`` with log-norm ``μ_M(J)=λ_max(sym(D J D⁻¹))≤0`` would give a finite,
τ-uniform constant; the question is whether one exists with non-vacuous conditioning.

Results here:
  * the abstract CMC class member WITHOUT transport (`bayes/cmc_toy.py`) has Euclidean μ₂ < 0, so the
    a-priori bound is non-vacuous for it directly (`cmc_apriori_certificate`) — a definite inequality for
    normal/contracting (non-transport dissipative) dynamics;
  * for SMA, a per-species diagonal metric (`sma_metric_report`) reduces μ_M but does not reach
    contraction; the operator analysis (`apriori_vacuity_report`) shows WHY: it is pointwise marginally
    stable (``sup_t α≈0`` — partly a smooth-clip subspace artefact, NOT a conserved transport mode) with
    eigenvector conditioning ``cond(V)``≈1e5–1e6, so the CONSTANT/per-species-DIAGONAL family provably
    fails (``inf_P μ_P(J)=α``; Ström 1975) and the per-step-optimal time-varying metric is numerically
    vacuous (κ_S≫κ_eff).  A general off-diagonal / time-varying (Lohmiller–Slotine 1998) metric is **not**
    ruled out in closed form → the a-priori κ_S stays open; the O(ε) law does not depend on it.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg as sla
import torch

from cex_model.bayes.cmc_toy import CMCConfig, integrate as _cmc_integrate
from cex_model.bayes.groenwall import groenwall_constant, log_norm2, _forward_full_state

__all__ = ["weighted_log_norm", "full_log_norm", "optimize_diagonal_metric", "cmc_apriori_certificate",
           "sma_metric_report", "apriori_vacuity_report", "certify_vacuity", "certify_vacuity_synthetic"]


# --------------------------------------------------------------------------- pure helpers

def weighted_log_norm(J, d) -> float:
    """``μ_M(J) = λ_max(sym(D J D⁻¹))`` for the metric ``M = diag(d)``, ``D = diag(√d)`` (d > 0)."""
    J = np.asarray(J, float)
    s = np.sqrt(np.asarray(d, float))
    Jt = (s[:, None] * J) / s[None, :]               # D J D⁻¹
    return float(np.linalg.eigvalsh(0.5 * (Jt + Jt.T))[-1])


def optimize_diagonal_metric(Js, groups, *, maxiter: int = 300):
    """Minimise ``max_t μ_M(J_t)`` over a per-GROUP diagonal metric (one log-weight per group; weights
    are constant within a group, e.g. one per species replicated over spatial nodes).

    ``groups`` is a list of index arrays partitioning the state.  Returns
    ``(weights, sup_mu_metric, sup_mu_euclid)`` — the optimal per-group weights and the worst-case
    metric vs Euclidean log-norm over the sampled ``Js``.  Derivative-free (Nelder–Mead) on log-weights;
    the max-eigenvalue objective is non-smooth but low-dimensional."""
    from scipy.optimize import minimize
    Js = [np.asarray(J, float) for J in Js]
    dim = Js[0].shape[0]
    sup_euclid = max(log_norm2(J) for J in Js)

    def expand(logw):
        d = np.ones(dim)
        for g, lw in zip(groups, logw):
            d[g] = math.exp(lw)
        return d

    def obj(logw):
        d = expand(logw)
        return max(weighted_log_norm(J, d) for J in Js)

    x0 = np.zeros(len(groups))
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6})
    return expand(res.x), float(res.fun), float(sup_euclid)


# --------------------------------------------------------------------- abstract CMC (no transport)

def _cmc_jac(c: CMCConfig, y, tt):
    nusig = c.nu + c.phi
    yk = y.clone().requires_grad_(True)

    def rhs(yy):
        R = (c.R0 - (nusig * c.beta * yy).sum()).clamp(min=1e-6)
        s = c.s_start + (c.s_end - c.s_start) * (tt / c.T)
        return c.g * R ** c.nu * s - c.delta * yy
    return torch.autograd.functional.jacobian(rhs, yk).detach().numpy()


def cmc_apriori_certificate(c: CMCConfig | None = None, *, fd_delta: float = 1e-5) -> dict:
    """A-priori Grönwall constant for the (contracting, no-transport) toy CMC: μ̄ = sup μ₂(J) < 0 ⇒
    ``κ_S = B₀(1−e^{μ̄τ})/|μ̄|`` finite and τ-uniform; verifies ``‖∂y(τ)/∂φ‖ ≤ ε·κ_S`` NON-vacuously."""
    c = c or CMCConfig()
    n, tau = c.n, c.T
    with torch.no_grad():
        t, Y = _cmc_integrate(c, phi_eff=c.phi)
    # per-step Euclidean log-norm and φ-forcing ‖∂f/∂φ‖
    mu, normF = [], []
    for k in range(0, Y.shape[0], 2):
        J = _cmc_jac(c, Y[k], t[k])
        mu.append(log_norm2(J))
        yk = Y[k]
        # ∂f_i/∂φ_j = g_i ν_i R^{ν_i-1} s · (−β_j y_j)
        R = float((c.R0 - (c.nu + c.phi) * c.beta * yk).sum().clamp(min=1e-6))
        s = c.s_start + (c.s_end - c.s_start) * float(t[k]) / c.T
        coef = (c.g * c.nu * R ** (c.nu - 1.0) * s).numpy()           # (n,)
        F = -np.outer(coef, (c.beta * yk).numpy())                    # (n, n) = ∂f/∂φ
        normF.append(float(np.linalg.norm(F)))
    mu = np.array(mu)
    eps = float(((c.beta * Y) / (c.R0 - (Y * (c.nu + c.phi) * c.beta).sum(1, keepdim=True)).clamp(min=1e-6)).amax())
    mu_bar = float(mu.max())
    B0 = float(max(normF)) / eps if eps > 0 else float("nan")
    kappa_S = groenwall_constant(B0, mu_bar, tau)
    bound = eps * kappa_S
    # actual ‖∂y(τ)/∂φ‖_F via central FD
    cols = []
    for j in range(n):
        pp = c.phi.clone(); pp[j] += fd_delta
        pm = c.phi.clone(); pm[j] -= fd_delta
        with torch.no_grad():
            yp = _cmc_integrate(c, phi_eff=pp)[1][-1].numpy()
            ym = _cmc_integrate(c, phi_eff=pm)[1][-1].numpy()
        cols.append((yp - ym) / (2 * fd_delta))
    actual = float(np.linalg.norm(np.stack(cols, 1)))
    return {"model": "toy CMC (no transport)", "tau": tau, "epsilon": eps, "mu_bar": mu_bar,
            "contracting": bool(mu_bar < 0), "B0": B0, "kappa_S": kappa_S, "bound": bound,
            "actual_state_sensitivity": actual,
            "holds": bool(np.isfinite(bound) and actual <= bound * (1 + 1e-6)),
            "tightness_ratio": (actual / bound) if bound > 0 else float("nan"),
            "vacuous": bool((not np.isfinite(bound)) or bound > 1e3 * max(actual, 1e-30))}


# --------------------------------------------------------------------- SMA (convection PDE)

def sma_metric_report(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                      fractions=(70.0, 30.0), n_steps: int = 60, stride: int = 4) -> dict:
    """Optimise a per-species diagonal contraction metric on the SMA variational Jacobian and report
    the worst-case metric vs Euclidean log-norm + the resulting a-priori κ_S factor."""
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.likelihood import unpack_u
    from cex_model.bayes.synthetic import synthetic_sma_bundle
    from cex_model.diffsolver.torch_solver import DTYPE
    from cex_model.bayes.loading_sweep import _run_full_state

    bundle, comps, u = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=True,
                                            nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    e0 = bundle.experiments[0]
    op = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    sim = _sim_for_op(bundle, op, n_steps)
    keq, kkin, nu_t, sig = unpack_u(torch.tensor(np.asarray(u, float), dtype=DTYPE), n)
    with torch.no_grad():
        Y = _run_full_state(sim, keq, kkin, nu_t, sig)
    gs, nspec = sim.gs, 2 * sim.nc - 1                              # state = nspec species × gs nodes
    groups = [np.arange(s * gs, (s + 1) * gs) for s in range(nspec)]   # one weight per species
    Js = [sim.analytic_jac(Y[k], keq, kkin, nu_t, sig).detach().numpy()
          for k in range(0, Y.shape[0], max(1, stride))]
    weights, sup_mu_metric, sup_mu_euclid = optimize_diagonal_metric(Js, groups)
    tau = float(sim.t[-1])
    return {"model": "SMA (convection PDE)", "n_comp": n_comp, "n_steps": n_steps, "tau": tau,
            "n_species": nspec, "sup_mu_euclid": sup_mu_euclid, "sup_mu_metric": sup_mu_metric,
            "species_weights": [float(weights[g[0]]) for g in groups],
            "reduced_to_contracting": bool(sup_mu_metric <= 0),
            "mu_bar_reduction": float(sup_mu_euclid - sup_mu_metric),
            "apriori_kappaS_euclid_vacuous": bool(sup_mu_euclid * tau > 50),
            "apriori_kappaS_metric_factor": (None if sup_mu_metric > 0
                                             else f"finite, τ-uniform ≤ B₀/{abs(sup_mu_metric):.3g}")}


# --------------------------------------------------------------- a-priori vacuity certificate (general)

def full_log_norm(A, P) -> float:
    """Logarithmic norm in a FULL SPD metric ``M = P``:  ``μ_P(A) = λ_max(sym_P(A))`` via the
    symmetric-definite generalized eigenproblem ``½(P A + Aᵀ P) v = λ P v``.  Generalizes
    ``weighted_log_norm`` (diagonal metric) to an arbitrary (off-diagonal) contraction metric."""
    A = np.asarray(A, float); P = np.asarray(P, float)
    S = 0.5 * (P @ A + A.T @ P)
    return float(sla.eigh(S, P, eigvals_only=True)[-1])


def apriori_vacuity_report(sim, keq, kkin, nu, sig, *, t_stride: int = 4) -> dict:
    """Diagnose whether a contraction metric can give a non-vacuous a-priori κ_S for the
    decision-sensitivity propagation ``J(t)=∂f/∂y`` along the nominal trajectory.

    **Scope (honest):** this rules out the CONSTANT and per-species-DIAGONAL contraction-metric family
    *by proof*, and finds the per-step-optimal *time-varying* metric numerically vacuous; a general
    off-diagonal / time-varying (Lohmiller–Slotine) metric is **not** ruled out in closed form, so the
    a-priori κ_S remains *open*.  (The O(ε) decision-null law does **not** depend on κ_S — it rests on the
    exact L1 forcing + the σ-channel STRICT test + the grid-stable measured κ_eff + the CMC second class.)

    PROVEN (constant/diagonal family): a single Grönwall metric must dominate ``sup_t α(J(t))`` because
    ``inf_{P≻0} μ_P(J)=α(J)`` holds per *fixed* J (Ström 1975; Desoer–Haneda).  Here ``sup_t α ≈ 0`` ⇒ no
    constant/diagonal metric yields a τ-uniform contraction rate below ≈0 ⇒ best constant-metric κ_S ~ B₀·τ
    (large for τ~10³–10⁴ s).  *Caveat:* part of ``α≈0`` is a smooth-clip subspace artefact (``analytic_jac``
    scales J by the clip derivative ``s_flat``, zeroing near-saturated columns → spurious ~1e-9 eigenvalues;
    clip-free ``α``≈−1e-4); it is **not** a conserved transport mode — the transport block ``Tliq`` is
    strictly stable (``α(Tliq)<0``, ≈−0.02 to −0.12).

    EMPIRICAL (margin-invariant): the non-normality is the stiff **adsorption reaction** coupling
    (``μ₂(reaction)``≫``μ₂(transport)≈0``; transport has small log-norm / is dissipative, not literally
    normal), with eigenvector conditioning ``cond(V)`` ≈1e5 (synthetic, ~mild) to ≈1e6 (real, strongly
    non-normal).  ``cond(V)`` is the margin/clip-invariant non-normality measure (NOT ``cond(M)≈cond(V)²``,
    which on a marginally-stable J is a 1/margin² near-singular-Lyapunov ARTEFACT that collapses ~3–4
    orders under an infinitesimal shift — so it is *not* quoted as the mechanism).  The per-step-optimal
    Lyapunov metric (``cond_Mt_*`` over the *stable* subset only) gives ``κ_S≳1e3–1e5`` ≫ measured
    κ_eff≈1e-4–1e-3.
    """
    ndim = sim.ndim
    Tliq = sim.T.clone()
    Tliq[:, 0] = Tliq[:, 0] - sim.vb * sim.N0                       # inlet-BC col, as _build_jac_structure
    mu2_transport = log_norm2(Tliq.detach().numpy())
    transport_spectral_abscissa = float(np.linalg.eigvals(Tliq.detach().numpy()).real.max())  # strictly <0

    Y = _forward_full_state(sim, keq, kkin, nu, sig)
    Jc = sim._J_const.detach().numpy()                             # transport (block-diag across C species)
    nt = Y.shape[0]
    tidx = list(range(0, nt, max(1, t_stride)))
    Js = [sim.analytic_jac(Y[k], keq, kkin, nu, sig).detach().numpy() for k in tidx]
    mu2 = np.array([log_norm2(J) for J in Js])
    kb = int(mu2.argmax()); Jk = Js[kb]
    mu2_reaction = log_norm2(Jk - Jc)                              # adsorption coupling at the worst step
    cond_V = float(np.linalg.cond(np.linalg.eig(Jk)[1]))          # eigenvector conditioning (margin-invariant)

    # per-step spectral abscissa α(J(t)) — the τ-uniform contraction FLOOR for a constant/diagonal metric
    # is sup_t α (inf_P μ_P(J)=α per fixed J), and the per-step-optimal Lyapunov-metric conditioning
    # cond(M(t)) (margin-sensitive; over the STABLE subset only).
    alphas, conds, n_nonstable = [], [], 0
    for J in Js:
        a = float(np.linalg.eigvals(J).real.max()); alphas.append(a)
        if a < 0:
            P = sla.solve_continuous_lyapunov(J.T, -np.eye(ndim)); P = 0.5 * (P + P.T)
            e = np.linalg.eigvalsh(P)
            conds.append(float(e[-1] / e[0]) if e[0] > 0 else float("inf"))
        else:
            n_nonstable += 1
    sup_t_alpha = float(max(alphas))
    fin = np.array([c for c in conds if np.isfinite(c)])
    cond_Mt_max = float(fin.max()) if fin.size else float("inf")
    cond_Mt_median = float(np.median(fin)) if fin.size else float("inf")
    kappaS_perstep_lyap = float(np.sqrt(cond_Mt_max)) if np.isfinite(cond_Mt_max) else float("inf")

    tau = float(sim.t[-1])
    return {
        "tau": tau, "ndim": int(ndim), "n_sampled": len(tidx),
        "mu2_transport": mu2_transport, "mu2_full_sup": float(mu2.max()), "mu2_reaction": mu2_reaction,
        "transport_spectral_abscissa": transport_spectral_abscissa,     # <0 (≈−0.02 to −0.12): transport strictly stable
        "nonnormality_is_reaction": bool(mu2_reaction > 10.0 * max(mu2_transport, 1e-12)),
        "sup_t_alpha": sup_t_alpha,                                     # ≈0 ⇒ no constant/diagonal contraction
        "cond_V_worststep": cond_V,                                    # margin-invariant non-normality (1e5–1e6)
        "cond_Mt_median_stable": cond_Mt_median, "cond_Mt_max_stable": cond_Mt_max,  # margin-sensitive, stable subset
        "kappaS_perstep_lyap": kappaS_perstep_lyap, "n_nonstable": int(n_nonstable),
        # vacuity verdict — PROVEN for the constant/diagonal family (sup_t α≈0) AND the per-step-optimal
        # time-varying metric is numerically vacuous (κ_S≫κ_eff); general off-diagonal/TV metric stays open.
        "const_diag_metric_fails": bool(sup_t_alpha > -1.0 / tau),     # margin too small to help over τ
        "apriori_kappaS_family_vacuous": bool((not np.isfinite(kappaS_perstep_lyap)) or kappaS_perstep_lyap > 1e3),
    }


def _vacuity_with_measured(out: dict, kappa_eff: float) -> dict:
    """Attach the measured grid-stable constant (groenwall κ_eff) and the bound-vs-measured slack."""
    out = dict(out)
    out["kappa_eff_measured"] = float(kappa_eff)
    out["apriori_vs_measured_ratio"] = (out["kappaS_perstep_lyap"] / kappa_eff
                                        if kappa_eff > 0 and np.isfinite(out["kappaS_perstep_lyap"])
                                        else float("inf"))
    return out


def certify_vacuity(product, *, in_dir="results/bayes", n_steps: int = 120, t_stride: int = 4,
                    with_kappa_eff: bool = True) -> dict:
    """a-priori-vacuity certificate on a real product's committed operator (mirrors groenwall.certify)."""
    import json
    from pathlib import Path
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.compare import filter_experiments
    from cex_model.bayes.likelihood import unpack_u
    from cex_model.bayes.posterior import Posterior
    from cex_model.bayes.groenwall import _PRODUCT_MAP, certify as _certify_eff
    from cex_model.diffsolver.torch_solver import DTYPE
    import cex_model.app_support as A

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
    out = apriori_vacuity_report(sim, keq, kkin, nu, sig, t_stride=t_stride)
    out.update({"product": product, "n_steps": n_steps,
                "gs": int(sim.gs), "nc": int(sim.nc), "n_protein": int(n)})
    if with_kappa_eff:
        out = _vacuity_with_measured(out, _certify_eff(product, in_dir=str(in_dir), n_steps=n_steps)["kappa_eff"])
    return out


def certify_vacuity_synthetic(*, n_comp: int = 2, nu=4.0, sigma=30.0, loading: float = 8.0,
                              fractions=(70.0, 30.0), n_steps: int = 60, t_stride: int = 4,
                              with_kappa_eff: bool = True) -> dict:
    """a-priori-vacuity certificate on a synthetic bundle (CI-friendly; mirrors certify_synthetic)."""
    from cex_model.bayes.active import _sim_for_op
    from cex_model.bayes.likelihood import unpack_u
    from cex_model.bayes.synthetic import synthetic_sma_bundle
    from cex_model.bayes.groenwall import certify_synthetic as _certify_eff
    from cex_model.diffsolver.torch_solver import DTYPE

    bundle, comps, u_true = synthetic_sma_bundle(n_comp, fractions=list(fractions), keq_ladder=True,
                                                 nu=nu, sigma=sigma, loading=loading)
    n = comps.n_protein
    e0 = bundle.experiments[0]
    op = [e0.loading_g_l, e0.gradient_start_pct, e0.gradient_end_pct, e0.elution_cv]
    sim = _sim_for_op(bundle, op, n_steps)
    keq, kkin, nu_t, sig = unpack_u(torch.tensor(np.asarray(u_true, float), dtype=DTYPE), n)
    out = apriori_vacuity_report(sim, keq, kkin, nu_t, sig, t_stride=t_stride)
    out.update({"product": f"SYN{n_comp}", "n_steps": n_steps, "gs": int(sim.gs), "nc": int(sim.nc)})
    if with_kappa_eff:
        eff = _certify_eff(n_comp=n_comp, nu=nu, sigma=sigma, loading=loading, fractions=fractions,
                           n_steps=n_steps)["kappa_eff"]
        out = _vacuity_with_measured(out, eff)
    return out
