"""Differentiable EDM+SMA forward solver in PyTorch (fixed-step BDF).

Mirrors ``cex_model.sma.tran_ode_rhs`` (EDM convection-dispersion + SMA isotherm/kinetics)
in torch tensors, integrated by a fixed-step BDF (BDF1 first step, then BDF2). Each implicit
step is solved by Newton under ``no_grad`` and made differentiable w.r.t. the SMA parameters
by the implicit-function theorem (a single linear correction with a detached Jacobian) — so
the graph does NOT unroll Newton, and ``d(loss)/d(keq,kkin,nu,sigma)`` flows at ~one-solve cost.

The non-negativity uses the smooth clip (``0.5*(Y+sqrt(Y^2+eps^2))``) — the same differentiable
path the stiff scipy solvers use — so the RHS is smooth (the hard ``max(Y,0)`` kink would block
both Newton and autograd).

Constants (transport operator T, inlet schedule, loading-correction gamma) do not depend on the
calibrated parameters and are precomputed in numpy via the existing builders, then frozen as torch
tensors. State packing matches sma.py exactly: Fortran-order ``[C (gs x nc), Q (gs x npr)]``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as _ckpt

from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet
from cex_model.corrections import LoadingCorrection, LinearLoadingCorrection
from cex_model.edm import generate_edm_matrices
from cex_model.gradients import InletProfile
from cex_model.sma import MOL_TO_G_L

DTYPE = torch.float64


def params_from_components(components: ComponentSet, *, requires_grad: bool = False):
    """Protein SMA parameter tensors (physical units, salt excluded): keq, kkin, nu, sigma."""
    keq = torch.tensor(components.keq_array()[1:], dtype=DTYPE, requires_grad=requires_grad)
    kkin = torch.tensor(components.kkin_array()[1:], dtype=DTYPE, requires_grad=requires_grad)
    nu = torch.tensor(components.nu_array()[1:], dtype=DTYPE, requires_grad=requires_grad)
    sigma = torch.tensor(components.sigma_array()[1:], dtype=DTYPE, requires_grad=requires_grad)
    return keq, kkin, nu, sigma


class TorchSimulator:
    """Fixed-step differentiable EDM+SMA integrator for one experiment."""

    def __init__(
        self,
        column: ColumnParameters,
        components: ComponentSet,
        inlet: InletProfile,
        loading_g_l: float,
        correction: LoadingCorrection | None = None,
        *,
        n_steps: int = 600,
        smooth_eps: float = 1e-8,
        newton_iters: int = 12,
        newton_tol: float = 1e-9,
    ) -> None:
        correction = correction or LinearLoadingCorrection()
        self.gs = int(column.grid_size)
        self.nc = int(components.n_total)
        self.npr = self.nc - 1
        self.ndim = self.gs * (2 * self.nc - 1)
        self.smooth_eps = float(smooth_eps)
        self.newton_iters = int(newton_iters)
        self.newton_tol = float(newton_tol)
        self.rt_min = float(column.rt)  # residence time per CV (min); for the differentiable op inlet
        self.n_steps = int(n_steps)

        edm = generate_edm_matrices(column)
        T = edm.T if edm.T is not None else edm.N @ edm.A_C
        self.T = torch.tensor(np.asarray(T), dtype=DTYPE)            # (gs, gs)
        self.N0 = torch.tensor(np.asarray(edm.N[:, 0]), dtype=DTYPE)  # (gs,)
        self.vb = float(column.velocity / column.epsbed)
        self.c1 = float((1.0 - column.epsbed) / column.epsbed)
        self.inocap = float(column.inocap)
        # gamma (loading correction) is independent of the calibrated SMA params.
        gamma = np.asarray(correction.gamma(loading_g_l, self.nc), dtype=float)
        self.gamma_p = torch.tensor(gamma[1:], dtype=DTYPE)          # (npr,)
        self.eye = torch.eye(self.ndim, dtype=DTYPE)
        self._build_jac_structure()

        # Fixed time grid + precomputed inlet (constant w.r.t. params).
        self.t = torch.linspace(0.0, float(inlet.duration), self.n_steps_grid(n_steps), dtype=DTYPE)
        ic = np.stack([inlet.inlet_concentrations(float(tk)) for tk in self.t.numpy()])  # (n, nc)
        self.inlet_c = torch.tensor(ic, dtype=DTYPE)                 # (n, nc)
        # Initial state: equilibrated salt everywhere, proteins/solid zero (matches simulator).
        y0 = np.zeros(self.ndim)
        y0[: self.gs] = float(inlet.segments[0, 0])
        self.y0 = torch.tensor(y0, dtype=DTYPE)
        self.feed_end_t = float(inlet.segments[0, 3])

    @staticmethod
    def n_steps_grid(n_steps: int) -> int:
        return int(n_steps) + 1

    def _build_jac_structure(self) -> None:
        """Precompute the constant transport block of J and the reaction-entry indices.

        The Jacobian = constant transport (the per-component Tliq blocks, state-independent)
        + per-step reaction couplings (all node-wise diagonals). Precomputing the former and
        the fixed (row, col) index lists turns per-step assembly into one vectorised scatter
        instead of ~grid_size index_put_ calls (the profiled bottleneck, 5 ms -> <1 ms).
        """
        gs, nc, npr = self.gs, self.nc, self.npr
        Tliq = self.T.clone()
        Tliq[:, 0] = Tliq[:, 0] - self.vb * self.N0
        Jc = torch.zeros(self.ndim, self.ndim, dtype=DTYPE)
        for c in range(nc):
            Jc[c * gs:c * gs + gs, c * gs:c * gs + gs] = Tliq
        self._J_const = Jc
        d = torch.arange(gs)
        R, C = [], []
        for c in range(1, nc):              # E1 dC_c/dC_c
            R.append(c * gs + d); C.append(c * gs + d)
        for q in range(npr):                # E2 dQ_q/dC_{q+1}
            R.append((nc + q) * gs + d); C.append((q + 1) * gs + d)
        for q in range(npr):                # E3 dQ_q/dC_0
            R.append((nc + q) * gs + d); C.append(d)
        for q in range(npr):                # E4 dC_{q+1}/dC_0
            R.append((q + 1) * gs + d); C.append(d)
        for q in range(npr):                # E5 dQ_q/dQ_jj
            for jj in range(npr):
                R.append((nc + q) * gs + d); C.append((nc + jj) * gs + d)
        for q in range(npr):                # E6 dC_{q+1}/dQ_jj
            for jj in range(npr):
                R.append((q + 1) * gs + d); C.append((nc + jj) * gs + d)
        self._react_rows = torch.cat(R)
        self._react_cols = torch.cat(C)

    # --- RHS (mirrors sma.tran_ode_rhs, smooth-clip path) ---
    def _smooth(self, Y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (Y + torch.sqrt(Y * Y + self.smooth_eps * self.smooth_eps))

    def rhs(self, y, inlet_c, keq_p, kkin_p, nu_p, nusig_p, iso_nn=None):
        gs, nc = self.gs, self.nc
        Y = self._smooth(y.reshape(2 * nc - 1, gs).t())  # Fortran reshape -> (gs, 2nc-1)
        C = Y[:, :nc]
        Qp = Y[:, nc:]
        dC = self.T @ C
        dC = dC + torch.outer(self.N0, self.vb * (inlet_c - C[0, :]))
        sma_sum = torch.clamp(self.inocap - (self.gamma_p * Qp) @ nusig_p, min=1e-12)  # (gs,)
        csalt = C[:, 0]
        ads = keq_p * sma_sum[:, None] ** nu_p[None, :] * C[:, 1:]
        if iso_nn is not None:
            # DPSOL gray-box: a shared NN corrects the adsorption isotherm (Stage B). Zero-init
            # => (1+r)==1 at start, so this is identity until trained. Transport/mass balance
            # and the mechanistic params are untouched.
            from cex_model.diffsolver.isotherm_nn import adsorption_features
            feats = adsorption_features(C[:, 1:], Qp, csalt, sma_sum, keq_p, nu_p,
                                        nusig_p - nu_p, self.inocap)
            ads = ads * torch.clamp(1.0 + iso_nn(feats), min=0.0)
        des = self.gamma_p * Qp * csalt[:, None] ** nu_p[None, :]
        dQ = (ads - des) / kkin_p
        dC = torch.cat([dC[:, :1], dC[:, 1:] - self.c1 * dQ], dim=1)
        return torch.cat([dC, dQ], dim=1).t().reshape(-1)  # Fortran ravel

    def _jac(self, y, inlet_c, keq_p, kkin_p, nu_p, nusig_p) -> torch.Tensor:
        """Dense ∂f/∂y by autograd (detached; reference / fallback for the analytic one)."""
        y_ = y.detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(
            lambda yy: self.rhs(yy, inlet_c, keq_p.detach(), kkin_p.detach(),
                                nu_p.detach(), nusig_p.detach()),
            y_, create_graph=False, vectorize=True,
        )
        return J.detach()

    def analytic_jac(self, y, keq_p, kkin_p, nu_p, sigma_p) -> torch.Tensor:
        """Dense ∂f/∂y in closed form (torch port of sma.tran_ode_jac, smooth-clip path).

        ~grid_size× cheaper than the autograd Jacobian (no per-row backward pass), which
        is what makes the implicit BDF fast enough for full-scale calibration. Returned
        detached (only used to form the Newton/implicit linear operator).
        """
        gs, nc, npr = self.gs, self.nc, self.npr
        with torch.no_grad():
            nusig_p = nu_p + sigma_p
            Yraw = y.reshape(2 * nc - 1, gs).t()
            root = torch.sqrt(Yraw * Yraw + self.smooth_eps * self.smooth_eps)
            Yc = 0.5 * (Yraw + root)
            s_flat = (0.5 * (1.0 + Yraw / root)).t().reshape(-1)  # Fortran ravel
            C = Yc[:, :nc]
            Q = Yc[:, nc:]
            gp = self.gamma_p
            csalt = C[:, 0]
            raw_sma = self.inocap - (gp * Q) @ nusig_p
            active = (raw_sma > 1e-12).to(DTYPE)
            sma_sum = torch.clamp(raw_sma, min=1e-12)
            pa = sma_sum[:, None] ** nu_p[None, :]
            sma_m1 = sma_sum[:, None] ** (nu_p[None, :] - 1.0)
            cs_pow = csalt[:, None] ** nu_p[None, :]
            cs_m1 = csalt[:, None] ** (nu_p[None, :] - 1.0)
            ads_dC = keq_p * pa / kkin_p                  # (gs, npr)
            dQ_dC0 = -gp * Q * nu_p * cs_m1 / kkin_p
            ads_dsma = keq_p * C[:, 1:] * nu_p * sma_m1
            desorp_diag = gp * cs_pow / kkin_p
            # coup[:, q, jj] = -(ads_dsma[:,q]/kkin_p[q]) * gp[jj]*nusig_p[jj] * active
            #                  (minus desorp_diag[:,q] on the j==q diagonal)
            coup = (-(ads_dsma / kkin_p)[:, :, None]
                    * (gp * nusig_p)[None, None, :] * active[:, None, None])
            ar = torch.arange(npr)
            coup[:, ar, ar] = coup[:, ar, ar] - desorp_diag
            c1 = self.c1
            # values concatenated in the exact precomputed (E1..E6) index order
            vals = torch.cat([
                (-c1 * ads_dC).t().reshape(-1),               # E1 dC_c/dC_c
                ads_dC.t().reshape(-1),                       # E2 dQ_q/dC_{q+1}
                dQ_dC0.t().reshape(-1),                       # E3 dQ_q/dC_0
                (-c1 * dQ_dC0).t().reshape(-1),               # E4 dC_{q+1}/dC_0
                coup.permute(1, 2, 0).reshape(-1),            # E5 dQ_q/dQ_jj
                (-c1 * coup).permute(1, 2, 0).reshape(-1),    # E6 dC_{q+1}/dQ_jj
            ])
            J = self._J_const.clone()
            J.index_put_((self._react_rows, self._react_cols), vals, accumulate=True)
            return J * s_flat[None, :]  # chain rule for the smooth clip (J @ diag(s))

    def _newton(self, y_guess, inlet_c, a0, hist, h, keq_p, kkin_p, nu_p, sigma_p, iso_nn=None):
        """Solve a0*y - h*f(y) = hist for y (detached, no graph).

        Modified Newton: factor the iteration matrix ``a0*I - h*∂f/∂y`` once at the initial
        guess and reuse the LU across iterations (the Jacobian barely changes over a step) —
        a triangular solve per iter instead of a fresh build+factor, ~3x fewer flops/step.
        """
        kd, kkd, nd, sd = (keq_p.detach(), kkin_p.detach(), nu_p.detach(), sigma_p.detach())
        nsd = nd + sd
        y = y_guess.detach().clone()

        def factor(at):
            J = self.analytic_jac(at, kd, kkd, nd, sd)
            return torch.linalg.lu_factor(a0 * self.eye - h * J)

        with torch.no_grad():
            LU, piv = factor(y)
            prev = None
            for _ in range(self.newton_iters):
                f = self.rhs(y, inlet_c, kd, kkd, nd, nsd, iso_nn)
                G = a0 * y - h * f - hist
                dy = torch.linalg.lu_solve(LU, piv, (-G).unsqueeze(1)).squeeze(1)
                nrm = torch.linalg.vector_norm(dy)
                # The frozen-Jacobian (chord) step can diverge for a stiff/nonlinear
                # coarse step; re-linearise (full Newton) when it does, then retry.
                if (not torch.isfinite(nrm)) or (prev is not None and nrm > prev):
                    LU, piv = factor(y)
                    dy = torch.linalg.lu_solve(LU, piv, (-G).unsqueeze(1)).squeeze(1)
                    nrm = torch.linalg.vector_norm(dy)
                y = y + dy
                prev = nrm
                if nrm <= self.newton_tol * (1.0 + torch.linalg.vector_norm(y)):
                    break
        return y

    def _diff_step(self, y_prev, y_pprev, keq_p, kkin_p, nu_p, sigma_p, h, ic, a0, iso_nn):
        """One implicitly-differentiated BDF step -> y (value == Newton solution; IFT gradient).

        Factored out of :meth:`integrate` so a step can be wrapped in gradient checkpointing.
        ``a0`` selects BDF1 (1.0) / BDF2 (1.5); ``y_pprev`` is ignored for BDF1. The implicit-
        function-theorem correction makes the value == ``y_star`` while the gradient flows through
        ``f(params/NN)`` and ``hist`` with the Jacobian held constant.
        """
        hist = y_prev if a0 == 1.0 else (2.0 * y_prev - 0.5 * y_pprev)
        y_star = self._newton(y_prev, ic, a0, hist.detach(), h, keq_p, kkin_p, nu_p, sigma_p, iso_nn)
        nusig_p = nu_p + sigma_p
        J = self.analytic_jac(y_star, keq_p, kkin_p, nu_p, sigma_p)
        M = (a0 * self.eye - h * J).detach()
        f = self.rhs(y_star, ic, keq_p, kkin_p, nu_p, nusig_p, iso_nn)
        G = a0 * y_star - h * f - hist
        return y_star - torch.linalg.solve(M, G)

    def integrate(self, keq_p, kkin_p, nu_p, sigma_p, *, differentiable: bool = True, iso_nn=None,
                  checkpoint: bool = False):
        """Integrate to the fixed time grid; return outlet curve [t, salt, prot_g_l] (n, 1+nc).

        ``iso_nn`` (optional) is the DPSOL isotherm-correction NN (Stage B); its parameters are
        the differentiated quantities then (the mechanistic keq/kkin/nu/sigma are passed frozen).
        The Newton/implicit Jacobian stays the mechanistic one (the NN correction is small) — a
        Gauss-Newton-style approximation that keeps the solve fast while the NN gradient still
        flows through the differentiable correction's f-term.

        ``checkpoint`` (gradient checkpointing) recomputes each BDF step's forward during the
        backward pass instead of storing its activations — it trades ~2-3x compute for not holding
        the whole n_steps-long autograd graph, so long fits (large ``n_steps``, e.g. the HLXSYN
        n>=1500 that OOM'd) fit in memory. The solver is deterministic, so the gradients are
        identical to the non-checkpointed path (see ``test_checkpoint_gradient_matches_noncheckpoint``);
        only relevant when ``differentiable``.
        """
        ys = [self.y0]
        y_prev, y_pprev = self.y0, None
        for k in range(1, self.t.shape[0]):
            h = self.t[k] - self.t[k - 1]
            ic = self.inlet_c[k]
            a0 = 1.0 if y_pprev is None else 1.5
            if not differentiable:
                hist = y_prev if y_pprev is None else (2.0 * y_prev - 0.5 * y_pprev)
                y = self._newton(y_prev, ic, a0, hist.detach(), h, keq_p, kkin_p, nu_p, sigma_p, iso_nn)
            else:
                ypp = y_prev if y_pprev is None else y_pprev  # placeholder for BDF1 (unused there)
                if checkpoint and y_pprev is not None:
                    y = _ckpt(self._diff_step, y_prev, ypp, keq_p, kkin_p, nu_p, sigma_p,
                              h, ic, a0, iso_nn, use_reentrant=False)
                else:
                    y = self._diff_step(y_prev, ypp, keq_p, kkin_p, nu_p, sigma_p, h, ic, a0, iso_nn)
            y_pprev, y_prev = y_prev, y
            ys.append(y)
        Y = torch.stack(ys)  # (n, ndim)
        gs = self.gs
        salt = Y[:, gs - 1]
        prot = torch.stack([Y[:, (c + 1) * gs - 1] for c in range(1, self.npr + 1)], dim=1) * MOL_TO_G_L
        return torch.cat([self.t[:, None], salt[:, None], prot], dim=1)

    def elution_curve(self, keq_p, kkin_p, nu_p, sigma_p, *, differentiable: bool = True, iso_nn=None,
                      checkpoint: bool = False):
        """Outlet curve from elution start, re-zeroed in time (matches SimulationResult)."""
        full = self.integrate(keq_p, kkin_p, nu_p, sigma_p, differentiable=differentiable,
                              iso_nn=iso_nn, checkpoint=checkpoint)
        i0 = int(torch.searchsorted(self.t, torch.as_tensor(self.feed_end_t, dtype=DTYPE)).item())
        i0 = max(0, min(i0, full.shape[0] - 1))
        out = full[i0:].clone()
        out[:, 0] = out[:, 0] - out[0, 0]
        return out


def simulate_torch(column, components, inlet, loading_g_l, correction=None, *,
                   n_steps: int = 600) -> np.ndarray:
    """Convenience: full outlet curve [t_s, salt, prot...] (numpy), no grad."""
    sim = TorchSimulator(column, components, inlet, loading_g_l, correction, n_steps=n_steps)
    keq_p, kkin_p, nu_p, sigma_p = params_from_components(components)
    with torch.no_grad():
        return sim.integrate(keq_p, kkin_p, nu_p, sigma_p, differentiable=False).numpy()
