"""SMA adsorption kinetics within the EDM framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet
from cex_model.corrections import LoadingCorrection, LinearLoadingCorrection
from cex_model.edm import EDMMatrices, generate_edm_matrices
from cex_model.gradients import InletProfile


MOL_TO_G_L = 150_000.0


@dataclass
class ModelState:
    """Packed arrays for ODE integration.

    ``nonneg`` selects how the solid/liquid concentrations are kept non-negative
    inside the right-hand side. ``"hard"`` reproduces MATLAB's ``Y(Y<0)=0`` clip
    bit-for-bit (default, used for MATLAB-parity validation). ``"smooth"`` uses a
    C-infinity approximation ``0.5*(Y + sqrt(Y^2 + eps^2))`` that keeps the RHS
    differentiable; this is what lets implicit/stiff solvers (BDF/Radau) integrate
    this stiff system efficiently (the hard clip's kink defeats their Newton
    iteration). The two agree to O(eps).
    """

    column: ColumnParameters
    components: ComponentSet
    edm: EDMMatrices
    inlet: InletProfile
    nu: np.ndarray
    keq: np.ndarray
    kkin: np.ndarray
    sigma: np.ndarray
    loading_g_l: float
    correction: LoadingCorrection
    # Precomputed once (were recomputed every RHS call before):
    gamma: np.ndarray = field(default=None)  # type: ignore[assignment]
    nusig_protein: np.ndarray = field(default=None)  # type: ignore[assignment]
    nonneg: str = "hard"
    smooth_eps: float = 1e-8


def build_model_state(
    column: ColumnParameters,
    components: ComponentSet,
    inlet: InletProfile,
    loading_g_l: float,
    correction: LoadingCorrection | None = None,
    *,
    nonneg: str = "hard",
    smooth_eps: float = 1e-8,
) -> ModelState:
    correction = correction or LinearLoadingCorrection()
    nc = components.n_total
    nu = components.nu_array()
    sigma = components.sigma_array()
    return ModelState(
        column=column,
        components=components,
        edm=generate_edm_matrices(column),
        inlet=inlet,
        nu=nu,
        keq=components.keq_array(),
        kkin=components.kkin_array(),
        sigma=sigma,
        loading_g_l=loading_g_l,
        correction=correction,
        gamma=correction.gamma(loading_g_l, nc),
        nusig_protein=(nu + sigma)[1:],
        nonneg=nonneg,
        smooth_eps=smooth_eps,
    )


def _compute_sma_sum(
    Q: np.ndarray, nu: np.ndarray, sigma: np.ndarray, inocap: float
) -> np.ndarray:
    """Compute SMA free-site term using MATLAB matrix multiplication.

    ``Q`` follows the state convention ``(n_grid, n_components)``. ``nu`` and
    ``sigma`` are one-dimensional component arrays, including salt at index 0.
    This mirrors MATLAB ``inocap - ((nu + sigma) * Q')'``.
    """
    Q = np.asarray(Q, dtype=float)
    nu = np.asarray(nu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)

    if Q.ndim != 2:
        raise ValueError(
            f"Q must be 2D with shape (n_grid, n_components), got {Q.shape}"
        )
    if nu.shape != sigma.shape:
        raise ValueError(
            f"nu and sigma must have the same shape, got nu.shape={nu.shape}, "
            f"sigma.shape={sigma.shape}"
        )
    if Q.shape[1] != nu.shape[0]:
        raise ValueError(
            "Q second dimension must match number of components. "
            f"Got Q.shape={Q.shape}, nu.shape={nu.shape}, sigma.shape={sigma.shape}"
        )

    return float(inocap) - Q @ (nu + sigma)


def _apply_nonneg(Y: np.ndarray, state: ModelState) -> np.ndarray:
    """Enforce non-negativity (load-bearing: ``C_salt**nu`` needs a real base)."""
    if state.nonneg == "smooth":
        eps2 = state.smooth_eps * state.smooth_eps
        return 0.5 * (Y + np.sqrt(Y * Y + eps2))
    return np.maximum(Y, 0.0)


def tran_ode_rhs(t: float, y: np.ndarray, state: ModelState) -> np.ndarray:
    """ODE right-hand side (MATLAB ``tran_ode.m``), vectorized over components.

    State vector layout (MATLAB column-major ``Y(:)``):
    ``[C (gs x n_comp), Q_protein (gs x n_protein)]`` reshaped Fortran-order.
    """
    gs = state.column.grid_size
    nc = state.components.n_total

    Y = y.reshape(gs, nc * 2 - 1, order="F")
    Y = _apply_nonneg(Y, state)

    C = Y[:, :nc]               # liquid phase (col 0 = salt)
    Q_protein = Y[:, nc:]       # stored solid phase (proteins only)

    col = state.column
    N = state.edm.N
    # Precomputed transport operator T = N @ A_C (cached per geometry); fall back
    # for a hand-built EDMMatrices without T.
    T = state.edm.T if state.edm.T is not None else N @ state.edm.A_C
    vb = col.velocity / col.epsbed
    c1 = (1.0 - col.epsbed) / col.epsbed
    gamma = state.gamma if state.gamma is not None else state.correction.gamma(
        state.loading_g_l, nc
    )
    nusig_p = (
        state.nusig_protein
        if state.nusig_protein is not None
        else (state.nu + state.sigma)[1:]
    )

    # Convection + dispersion: dC = (N@A_C)@C plus the inlet boundary applied via
    # column 0 of N (was A_C@C then N@pre; one matvec now).
    inlet_c = state.inlet.inlet_concentrations(t)
    dC = T @ C
    dC += np.outer(N[:, 0], vb * (inlet_c - C[0, :]))

    # SMA free-site term, then the protein kinetics vectorized over components
    # (was a per-protein Python loop).
    Qeff_protein = gamma[1:] * Q_protein
    sma_sum = np.maximum(col.inocap - Qeff_protein @ nusig_p, 1e-12)
    csalt = C[:, 0]

    nu_p = state.nu[1:]
    keq_p = state.keq[1:]
    kkin_p = state.kkin[1:]
    gamma_p = gamma[1:]
    adsorption = keq_p * (sma_sum[:, None] ** nu_p[None, :]) * C[:, 1:]
    desorption = gamma_p * Q_protein * (csalt[:, None] ** nu_p[None, :])
    dQ = (adsorption - desorption) / kkin_p
    dC[:, 1:] -= c1 * dQ

    return np.column_stack([dC, dQ]).ravel(order="F")


@lru_cache(maxsize=16)
def build_jacobian_sparsity(gs: int, nc: int):
    """Boolean Jacobian sparsity for ``tran_ode_rhs`` (safe structural superset).

    Lets stiff solvers (BDF/Radau) build the Jacobian by sparse finite differences
    with a handful of colors instead of ``ndim`` RHS evaluations, which is the
    difference between a stiff solver being slower than RK23 and ~4x faster.

    Structure (state laid out Fortran-order as ``[C_0..C_{nc-1}, Q_1..Q_{nc-1}]``):
      * each liquid block ``C_c`` is dense in itself (``N = inv(M)`` is dense);
      * ``dC_c`` (c>=1) and ``dQ_q`` couple node-wise to salt ``C_0``, the matching
        ``C_c`` and to every ``Q_j`` through the shared free-site term.
    The smooth-clip chain rule is diagonal and does not change the pattern.
    """
    from scipy.sparse import csr_matrix

    ndim = gs * (2 * nc - 1)
    P = np.zeros((ndim, ndim), dtype=bool)
    d = np.arange(gs)

    def blk(c: int) -> slice:
        return slice(c * gs, c * gs + gs)

    # Dense liquid self-blocks for every component (incl. salt).
    for c in range(nc):
        P[blk(c), blk(c)] = True

    # dC_c (c>=1): node-wise coupling to salt and to every protein Q_j.
    for c in range(1, nc):
        rows = c * gs + d
        P[rows, 0 * gs + d] = True
        for j in range(1, nc):
            P[rows, (nc + j - 1) * gs + d] = True

    # dQ_q rows: node-wise coupling to salt, own liquid C_q and every Q_j.
    for q in range(1, nc):
        rows = (nc + q - 1) * gs + d
        P[rows, 0 * gs + d] = True
        P[rows, q * gs + d] = True
        for j in range(1, nc):
            P[rows, (nc + j - 1) * gs + d] = True

    return csr_matrix(P)


def tran_ode_jac(t: float, y: np.ndarray, state: ModelState):
    """Analytic sparse Jacobian of :func:`tran_ode_rhs` (smooth-clip path only).

    Exact derivative of the EDM+SMA RHS. Passed as ``jac=`` to the implicit
    solvers (BDF/Radau) so they skip the ~grid_size finite-difference RHS
    evaluations per Jacobian build and get exact derivatives — faster Newton, no
    change to the accuracy gate (rtol/atol). Requires ``nonneg='smooth'`` (the
    only path that calls it). Nonzero pattern is a subset of
    :func:`build_jacobian_sparsity`.
    """
    from scipy.sparse import csr_matrix, diags

    if state.nonneg != "smooth":
        raise ValueError("tran_ode_jac requires state.nonneg == 'smooth'")

    gs = state.column.grid_size
    nc = state.components.n_total
    npr = nc - 1
    ndim = gs * (2 * nc - 1)

    # Smooth clip and its diagonal derivative s = dY_clip/dY_raw.
    Yraw = y.reshape(gs, 2 * nc - 1, order="F")
    root = np.sqrt(Yraw * Yraw + state.smooth_eps * state.smooth_eps)
    Yc = 0.5 * (Yraw + root)
    s_flat = (0.5 * (1.0 + Yraw / root)).ravel(order="F")

    C = Yc[:, :nc]
    Q = Yc[:, nc:]
    col = state.column
    N = state.edm.N
    T = state.edm.T if state.edm.T is not None else N @ state.edm.A_C
    vb = col.velocity / col.epsbed
    c1 = (1.0 - col.epsbed) / col.epsbed
    gamma = state.gamma
    nusig_p = state.nusig_protein
    nu_p, keq_p, kkin_p, gamma_p = state.nu[1:], state.keq[1:], state.kkin[1:], gamma[1:]
    csalt = C[:, 0]

    raw_sma = col.inocap - (gamma_p * Q) @ nusig_p
    active = (raw_sma > 1e-12).astype(float)            # 0 where sma_sum clamped
    sma_sum = np.maximum(raw_sma, 1e-12)
    pa = sma_sum[:, None] ** nu_p[None, :]
    sma_m1 = sma_sum[:, None] ** (nu_p[None, :] - 1.0)
    cs_pow = csalt[:, None] ** nu_p[None, :]
    cs_m1 = csalt[:, None] ** (nu_p[None, :] - 1.0)

    ads_dC = keq_p * pa / kkin_p                         # dQ_q/dC_{q+1}
    dQ_dC0 = -gamma_p * Q * nu_p * cs_m1 / kkin_p        # dQ_q/dC_0
    ads_dsma = keq_p * C[:, 1:] * nu_p * sma_m1          # d(adsorption_q)/d(sma_sum)
    desorp_diag = gamma_p * cs_pow / kkin_p              # self j==q desorption term

    Tliq = T.copy()
    Tliq[:, 0] -= vb * N[:, 0]                           # inlet boundary correction

    d = np.arange(gs)
    rr, cc = np.meshgrid(d, d, indexing="ij")
    rows, cols, data = [], [], []

    def add_dense(br, bc, block):
        rows.append((br + rr).ravel()); cols.append((bc + cc).ravel()); data.append(block.ravel())

    def add_diag(br, bc, vec):
        rows.append(br + d); cols.append(bc + d); data.append(np.asarray(vec, dtype=float))

    # Liquid self-blocks dC_c/dC_c = Tliq (+ reaction diagonal for proteins)
    for c in range(nc):
        add_dense(c * gs, c * gs, Tliq)
        if c >= 1:
            add_diag(c * gs, c * gs, -c1 * ads_dC[:, c - 1])

    for q in range(npr):
        i = q + 1
        rQ = (nc + q) * gs
        rC = i * gs
        add_diag(rQ, i * gs, ads_dC[:, q])               # dQ_q/dC_i
        add_diag(rQ, 0, dQ_dC0[:, q])                    # dQ_q/dC_0
        add_diag(rC, 0, -c1 * dQ_dC0[:, q])              # dC_i/dC_0
        for jj in range(npr):
            coup = -ads_dsma[:, q] * gamma_p[jj] * nusig_p[jj] * active / kkin_p[q]
            if jj == q:
                coup = coup - desorp_diag[:, q]
            add_diag(rQ, (nc + jj) * gs, coup)           # dQ_q/dQ_j
            add_diag(rC, (nc + jj) * gs, -c1 * coup)     # dC_i/dQ_j

    J = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndim, ndim),
    )
    # Chain rule for the smooth clip: scale each column k by s[k].
    return J @ diags(s_flat)
