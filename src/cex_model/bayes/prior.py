"""Product-agnostic weak physical prior over the SMA parameters (u-space).

The prior is a diagonal Gaussian in the inference space
``u = [log10(keq), log10(kkin), nu, sigma]`` (model units), i.e. a broad
log-normal on keq/kkin and a broad normal on nu/sigma.  It depends ONLY on the
number of protein components (and optional bound overrides) -- **never** on the
target product's calibrated parameters, so evaluating a held-out product cannot
leak its fitted values (plan M1).

The default bounds are deliberately broad ("weak").  They should be
sanity-checked against the *envelope of the OTHER products'* fitted parameters,
not anchored to the product being inferred.  Bounds are given in **model units**
(``keq_model = keq_table * 1e-2``, ``kkin_model = kkin_table * 1e-5``) to match
the differentiable solver / :func:`cex_model.diffsolver.torch_solver.params_from_components`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cex_model.inverse.encoding import LOG_PARAMS, PARAM_ROWS, log_mask, param_names

__all__ = ["DEFAULT_BOUNDS", "PhysicalPrior", "physical_prior", "components_to_u",
           "physical_u_bounds", "SOLVER_GUARD_BOUNDS", "solver_guard_bounds",
           "assert_guard_contains"]

# Per-row weak physical bounds in MODEL units (lo, hi); see module docstring.
# keq/kkin span decades (log10); nu/sigma are linear. ~2 sigma reaches each bound.
# Centered/scaled to cover the cross-product fitted envelope (HLXSYN/03/04) within
# ~+-1.85 sigma -- weak but PHYSICAL, never anchored to the target product.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "keq": (1e-7, 1e-1),    # model units (table 1e-5..10)
    "kkin": (1e-11, 1e-3),  # model units (table 1e-6..1e2)
    "nu": (1.0, 18.0),      # characteristic charge (mAb CEX ~ 6-13)
    "sigma": (1.0, 100.0),  # steric shielding factor (mAb CEX ~ 10-95)
}


# The support in which the SMA solver is DEFINED, which is a different question from where the prior puts
# its mass. Nothing here is a modelling statement, so every row sits far outside any plausible fit -- in
# particular outside every product's fitted reference point, which DEFAULT_BOUNDS does not guarantee,
# since a Gaussian prior puts no bound on where the data may pull the fit. Using DEFAULT_BOUNDS as the
# guard clipped 79% of mAb C's draws in one coordinate whose floor sat above that product's own value.
#
# The characteristic charge has two thresholds, not one. The right-hand side needs ``csalt ** nu`` finite,
# so nu >= 0; the analytic Jacobian forms ``csalt ** (nu - 1)`` (diffsolver/torch_solver.py, sma.py) and
# so needs nu >= 1 wherever the salt state can reach exactly zero, which the smooth non-negativity does
# once the raw state falls below about -0.98 at the default smoothing. The floor is set at the binding
# one. Every posterior in the tree puts at most 3e-7 mass below it, so this costs nothing.
SOLVER_GUARD_BOUNDS: dict[str, tuple[float, float]] = {
    "keq": (1e-30, 1e10),
    "kkin": (1e-30, 1e10),
    "nu": (1.0, 50.0),
    "sigma": (0.0, 500.0),
}


def _row_mean_std(row: str, lo: float, hi: float) -> tuple[float, float]:
    """Gaussian (mean, std) in u-space for one parameter row from its (lo, hi)."""
    if row in LOG_PARAMS:
        lo, hi = np.log10(lo), np.log10(hi)
    mean = 0.5 * (lo + hi)
    std = (hi - lo) / 4.0  # +-2 std spans [lo, hi]
    return float(mean), float(std)


@dataclass
class PhysicalPrior:
    """Diagonal Gaussian prior over ``u`` (length ``4*n_protein``)."""

    mean: np.ndarray   # (dim,)
    std: np.ndarray    # (dim,)
    names: list[str]
    n_protein: int

    @property
    def dim(self) -> int:
        return self.mean.size

    def precision(self) -> np.ndarray:
        """Diagonal prior precision ``diag(1/std**2)`` -- the Laplace prior term."""
        return np.diag(1.0 / self.std**2)

    def log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """Gaussian log-density of ``u`` (torch, differentiable for the MAP fit)."""
        mean = torch.as_tensor(self.mean, dtype=u.dtype)
        std = torch.as_tensor(self.std, dtype=u.dtype)
        z = (u - mean) / std
        return -0.5 * torch.sum(z**2) - torch.sum(torch.log(std)) - 0.5 * u.numel() * np.log(2.0 * np.pi)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return self.mean + self.std * rng.standard_normal((n, self.dim))


def physical_prior(n_protein: int, *, bounds: dict[str, tuple[float, float]] | None = None) -> PhysicalPrior:
    """Weak product-agnostic prior for ``n_protein`` components (no product params)."""
    bnd = {**DEFAULT_BOUNDS, **(bounds or {})}
    mean, std = [], []
    for row in PARAM_ROWS:  # (keq, kkin, nu, sigma) -- matches the packed u order
        m, s = _row_mean_std(row, *bnd[row])
        mean.extend([m] * n_protein)
        std.extend([s] * n_protein)
    names = param_names([f"c{j + 1}" for j in range(n_protein)])
    return PhysicalPrior(np.asarray(mean, float), np.asarray(std, float), names, n_protein)


def components_to_u(components) -> np.ndarray:
    """Pack a :class:`ComponentSet` into ``u`` (model units, log10 keq/kkin)."""
    keq = np.log10(components.keq_array()[1:])
    kkin = np.log10(components.kkin_array()[1:])
    nu = components.nu_array()[1:]
    sigma = components.sigma_array()[1:]
    return np.concatenate([keq, kkin, nu, sigma])


def _u_bounds(bnd: dict[str, tuple[float, float]], n_protein: int):
    lo: list[float] = []
    hi: list[float] = []
    for row in PARAM_ROWS:  # (keq, kkin, nu, sigma) -- matches the packed u order
        a, b = bnd[row]
        if row in LOG_PARAMS:
            a, b = float(np.log10(a)), float(np.log10(b))
        lo.extend([a] * n_protein)
        hi.extend([b] * n_protein)
    return np.asarray(lo, float), np.asarray(hi, float)


def physical_u_bounds(n_protein: int, *, bounds: dict[str, tuple[float, float]] | None = None):
    """``(lo, hi)`` arrays over ``u`` (length ``4*n_protein``) = the span the PRIOR is scaled to,
    in the packed u order (keq, kkin, nu, sigma; keq/kkin in log10).

    This is the interval :func:`physical_prior` turns into a Gaussian mean and standard deviation,
    so it describes where the prior puts its mass, not where the model is defined. It is NOT a
    constraint: the prior is unbounded, and the data may pull a fitted parameter outside this span.
    To clip draws before the solver, use :func:`solver_guard_bounds` instead.
    """
    return _u_bounds({**DEFAULT_BOUNDS, **(bounds or {})}, n_protein)


def solver_guard_bounds(n_protein: int, *, bounds: dict[str, tuple[float, float]] | None = None):
    """``(lo, hi)`` arrays over ``u`` = the support in which the SMA solver is defined.

    Draws from the *unbounded* Gaussian posterior can land where the model is not computable -- the
    right-hand side needs ``nu >= 0`` for ``csalt ** nu`` to stay finite, and the analytic Jacobian
    needs ``nu >= 1`` for ``csalt ** (nu - 1)`` -- so draws are clipped here before simulating. The interval is deliberately far wider than any
    plausible fit, because its only job is to keep the solver defined; using the prior's span for
    this instead would clip the bulk of a product's draws whenever the data pull a coordinate past
    the prior's nominal two-standard-deviation reach.
    """
    return _u_bounds({**SOLVER_GUARD_BOUNDS, **(bounds or {})}, n_protein)


def assert_guard_contains(u_map, n_protein: int, *, what: str = "the fitted point") -> None:
    """Fail loudly if the clipping guard excludes the point draws are taken around.

    Clipping a posterior is a tail operation and stays one only while the guard contains the centre.
    When it does not, every draw in that coordinate is displaced to a wall the fit never chose, and the
    pushforward stops being a censored version of the posterior and becomes a different distribution --
    which is how a mis-set guard once manufactured a nonlinearity verdict on one product. This is the
    cheapest check that would have failed at the moment the guard was first applied.
    """
    lo, hi = solver_guard_bounds(n_protein)
    u = np.asarray(u_map, float)
    bad = np.flatnonzero((u < lo) | (u > hi))
    if bad.size:
        rows = ", ".join(f"u[{j}]={u[j]:.4g} outside [{lo[j]:.4g}, {hi[j]:.4g}]" for j in bad)
        raise ValueError(f"solver guard does not contain {what}: {rows}")


# Sanity guard: u packing must agree with the log-transform mask shared with the
# inverse encoder (keq/kkin are the log10 rows; nu/sigma are linear).
assert log_mask(1).tolist() == [True, True, False, False]
