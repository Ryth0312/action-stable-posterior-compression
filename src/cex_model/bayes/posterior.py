"""Per-product posterior over SMA parameters: MAP + Laplace (Gaussian) engine.

``map_fit`` is the existing differentiable Adam calibration with an added
log-prior (so it reuses the exact forward pass).  ``laplace_posterior`` forms the
Gauss-Newton approximation of the negative-log-posterior Hessian at the MAP --
``H = JᵀJ / sigma² + prior_precision`` with ``J = d(model)/du`` from the
differentiable solver -- giving a Gaussian posterior ``N(MAP, H⁻¹)``.  This is the
cheap, zero-extra-dependency engine; SVI / NUTS (Pyro) refine it later.

The ``J`` Jacobian here is the expensive part on real data (one reverse pass per
residual point) -- run those on Colab; tests use a cheap analytic model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
import torch

from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, gaussian_loglik
from cex_model.bayes.prior import PhysicalPrior
from cex_model.components import ComponentSet, ComponentType, SMAComponent
from cex_model.diffsolver.torch_solver import DTYPE
from cex_model.inverse.encoding import log_mask

__all__ = ["Posterior", "map_fit", "laplace_posterior"]


def map_fit(u_init, predict_fn, obs, prior: PhysicalPrior, *,
            sigma_obs: float = AKTA_NOISE_FLOOR_G_L, iters: int = 300, lr: float = 0.05,
            progress: bool = False):
    """Adam MAP of ``-log p(obs|u) - log prior(u)``; returns ``(u_map, history)``."""
    u = torch.tensor(np.asarray(u_init, float), dtype=DTYPE, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    hist = {"loss": [], "nll": []}
    for it in range(iters):
        opt.zero_grad()
        nll = -gaussian_loglik(u, predict_fn, obs, sigma_obs)
        loss = nll - prior.log_prob(u)
        loss.backward()
        opt.step()
        hist["loss"].append(float(loss.item()))
        hist["nll"].append(float(nll.item()))
        if progress and (it % max(1, iters // 10) == 0 or it == iters - 1):
            print(f"  iter {it:3d}: -logpost={loss.item():.5e}")
    return u.detach().numpy(), hist


def _ggn_hessian(predict_fn, u_map, sigma_obs: float) -> np.ndarray:
    """Gauss-Newton ``JᵀJ / sigma²`` of the negative log-likelihood at ``u_map``.

    The Jacobian uses REVERSE mode (one pass per residual point).  Forward-mode
    JVP / double-backward are *incorrect* through the solver's implicit-function
    correction (the detached Newton Jacobian carries no tangent / 2nd-order info;
    verified: forward-mode JVP disagrees with finite differences by ~1e-1), so
    the cheaper column-wise build is not available -- this O(n_resid) reverse
    sweep is the dominant Laplace cost on real data (run it on Colab).
    """
    u = torch.tensor(np.asarray(u_map, float), dtype=DTYPE)
    J = torch.autograd.functional.jacobian(predict_fn, u)  # (n_resid, dim), reverse-mode
    J = J.detach().numpy()
    return (J.T @ J) / (sigma_obs**2)


def laplace_posterior(u_map, predict_fn, obs, prior: PhysicalPrior, *, template: ComponentSet,
                      sigma_obs: float = AKTA_NOISE_FLOOR_G_L, jitter: float = 1e-9) -> "Posterior":
    """Gaussian posterior ``N(MAP, (GGN + prior_precision)⁻¹)`` at the MAP."""
    H = _ggn_hessian(predict_fn, u_map, sigma_obs) + prior.precision()
    H = 0.5 * (H + H.T) + jitter * np.eye(H.shape[0])
    cov = np.linalg.inv(H)
    cov = 0.5 * (cov + cov.T)
    return Posterior.from_template(
        mean=np.asarray(u_map, float), cov=cov, u_map=np.asarray(u_map, float),
        prior=prior, template=template, sigma_obs=sigma_obs, engine="laplace")


@dataclass
class Posterior:
    """Gaussian posterior over ``u`` plus the metadata to rebuild a ComponentSet."""

    mean: np.ndarray            # (dim,)
    cov: np.ndarray             # (dim, dim)
    u_map: np.ndarray           # (dim,)
    names: list[str]            # parameter names (len dim)
    n_protein: int
    comp_names: list[str]
    comp_types: list[str]
    fractions: np.ndarray       # (n_protein,)
    loading_correction: dict | None
    sigma_obs: float
    engine: str
    samples_u: np.ndarray | None = None  # (n_draws, dim) for sample-based engines (NUTS)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_template(cls, *, mean, cov, u_map, prior: PhysicalPrior, template: ComponentSet,
                      sigma_obs: float, engine: str, samples_u=None) -> "Posterior":
        return cls(
            mean=np.asarray(mean, float), cov=np.asarray(cov, float), u_map=np.asarray(u_map, float),
            names=list(prior.names), n_protein=template.n_protein,
            comp_names=[c.name for c in template.components],
            comp_types=[c.component_type.value for c in template.components],
            fractions=template.fraction_array().astype(float),
            loading_correction=template.loading_correction, sigma_obs=float(sigma_obs), engine=engine,
            samples_u=None if samples_u is None else np.asarray(samples_u, float))

    # ---- summaries --------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.mean.size

    def std_u(self) -> np.ndarray:
        """Marginal posterior std in u-space (empirical if sample-based)."""
        if self.samples_u is not None:
            return np.std(self.samples_u, axis=0)
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    def samples(self, n: int, seed: int = 0) -> np.ndarray:
        """Draw ``n`` posterior samples in u-space -> (n, dim).

        Resamples the stored draws for sample-based engines (NUTS); otherwise
        draws from the Gaussian ``N(mean, cov)`` (Laplace / SVI).
        """
        rng = np.random.default_rng(seed)
        if self.samples_u is not None:
            idx = rng.integers(0, self.samples_u.shape[0], size=n)
            return self.samples_u[idx]
        return rng.multivariate_normal(self.mean, self.cov, size=n)

    def credible_interval(self, level: float = 0.9) -> dict[str, tuple[float, float]]:
        """Per-parameter credible interval in PHYSICAL units (keq/kkin un-logged).

        Empirical quantiles for sample-based engines; Gaussian ``mean +- z*std``
        otherwise.  Endpoints are monotone-transformed, so log dims stay valid.
        """
        is_log = log_mask(self.n_protein)
        if self.samples_u is not None:
            lo_u = np.quantile(self.samples_u, 0.5 * (1.0 - level), axis=0)
            hi_u = np.quantile(self.samples_u, 0.5 * (1.0 + level), axis=0)
        else:
            z = NormalDist().inv_cdf(0.5 * (1.0 + level))
            std = self.std_u()
            lo_u, hi_u = self.mean - z * std, self.mean + z * std
        out: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(self.names):
            lo, hi = lo_u[i], hi_u[i]
            if is_log[i]:
                lo, hi = 10.0**lo, 10.0**hi
            out[name] = (float(lo), float(hi))
        return out

    # ---- back to physics --------------------------------------------------
    def to_components(self, u: np.ndarray) -> ComponentSet:
        """Rebuild a :class:`ComponentSet` (table units) from a u-vector."""
        u = np.asarray(u, float)
        n = self.n_protein
        keq_model, kkin_model = 10.0 ** u[0:n], 10.0 ** u[n : 2 * n]
        nu, sigma = u[2 * n : 3 * n], u[3 * n : 4 * n]
        comps = [
            SMAComponent(
                name=self.comp_names[j], nu=float(nu[j]),
                keq=float(keq_model[j]) / 1e-2, sigma=float(sigma[j]),
                kkin=float(kkin_model[j]) / 1e-5, fraction=float(self.fractions[j]),
                component_type=ComponentType(self.comp_types[j]))
            for j in range(n)
        ]
        return ComponentSet(components=comps, loading_correction=self.loading_correction)

    def map_components(self) -> ComponentSet:
        return self.to_components(self.u_map)

    # ---- persistence ------------------------------------------------------
    def save(self, path) -> None:
        meta = {
            "names": self.names, "n_protein": self.n_protein, "comp_names": self.comp_names,
            "comp_types": self.comp_types, "loading_correction": self.loading_correction,
            "sigma_obs": self.sigma_obs, "engine": self.engine,
            "has_samples": self.samples_u is not None,
        }
        arrays = dict(mean=self.mean, cov=self.cov, u_map=self.u_map,
                      fractions=self.fractions, meta=np.array(json.dumps(meta)))
        if self.samples_u is not None:
            arrays["samples_u"] = self.samples_u
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> "Posterior":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        return cls(
            mean=d["mean"], cov=d["cov"], u_map=d["u_map"], names=list(meta["names"]),
            n_protein=int(meta["n_protein"]), comp_names=list(meta["comp_names"]),
            comp_types=list(meta["comp_types"]), fractions=d["fractions"],
            loading_correction=meta["loading_correction"], sigma_obs=float(meta["sigma_obs"]),
            engine=str(meta["engine"]),
            samples_u=d["samples_u"] if meta.get("has_samples") else None)
