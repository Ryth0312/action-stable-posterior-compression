"""SVI and NUTS posterior engines (Pyro) on the differentiable solver.

Both share one probabilistic model: a weak Gaussian prior over ``u`` times the
Gaussian observation likelihood, with the differentiable ``predict_fn`` as the
deterministic forward map (so the same solver graph used by MAP/Laplace is
reused, no rewrite).

- ``svi_posterior``: full-covariance Gaussian guide (``AutoMultivariateNormal``)
  optimised by ELBO -- the production posterior on real data (captures the
  parameter correlations a diagonal fit would miss).
- ``nuts_posterior``: gradient-based MCMC, the *gold standard*.  Each leapfrog
  step needs one reverse-mode solver gradient, so on the real solver it is
  ~prohibitive (see plan) -- use it on cheap synthetic / tiny-grid models to
  validate that SVI / Laplace are trustworthy.

Inference runs under float64 (the solver's dtype) so Pyro's guide / kernel
parameters match the solver tensors.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L
from cex_model.bayes.posterior import Posterior
from cex_model.bayes.prior import PhysicalPrior
from cex_model.components import ComponentSet
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = ["svi_posterior", "nuts_posterior"]


@contextlib.contextmanager
def _default_float64():
    """Make Pyro create its guide/kernel params in float64 (the solver dtype)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _make_model(predict_fn, obs, prior: PhysicalPrior, sigma_obs: float):
    import pyro
    import pyro.distributions as dist

    mean = torch.as_tensor(prior.mean, dtype=DTYPE)
    std = torch.as_tensor(prior.std, dtype=DTYPE)
    obs = torch.as_tensor(obs, dtype=DTYPE)

    def model():
        u = pyro.sample("u", dist.Normal(mean, std).to_event(1))
        pred = predict_fn(u)
        pyro.sample("obs", dist.Normal(pred, sigma_obs).to_event(1), obs=obs)

    return model


def svi_posterior(predict_fn, obs, prior: PhysicalPrior, *, template: ComponentSet,
                  sigma_obs: float = AKTA_NOISE_FLOOR_G_L, steps: int = 1500, lr: float = 0.02,
                  init_loc=None, seed: int = 0, progress: bool = False):
    """Full-covariance Gaussian variational posterior (ELBO). Returns ``(Posterior, history)``."""
    import pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoMultivariateNormal
    from pyro.infer.autoguide.initialization import init_to_value
    from pyro.optim import Adam

    with _default_float64():
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        model = _make_model(predict_fn, obs, prior, sigma_obs)
        init_fn = None
        if init_loc is not None:
            init_fn = init_to_value(values={"u": torch.as_tensor(np.asarray(init_loc, float), dtype=DTYPE)})
        guide = AutoMultivariateNormal(model, init_loc_fn=init_fn) if init_fn else AutoMultivariateNormal(model)
        svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
        losses = []
        for it in range(steps):
            losses.append(float(svi.step()))
            if progress and (it % max(1, steps // 10) == 0 or it == steps - 1):
                print(f"  svi step {it:4d}: ELBO loss={losses[-1]:.5e}")
        mvn = guide.get_posterior()
        mean = mvn.mean.detach().numpy()
        cov = mvn.covariance_matrix.detach().numpy()

    post = Posterior.from_template(mean=mean, cov=cov, u_map=mean, prior=prior, template=template,
                                   sigma_obs=sigma_obs, engine="svi")
    return post, {"loss": losses}


def nuts_posterior(predict_fn, obs, prior: PhysicalPrior, *, template: ComponentSet,
                   sigma_obs: float = AKTA_NOISE_FLOOR_G_L, num_samples: int = 400,
                   warmup: int = 400, init_loc=None, seed: int = 0, progress: bool = False,
                   max_tree_depth: int = 10, target_accept_prob: float = 0.8):
    """Gold-standard NUTS posterior (synthetic / tiny-grid only). Returns ``(Posterior, history)``.

    ``max_tree_depth`` caps the leapfrog steps per draw at ``2**max_tree_depth``; each leapfrog is
    one reverse-mode solver gradient, so on the near-flat keq<->nu ridge the default 10 saturates
    -> up to ~1e3 differentiable solves *per draw*, the dominant cost.  Lowering it (7-8) bounds the
    per-draw work; the returned ``diagnostics`` report how often it is hit -- if that is most draws
    the trees are being truncated (raise it / loosen the geometry) rather than terminating on a
    U-turn.  ``target_accept_prob`` higher -> smaller steps -> deeper trees (slower, more accurate).
    ``progress=True`` shows the live it/s + ETA bar -- essential for a multi-hour run.
    """
    import pyro
    from pyro.infer import MCMC, NUTS

    with _default_float64():
        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        model = _make_model(predict_fn, obs, prior, sigma_obs)
        kernel = NUTS(model, jit_compile=False, max_tree_depth=max_tree_depth,
                      target_accept_prob=target_accept_prob)
        init_params = None
        if init_loc is not None:
            init_params = {"u": torch.as_tensor(np.asarray(init_loc, float), dtype=DTYPE)}
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup, num_chains=1,
                    initial_params=init_params, disable_progbar=not progress)
        mcmc.run()
        samples_u = mcmc.get_samples()["u"].detach().numpy()

    mean = samples_u.mean(axis=0)
    cov = np.cov(samples_u, rowvar=False)
    post = Posterior.from_template(mean=mean, cov=np.atleast_2d(cov), u_map=mean, prior=prior,
                                   template=template, sigma_obs=sigma_obs, engine="nuts",
                                   samples_u=samples_u)
    return post, {"n_samples": int(num_samples), "diagnostics": mcmc.diagnostics()}
