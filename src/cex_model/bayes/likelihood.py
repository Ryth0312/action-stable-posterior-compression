"""Gaussian observation likelihood for gradient-based Bayesian SMA calibration.

The inference parameter is the SAME ``log10(model-unit)`` vector that
:func:`cex_model.diffsolver.calibrate_diff.fit_sma_adam` already optimises::

    u = [log10(keq), log10(kkin), nu, sigma]      (per protein, model units)

so a MAP fit here is the existing Adam calibration *plus* a log-prior term, and
the differentiable solver propagates ``d(curve)/d(u)`` for free.  The detector
residuals are treated as i.i.d. Gaussian with the AKTA noise floor as the
standard deviation -- the same per-point error already minimised (as an MSE) by
the calibration, now written as a proper log-likelihood.
"""

from __future__ import annotations

import math

import torch

from cex_model.diffsolver.calibrate_diff import _grouped_curve
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = ["AKTA_NOISE_FLOOR_G_L", "unpack_u", "mechanistic_model", "gaussian_loglik"]

# AKTA UV detector noise floor (g/L); the observation-noise sigma of the likelihood.
AKTA_NOISE_FLOOR_G_L = 0.148


def unpack_u(u: torch.Tensor, n_protein: int):
    """Split the packed vector ``u`` into physical (model-unit) SMA tensors.

    ``u`` is ``[log10(keq).., log10(kkin).., nu.., sigma..]`` (length ``4*n``);
    returns ``(keq, kkin, nu, sigma)`` in model units (keq/kkin un-logged).
    """
    n = n_protein
    log_keq, log_kkin = u[0:n], u[n : 2 * n]
    nu, sigma = u[2 * n : 3 * n], u[3 * n : 4 * n]
    return 10.0**log_keq, 10.0**log_kkin, nu, sigma


def mechanistic_model(targets, n_protein, *, differentiable: bool = True, checkpoint: bool = False):
    """Build ``(predict_fn, obs)`` for a product's :class:`ExperimentTarget` list.

    ``predict_fn(u)`` returns the flattened model concentration vector
    (g/L, concatenated over experiments and observed groups); ``obs`` is the
    matching measured vector.  Reuses the calibration's ``_grouped_curve`` and
    each target's differentiable ``TorchSimulator`` (so this is byte-consistent
    with ``fit_sma_adam``'s forward pass).
    """
    obs = torch.cat([tg.values.reshape(-1).to(DTYPE) for tg in targets])

    def predict_fn(u: torch.Tensor) -> torch.Tensor:
        keq, kkin, nu, sigma = unpack_u(u, n_protein)
        cols = []
        for tg in targets:
            curve = tg.sim.elution_curve(
                keq, kkin, nu, sigma, differentiable=differentiable, checkpoint=checkpoint
            )
            cols.append(_grouped_curve(curve, tg.times_s, tg.groups, n_protein).reshape(-1))
        return torch.cat(cols)

    return predict_fn, obs


def gaussian_loglik(u, predict_fn, obs, sigma_obs: float = AKTA_NOISE_FLOOR_G_L) -> torch.Tensor:
    """Diagonal-Gaussian log-likelihood ``log p(obs | u)`` at observation noise ``sigma_obs``."""
    pred = predict_fn(u)
    resid = (pred - obs) / sigma_obs
    n = obs.numel()
    const = -0.5 * n * math.log(2.0 * math.pi) - n * math.log(float(sigma_obs))
    return -0.5 * torch.sum(resid**2) + const
