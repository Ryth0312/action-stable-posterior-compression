"""Profile likelihood -- a Laplace-free identifiability check.

For one parameter, fix it on a grid and re-optimise all others; the profile of the
negative log-posterior is FLAT along an unidentified direction and parabolic along
an identified one.  This independently validates the Laplace covariance's "wide"
directions (the keq<->nu ridge, sigma) without assuming a Gaussian posterior, and
answers the reviewer concern that a local Gaussian approximation might mis-state
uncertainty.
"""

from __future__ import annotations

import numpy as np
import torch

from cex_model.bayes.likelihood import AKTA_NOISE_FLOOR_G_L, gaussian_loglik
from cex_model.diffsolver.torch_solver import DTYPE

__all__ = ["profile_likelihood"]


def profile_likelihood(predict_fn, obs, prior, u_map, param_idx: int, *,
                       sigma_obs: float = AKTA_NOISE_FLOOR_G_L, n: int = 11, span: float = 2.0,
                       iters: int = 300, lr: float = 0.05, flat_tol: float = 0.5) -> dict:
    """1-D profile of the negative log-posterior along ``u[param_idx]``.

    Returns the grid, the (min-subtracted) profile NLP, and a ``flat`` flag
    (profile rises by < ``flat_tol`` over ``+/- span`` => effectively unidentified;
    ``flat_tol=0.5`` ~ a 1-sigma chi-square rise).
    """
    u_map = np.asarray(u_map, float)
    dim = u_map.size
    free = [j for j in range(dim) if j != param_idx]
    free_t = torch.tensor(free, dtype=torch.long)
    pidx_t = torch.tensor([param_idx], dtype=torch.long)
    grid = u_map[param_idx] + np.linspace(-span, span, n)

    nlp = []
    for v in grid:
        u_free = torch.tensor(u_map[free], dtype=DTYPE, requires_grad=True)
        v_t = torch.tensor([float(v)], dtype=DTYPE)
        opt = torch.optim.Adam([u_free], lr=lr)
        loss = None
        for _ in range(iters):
            opt.zero_grad()
            full = torch.zeros(dim, dtype=DTYPE).index_add(0, free_t, u_free).index_add(0, pidx_t, v_t)
            loss = -gaussian_loglik(full, predict_fn, obs, sigma_obs) - prior.log_prob(full)
            loss.backward()
            opt.step()
        nlp.append(float(loss.item()))

    nlp = np.asarray(nlp) - float(np.min(nlp))
    return {"param_idx": int(param_idx), "grid": grid.tolist(), "nlp": nlp.tolist(),
            "flat": bool(nlp.max() < flat_tol), "rise": float(nlp.max())}
