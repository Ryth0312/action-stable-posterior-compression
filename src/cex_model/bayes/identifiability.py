"""Identifiability diagnostics from a Gaussian posterior covariance.

Turns the posterior covariance into the formal "what is and isn't constrained by
the data" statement (plan M3): the sloppy/stiff eigen-spectrum, the data-driven
shrinkage of each parameter relative to its prior, and the strongest pairwise
posterior correlations (which surface the known degeneracies, e.g. basics
keq <-> nu).  Pure numpy -- no solver, no torch.
"""

from __future__ import annotations

import numpy as np

from cex_model.bayes.prior import PhysicalPrior
from cex_model.bayes.posterior import Posterior

__all__ = ["eigen_identifiability", "shrinkage", "correlation_pairs"]


def eigen_identifiability(posterior: Posterior, *, top_params: int = 3) -> dict:
    """Eigen-decompose the posterior covariance into sloppy -> stiff directions.

    Large eigenvalue = wide (poorly constrained) direction.  Each direction is
    annotated with the parameters carrying most of its weight.
    """
    cov = posterior.cov
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]  # widest (most uncertain) first
    directions = []
    for k in order:
        w = np.abs(vecs[:, k])
        top = np.argsort(w)[::-1][:top_params]
        directions.append({
            "std": float(np.sqrt(max(vals[k], 0.0))),
            "top_params": [posterior.names[i] for i in top],
            "weights": [float(vecs[i, k]) for i in top],
        })
    return {"eigenvalues": [float(v) for v in vals[order]], "directions": directions}


def shrinkage(posterior: Posterior, prior: PhysicalPrior) -> dict[str, float]:
    """Per-parameter ``posterior_std / prior_std`` in u-space.

    Ratio << 1 => the parameter is pinned by the data; ratio ~ 1 => it is still
    prior-dominated (i.e. effectively unidentified from the curves).
    """
    post_std = posterior.std_u()
    return {name: float(post_std[i] / prior.std[i]) for i, name in enumerate(posterior.names)}


def correlation_pairs(posterior: Posterior, *, k: int = 5) -> list[dict]:
    """Top-``k`` strongest off-diagonal posterior correlations (the degeneracies)."""
    std = posterior.std_u()
    denom = np.outer(std, std)
    corr = np.divide(posterior.cov, denom, out=np.zeros_like(posterior.cov), where=denom > 0)
    dim = corr.shape[0]
    pairs = [
        (abs(corr[i, j]), i, j)
        for i in range(dim) for j in range(i + 1, dim)
    ]
    pairs.sort(reverse=True)
    return [
        {"params": (posterior.names[i], posterior.names[j]), "corr": float(corr[i, j])}
        for _, i, j in pairs[:k]
    ]
