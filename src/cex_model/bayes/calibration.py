"""Simulation-based calibration (SBC, Talts et al. 2018).

Draw ``theta ~ prior``, simulate data, infer a posterior, and rank the true
``theta`` among the posterior draws.  If inference is calibrated the ranks are
uniform -- a stronger check than marginal coverage.  Cheap by design (intended
for a small synthetic model, no RK23): it certifies the SVI / Laplace / NUTS
engines before they are trusted on real data.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["sbc"]


def sbc(sample_prior, simulate, infer, *, n_datasets: int = 200, n_post: int = 100,
        level: float = 0.9, seed: int = 0) -> dict:
    """Run SBC and return rank statistics + uniformity + central-interval coverage.

    Parameters
    ----------
    sample_prior(rng) -> u_true        : draw one ground-truth parameter vector (dim,)
    simulate(u_true, rng) -> obs       : simulate one dataset under u_true
    infer(obs) -> samples (n_post, dim): posterior draws in u-space for that dataset
    """
    rng = np.random.default_rng(seed)
    ranks, covered = [], 0.0
    dim = None
    for _ in range(n_datasets):
        u_true = np.asarray(sample_prior(rng), float)
        dim = u_true.size
        samp = np.asarray(infer(simulate(u_true, rng)), float)  # (n_post, dim)
        ranks.append((samp < u_true).sum(axis=0))  # rank in [0, n_post] per dim
        lo = np.quantile(samp, 0.5 * (1.0 - level), axis=0)
        hi = np.quantile(samp, 0.5 * (1.0 + level), axis=0)
        covered += float(np.mean((u_true >= lo) & (u_true <= hi)))
    ranks = np.asarray(ranks)  # (n_datasets, dim)

    ks_p = []
    for d in range(dim):
        # jitter discrete ranks to a continuous [0,1] before the uniformity KS test
        u = (ranks[:, d] + rng.random(ranks.shape[0])) / (n_post + 1)
        ks_p.append(float(stats.kstest(u, "uniform").pvalue))
    return {
        "ranks": ranks, "n_post": n_post, "ks_pvalue": ks_p, "ks_pvalue_min": float(min(ks_p)),
        "coverage": float(covered / n_datasets), "level": level, "n_datasets": n_datasets,
    }
