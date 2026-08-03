"""Torch-free decision-calibration metrics shared by the R5b and phase-2 calibration studies.

Extracted so the SBC/reliability scoring has a single source of truth: ``step5_decision_sbc.py`` (which needs
torch for the nonlinear decision solve) and ``step6_fullpipeline_calibration.py`` (pure numpy) both import these
rather than keeping divergent copies.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["_reliability", "_cal_intercept_slope", "_ece", "_score_block", "_rank_stats"]


def _reliability(p, y, n_bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if m.sum():
            rows.append({"bin": [float(edges[i]), float(edges[i + 1])], "n": int(m.sum()),
                         "p_mean": float(p[m].mean()), "y_freq": float(y[m].mean())})
    return rows


def _cal_intercept_slope(p, y, *, cap=25.0):
    """Logistic recalibration fit y ~ Bern(sigmoid(a + b logit p)); calibrated iff (a, b) = (0, 1).

    Damped Newton with a coefficient cap: under perfect separation the MLE diverges, so we report NaN
    with a ``separation`` note instead of a meaningless 1e8 coefficient.
    """
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.clip(np.log(p / (1 - p)), -12.0, 12.0)          # cap the design to keep exp() finite
    y = np.asarray(y, float)
    if np.unique(y).size < 2:
        return {"intercept": float("nan"), "slope": float("nan"), "cal_note": "single outcome class"}
    # A saturated/bimodal p-hat gives a rank-deficient design: the fit is unidentifiable and Newton
    # never leaves (0, 1), which would be MISREAD as "perfectly calibrated". Refuse to report it.
    if np.unique(np.round(x, 6)).size < 3 or np.std(x) < 1e-6:
        return {"intercept": float("nan"), "slope": float("nan"),
                "cal_note": "degenerate design: p-hat saturated/bimodal (use ECE, Brier and tail hit rates)"}
    X = np.column_stack([np.ones_like(x), x])

    def nll(a, b):                                          # stable -log-likelihood
        eta = np.clip(a + b * x, -30.0, 30.0)
        return float(-np.sum(y * eta - np.logaddexp(0.0, eta)))

    a, b, converged = 0.0, 1.0, False
    for _ in range(100):
        eta = np.clip(a + b * x, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        d = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(2), X.T @ (y - mu))
        f0, t = nll(a, b), 1.0
        while t > 1e-10 and nll(a + t * d[0], b + t * d[1]) > f0 - 1e-12:
            t *= 0.5                                        # backtracking: a fixed step clip can 2-cycle
        if t <= 1e-10:
            converged = True
            break
        a, b = a + t * d[0], b + t * d[1]
        if abs(a) > cap or abs(b) > cap:
            return {"intercept": float("nan"), "slope": float("nan"),
                    "cal_note": "separation: logistic MLE diverges (use ECE, Brier and tail hit rates)"}
        if np.max(np.abs(t * d)) < 1e-9:
            converged = True
            break
    if not converged:
        return {"intercept": float("nan"), "slope": float("nan"),
                "cal_note": "logistic fit did not converge (use ECE, Brier and tail hit rates)"}
    return {"intercept": float(a), "slope": float(b)}


def _ece(rel, n_total):
    """Expected calibration error: bin-count-weighted mean |p_mean - y_freq|. Unlike the logistic
    intercept/slope it stays meaningful when p-hat is saturated, so it is the primary calibration read."""
    if not rel or n_total <= 0:
        return float("nan")
    return float(sum(r["n"] * abs(r["p_mean"] - r["y_freq"]) for r in rel) / n_total)


def _score_block(p, y, label):
    p, y = np.asarray(p, float), np.asarray(y, float)
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    hi, lo = p >= 0.95, p <= 0.05
    rel = _reliability(p, y)
    return {"label": label, "n": int(p.size), "mean_p": float(p.mean()), "base_rate": float(y.mean()),
            "brier": float(np.mean((p - y) ** 2)), "ece": _ece(rel, p.size),
            "log_score": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            **_cal_intercept_slope(p, y),
            "tail_hi_n": int(hi.sum()), "tail_hi_hit_freq": float(y[hi].mean()) if hi.any() else float("nan"),
            "tail_lo_n": int(lo.sum()), "tail_lo_hit_freq": float(y[lo].mean()) if lo.any() else float("nan"),
            "reliability": rel}


def _rank_stats(ranks, n_post, label, seed=0):
    ranks = np.asarray(ranks, float)
    if ranks.size == 0:
        return {"label": label, "ks_pvalue": float("nan")}
    u = (ranks + np.random.default_rng(seed).random(ranks.size)) / (n_post + 1)
    return {"label": label, "ks_pvalue": float(stats.kstest(u, "uniform").pvalue),
            "mean_rank_frac": float(np.mean(ranks / n_post))}
