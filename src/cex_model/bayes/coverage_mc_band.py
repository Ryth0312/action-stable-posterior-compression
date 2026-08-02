"""MC + DKW finite-sample coverage fallback (kappa closure brief, Deliverable C).

When the posterior-whitened `kappa_q` curvature diagnostic (`bayes/whitened_curvature.py`) cannot be
certified (`WARN_DIAGNOSTIC_ONLY` / `FAIL_DIAGNOSTIC_ONLY`), the decision-interval coverage claim
should not rest on an analytic delta-method + curvature correction at all. This module instead
reports coverage the way `bayes.decision.decision_covariance_mc` / the project's existing
`relFrob`/MC checks already do -- draw nonlinear posterior samples, evaluate the pooled-window QoI
on each, and check how often the LINEARIZED (delta-method) nominal interval actually contains the
nonlinear draw -- with an explicit finite-sample uncertainty band via the Dworetzky-Kiefer-Wolfowitz
(DKW) inequality (a proven, distribution-free uniform CDF bound, not an asymptotic approximation).

This is always reported as an a-posteriori empirical DIAGNOSTIC (`PASS_EMPIRICAL_DIAGNOSTIC`), never
as an a-priori certificate -- the DKW band is a rigorous statement about the SAMPLING uncertainty of
the empirical coverage estimate, not a claim that coverage is exactly nominal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from cex_model.bayes.decision import decision_covariance, decision_covariance_mc, decision_jacobian
from cex_model.bayes.loading_sweep import _load_product_setup

__all__ = [
    "CoverageBandConfig",
    "compute_linear_interval",
    "sample_nonlinear_qoi",
    "coverage_with_dkw",
    "run_coverage_band_report",
]

_QOI_INDEX = {"purity": 0, "yield": 1}


@dataclass
class CoverageBandConfig:
    products: list[str]
    qois: tuple[str, ...] = ("purity", "yield")
    draws: int = 1000                   # N=1000 default (DKW half-band +-0.043); brief/paper coverage evidence
    nominal_alpha: float = 0.05
    dkw_eta: float = 0.05
    posterior_source: str = "laplace"   # or "nuts" if a {product}_nuts_posterior_M.npz is committed
    n_steps: int = 120
    seed: int = 0
    # per-cell draw override keyed "PRODUCT:qoi" (e.g. {"HLXSYN:yield": 5000}) -- lets a flagged QoI run
    # at N=5000 without a second whole-file run overwriting the shared out-path.
    flagged_draws: dict | None = None
    in_dir: str = "results/bayes"
    out_path: str = "results/bayes/coverage_mc_band.json"


def _load_posterior(product: str, posterior_source: str, in_dir: str):
    from cex_model.bayes.posterior import Posterior
    in_dir = Path(in_dir)
    if posterior_source == "nuts":
        nuts_path = in_dir / f"{product}_nuts_posterior_M.npz"
        if not nuts_path.exists():
            raise FileNotFoundError(f"posterior_source='nuts' requested but {nuts_path} not committed "
                                    f"(NUTS posteriors exist only for HLXSYN's main-peak anchor run)")
        return Posterior.load(nuts_path)
    if posterior_source != "laplace":
        raise ValueError(f"posterior_source must be 'laplace' or 'nuts', got {posterior_source!r}")
    return Posterior.load(in_dir / f"{product}_posterior.npz")


def _interval_from_mu_std(mu: float, std: float, z: float, link: str) -> tuple[float, float]:
    """Delta-method ``(1-alpha)`` interval from ``(mu, std)``. ``link="linear"`` = ``mu +- z*std`` (may
    leave [0,1]); ``link="logit"`` = ``expit(logit(p) +- z*std/(p(1-p)))`` -- respects [0,1] and is
    asymmetric (the transformed-QoI interval, brief/Task-3). Guards ``p`` off the {0,1} boundary and a
    degenerate ``std=0`` (returns the point interval)."""
    if link == "logit":
        p = float(np.clip(mu, 1e-6, 1.0 - 1e-6))
        if std <= 0.0:
            return (p, p)
        sd_logit = std / (p * (1.0 - p))
        return (float(expit(logit(p) - z * sd_logit)), float(expit(logit(p) + z * sd_logit)))
    return (mu - z * std, mu + z * std)


def compute_linear_interval(product: str, qoi: str, alpha: float, *, n_steps: int = 120,
                            in_dir: str = "results/bayes", posterior_source: str = "laplace",
                            link: str = "linear") -> tuple[float, float]:
    """Delta-method nominal ``(1-alpha)`` interval ``[a, b]`` for the pooled-window QoI, from
    ``G Sigma G^T`` (``bayes.decision``) -- the SAME linearization ``bayes.whitened_curvature``
    diagnoses the curvature of. ``link="logit"`` returns the [0,1]-respecting transformed interval
    (reused by the HLXSYN yield triage, ``bayes.yield_triage``)."""
    if qoi not in _QOI_INDEX:
        raise ValueError(f"qoi must be one of {list(_QOI_INDEX)}, got {qoi!r}")
    idx = _QOI_INDEX[qoi]
    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    posterior = post if posterior_source == "laplace" else _load_posterior(product, posterior_source, in_dir)
    G, g_map, _ = decision_jacobian(bundle, base_op, posterior.u_map, n_steps=n_steps, return_extra=True)
    C = decision_covariance(G, posterior.cov)
    std = float(np.sqrt(max(C[idx, idx], 0.0)))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    mu = float(g_map[idx])
    return _interval_from_mu_std(mu, std, z, link)


def sample_nonlinear_qoi(product: str, qoi: str, draws: int, posterior_source: str = "laplace", *,
                         n_steps: int = 120, in_dir: str = "results/bayes", seed: int = 0) -> np.ndarray:
    """Draw ``draws`` nonlinear posterior samples of the pooled-window QoI on the FIXED MAP window
    (reuses ``decision.decision_covariance_mc``'s sampling path -- the same estimand the linearized
    ``C`` targets, not a re-optimized-window variant)."""
    if qoi not in _QOI_INDEX:
        raise ValueError(f"qoi must be one of {list(_QOI_INDEX)}, got {qoi!r}")
    idx = _QOI_INDEX[qoi]
    post, base_op, tol, n, prior, bundle = _load_product_setup(product, in_dir)
    posterior = post if posterior_source == "laplace" else _load_posterior(product, posterior_source, in_dir)
    out = decision_covariance_mc(posterior, bundle, base_op, n_samples=draws, seed=seed,
                                 n_steps=n_steps, reselect=False, return_gs=True)
    gs = np.asarray(out.get("gs", []), float)
    if gs.size == 0:
        return np.array([])
    return gs[:, idx]


def coverage_with_dkw(samples: np.ndarray, interval: tuple[float, float], eta: float) -> dict:
    """Empirical coverage of ``interval`` over ``samples`` + the DKW finite-sample band, clipped to
    ``[0, 1]`` (brief §5.1: ``eps_DKW = sqrt(log(2/eta) / (2N))``, ``c_true in [c_hat-2eps, c_hat+2eps]``)."""
    samples = np.asarray(samples, float)
    n = int(samples.size)
    if n == 0:
        return {"empirical_coverage": float("nan"), "dkw_eps": float("nan"),
               "coverage_band": [0.0, 1.0], "n": 0}
    a, b = interval
    c_hat = float(np.mean((samples >= a) & (samples <= b)))
    eps_dkw = float(np.sqrt(np.log(2.0 / eta) / (2.0 * n)))
    lo = float(np.clip(c_hat - 2.0 * eps_dkw, 0.0, 1.0))
    hi = float(np.clip(c_hat + 2.0 * eps_dkw, 0.0, 1.0))
    return {"empirical_coverage": c_hat, "dkw_eps": eps_dkw, "coverage_band": [lo, hi], "n": n}


def _wilson_interval(c_hat: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Binomial Wilson score interval (brief §5.4: "report a binomial/Wilson interval if already
    available, but keep DKW because it is the theorem-backed uniform CDF band")."""
    if n == 0:
        return (0.0, 1.0)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    denom = 1.0 + z * z / n
    center = (c_hat + z * z / (2 * n)) / denom
    half = (z * np.sqrt(c_hat * (1 - c_hat) / n + z * z / (4 * n * n))) / denom
    return (float(np.clip(center - half, 0.0, 1.0)), float(np.clip(center + half, 0.0, 1.0)))


def run_coverage_band_report(config: CoverageBandConfig) -> dict:
    """Run all ``config.products`` x ``config.qois`` and write the MC/DKW coverage report. Runs even
    when curvature validation has failed elsewhere -- no dependency on ``whitened_curvature.json``."""
    flagged = config.flagged_draws or {}
    products_out: dict = {}
    for p in config.products:
        products_out[p] = {}
        for qoi in config.qois:
            draws_pq = int(flagged.get(f"{p}:{qoi}", config.draws))
            try:
                interval = compute_linear_interval(p, qoi, config.nominal_alpha, n_steps=config.n_steps,
                                                   in_dir=config.in_dir, posterior_source=config.posterior_source,
                                                   link="linear")
                logit_interval = compute_linear_interval(p, qoi, config.nominal_alpha, n_steps=config.n_steps,
                                                         in_dir=config.in_dir, posterior_source=config.posterior_source,
                                                         link="logit")
                samples = sample_nonlinear_qoi(p, qoi, draws_pq, config.posterior_source,
                                               n_steps=config.n_steps, in_dir=config.in_dir, seed=config.seed)
                cov = coverage_with_dkw(samples, interval, config.dkw_eta)
                cov_logit = coverage_with_dkw(samples, logit_interval, config.dkw_eta)
                wilson = _wilson_interval(cov["empirical_coverage"], cov["n"]) if cov["n"] > 0 else None
                products_out[p][qoi] = {
                    "linear_interval": [float(interval[0]), float(interval[1])],
                    "empirical_coverage": cov["empirical_coverage"], "dkw_eps": cov["dkw_eps"],
                    "coverage_band": cov["coverage_band"], "n_draws_used": cov["n"], "draws_requested": draws_pq,
                    "wilson_interval": list(wilson) if wilson else None,
                    "logit_interval": [float(logit_interval[0]), float(logit_interval[1])],
                    "logit_empirical_coverage": cov_logit["empirical_coverage"],
                    "status": "PASS_EMPIRICAL_DIAGNOSTIC" if cov["n"] >= 20 else "WARN_TOO_FEW_DRAWS",
                }
            except Exception as exc:
                products_out[p][qoi] = {"status": "FAIL_COVERAGE_DIAGNOSTIC", "error": str(exc)}

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "draws": config.draws, "dkw_eta": config.dkw_eta,
            "nominal_coverage": 1.0 - config.nominal_alpha,
            "posterior_source": config.posterior_source, "seed": config.seed,
            "flagged_draws": dict(flagged) if flagged else None,
            "claim_level": "coverage_diagnostic_not_certificate",
        },
        "products": products_out,
    }
    out_path = Path(config.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
