"""Gradient-based Bayesian calibration of SMA parameters from real curves.

Per-product posterior ``p(theta | curves, ops)`` over ``theta = keq/kkin/nu/sigma``
built on the existing differentiable BDF solver (``cex_model.diffsolver``): a
Gaussian observation likelihood at the AKTA noise floor, a weak product-agnostic
physical prior, and a MAP + Laplace posterior with identifiability diagnostics.
No per-product point calibration is required up front, and no synthetic training
distribution is involved -- the likelihood fits the real curves directly.

Predictive validation is intended to run on the RK23 truth path (the project's
lossless contract); SVI / NUTS (Pyro) and posterior-driven experiment design are
layered on top of this foundation.
"""

from cex_model.bayes.active import identifiability_report, run_adaptive_design, simulate_target
from cex_model.bayes.calibration import sbc
from cex_model.bayes.decision import (
    DECISION_NAMES,
    decision_covariance,
    decision_covariance_mc,
    decision_jacobian,
    decision_oed_score,
    decision_report,
)
from cex_model.bayes.design import expected_info_gain, recommend_experiments_bayes
from cex_model.bayes.engines import nuts_posterior, svi_posterior
from cex_model.bayes.pbp import pbp_applicability, yamamoto_nu_keq
from cex_model.bayes.identifiability import correlation_pairs, eigen_identifiability, shrinkage
from cex_model.bayes.likelihood import (
    AKTA_NOISE_FLOOR_G_L,
    gaussian_loglik,
    mechanistic_model,
    unpack_u,
)
from cex_model.bayes.posterior import Posterior, laplace_posterior, map_fit
from cex_model.bayes.predictive import posterior_predictive_rk23, predict_experiment
from cex_model.bayes.prior import PhysicalPrior, components_to_u, physical_prior
from cex_model.bayes.profile import profile_likelihood
from cex_model.bayes.sigma_ablation import run_ablation as sigma_freeze_or_compress_ablation
from cex_model.bayes.spectral_transfer import decompose_decision
from cex_model.bayes.synthetic import synthetic_sma_bundle
from cex_model.bayes.adequacy_gate import (
    DEFAULT_LOADING_CAP_G_PER_L,
    SupportedDomain,
    default_domains,
    eligibility,
    gate_operating_window,
    gated_action,
    gated_widest_adequate,
)
from cex_model.bayes.validation import (
    decision_leave_one_experiment_out,
    design_metrics,
    empirical_sigma_obs,
    fisher_ablation,
    flag_outlier_experiments,
    keq_nu_ridge_width,
    leave_one_experiment_out,
    n_steps_sensitivity,
    prior_sensitivity,
    real_design_loop,
    retrospective_design,
    retrospective_eig,
    sigma_obs_sensitivity,
)

__all__ = [
    "AKTA_NOISE_FLOOR_G_L",
    "gaussian_loglik",
    "mechanistic_model",
    "unpack_u",
    "PhysicalPrior",
    "physical_prior",
    "components_to_u",
    "map_fit",
    "laplace_posterior",
    "svi_posterior",
    "nuts_posterior",
    "Posterior",
    "eigen_identifiability",
    "shrinkage",
    "correlation_pairs",
    "sbc",
    "posterior_predictive_rk23",
    "predict_experiment",
    "recommend_experiments_bayes",
    "expected_info_gain",
    "identifiability_report",
    "run_adaptive_design",
    "simulate_target",
    "DECISION_NAMES",
    "decision_report",
    "decision_jacobian",
    "decision_covariance",
    "decision_covariance_mc",
    "decision_oed_score",
    "decompose_decision",
    "sigma_freeze_or_compress_ablation",
    "yamamoto_nu_keq",
    "pbp_applicability",
    "profile_likelihood",
    "synthetic_sma_bundle",
    "flag_outlier_experiments",
    "leave_one_experiment_out",
    "decision_leave_one_experiment_out",
    "SupportedDomain",
    "default_domains",
    "eligibility",
    "gate_operating_window",
    "gated_widest_adequate",
    "gated_action",
    "DEFAULT_LOADING_CAP_G_PER_L",
    "retrospective_eig",
    "retrospective_design",
    "fisher_ablation",
    "keq_nu_ridge_width",
    "real_design_loop",
    "design_metrics",
    "empirical_sigma_obs",
    "sigma_obs_sensitivity",
    "n_steps_sensitivity",
    "prior_sensitivity",
]
