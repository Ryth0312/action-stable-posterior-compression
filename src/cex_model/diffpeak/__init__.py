"""D4.1 -- real-data-trained differentiable peak/process model.

A minimal, physically-constrained, differentiable chromatographic peak model fit DIRECTLY to measured outlet
curves: NO RK23 labels, NO pre-calibrated SMA parameters. Operating conditions -> per-(observed-)peak EMG
parameters (area / retention / width / skew) via a LOW-ORDER parameterisation (3-6 experiments per product
cannot identify an MLP); differentiable EMGs reconstruct the curve; purity/recovery/yield come from
integrating it. The goal of D4.1 is differentiable curve fitting + HONEST identifiability diagnostics, not
beating the SMA Phase-2 optimiser.
"""

from cex_model.diffpeak.data import OP_FEATURES, Experiment, PeakData, load_peak_data
from cex_model.diffpeak.design import (
    OP_LIMITS,
    candidate_pool,
    recommend_experiments,
    safe_bounds,
)
from cex_model.diffpeak.emg import emg, erfcx
from cex_model.diffpeak.evaluate import (
    bootstrap_uncertainty,
    leave_one_out,
    op_coverage,
    perturbation_trends,
)
from cex_model.diffpeak.fit import FitConfig, fit_diffpeak, peak_metrics
from cex_model.diffpeak.model import DiffPeakModel

__all__ = ["emg", "erfcx", "Experiment", "PeakData", "load_peak_data", "OP_FEATURES",
           "DiffPeakModel", "FitConfig", "fit_diffpeak", "peak_metrics",
           "leave_one_out", "op_coverage", "perturbation_trends", "bootstrap_uncertainty",
           "OP_LIMITS", "safe_bounds", "candidate_pool", "recommend_experiments"]
