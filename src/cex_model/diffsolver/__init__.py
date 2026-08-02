"""Differentiable PyTorch EDM+SMA solver (DPSOL groundwork).

A torch re-implementation of the EDM transport + SMA isotherm ODE (``sma.tran_ode_rhs``)
integrated with a fixed-step, implicitly-differentiated BDF, so that ``d(loss)/d(params)``
is available by autograd at ~one-solve cost — the enabler for fast gradient-based
calibration and (later) a solver-in-the-loop isotherm NN (Chen 2025 DPSOL).

Requires torch (optional dependency, the ``[surrogate]`` extra). The numpy/scipy core
(sma.py, simulator.py) is untouched and remains the reference/validator.
"""

from cex_model.diffsolver.calibrate_diff import (
    ExperimentTarget,
    fit_sma_adam,
    predict_curve,
    targets_from_bundle,
    train_isotherm_nn,
)
from cex_model.diffsolver.collection_objective import (
    FixedWindowObjectiveResult,
    GradientWindowSelection,
    differentiable_fixed_window_objective,
    select_window_for_gradient,
)
from cex_model.diffsolver.isotherm_nn import IsothermNN
from cex_model.diffsolver.operating import bind_operating_conditions, linear_gamma_coeffs
from cex_model.diffsolver.torch_solver import (
    TorchSimulator,
    params_from_components,
    simulate_torch,
)

__all__ = [
    "TorchSimulator",
    "params_from_components",
    "simulate_torch",
    "ExperimentTarget",
    "fit_sma_adam",
    "train_isotherm_nn",
    "predict_curve",
    "targets_from_bundle",
    "IsothermNN",
    "bind_operating_conditions",
    "linear_gamma_coeffs",
    "select_window_for_gradient",
    "differentiable_fixed_window_objective",
    "FixedWindowObjectiveResult",
    "GradientWindowSelection",
]
