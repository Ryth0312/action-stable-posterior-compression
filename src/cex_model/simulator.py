"""Chromatography simulator using scipy ODE solvers."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from cex_model.backends import SimulationBackend
from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet
from cex_model.corrections import LoadingCorrection, LinearLoadingCorrection
from cex_model.gradients import InletProfile, build_fitting_inlet, build_simulation_inlet
from cex_model.metrics import compute_rmse
from cex_model.sma import (
    MOL_TO_G_L,
    ModelState,
    build_jacobian_sparsity,
    build_model_state,
    tran_ode_jac,
    tran_ode_rhs,
)

# scipy implicit solvers that benefit from a Jacobian sparsity pattern.
_STIFF_METHODS = frozenset({"BDF", "Radau"})

logger = logging.getLogger(__name__)


# Optional progress hook: a no-arg callable invoked once per successful ODE solve
# (main process only). Lets a calibration drive a progress bar without threading a
# callback through every solver call site. Set via the count_solves() context manager.
_solve_hook = None


@contextlib.contextmanager
def count_solves(hook):
    """Temporarily route a per-solve tick to ``hook`` (pass None to disable)."""
    global _solve_hook
    prev = _solve_hook
    _solve_hook = hook
    try:
        yield
    finally:
        _solve_hook = prev


@dataclass
class SimulationResult:
    """Outlet chromatogram from simulation."""

    time_s: np.ndarray
    salt_mol_l: np.ndarray
    protein_mol_l: np.ndarray  # shape (n_time, n_protein)
    feed_end_index: int
    raw_solution: object | None = None

    @property
    def n_protein(self) -> int:
        return self.protein_mol_l.shape[1]

    def elution_curve(self) -> np.ndarray:
        """Return curve from elution start: [time, salt, protein...] in g/L."""
        t = self.time_s[self.feed_end_index :] - self.time_s[self.feed_end_index]
        salt = self.salt_mol_l[self.feed_end_index :]
        prot = self.protein_mol_l[self.feed_end_index :] * MOL_TO_G_L
        return np.column_stack([t, salt, prot])

    def outlet_g_l(self) -> np.ndarray:
        """Protein concentrations at column outlet in g/L."""
        return self.protein_mol_l * MOL_TO_G_L


@dataclass
class ChromatographySimulator:
    """Forward simulator for CEX/IEC EDM+SMA model."""

    column: ColumnParameters
    components: ComponentSet
    correction: LoadingCorrection = field(default_factory=LinearLoadingCorrection)
    rtol: float = 1e-5
    method: str = "RK23"  # matches MATLAB ode23 default path
    n_time_points: int = 1000
    backend: SimulationBackend | None = None
    # Non-negativity handling inside the RHS. "hard" reproduces MATLAB's Y(Y<0)=0
    # (default, for parity). "smooth" is differentiable and required for stiff
    # solvers (BDF/Radau) to be efficient; see ModelState docstring.
    nonneg: str = "hard"
    smooth_eps: float = 1e-8
    # Optional exact analytic Jacobian (tran_ode_jac) for the stiff path. Verified
    # correct + accuracy-preserving, but OFF by default: for this dense-block stiff
    # system it does not beat scipy's sparse finite-difference Jacobian (the stiff
    # stepping + LU solves dominate, not Jacobian construction). The far bigger
    # lossless win is using RK23 (the MATLAB-parity default), ~10x faster than BDF
    # for realistic parameters. Set True to use the analytic Jacobian with BDF/Radau.
    use_analytic_jac: bool = False

    def simulate(
        self,
        inlet: InletProfile,
        loading_g_l: float,
        *,
        n_time_points: int | None = None,
        method: str | None = None,
        t_start: float = 0.0,
    ) -> SimulationResult:
        """Integrate ODE from t=t_start to end of inlet profile."""
        if self.backend is not None:
            return self.backend.simulate(
                simulator=self,
                inlet=inlet,
                loading_g_l=loading_g_l,
                n_time_points=n_time_points,
                method=method,
                t_start=t_start,
            )

        n_tp = n_time_points or self.n_time_points
        meth = method or self.method

        # The hard Y(Y<0)=0 clip is non-smooth and defeats implicit solvers'
        # Newton iteration (BDF NaNs, LSODA is slower than RK23). When a stiff
        # method is requested, fall back to the smooth non-negativity unless the
        # caller explicitly asked for a non-default value.
        nonneg = self.nonneg
        if meth in _STIFF_METHODS and nonneg == "hard":
            nonneg = "smooth"

        state = build_model_state(
            self.column,
            self.components,
            inlet,
            loading_g_l,
            self.correction,
            nonneg=nonneg,
            smooth_eps=self.smooth_eps,
        )
        gs = self.column.grid_size
        nc = self.components.n_total
        y0 = np.zeros(gs * (nc * 2 - 1))
        y0[:gs] = inlet.segments[0, 0]  # equilibrated salt

        t_eval = np.linspace(t_start, inlet.duration, n_tp)
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 1.0
        # MATLAB uses 1-based ``ceil(feedtime / t(2))`` and then slices from
        # that row. Convert the selected row to Python's 0-based indexing.
        feed_end_index = int(np.ceil((inlet.segments[0, 3] - t_start) / dt)) - 1
        feed_end_index = max(0, min(feed_end_index, len(t_eval) - 1))

        def rhs(t, y):
            return tran_ode_rhs(t, y, state)

        solver_kwargs: dict = dict(method=meth, rtol=self.rtol, atol=1e-8)
        if meth in _STIFF_METHODS:
            if self.use_analytic_jac and nonneg == "smooth":
                # Exact analytic Jacobian: skips the colored finite-difference RHS
                # evals entirely (the dense N makes that ~grid_size evals/build).
                solver_kwargs["jac"] = lambda tt, yy: tran_ode_jac(tt, yy, state)
            else:
                # A sparse Jacobian pattern turns the implicit Jacobian build from
                # ``ndim`` RHS evaluations into a handful (~4x speedup overall).
                solver_kwargs["jac_sparsity"] = build_jacobian_sparsity(gs, nc)
        else:
            # Starting at half the output interval best matches MATLAB ode23
            # for the reference tspan-vector workflow (explicit RK only).
            solver_kwargs["first_step"] = 0.5 * dt

        sol = solve_ivp(
            rhs,
            (t_start, inlet.duration),
            y0,
            t_eval=t_eval,
            **solver_kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        if _solve_hook is not None:
            _solve_hook()

        y = sol.y.T
        gs = self.column.grid_size
        n_protein = self.components.n_protein

        salt = y[:, gs - 1]  # outlet node for salt (component 0)
        protein = np.column_stack(
            [y[:, (i + 1) * gs - 1] for i in range(1, n_protein + 1)]
        )

        return SimulationResult(
            time_s=sol.t,
            salt_mol_l=salt,
            protein_mol_l=protein,
            feed_end_index=feed_end_index,
            raw_solution=sol,
        )

    def run_fitting_case(
        self,
        *,
        buffer_a: float,
        buffer_b: float,
        gradient_start_pct: float,
        gradient_end_pct: float,
        elution_cv: float,
        load_amount_g_l: float,
        experimental_data: np.ndarray | None = None,
        lag_s: float = 0.0,
        min_conc_g_l: float = 0.3,
        observation_groups: list[list[int]] | None = None,
    ) -> tuple[SimulationResult, np.ndarray | None]:
        """Run simulation for parameter fitting (MATLAB ``SMA.m`` workflow)."""
        fractions = self.components.fraction_array()
        inlet = build_fitting_inlet(
            buffer_a=buffer_a,
            buffer_b=buffer_b,
            gradient_start_pct=gradient_start_pct,
            gradient_end_pct=gradient_end_pct,
            elution_cv=elution_cv,
            rt_min=self.column.rt,
            load_amount_g_l=load_amount_g_l,
            component_fractions_pct=fractions,
        )
        result = self.simulate(inlet, load_amount_g_l)
        curve = result.elution_curve()

        mse = None
        if experimental_data is not None:
            mse = compute_rmse(curve, experimental_data, lag_s, min_conc_g_l, observation_groups)
        return result, mse

    def run_forward_case(
        self,
        *,
        buffer_a: float,
        buffer_b: float,
        gradient_start_pct: float,
        gradient_end_pct: float,
        elution_cv: float,
        load_amount_g_l: float,
        n_time_points: int = 2000,
    ) -> SimulationResult:
        """Run forward simulation (MATLAB ``generate_curve.m``)."""
        fractions = self.components.fraction_array()
        inlet = build_simulation_inlet(
            buffer_a=buffer_a,
            buffer_b=buffer_b,
            gradient_start_pct=gradient_start_pct,
            gradient_end_pct=gradient_end_pct,
            elution_cv=elution_cv,
            rt_min=self.column.rt,
            load_amount_g_l=load_amount_g_l,
            component_fractions_pct=fractions,
        )
        return self.simulate(
            inlet,
            load_amount_g_l,
            n_time_points=n_time_points,
            # Honor the simulator's method (default RK23 = MATLAB generate_curve.m
            # parity; BDF for fast gradient/collection optimization sweeps).
            method=self.method,
            t_start=1.0,  # matches MATLAB generate_curve.m linspace(1, ...)
        )
