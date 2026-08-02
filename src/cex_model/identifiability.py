"""Sensitivity-based identifiability analysis (differentiable-physics diagnostic).

This is a *read-only* diagnostic. It does NOT change any fitting/optimization
path. The motivation (see ``PROJECT_STATE.md`` §4.4/§7.0): high-precision DE
fails to recover the basic components' SMA parameters, and the suspected root
cause is *identifiability*, not optimizer strength -- ``keq`` and ``nu`` are
degenerate under the single-gradient (loading-varied) data, so many parameter
sets fit the curve about equally well. An optimizer cannot recover information
the data does not contain.

This module turns that qualitative suspicion into numbers using the standard
sensitivity / Fisher-information approach:

1. ``compute_sensitivities`` evaluates the *relative* sensitivity matrix
   ``S[i, j] = d(curve_i) / d(ln theta_j)`` at a nominal parameter point, by
   central differences on the **differentiable (smooth-clip) simulator** (the
   ``nonneg="smooth"`` path exists precisely so the right-hand side -- and hence
   the curve -- is differentiable; central differences on it approximate the
   true derivative rather than differentiating across the hard clip's kink).
   Each observed component is normalized by its own peak height so components of
   very different magnitude contribute on an equal footing.

2. ``fisher_analysis`` forms the Fisher information ``FIM = S^T S`` and reports:
   * the eigenvalue spectrum and condition number (large => ill-conditioned =>
     some parameter combination is unidentifiable);
   * the eigenvectors of the smallest eigenvalues (the degenerate directions --
     e.g. ``+0.7 keq[B1] - 0.7 nu[B1]`` *is* the keq<->nu degeneracy);
   * the parameter correlation matrix ``corr = D^-1 (FIM)^+ D^-1`` (a pair near
     ``+-1`` is practically unidentifiable as a pair);
   * per-parameter influence (``||S[:, j]||``) and a Cramer-Rao style relative
     standard error (larger => less identifiable).

Working in ``ln theta`` makes every quantity scale-free, which matters because
``keq`` spans orders of magnitude across components (HLXSYN: 6e-4 .. 0.14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cex_model.column import ColumnParameters
from cex_model.components import ComponentSet
from cex_model.corrections import LoadingCorrection
from cex_model.simulator import ChromatographySimulator

PARAM_NAMES = ("keq", "kkin", "nu", "sigma")
# Row of each fitted parameter in ``ComponentSet.to_parameter_table`` (row 0 is
# the fixed mass fraction, which is not fitted).
_PARAM_ROW = {"keq": 1, "kkin": 2, "nu": 3, "sigma": 4}


@dataclass
class SensitivityResult:
    """Relative sensitivity matrix of the outlet curve w.r.t. ln(parameters)."""

    S: np.ndarray             # (n_obs, n_param): d(curve)/d(ln theta), peak-normalized
    param_labels: list[str]   # e.g. "keq[B1]", aligned with S columns
    nominal: np.ndarray       # nominal theta (table units), aligned with S columns


@dataclass
class IdentifiabilityReport:
    """Fisher-information identifiability summary at a nominal parameter point."""

    param_labels: list[str]
    eigenvalues: np.ndarray    # FIM eigenvalues, descending
    eigenvectors: np.ndarray   # columns aligned with ``eigenvalues``
    condition_number: float    # eigenvalue_max / eigenvalue_min (inf if singular)
    correlation: np.ndarray    # (n_param, n_param) parameter correlation
    sensitivity_norm: np.ndarray   # ||S[:, j]|| per parameter (influence)
    cramer_rao_std: np.ndarray     # sqrt(diag(pinv(FIM))), relative (up to noise scale)

    def degenerate_pairs(self, threshold: float = 0.95) -> list[tuple[str, str, float]]:
        """Parameter pairs whose correlation magnitude exceeds ``threshold``.

        A pair near +-1 is practically unidentifiable *as a pair*: the data
        constrains a combination of them, not each one. Sorted most-degenerate
        first.
        """
        n = len(self.param_labels)
        out: list[tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                c = float(self.correlation[i, j])
                if abs(c) >= threshold:
                    out.append((self.param_labels[i], self.param_labels[j], c))
        out.sort(key=lambda t: abs(t[2]), reverse=True)
        return out

    def worst_directions(self, n_dirs: int = 3, top: int = 4) -> list[list[tuple[str, float]]]:
        """Top contributors to the ``n_dirs`` least-constrained eigen-directions.

        The eigenvector of the smallest eigenvalue is the combination of
        parameters the data constrains *least*; its largest-magnitude entries
        (with sign) name the degenerate combination.
        """
        dirs: list[list[tuple[str, float]]] = []
        for k in range(min(n_dirs, self.eigenvectors.shape[1])):
            vec = self.eigenvectors[:, -(k + 1)]
            order = np.argsort(np.abs(vec))[::-1][:top]
            dirs.append([(self.param_labels[i], float(vec[i])) for i in order])
        return dirs

    def least_identifiable(self, k: int = 5) -> list[tuple[str, float]]:
        """Parameters with the largest Cramer-Rao relative uncertainty."""
        order = np.argsort(self.cramer_rao_std)[::-1][:k]
        return [(self.param_labels[i], float(self.cramer_rao_std[i])) for i in order]

    def summary(self, corr_threshold: float = 0.95) -> str:
        """Human-readable verdict for printing/logging."""
        lines = []
        lines.append(
            f"Fisher condition number: {self.condition_number:.3e}  "
            f"(eigenvalues {self.eigenvalues[0]:.3e} .. {self.eigenvalues[-1]:.3e})"
        )
        lines.append("")
        lines.append(f"Degenerate parameter pairs (|corr| >= {corr_threshold}):")
        pairs = self.degenerate_pairs(corr_threshold)
        if pairs:
            for a, b, c in pairs:
                lines.append(f"  {a:<12} <-> {b:<12}  corr = {c:+.4f}")
        else:
            lines.append("  (none)")
        lines.append("")
        lines.append("Least-constrained directions (smallest eigenvalues):")
        for k, d in enumerate(self.worst_directions()):
            terms = "  ".join(f"{w:+.2f}*{name}" for name, w in d)
            lines.append(f"  #{k+1} (eig={self.eigenvalues[-(k+1)]:.3e}): {terms}")
        lines.append("")
        lines.append("Least identifiable parameters (Cramer-Rao relative std):")
        for name, s in self.least_identifiable():
            lines.append(f"  {name:<12}  std ~ {s:.3e}")
        return "\n".join(lines)


def _simulate_protein_curve(sim: ChromatographySimulator, case: dict) -> np.ndarray:
    """Outlet protein concentrations (g/L) over the elution window, ``(n_time, n_protein)``."""
    result, _ = sim.run_fitting_case(
        buffer_a=case["buffer_a"],
        buffer_b=case["buffer_b"],
        gradient_start_pct=case["gradient_start_pct"],
        gradient_end_pct=case["gradient_end_pct"],
        elution_cv=case["elution_cv"],
        load_amount_g_l=case["load_amount_g_l"],
    )
    return result.elution_curve()[:, 2:]  # drop time + salt columns


def compute_sensitivities(
    column: ColumnParameters,
    components: ComponentSet,
    experiments: list[dict],
    *,
    params: tuple[str, ...] = PARAM_NAMES,
    rel_step: float = 5e-3,
    method: str = "RK23",
    n_time_points: int = 600,
    rtol: float = 1e-7,
    scheme: str = "central",
    correction: LoadingCorrection | None = None,
    peak_floor: float = 1e-3,
    progress: Callable[[float], None] | None = None,
) -> SensitivityResult:
    """Relative parameter sensitivities of the outlet curve at a nominal point.

    Parameters
    ----------
    experiments
        Operating-condition dicts (``buffer_a``, ``buffer_b``,
        ``gradient_start_pct``, ``gradient_end_pct``, ``elution_cv``,
        ``load_amount_g_l``) -- the same keys the fitting path consumes. The
        experimental *data* is not needed: identifiability is a property of the
        model + operating conditions, not of one noisy measurement.
    params
        Which SMA parameters to include per component.
    rel_step
        Difference step in ``ln theta`` (multiplicative perturbation
        ``theta * exp(+-rel_step)``). The derivative error is ``O(rel_step^2)``
        truncation (``O(rel_step)`` for ``scheme='forward'``) plus
        ``O(rtol / rel_step)`` solver roundoff, so ``rtol`` must be well below
        ``rel_step`` or the smallest Fisher eigenvalues drown in roundoff (and
        the problem looks better-conditioned than it is).
    rtol
        ODE relative tolerance for the sensitivity solves. Tightened well below
        the default ``1e-5`` so the differences are clean; with the default
        ``1e-7`` the roundoff floor at ``rel_step=5e-3`` is ~2e-5.
    scheme
        ``"central"`` (default, ``1 + 2*n_param`` solves/experiment, accurate) or
        ``"forward"`` (``1 + n_param`` solves/experiment, ~2x faster, less
        accurate) -- the latter is useful for an interactive run.
    progress
        Optional callback invoked with a 0..1 fraction after each ODE solve, to
        drive a UI progress bar.
    """
    if scheme not in ("central", "forward"):
        raise ValueError(f"scheme must be 'central' or 'forward', got {scheme!r}")
    sim = ChromatographySimulator(
        column=column,
        components=components,
        method=method,
        rtol=rtol,
        n_time_points=n_time_points,
        nonneg="smooth",  # differentiable right-hand side
        **({"correction": correction} if correction is not None else {}),
    )
    base_table = components.to_parameter_table()  # (5, n_protein)
    cnames = components.names

    # Component-major parameter order so each component's params (and its
    # keq<->nu block) sit adjacently in the correlation matrix.
    labels: list[str] = []
    idx: list[tuple[int, int]] = []
    nominal: list[float] = []
    for c in range(components.n_protein):
        for p in params:
            r = _PARAM_ROW[p]
            labels.append(f"{p}[{cnames[c]}]")
            idx.append((r, c))
            nominal.append(float(base_table[r, c]))

    solves_per_exp = 1 + (2 if scheme == "central" else 1) * len(idx)
    total_solves = max(len(experiments) * solves_per_exp, 1)
    done = 0

    def _tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(min(done / total_solves, 1.0))

    blocks: list[np.ndarray] = []
    for case in experiments:
        sim.components = ComponentSet.from_parameter_table(base_table)
        y0 = _simulate_protein_curve(sim, case)
        _tick()
        scale = np.maximum(y0.max(axis=0), peak_floor)  # per-component peak height

        cols: list[np.ndarray] = []
        for r, c in idx:
            tp = base_table.copy()
            tp[r, c] *= np.exp(rel_step)
            sim.components = ComponentSet.from_parameter_table(tp)
            yp = _simulate_protein_curve(sim, case)
            _tick()
            if scheme == "central":
                tm = base_table.copy()
                tm[r, c] *= np.exp(-rel_step)
                sim.components = ComponentSet.from_parameter_table(tm)
                ym = _simulate_protein_curve(sim, case)
                _tick()
                dyd = (yp - ym) / (2.0 * rel_step)  # d(curve)/d(ln theta)
            else:
                dyd = (yp - y0) / rel_step
            cols.append((dyd / scale).ravel(order="F"))
        blocks.append(np.column_stack(cols))

    S = np.vstack(blocks)
    return SensitivityResult(S=S, param_labels=labels, nominal=np.asarray(nominal))


def fisher_analysis(sens: SensitivityResult) -> IdentifiabilityReport:
    """Fisher-information eigen / correlation analysis of a sensitivity matrix."""
    S = sens.S
    fim = S.T @ S
    evals, evecs = np.linalg.eigh(fim)  # ascending, symmetric
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    emin = float(evals[-1])
    cond = float(evals[0] / emin) if emin > 0 else float("inf")

    # Covariance via pseudo-inverse: the FIM is (near-)singular exactly when a
    # parameter combination is unidentifiable, so a plain inverse would blow up.
    cov = np.linalg.pinv(fim)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    return IdentifiabilityReport(
        param_labels=list(sens.param_labels),
        eigenvalues=evals,
        eigenvectors=evecs,
        condition_number=cond,
        correlation=corr,
        sensitivity_norm=np.linalg.norm(S, axis=0),
        cramer_rao_std=d,
    )
