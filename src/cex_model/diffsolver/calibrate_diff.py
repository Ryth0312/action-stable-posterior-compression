"""Gradient-based SMA calibration through the differentiable BDF solver.

Replaces the derivative-free differential evolution (thousands of ODE solves, the source
of the 10+ h / Colab-timeout pain) with Adam on exact autograd gradients from
``TorchSimulator`` — a handful of solves per step. keq/kkin are optimised in log space
(they span decades); nu/sigma linearly. The loss is the per-observed-component MSE between
the simulated elution curve and the experimental fraction points, summed over experiments,
with the product's observation-group mapping (e.g. HLXSYN ``[[0],[1],[2],[3,4]]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cex_model.components import ComponentSet
from cex_model.diffsolver.torch_solver import DTYPE, TorchSimulator

__all__ = ["ExperimentTarget", "fit_sma_adam", "train_isotherm_nn", "predict_curve",
           "targets_from_bundle"]


@dataclass
class ExperimentTarget:
    """One experiment for the differentiable fit: its solver + sampled measurements."""

    sim: TorchSimulator
    times_s: torch.Tensor          # (m,) sample times on the elution-relative axis
    values: torch.Tensor           # (m, n_obs) measured g/L per observed group
    groups: list[list[int]] | None  # model-protein -> observed-column mapping (0-based protein idx)


def targets_from_bundle(bundle, n_steps: int) -> list[ExperimentTarget]:
    """Build per-experiment ExperimentTargets from a product bundle (HLXSYN/03/04, product-agnostic)."""
    from cex_model.gradients import build_fitting_inlet
    fr = bundle.components.fraction_array()
    targets = []
    for e in bundle.experiments:
        inlet = build_fitting_inlet(
            buffer_a=e.buffer_a, buffer_b=e.buffer_b,
            gradient_start_pct=e.gradient_start_pct, gradient_end_pct=e.gradient_end_pct,
            elution_cv=e.elution_cv, rt_min=bundle.column.rt,
            load_amount_g_l=e.loading_g_l, component_fractions_pct=fr)
        sim = TorchSimulator(bundle.column, bundle.components, inlet, e.loading_g_l,
                             bundle.correction, n_steps=n_steps)
        targets.append(ExperimentTarget(
            sim=sim, times_s=torch.tensor(e.curve[:, 0], dtype=DTYPE),
            values=torch.tensor(e.curve[:, 1:], dtype=DTYPE), groups=bundle.observation_groups))
    return targets


def _interp(x, xp, fp):
    """Differentiable 1-D linear interpolation (fp is (n,)), clamped at the ends."""
    idx = torch.clamp(torch.searchsorted(xp.contiguous(), x.contiguous(), right=True),
                      1, xp.shape[0] - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = torch.clamp((x - x0) / (x1 - x0 + 1e-12), 0.0, 1.0)
    return y0 + w * (y1 - y0)


def _grouped_curve(curve, times, groups, n_protein):
    """Model protein outlet (g/L) at ``times``, summed per observation group -> (m, n_obs)."""
    t_model = curve[:, 0]
    cols = []
    obs = groups if groups is not None else [[j] for j in range(n_protein)]
    for grp in obs:
        s = 0.0
        for j in grp:
            s = s + _interp(times, t_model, curve[:, 2 + j])
        cols.append(s)
    return torch.stack(cols, dim=1)


def fit_sma_adam(targets: list[ExperimentTarget], init: ComponentSet, *,
                 iters: int = 150, lr: float = 0.05, fit_sigma: bool = False,
                 differentiable: bool = True, progress: bool = False, checkpoint: bool = False):
    """Adam fit of per-component SMA params to the targets; returns (ComponentSet, history).

    Counts ODE solves so the cost can be compared to differential evolution. ``checkpoint``
    enables gradient checkpointing in the solver (~2-3x slower, but avoids OOM at large n_steps).
    """
    keq0 = torch.tensor(init.keq_array()[1:], dtype=DTYPE)
    kkin0 = torch.tensor(init.kkin_array()[1:], dtype=DTYPE)
    nu0 = torch.tensor(init.nu_array()[1:], dtype=DTYPE)
    sigma0 = torch.tensor(init.sigma_array()[1:], dtype=DTYPE)
    log_keq = torch.log10(keq0).clone().requires_grad_(True)
    log_kkin = torch.log10(kkin0).clone().requires_grad_(True)
    nu = nu0.clone().requires_grad_(True)
    params = [log_keq, log_kkin, nu]
    if fit_sigma:
        sigma = sigma0.clone().requires_grad_(True)
        params.append(sigma)
    else:
        sigma = sigma0
    opt = torch.optim.Adam(params, lr=lr)

    n_protein = init.n_protein
    hist = {"loss": [], "solves": 0}
    for it in range(iters):
        opt.zero_grad()
        keq = 10.0 ** log_keq
        kkin = 10.0 ** log_kkin
        loss = 0.0
        for tg in targets:
            curve = tg.sim.elution_curve(keq, kkin, nu, sigma, differentiable=differentiable,
                                         checkpoint=checkpoint)
            hist["solves"] += 1
            model = _grouped_curve(curve, tg.times_s, tg.groups, n_protein)
            loss = loss + torch.mean((model - tg.values) ** 2)
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        if progress and (it % max(1, iters // 10) == 0 or it == iters - 1):
            print(f"  iter {it:3d}: loss={loss.item():.5e}")

    keq_f = (10.0 ** log_keq).detach().numpy()
    kkin_f = (10.0 ** log_kkin).detach().numpy()
    nu_f = nu.detach().numpy()
    sigma_f = (sigma.detach().numpy() if fit_sigma else sigma0.numpy())
    fitted = _rebuild_components(init, keq_f, kkin_f, nu_f, sigma_f)
    return fitted, hist


def _frozen_params(components: ComponentSet):
    return (torch.tensor(components.keq_array()[1:], dtype=DTYPE),
            torch.tensor(components.kkin_array()[1:], dtype=DTYPE),
            torch.tensor(components.nu_array()[1:], dtype=DTYPE),
            torch.tensor(components.sigma_array()[1:], dtype=DTYPE))


def predict_curve(tg: ExperimentTarget, components: ComponentSet, iso_nn=None):
    """Grouped model curve (g/L) at the target's sample times — pure mechanistic if iso_nn=None."""
    keq, kkin, nu, sigma = _frozen_params(components)
    with torch.no_grad():
        curve = tg.sim.elution_curve(keq, kkin, nu, sigma, differentiable=False, iso_nn=iso_nn)
    return _grouped_curve(curve, tg.times_s, tg.groups, components.n_protein)


def train_isotherm_nn(targets: list[ExperimentTarget], components: ComponentSet, *,
                      iters: int = 200, lr: float = 0.01, hidden: int = 14, layers: int = 2,
                      progress: bool = False, checkpoint: bool = False):
    """DPSOL Stage B: train the shared isotherm-correction NN with the mechanistic params FROZEN.

    Solver-in-the-loop — the NN is applied inside the BDF rollout and its weights are optimised by
    backprop through the differentiable solver (Adam). Zero-init => starts == pure mechanistic.
    Product-agnostic (works for HLXSYN/03/04 via their ExperimentTargets + ComponentSet). Returns
    (IsothermNN, history). Adopt only if it improves held-out RMSE (LOO; see scripts/train_dpsol).
    """
    from cex_model.diffsolver.isotherm_nn import IsothermNN

    keq, kkin, nu, sigma = _frozen_params(components)
    net = IsothermNN(hidden=hidden, layers=layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_protein = components.n_protein
    hist = {"loss": [], "solves": 0}
    for it in range(iters):
        opt.zero_grad()
        loss = 0.0
        for tg in targets:
            curve = tg.sim.elution_curve(keq, kkin, nu, sigma, differentiable=True, iso_nn=net,
                                         checkpoint=checkpoint)
            hist["solves"] += 1
            model = _grouped_curve(curve, tg.times_s, tg.groups, n_protein)
            loss = loss + torch.mean((model - tg.values) ** 2)
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        if progress and (it % max(1, iters // 10) == 0 or it == iters - 1):
            print(f"  iter {it:3d}: loss={loss.item():.5e}")
    return net, hist


def _rebuild_components(init: ComponentSet, keq, kkin, nu, sigma) -> ComponentSet:
    """Rebuild a ComponentSet with new protein params; ``components`` is proteins-only.

    keq/kkin arrive in model units (10**log); convert back to the table units stored on
    SMAComponent (table = model / 1e-2 for keq, / 1e-5 for kkin).
    """
    from dataclasses import replace
    out = [replace(c, keq=float(keq[j]) / 1e-2, kkin=float(kkin[j]) / 1e-5,
                   nu=float(nu[j]), sigma=float(sigma[j]))
           for j, c in enumerate(init.components)]
    return replace(init, components=out)
