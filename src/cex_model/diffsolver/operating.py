"""Differentiable inlet / time grid / loading-correction as functions of the Phase-2 operating
conditions (D2.1).

Makes :class:`~cex_model.diffsolver.torch_solver.TorchSimulator` differentiable w.r.t. the
decision variables {loading_g_l, gradient_start_pct, gradient_end_pct, gradient_cv, feed_cv}
(flow is fixed per product) by rebuilding ``sim.t`` / ``sim.inlet_c`` / ``sim.gamma_p`` /
``sim.feed_end_t`` in torch from those (tensor) variables. The existing implicit-function-theorem
BDF step then propagates ``d(curve)/d(op)`` unchanged -- ``rhs`` already reads ``sim.gamma_p`` and
the per-step ``inlet_c`` / step ``h``, so once those are torch functions of the op tensors the
gradient flows with no change to the solver.

The salt schedule reconstructs ``gradients.build_fitting_inlet`` exactly: a constant wash level
(1e-4) over the load phase ``[0, feedtime]``, a linear ramp ``gradient_start -> gradient_end`` over
the gradient ``[feedtime, grad_end_t]``, then a hold at ``gradient_end`` -- selected by ``torch.where``
(exact; valid subgradients at the measure-zero phase boundaries, which the dispersive ODE smooths in
the output anyway). Proteins are fed at a constant loading-scaled concentration over ``[0, feedtime]``.
The initial column salt (``y0`` = 1e-4) is operating-condition-independent and left untouched.
"""

from __future__ import annotations

import numpy as np
import torch

from cex_model.corrections import LinearLoadingCorrection, LoadingCorrection
from cex_model.diffsolver.torch_solver import DTYPE, TorchSimulator

LOAD_SALT = 1e-4  # wash/equilibration salt during the load phase (build_fitting_inlet inlet[0])


def linear_gamma_coeffs(correction: LoadingCorrection, nc: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-protein (a_p, b_p) so ``gamma_p(loading) = a_p * loading + b_p`` reproduces
    ``correction.gamma(loading, nc)[1:]`` (affine in loading; identity / gamma==1 -> a=0, b=1)."""
    npr = nc - 1
    if isinstance(correction, LinearLoadingCorrection):
        return (np.asarray(correction.a[1:nc], dtype=float),
                np.asarray(correction.b[1:nc], dtype=float))
    return np.zeros(npr), np.ones(npr)  # IdentityCorrection / any constant gamma == 1


def _scalar(x) -> torch.Tensor:
    """0-dim DTYPE tensor; pass-through for tensors (preserves requires_grad), wrap for floats."""
    return x if torch.is_tensor(x) else torch.tensor(float(x), dtype=DTYPE)


def bind_operating_conditions(
    sim: TorchSimulator, *, loading, gradient_start_pct, gradient_end_pct, gradient_cv, feed_cv,
    buffer_a: float, buffer_b: float, fractions, hold_cv: float,
    correction: LoadingCorrection | None = None, mol_to_g_l: float = 150_000.0,
) -> TorchSimulator:
    """Set ``sim``'s grid / inlet / gamma to differentiable torch functions of the operating conditions.

    The 5 op inputs may be 0-dim tensors (``requires_grad=True`` for Phase-2 gradients) or floats.
    ``buffer_a`` / ``buffer_b`` / ``fractions`` (protein mass %, length npr) / ``hold_cv`` are fixed
    per product. Reconstructs ``build_fitting_inlet`` exactly (forward parity); a later
    ``sim.integrate(keq, kkin, nu, sigma)`` with fixed params then yields ``d(curve)/d(op)``.
    Mutates and returns ``sim`` (``sim.y0`` -- the op-independent column salt -- is left untouched).
    """
    correction = correction or LinearLoadingCorrection()
    loading = _scalar(loading)
    gs_pct, ge_pct = _scalar(gradient_start_pct), _scalar(gradient_end_pct)
    gcv, fcv = _scalar(gradient_cv), _scalar(feed_cv)

    rt_s = sim.rt_min * 60.0  # seconds per CV (flow fixed -> constant)
    feedtime = fcv * rt_s
    grad_end_t = feedtime + gcv * rt_s
    sim_end_t = grad_end_t + hold_cv * rt_s
    gstart = buffer_a * (1.0 - gs_pct / 100.0) + buffer_b * (gs_pct / 100.0)
    gend = buffer_a * (1.0 - ge_pct / 100.0) + buffer_b * (ge_pct / 100.0)

    # fixed N points stretched to [0, sim_end_t] (matches linspace(0, inlet.duration, N))
    u = torch.linspace(0.0, 1.0, sim.n_steps + 1, dtype=DTYPE)
    t = u * sim_end_t
    frac = torch.clamp((t - feedtime) / torch.clamp(grad_end_t - feedtime, min=1e-9), 0.0, 1.0)
    salt_ramp = gstart + frac * (gend - gstart)
    salt = torch.where(t <= feedtime, torch.full_like(t, LOAD_SALT), salt_ramp)

    fr = torch.as_tensor(np.asarray(fractions, dtype=float), dtype=DTYPE)  # (npr,)
    prot_conc = fr / mol_to_g_l * loading / 100.0 / 10.0                   # (npr,), ~ loading
    feed_mask = (t <= feedtime).to(DTYPE)
    inlet_c = torch.cat([salt[:, None], feed_mask[:, None] * prot_conc[None, :]], dim=1)  # (n, nc)

    a_p, b_p = linear_gamma_coeffs(correction, sim.nc)
    gamma_p = torch.as_tensor(a_p, dtype=DTYPE) * loading + torch.as_tensor(b_p, dtype=DTYPE)  # (npr,)

    sim.t, sim.inlet_c, sim.gamma_p, sim.feed_end_t = t, inlet_c, gamma_p, feedtime
    return sim
