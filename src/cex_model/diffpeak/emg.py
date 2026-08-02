"""Differentiable exponentially-modified Gaussian (EMG) peak shape (D4.1).

A chromatographic peak that tails: a Gaussian convolved with a one-sided exponential. Parameterised by
``area`` (= the time-integral of the profile), Gaussian centre ``mu``, Gaussian width ``sigma`` and the
exponential tail constant ``tau`` (the skew). As ``tau -> 0`` it reduces to a Gaussian(area, mu, sigma), so a
Gaussian fit is the ``tau -> 0`` ablation. Everything is torch + float64 so the whole peak/process model is
differentiable end-to-end and trains by gradient descent directly on real outlet curves (no RK23, no SMA).
"""

from __future__ import annotations

import torch

_SQRT2 = 2.0 ** 0.5


def erfcx(x: torch.Tensor) -> torch.Tensor:
    """Scaled complementary error function exp(x^2)*erfc(x). Uses ``torch.special.erfcx`` when available
    (stable for all x); otherwise exp(x^2)*erfc(x) with the exponent clamped -- safe here because a peak whose
    centre/width sit inside the elution window keeps |x| small (the clamp only guards pathological optimiser
    steps)."""
    fn = getattr(torch.special, "erfcx", None)
    if fn is not None:
        return fn(x)
    return torch.exp(torch.clamp(x * x, max=60.0)) * torch.erfc(x)


def emg(t: torch.Tensor, area: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
        tau: torch.Tensor) -> torch.Tensor:
    """EMG concentration profile c(t) (tailing, ``tau`` > 0), differentiable and numerically stable.

    c(t) = area/(2*tau) * exp(-0.5*((t-mu)/sigma)^2) * erfcx( (sigma/tau + (mu-t)/sigma)/sqrt(2) )

    which is the standard EMG factored through ``erfcx`` so the exp/erfc product never overflows. The
    time-integral of c is exactly ``area``; ``mu`` is the Gaussian centre (the EMG mean is ``mu + tau``).
    Non-negative for area, sigma, tau > 0.
    """
    sigma = sigma.clamp_min(1e-6)
    tau = tau.clamp_min(1e-6)
    z = (sigma / tau + (mu - t) / sigma) / _SQRT2
    gauss = torch.exp(-0.5 * ((t - mu) / sigma) ** 2)
    return area / (2.0 * tau) * gauss * erfcx(z)
