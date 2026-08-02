"""Loading-dependent solid-phase activity corrections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class LoadingCorrection(ABC):
    """Interface for loading-dependent solid-phase corrections."""

    @abstractmethod
    def gamma(self, loading_g_l: float, n_components: int) -> np.ndarray:
        """Return multiplicative factors for solid phase Q per component."""


@dataclass
class IdentityCorrection(LoadingCorrection):
    """No correction (gamma = 1 for all components)."""

    def gamma(self, loading_g_l: float, n_components: int) -> np.ndarray:
        return np.ones(n_components)


@dataclass
class LinearLoadingCorrection(LoadingCorrection):
    """Linear loading correction: q' = q * (loading * a + b).

    Default coefficients match MATLAB ``tran_ode.m`` HLXSYN/04 hard-coded values.
    Salt (index 0) is always 1.0.
    """

    a: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.02, 0.005, 0.0, -0.0025, 0.0])
    )
    b: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.725, 1.0, 1.1125, 1.0])
    )

    def gamma(self, loading_g_l: float, n_components: int) -> np.ndarray:
        a = self.a[:n_components]
        b = self.b[:n_components]
        g = np.ones(n_components)
        g[0] = 1.0
        g[1:] = loading_g_l * a[1:] + b[1:]
        return g

    @property
    def n_proteins(self) -> int:
        """Number of proteins these coefficients were designed for (salt excluded)."""
        return len(self.a) - 1

    @classmethod
    def from_config(cls, cfg: dict) -> LinearLoadingCorrection:
        return cls(
            a=np.array(cfg.get("a", [0.0, 0.02, 0.005, 0.0, -0.0025, 0.0])),
            b=np.array(cfg.get("b", [1.0, 0.0, 0.725, 1.0, 1.1125, 1.0])),
        )


def make_loading_correction(
    mode: str = "auto",
    n_proteins: int | None = None,
    coeffs: dict | None = None,
) -> LoadingCorrection:
    """Resolve the loading correction for a product.

    The default ``a``/``b`` coefficients are hard-coded for a **5-protein** product
    (HLXSYN). Applying them to a product with a different number of proteins silently
    mis-maps each component's coefficient and corrupts the physics — so ``"auto"``
    only enables the linear correction when the coefficient count matches.

    ``coeffs`` lets a product override the defaults with its own fitted
    ``{"a": [...], "b": [...]}`` (e.g. HLXSYN, whose high-load acid over-sharpens
    under the HLXSYN coefficients — see scripts/refit_loading_correction.py). Pass
    ``components.loading_correction`` here.

    mode:
      * ``"off"``  -> :class:`IdentityCorrection` (gamma = 1).
      * ``"on"``   -> :class:`LinearLoadingCorrection` (caller asserts it fits).
      * ``"auto"`` -> linear only when ``n_proteins`` matches the coefficients,
        otherwise identity. This is the safe default.
    """
    mode = (mode or "auto").lower()
    if mode == "off":
        return IdentityCorrection()
    if coeffs and "a" in coeffs and "b" in coeffs:
        linear = LinearLoadingCorrection.from_config(coeffs)
    else:
        linear = LinearLoadingCorrection()
    if mode == "on":
        return linear
    if mode == "auto":
        if n_proteins is not None and n_proteins == linear.n_proteins:
            return linear
        return IdentityCorrection()
    raise ValueError(f"unknown loading-correction mode {mode!r}")
