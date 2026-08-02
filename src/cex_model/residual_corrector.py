"""ML residual layer: learn the model-vs-AKTA total residual and add it back (gated).

The mechanistic model's total-protein curve has a structured residual vs the real AKTA
total (diagnostic: ~2.7x the repeatability noise floor) that the physical levers cannot
remove without breaking the per-component calibration (Pareto conflict). This layer learns
that residual from the fitting experiments and adds it back on top of the *unchanged*
physical model -- the "physical base + ML residual" design, toggled on/off.

Model: nonparametric kernel regression (Nadaraya-Watson) over OPERATING CONDITIONS,
keeping each training experiment's full residual SHAPE (vs gradient fraction phi) and
blending neighbouring conditions by an RBF weight. This is deliberately low-capacity:
with only 3-5 distinct conditions per product a neural net / boosting would overfit. Two
properties make it Phase-2-safe:

* **Coverage gate** -- the unnormalised kernel mass falls off when the query condition is
  far from every training condition, so the correction decays to 0 outside the trained
  operating envelope instead of extrapolating wildly.
* **Peak normalisation** -- residuals are divided by the model peak height before blending
  and rescaled by the query's model peak, so the SHAPE (not the load-dependent magnitude)
  is what generalises across loading.

Scope: corrects the TOTAL curve only (the AKTA UV is single-channel). It improves
total-based Phase-2 outputs (collection window / yield); per-component PURITY still comes
from the physical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ResidualCorrector:
    """Condition-blended residual model on a common gradient-fraction grid."""

    phi_grid: np.ndarray        # (P,) gradient fraction (0=gradient start, 1=end of ramp)
    resid_norm: np.ndarray      # (K, P) peak-normalised residual per training experiment
    cond: np.ndarray            # (K, D) raw operating-condition features
    cond_mean: np.ndarray       # (D,)
    cond_std: np.ndarray        # (D,)
    bandwidth: float            # RBF bandwidth in normalised condition space
    mass_ref: float             # reference kernel mass (gate calibration)

    @classmethod
    def fit(cls, phi_grid, resid_curves, model_peaks, conditions, bandwidth=None):
        """Build from per-experiment residual curves (on a common ``phi_grid``).

        ``resid_curves`` (K,P) g/L, ``model_peaks`` (K,) g/L, ``conditions`` (K,D).
        """
        phi_grid = np.asarray(phi_grid, float)
        resid = np.asarray(resid_curves, float)
        peaks = np.maximum(np.asarray(model_peaks, float).reshape(-1, 1), 1e-6)
        resid_norm = resid / peaks
        cond = np.asarray(conditions, float)
        mean = cond.mean(axis=0)
        std = cond.std(axis=0)
        std[std == 0] = 1.0
        z = (cond - mean) / std

        dist = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))  # (K,K)
        if bandwidth is None:
            off = dist[~np.eye(len(z), dtype=bool)]
            bandwidth = float(np.median(off)) if off.size else 1.0
        bandwidth = max(bandwidth, 1e-3)
        # LOO kernel mass each training point sees from the others -> gate reference
        k = np.exp(-0.5 * (dist / bandwidth) ** 2)
        np.fill_diagonal(k, 0.0)
        loo_mass = k.sum(axis=1)
        mass_ref = float(np.median(loo_mass)) if np.any(loo_mass > 0) else 1.0
        mass_ref = max(mass_ref, 1e-6)

        return cls(phi_grid=phi_grid, resid_norm=resid_norm, cond=cond,
                   cond_mean=mean, cond_std=std, bandwidth=bandwidth, mass_ref=mass_ref)

    def _weights(self, condition) -> tuple[np.ndarray, float]:
        z = (np.asarray(condition, float) - self.cond_mean) / self.cond_std
        d2 = ((self.cond - self.cond_mean) / self.cond_std - z) ** 2
        w = np.exp(-0.5 * d2.sum(-1) / self.bandwidth ** 2)
        return w, float(w.sum())

    def _box_factor(self, condition) -> float:
        """In-hull coverage factor in [0, 1]: 1 inside the training condition box, decaying
        outside it per dimension. Essential with very few conditions, where the kernel-mass
        gate alone cannot tell interpolation from extrapolation (bandwidth ~ point spacing).
        """
        c = np.asarray(condition, float)
        lo, hi = self.cond.min(axis=0), self.cond.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        below = np.maximum(lo - c, 0.0) / (0.5 * span)
        above = np.maximum(c - hi, 0.0) / (0.5 * span)
        return float(np.prod(np.exp(-(below ** 2 + above ** 2))))

    def predict_norm(self, condition) -> tuple[np.ndarray, float]:
        """Peak-normalised residual curve on ``phi_grid`` + coverage gate in [0, 1].

        Gate = (kernel-mass coverage) x (in-hull box factor); -> 0 outside the trained
        operating envelope so the correction never extrapolates into Phase 2.
        """
        w, mass = self._weights(condition)
        if mass <= 0:
            return np.zeros_like(self.phi_grid), 0.0
        curve = (w[:, None] * self.resid_norm).sum(axis=0) / mass
        gate = float(min(1.0, mass / self.mass_ref) * self._box_factor(condition))
        return curve, gate

    def correction(self, condition, phi_query, model_peak) -> tuple[np.ndarray, float]:
        """Additive residual (g/L) at ``phi_query`` for one condition, plus the gate.

        ``correction = gate * model_peak * resid_norm(phi)``. Add it to the model total.
        """
        norm_curve, gate = self.predict_norm(condition)
        r = np.interp(phi_query, self.phi_grid, norm_curve, left=0.0, right=0.0)
        return gate * model_peak * r, gate

    def apply(self, model_total, condition, phi_query, model_peak) -> tuple[np.ndarray, float]:
        """Corrected total = max(model_total + gated residual, 0), plus the gate.

        Clipping at 0 enforces the physical floor: a learned residual must never drive the
        total protein concentration negative (which the raw additive correction can do on
        the leading edge, where the model is ~0 but the blended residual is negative).
        """
        corr, gate = self.correction(condition, phi_query, model_peak)
        return np.maximum(np.asarray(model_total, float) + corr, 0.0), gate

    def correct_curve(self, curve, condition) -> tuple[np.ndarray, float]:
        """Apply the gated total correction to a sim curve ``[time_s, salt, comp1..n]``.

        Returns ``(curve_corrected, gate)``: each protein column is scaled (per time point)
        so the components sum to the gated, clipped corrected total. Wherever the model has
        signal the scaled total equals the corrected total exactly (so yield and the
        total-protein collection-window triggers become ML-driven); component RATIOS at each
        time are preserved (a total-only signal cannot inform the acid/main/basic split --
        purity therefore only shifts via time-reweighting / the window, not a real per-component
        fix). The gradient fraction phi is derived from the simulated salt trace.
        """
        curve = np.asarray(curve, dtype=float)
        t_min = curve[:, 0] / 60.0
        salt = curve[:, 1]
        total = curve[:, 2:].sum(axis=1)
        lo, hi = float(salt.min()), float(salt.max())
        if hi - lo > 0:
            rise = np.where(salt > lo + 0.01 * (hi - lo))[0]
            start = float(t_min[rise[0]]) if rise.size else float(t_min[0])
            end = float(t_min[int(np.argmax(salt))])
        else:  # isocratic: no salt ramp to anchor phi
            start, end = float(t_min[0]), float(t_min[-1])
        phi = (t_min - start) / max(end - start, 1e-6)
        corrected_total, gate = self.apply(total, condition, phi, float(total.max()))
        scale = np.ones_like(total)
        m = total > 1e-9
        scale[m] = corrected_total[m] / total[m]
        out = curve.copy()
        out[:, 2:] = out[:, 2:] * scale[:, None]
        return out, gate

    def save(self, path: str | Path) -> None:
        np.savez(path, phi_grid=self.phi_grid, resid_norm=self.resid_norm, cond=self.cond,
                 cond_mean=self.cond_mean, cond_std=self.cond_std,
                 bandwidth=self.bandwidth, mass_ref=self.mass_ref)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualCorrector":
        d = np.load(path)
        return cls(phi_grid=d["phi_grid"], resid_norm=d["resid_norm"], cond=d["cond"],
                   cond_mean=d["cond_mean"], cond_std=d["cond_std"],
                   bandwidth=float(d["bandwidth"]), mass_ref=float(d["mass_ref"]))
