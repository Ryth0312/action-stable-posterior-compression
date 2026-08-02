"""Low-order, physically-constrained, differentiable peak/process model (D4.1).

Per observed peak ``j`` the four EMG parameters are LOW-ORDER functions of the operating conditions -- NOT an
MLP, because each product has only 3-6 experiments (an MLP would memorise them and tell us nothing). The
physical constraints are baked into the parameterisation rather than added as soft penalties where possible:

* area_j   = loading * frac_j * softplus(eta_j)        -> area >= 0 and PROPORTIONAL to loaded mass x fraction
                                                          (no gradient term: total eluted mass is ~conserved)
* mu_j     = mu0_j + beta_j . z(op)                    -> retention time, LINEAR in standardized OP (slopes
                                                          beta are the identifiability question)
* sigma_j  = softplus(sig0_j + delta_j . z(op))        -> width > 0, smooth in OP
* tau_j    = softplus(tau0_j)                          -> skew >= 0 (per-peak constant; regularised toward 0)

Everything internal works in STANDARDIZED time (t -> (t - t0)/t_scale) so mu/sigma/tau are O(1) and a single
learning rate conditions well (peaks sit at thousands of seconds; in raw units the optimiser crawls). The
predicted concentration VALUE is scale-invariant, so it is compared directly to the measured curve; only
area/retention are converted back to seconds for reporting (``peak_params_real``).

``beta`` / ``delta`` (the OP->parameter slopes) carry an L2 prior so directions the 3-6 experiments cannot
constrain shrink to zero -- which is exactly what the LOO / bootstrap diagnostics then read out.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cex_model.diffpeak.emg import emg

DTYPE = torch.float64
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2.0 renamed trapz -> trapezoid
# floors (in STANDARDIZED time) keep width/skew away from 0 with a smooth gradient -- a collapsing peak spikes
# the EMG and blows the gradient up (LOO folds diverged to nan without this). Well below real peak widths.
SIGMA_FLOOR, TAU_FLOOR = 0.02, 0.01


class DiffPeakModel(nn.Module):
    def __init__(self, n_peaks: int, frac, op_mean, op_std, t0: float, t_scale: float):
        super().__init__()
        n_op = len(op_mean)
        self.n_peaks = n_peaks
        self.register_buffer("frac", torch.as_tensor(np.asarray(frac), dtype=DTYPE))
        self.register_buffer("op_mean", torch.as_tensor(np.asarray(op_mean), dtype=DTYPE))
        self.register_buffer("op_std", torch.as_tensor(np.asarray(op_std), dtype=DTYPE))
        self.register_buffer("t0", torch.tensor(float(t0), dtype=DTYPE))
        self.register_buffer("t_scale", torch.tensor(float(t_scale), dtype=DTYPE))
        # baselines initialised in fit (data-dependent); slopes start at 0 (the OP-independent model)
        self.eta = nn.Parameter(torch.zeros(n_peaks, dtype=DTYPE))           # area coeff (pre-softplus)
        self.mu0 = nn.Parameter(torch.zeros(n_peaks, dtype=DTYPE))           # retention baseline (std time)
        self.beta = nn.Parameter(torch.zeros(n_peaks, n_op, dtype=DTYPE))    # retention OP-slopes
        self.sig0 = nn.Parameter(torch.zeros(n_peaks, dtype=DTYPE))          # width baseline (pre-softplus)
        self.delta = nn.Parameter(torch.zeros(n_peaks, n_op, dtype=DTYPE))   # width OP-slopes
        self.tau0 = nn.Parameter(torch.zeros(n_peaks, dtype=DTYPE))          # skew (pre-softplus)

    def _z(self, op: torch.Tensor) -> torch.Tensor:
        # clamp the standardized OP so an OOD experiment (e.g. an LOO held-out that varies an OP the train fold
        # held CONSTANT -> op_std ~ 0 -> |z| ~ 1e9) SATURATES the linear slope instead of blowing up to inf.
        return torch.clamp((op - self.op_mean) / self.op_std, -4.0, 4.0)

    def peak_params(self, op: torch.Tensor):
        """(area_z, mu_z, sigma_z, tau_z) per peak in STANDARDIZED time for one experiment. ``loading`` = op[0];
        area is in standardized-time units (concentration values are scale-invariant)."""
        z = self._z(op)
        area = op[0] * self.frac * nn.functional.softplus(self.eta)
        mu = self.mu0 + self.beta @ z
        sigma = SIGMA_FLOOR + nn.functional.softplus(self.sig0 + self.delta @ z)
        tau = TAU_FLOOR + nn.functional.softplus(self.tau0)
        return area, mu, sigma, tau

    def peak_params_real(self, op: torch.Tensor):
        """(area, mu, sigma, tau) in REAL units (seconds; area = g/L*s) for metrics / reporting."""
        area, mu, sigma, tau = self.peak_params(op)
        return area * self.t_scale, self.t0 + self.t_scale * mu, self.t_scale * sigma, self.t_scale * tau

    def predict_peaks(self, op: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """(n_peaks, T) predicted concentration per observed peak at ``times`` for one experiment."""
        area, mu, sigma, tau = self.peak_params(op)
        tz = ((times - self.t0) / self.t_scale).unsqueeze(0)  # (1, T) standardized time
        return emg(tz, area.unsqueeze(1), mu.unsqueeze(1), sigma.unsqueeze(1), tau.unsqueeze(1))

    def predict_total(self, op: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return self.predict_peaks(op, times).sum(dim=0)

    @torch.no_grad()
    def init_baselines_(self, data) -> None:
        """Warm-start baselines from the data (in standardized time): mu0 = mean centroid, sig0 ~ peak spread,
        eta so area ~ the observed peak integral, tau0 small. Keeps the optimiser out of flat EMG regions."""
        s = float(self.t_scale); t0 = float(self.t0)
        mu0 = np.zeros(self.n_peaks); sig0 = np.zeros(self.n_peaks); eta = np.zeros(self.n_peaks)
        inv_softplus = lambda y: float(np.log(np.expm1(np.clip(y, 1e-9, None))))  # noqa: E731
        for j in range(self.n_peaks):
            mus, sigs, etas = [], [], []
            for e in data.experiments:
                c = e.curves[j]; tot = c.sum()
                if tot <= 1e-9:
                    continue
                ctr = float((e.times * c).sum() / tot)
                var = float((e.times ** 2 * c).sum() / tot - ctr ** 2)
                area = float(_trapz(c, e.times))
                mus.append((ctr - t0) / s); sigs.append(np.sqrt(max(var, 1.0)) / s)
                etas.append((area / s) / max(e.loading * data.frac[j], 1e-9))
            mu0[j] = np.mean(mus) if mus else 0.0
            sig0[j] = inv_softplus(np.mean(sigs) - SIGMA_FLOOR) if sigs else 0.0   # sigma = SIGMA_FLOOR + softplus
            eta[j] = inv_softplus(np.mean(etas)) if etas else -5.0
        self.mu0.copy_(torch.as_tensor(mu0, dtype=DTYPE))
        self.sig0.copy_(torch.as_tensor(sig0, dtype=DTYPE))
        self.eta.copy_(torch.as_tensor(eta, dtype=DTYPE))
        self.tau0.copy_(torch.full((self.n_peaks,), inv_softplus(0.1 - TAU_FLOOR), dtype=DTYPE))  # small skew


def save_diffpeak(path, model: DiffPeakModel, peak_types, frac) -> None:
    """Persist a fitted peak model (the buffers frac/op stats/time scale are in ``state_dict``)."""
    torch.save({"state_dict": model.state_dict(), "n_peaks": int(model.n_peaks),
                "n_op": int(model.beta.shape[1]), "peak_types": list(peak_types),
                "frac": np.asarray(frac).tolist()}, path)


def load_diffpeak(path):
    """Inverse of :func:`save_diffpeak` → ``(model.eval(), peak_types)``."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    n_peaks, n_op = int(d["n_peaks"]), int(d["n_op"])
    model = DiffPeakModel(n_peaks, np.zeros(n_peaks), np.zeros(n_op), np.ones(n_op), 0.0, 1.0).to(DTYPE)
    model.load_state_dict(d["state_dict"])  # restores the real buffers (frac / op stats / t0 / t_scale)
    model.eval()
    return model, tuple(d["peak_types"])
