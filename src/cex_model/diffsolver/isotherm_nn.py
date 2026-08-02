"""DPSOL isotherm-correction neural network (Stage B).

A small, product-agnostic NN that corrects the SMA adsorption term inside the differentiable
solver — the "gray box" of Chen 2025 DPSOL. It is applied per (grid node, protein) and **shared
across components and products** (the component's own keq/nu/sigma are inputs), so it learns a
correction to the adsorption *isotherm* (a material property, function of local state) rather than
a per-condition residual — which is what lets it generalise from a few experiments (the §6f failure
mode of the output-residual layer). The last layer is zero-initialised, so at the start the NN
output is 0 → the corrected model == the pure mechanistic model (no perturbation to a good fit).

Transport (convection/dispersion), mass conservation and the mechanistic SMA parameters are
untouched; only the adsorption term gets the multiplicative ``(1 + r)`` correction. Trained
solver-in-the-loop with the mechanistic params frozen (see calibrate_diff.train_isotherm_nn).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cex_model.diffsolver.torch_solver import DTYPE


class IsothermNN(nn.Module):
    """Per-(node, component) adsorption correction r; corrected ads = ads * clamp(1 + r, >=0)."""

    N_FEAT = 7  # [c_i, q_i, csalt, sma_sum/inocap, log10 keq_i, nu_i/10, sigma_i/50]

    def __init__(self, hidden: int = 14, layers: int = 2):
        super().__init__()
        dims = [self.N_FEAT] + [hidden] * layers
        seq: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            seq += [nn.Linear(a, b), nn.ReLU()]
        out = nn.Linear(dims[-1], 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)  # zero-init -> r=0 -> identity correction at start
        seq.append(out)
        self.net = nn.Sequential(*seq).to(DTYPE)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


def adsorption_features(c_i, q_i, csalt, sma_sum, keq_p, nu_p, sigma_p, inocap: float):
    """Build the (gs, npr, N_FEAT) feature tensor for the shared isotherm NN.

    c_i, q_i: (gs, npr) liquid/solid protein conc; csalt, sma_sum: (gs,); keq/nu/sigma: (npr,).
    Concentrations are scaled by the ionic capacity and keq is log-scaled so the inputs sit in a
    comparable range; the zero-init output makes the exact normalisation non-critical.
    """
    gs, npr = c_i.shape
    cs = (csalt[:, None] / inocap).expand(gs, npr)
    ss = (sma_sum[:, None] / inocap).expand(gs, npr)
    logkeq = torch.log10(torch.clamp(keq_p, min=1e-12))[None, :].expand(gs, npr)
    nuf = (nu_p / 10.0)[None, :].expand(gs, npr)
    sgf = (sigma_p / 50.0)[None, :].expand(gs, npr)
    return torch.stack([c_i / inocap, q_i / inocap, cs, ss, logkeq, nuf, sgf], dim=-1)
