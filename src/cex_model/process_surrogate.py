"""Process-metric surrogate (D3.1): predict Phase-2 OUTCOMES (recovery / yield / purity / feasibility)
directly from operating conditions, trained on RK23 + ``optimize_collection_window`` labels.

This is the D3 pivot after D2.3 (differentiable Phase-2 refine) was recorded as an R&D negative result:
the gradient refiner was twice beaten by a plain pure-ODE DE (Adam counterproductive; the ODE winner was
always the Sobol-screen fallback). Instead of a curve surrogate (the Milestone-1-rejected ANN) or the
differentiable solver, D3.1 learns the PROCESS METRICS the optimiser actually scores, with a loss aimed at
top-K RANKING vs RK23 -- not global curve RMSE, which was the old surrogate's failure mode.

Scope (minimal closed loop): single product, OPERATING-condition features only (product-id / SMA-param
features are the multi-product extension). Labels are the best-PURITY-FEASIBLE collection window's metrics
for a FIXED purity spec; regression heads are trained MASKED to feasible samples (infeasible points carry
no meaningful recovery/yield/purity), while a separate feasibility head classifies everywhere. The surrogate
only PROPOSES -- its top-K candidates are always RK23-verified downstream (lossless), the same contract as
the differentiable/ANN paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

# operating-condition features (flow held fixed per product, so excluded). PROCESS_VARS names.
OP_FEATURES: tuple[str, ...] = ("loading_g_l", "feed_cv", "gradient_start_pct", "gradient_end_pct", "gradient_cv")
# D3.2 multi-spec: the purity spec becomes INPUT features (so one surrogate generalises across specs).
SPEC_FEATURES: tuple[str, ...] = ("acid_max", "main_min", "basic_max")
# D3.3 multi-product: PRODUCT-identity features by GROUP aggregation (so an UNSEEN product is described by its
# physics, not a one-hot id). Every component must be one of the charge-variant classes acid/main/basic; a
# product with a missing group -> count 0 + zeroed stats, an UNCLASSIFIED component -> error (guards the
# HLXSYN-style mis-typing where real acid/basic peaks had been left as `other`). Per-component SMA params are
# aggregated per group (charge == SMA nu; keq/kkin are log-scaled as they span orders of magnitude). Column
# constants appended once.
PRODUCT_GROUPS: tuple[str, ...] = ("acid", "main", "basic")
_SMA_PARAMS: tuple[str, ...] = ("nu", "log_keq", "sigma", "log_kkin")
COLUMN_FEATURES: tuple[str, ...] = ("ionic_capacity", "flow_rate", "column_volume", "total_porosity")
DTYPE = torch.float64


def append_spec_features(X_ops: np.ndarray, spec) -> np.ndarray:
    """Broadcast a (acid_max, main_min, basic_max) spec onto each operating-condition row → (N, n_ops+3).
    Turns a single-spec test set into the 8-feature layout a multi-spec surrogate expects."""
    X_ops = np.asarray(X_ops, dtype=float)
    spec = np.broadcast_to(np.asarray(spec, dtype=float).reshape(-1), (X_ops.shape[0], 3))
    return np.concatenate([X_ops, spec], axis=1)


def append_product_features(X: np.ndarray, prod_vec: np.ndarray) -> np.ndarray:
    """Broadcast a per-product feature vector (constant within a product) onto each row → (N, n+P)."""
    X = np.asarray(X, dtype=float)
    pv = np.asarray(prod_vec, dtype=float).reshape(-1)
    return np.concatenate([X, np.broadcast_to(pv, (X.shape[0], pv.shape[0]))], axis=1)


def _sma_param(c, p: str) -> float:
    if p == "log_keq":
        return float(np.log(max(float(c.keq), 1e-12)))
    if p == "log_kkin":
        return float(np.log(max(float(c.kkin), 1e-12)))
    return float(getattr(c, p))  # nu (== characteristic charge), sigma


def product_features(bundle, *, include_std: bool = True) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fixed-length PRODUCT-identity features (D3.3) by GROUP aggregation over the charge-variant classes
    acid/main/basic. A product that lacks a group is described STRUCTURALLY (count 0, present 0, zeroed param
    stats) rather than memorised by a one-hot id, so the surrogate can predict an UNSEEN product from its
    physics. Per group: count, total (normalised) fraction, present-indicator, and the fraction-weighted mean
    (+ optional std) of the SMA params (nu==charge, log keq, sigma, log kkin). Then a few column/process
    constants appended once. Every component MUST be acid/main/basic -- an unclassified one raises (so the
    HLXSYN-style mis-typing that left real acid/basic peaks as `other` can never silently corrupt the features).
    """
    comps = bundle.components.components
    frac = np.asarray(bundle.components.fraction_array(), dtype=float)
    frac = frac / frac.sum() if frac.sum() > 0 else frac  # -> proportion of feed (fraction_array is in percent)
    names: list[str] = []
    vals: list[float] = []
    assigned: set[int] = set()
    for g in PRODUCT_GROUPS:
        members = [i for i, c in enumerate(comps) if str(c.component_type.value).lower() == g]
        assigned.update(members)
        w = frac[members]
        names += [f"{g}_count", f"{g}_fraction", f"{g}_present"]
        vals += [float(len(members)), float(w.sum()), 1.0 if members else 0.0]
        ww = w / w.sum() if w.sum() > 0 else w  # within-group fraction weights for the param moments
        for p in _SMA_PARAMS:
            x = np.array([_sma_param(comps[i], p) for i in members], dtype=float)
            mean = float((ww * x).sum()) if len(members) else 0.0
            names.append(f"{g}_{p}_mean"); vals.append(mean)
            if include_std:
                std = float(np.sqrt(max(0.0, float((ww * (x - mean) ** 2).sum())))) if len(members) > 1 else 0.0
                names.append(f"{g}_{p}_std"); vals.append(std)
    if len(assigned) != len(comps):
        bad = [(comps[i].name, str(comps[i].component_type.value)) for i in range(len(comps)) if i not in assigned]
        raise ValueError(f"product_features: components not in {PRODUCT_GROUPS} (fix the product's typing): {bad}")
    for cf in COLUMN_FEATURES:
        names.append(f"col_{cf}"); vals.append(float(getattr(bundle.column, cf)))
    return np.asarray(vals, dtype=float), tuple(names)


@dataclass
class ProcessDataset:
    """RK23 Phase-2 labels for a fixed purity spec. ``feasible`` is 0/1; recovery/yield/purity are the
    best-feasible window's metrics (degenerate where ``feasible==0`` -> masked out of the regression)."""

    X: np.ndarray              # (N, n_features) raw operating features (OP_FEATURES order)
    recovery: np.ndarray       # (N,) window yield / whole-curve eluted yield, in [0, 1]
    yield_g: np.ndarray        # (N,) absolute collected protein (MATLAB units)
    acid: np.ndarray           # (N,) collected acid fraction
    main: np.ndarray
    basic: np.ndarray
    feasible: np.ndarray       # (N,) 1 if a purity-feasible window exists, else 0
    feature_names: tuple[str, ...] = OP_FEATURES
    purity: tuple[float, float, float] = (0.20, 0.70, 0.10)  # (acid_max, main_min, basic_max) baked in
    product: str = ""

    def save(self, path) -> None:
        np.savez(path, X=self.X, recovery=self.recovery, yield_g=self.yield_g, acid=self.acid,
                 main=self.main, basic=self.basic, feasible=self.feasible,
                 feature_names=np.array(self.feature_names), purity=np.array(self.purity, dtype=float),
                 product=np.array(self.product))

    @classmethod
    def load(cls, path) -> "ProcessDataset":
        d = np.load(path, allow_pickle=False)
        return cls(X=d["X"], recovery=d["recovery"], yield_g=d["yield_g"], acid=d["acid"], main=d["main"],
                   basic=d["basic"], feasible=d["feasible"], feature_names=tuple(d["feature_names"].tolist()),
                   purity=tuple(float(v) for v in d["purity"]), product=str(d["product"]))


@dataclass
class ProcessSurrogateConfig:
    # Defaults are the config validated on HLXSYN (top-16 overlap 0.81, recovery RMSE 0.039 on the viable
    # region). The top-16 recoveries span only ~0.03, so resolving them needs the deeper net + the strong
    # ranking weight; the tiny 64x64 / weight_ranking=0.5 net could not (RMSE 0.054 > the top spread).
    hidden: tuple[int, ...] = (128, 128, 128)
    lr: float = 1.5e-3
    epochs: int = 800
    weight_decay: float = 1e-5
    weight_recovery: float = 1.0
    weight_log_yield: float = 1.0
    weight_purity: float = 1.0
    weight_feasible: float = 1.0
    weight_ranking: float = 3.0     # pairwise hinge ranking on recovery (the top-K-overlap target)
    ranking_pairs: int = 4096       # random viable pairs per epoch for the ranking term
    val_fraction: float = 0.2
    seed: int = 0


@dataclass
class ProcessScaler:
    """z-score the features; standardise log-yield (fit on feasible TRAIN only)."""

    x_mean: np.ndarray
    x_std: np.ndarray
    logy_mean: float
    logy_std: float

    def x(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.x_mean) / self.x_std

    def logy_z(self, yield_g: np.ndarray) -> np.ndarray:
        return (np.log(np.clip(np.asarray(yield_g, dtype=float), 1e-9, None)) - self.logy_mean) / self.logy_std

    def inv_logy(self, z: np.ndarray) -> np.ndarray:
        return np.exp(np.asarray(z, dtype=float) * self.logy_std + self.logy_mean)


class ProcessMetricSurrogate(nn.Module):
    """MLP trunk + 4 heads: recovery (logit→sigmoid∈[0,1]), standardised log-yield (linear), purity
    (3 logits→softmax, acid/main/basic sum to 1), feasibility (logit→BCE)."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_recovery = nn.Linear(d, 1)
        self.head_log_yield = nn.Linear(d, 1)
        self.head_purity = nn.Linear(d, 3)
        self.head_feasible = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> dict:
        z = self.trunk(x)
        return {"recovery_logit": self.head_recovery(z).squeeze(-1),
                "log_yield_z": self.head_log_yield(z).squeeze(-1),
                "purity_logits": self.head_purity(z),
                "feasible_logit": self.head_feasible(z).squeeze(-1)}


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=float), dtype=DTYPE)


def _ranking_loss(pred: torch.Tensor, true: torch.Tensor, n_pairs: int, gen: torch.Generator) -> torch.Tensor:
    """Pairwise hinge: penalise predicted recovery that orders a random feasible pair against RK23."""
    m = pred.shape[0]
    if m < 2:
        return pred.new_zeros(())
    i = torch.randint(0, m, (n_pairs,), generator=gen)
    j = torch.randint(0, m, (n_pairs,), generator=gen)
    s = torch.sign(true[i] - true[j])
    return torch.relu(-s * (pred[i] - pred[j])).mean()


def fit_scaler(ds: ProcessDataset, *, min_yield: float = 0.0, idx: np.ndarray | None = None) -> ProcessScaler:
    """z-score features + standardise log-yield, fit on ``idx`` rows (default all), with the yield stats over
    VIABLE rows only. Exposed so D3.4 few-shot training (a fresh scaler on N curves) and the pretrained-scaler
    warm start share the EXACT normalisation the main trainer uses."""
    idx = np.arange(len(ds.X)) if idx is None else np.asarray(idx)
    X = ds.X[idx]
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    viable = (ds.feasible[idx] > 0.5) & (ds.yield_g[idx] >= min_yield)
    if viable.any():
        logy = np.log(np.clip(ds.yield_g[idx][viable], 1e-9, None))
        logy_mean, logy_std = float(logy.mean()), float(logy.std() + 1e-8)
    else:
        logy_mean, logy_std = 0.0, 1.0
    return ProcessScaler(x_mean, x_std, logy_mean, logy_std)


def _optimize(model: "ProcessMetricSurrogate", scaler: ProcessScaler, ds: ProcessDataset, tr_idx: np.ndarray,
              cfg: ProcessSurrogateConfig, *, lr: float, min_yield: float, gen: torch.Generator,
              val_idx: np.ndarray | None = None) -> list[dict]:
    """Full-batch optimisation loop shared by pretrain / fine-tune / scratch (so they use the IDENTICAL loss).
    Regression heads (recovery / log-yield / purity) + the ranking term are MASKED to the VIABLE region
    (feasible AND yield >= ``min_yield``); the feasibility head (BCE) trains on ALL rows. ``val_idx`` (if given)
    logs per-epoch val metrics into the returned history."""
    Xt = _t(scaler.x(ds.X))
    rec_t, feas_t, yld_t = _t(ds.recovery), _t(ds.feasible), _t(ds.yield_g)
    logyz_t = _t(scaler.logy_z(ds.yield_g))
    pur_t = _t(np.stack([ds.acid, ds.main, ds.basic], axis=1))
    tr = torch.as_tensor(np.asarray(tr_idx), dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt[tr])
        fmask = (feas_t[tr] > 0.5) & (yld_t[tr] >= min_yield)  # viable = feasible AND above the #4 floor
        loss = cfg.weight_feasible * bce(out["feasible_logit"], feas_t[tr])
        if fmask.any():
            rec_p = torch.sigmoid(out["recovery_logit"][fmask])
            loss = loss + cfg.weight_recovery * torch.mean((rec_p - rec_t[tr][fmask]) ** 2)
            loss = loss + cfg.weight_log_yield * torch.mean((out["log_yield_z"][fmask] - logyz_t[tr][fmask]) ** 2)
            pur_p = torch.softmax(out["purity_logits"][fmask], dim=-1)
            loss = loss + cfg.weight_purity * torch.mean((pur_p - pur_t[tr][fmask]) ** 2)
            if cfg.weight_ranking > 0:
                loss = loss + cfg.weight_ranking * _ranking_loss(rec_p, rec_t[tr][fmask], cfg.ranking_pairs, gen)
        loss.backward()
        opt.step()
        if val_idx is not None and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
            history.append({"epoch": epoch, "loss": loss.item(),
                            **evaluate_metrics(model, scaler, ds, val_idx, min_yield=min_yield)})
    return history


def train_process_surrogate(ds: ProcessDataset, config: ProcessSurrogateConfig | None = None,
                            *, min_yield: float = 0.0):
    """Full-batch train (dataset is small) on a train split; return (model, scaler, history, val_idx).

    Regression heads (recovery / log-yield / purity) + the ranking term train MASKED to the VIABLE region
    (feasible AND yield >= ``min_yield``, the #4 productivity floor) -- this excludes the degenerate
    recovery=1.0 micro-yield windows that #4 exists to reject and that otherwise put a cliff in the recovery
    target. The feasibility head (BCE) still trains on ALL samples. ``history`` logs per-epoch val metrics.
    """
    cfg = config or ProcessSurrogateConfig()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    N = len(ds.X)
    perm = rng.permutation(N)
    n_val = max(1, int(round(cfg.val_fraction * N)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    scaler = fit_scaler(ds, min_yield=min_yield, idx=tr_idx)
    model = ProcessMetricSurrogate(ds.X.shape[1], cfg.hidden).to(DTYPE)
    history = _optimize(model, scaler, ds, tr_idx, cfg, lr=cfg.lr, min_yield=min_yield, gen=gen, val_idx=val_idx)
    return model, scaler, history, val_idx


def finetune_process_surrogate(model: "ProcessMetricSurrogate", scaler: ProcessScaler, ds: ProcessDataset, *,
                               epochs: int, lr: float, min_yield: float = 0.0,
                               config: ProcessSurrogateConfig | None = None, seed: int = 0):
    """D3.4 warm-start: continue training a PRETRAINED model on a small new-product dataset, REUSING the
    pretrained ``scaler`` (so the new product enters the SAME normalised feature space) and the SAME loss as
    the pretrainer. No train/val split -- few-shot data is precious, so all rows train; evaluate on a DISJOINT
    test set. Trains in place and returns the model. (train-from-scratch = a fresh ``ProcessMetricSurrogate`` +
    ``fit_scaler`` on the same N curves + this, giving an apples-to-apples warm- vs cold-start comparison.)"""
    cfg = replace(config or ProcessSurrogateConfig(), epochs=epochs)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    _optimize(model, scaler, ds, np.arange(len(ds.X)), cfg, lr=lr, min_yield=min_yield, gen=gen, val_idx=None)
    return model


def predict(model: ProcessMetricSurrogate, scaler: ProcessScaler, X: np.ndarray) -> dict:
    """Surrogate forward → {recovery, yield_g, acid, main, basic, feasible_prob} as numpy arrays."""
    model.eval()
    with torch.no_grad():
        out = model(_t(scaler.x(X)))
        pur = torch.softmax(out["purity_logits"], dim=-1).cpu().numpy()
        return {"recovery": torch.sigmoid(out["recovery_logit"]).cpu().numpy(),
                "yield_g": scaler.inv_logy(out["log_yield_z"].cpu().numpy()),
                "acid": pur[:, 0], "main": pur[:, 1], "basic": pur[:, 2],
                "feasible_prob": torch.sigmoid(out["feasible_logit"]).cpu().numpy()}


def gated_score(recovery, yield_g, feasible, *, min_yield: float = 0.0, alpha: float = 0.05,
                yield_scale: float = 1.0) -> np.ndarray:
    """The optimiser's selection score as a rankable array: ``recovery + alpha*yield/yield_scale`` where
    feasible AND yield>=min_yield, else ``-inf`` (so infeasible / sub-floor points sink in any top-K)."""
    recovery = np.asarray(recovery, dtype=float)
    yield_g = np.asarray(yield_g, dtype=float)
    ok = (np.asarray(feasible, dtype=float) > 0.5) & (yield_g >= min_yield)
    score = recovery + alpha * yield_g / (yield_scale if yield_scale > 0 else 1.0)
    return np.where(ok, score, -np.inf)


def topk_overlap(true_score: np.ndarray, pred_score: np.ndarray, k: int) -> float:
    """Fraction of the surrogate's predicted top-K that is in RK23's true top-K (the D3 target metric).
    ``-inf`` (infeasible) entries are excluded from the true top-K so overlap isn't inflated by ties."""
    true_score, pred_score = np.asarray(true_score, dtype=float), np.asarray(pred_score, dtype=float)
    n_true = int(np.isfinite(true_score).sum())
    k = min(k, n_true)
    if k <= 0:
        return float("nan")
    t = set(np.argsort(true_score)[::-1][:k].tolist())
    p = set(np.argsort(pred_score)[::-1][:k].tolist())
    return len(t & p) / k


def evaluate_metrics(model: ProcessMetricSurrogate, scaler: ProcessScaler, ds: ProcessDataset,
                     idx: np.ndarray, *, k: int = 10, alpha: float = 0.05, min_yield: float = 0.0) -> dict:
    """Validation metrics on ``idx``. Regression RMSE (recovery / log-yield / purity) and the top-K overlap
    are computed on the VIABLE region (feasible AND yield >= ``min_yield``) so they match the floored search
    -- without the floor the "true top-K" is the degenerate recovery=1.0 micro-yield windows (an arbitrary
    tie-break the surrogate can't and shouldn't match). Feasibility accuracy + false-positive rate (claimed
    feasible but truly infeasible -- the dangerous error) are over all samples."""
    p = predict(model, scaler, ds.X[idx])
    feas_true = ds.feasible[idx] > 0.5
    feas_pred = p["feasible_prob"] >= 0.5
    acc = float(np.mean(feas_pred == feas_true)) if len(idx) else float("nan")
    fp = float(np.mean(feas_pred & ~feas_true)) if len(idx) else float("nan")  # claimed feasible, isn't
    viable = feas_true & (ds.yield_g[idx] >= min_yield)
    out = {"feas_acc": acc, "feas_false_pos": fp, "n_val": int(len(idx)),
           "n_val_feasible": int(feas_true.sum()), "n_val_viable": int(viable.sum())}
    if viable.any():
        m = viable
        out["recovery_rmse"] = float(np.sqrt(np.mean((p["recovery"][m] - ds.recovery[idx][m]) ** 2)))
        out["log_yield_rmse"] = float(np.sqrt(np.mean(
            (scaler.logy_z(p["yield_g"][m]) - scaler.logy_z(ds.yield_g[idx][m])) ** 2)))
        pur_p = np.stack([p["acid"], p["main"], p["basic"]], axis=1)[m]
        pur_t = np.stack([ds.acid[idx], ds.main[idx], ds.basic[idx]], axis=1)[m]
        out["purity_rmse"] = float(np.sqrt(np.mean((pur_p - pur_t) ** 2)))
    yscale = float(np.max(ds.yield_g[idx][viable])) if viable.any() else 1.0
    true_s = gated_score(ds.recovery[idx], ds.yield_g[idx], ds.feasible[idx], min_yield=min_yield,
                         alpha=alpha, yield_scale=yscale)
    pred_s = gated_score(p["recovery"], p["yield_g"], feas_pred, min_yield=min_yield,
                         alpha=alpha, yield_scale=yscale)
    out["topk_overlap"] = topk_overlap(true_s, pred_s, k)
    return out


def save_surrogate(path, model: ProcessMetricSurrogate, scaler: ProcessScaler, config: ProcessSurrogateConfig,
                   *, feature_names, purity, min_yield: float = 0.0, product: str = "") -> None:
    """Persist everything needed to reload for inference (weights + scaler + the feature/spec metadata) so
    the app / a benchmark can reuse a trained surrogate without re-running ``train_process_surrogate``."""
    torch.save({"state_dict": model.state_dict(), "n_features": model.trunk[0].in_features,
                "hidden": tuple(config.hidden), "x_mean": scaler.x_mean, "x_std": scaler.x_std,
                "logy_mean": scaler.logy_mean, "logy_std": scaler.logy_std,
                "feature_names": list(feature_names), "purity": list(purity),
                "min_yield": float(min_yield), "product": str(product)}, path)


def load_surrogate(path):
    """Inverse of :func:`save_surrogate` → ``(model.eval(), scaler, meta)`` where ``meta`` carries
    feature_names / purity / min_yield / product."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    model = ProcessMetricSurrogate(int(d["n_features"]), tuple(d["hidden"])).to(DTYPE)
    model.load_state_dict(d["state_dict"])
    model.eval()
    scaler = ProcessScaler(np.asarray(d["x_mean"]), np.asarray(d["x_std"]),
                           float(d["logy_mean"]), float(d["logy_std"]))
    meta = {"feature_names": tuple(d["feature_names"]), "purity": tuple(d["purity"]),
            "min_yield": float(d["min_yield"]), "product": str(d["product"])}
    return model, scaler, meta
