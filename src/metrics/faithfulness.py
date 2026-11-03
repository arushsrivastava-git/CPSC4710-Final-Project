"""Explanation faithfulness: fid+ and fid- (keep / remove the top-k attributed features).

Always evaluated against the **target** model ``f`` (never the surrogate), because the
question is whether the explanation is faithful to the original model.

Definitions (docs/DESIGN_DECISIONS.md): let ``s = sign(f(x0))`` (the side the target predicts)
and let a feature be "removed" by replacing it with its baseline value ``x'_i``.

* ``fid+(k) = s * (f(x0) - f(x0 with the top-k features removed))``: how much the prediction
  falls when the most important features are taken away. **Higher is better** (necessity).
* ``fid-(k) = s * (f(x0) - f(x0 with ONLY the top-k features kept))``: how much the prediction
  falls when everything else is taken away. **Lower is better** (sufficiency).

Features are ranked by ``|attribution|``, descending. Curves are computed for ``k = 0..d``.
The summary numbers are the mean over ``k = 1..d-1`` (area under the curve, as a fraction of
the k-axis), reported raw (logit units) and normalized by the total effect
``s * (f(x0) - f(x'))`` when that is large enough to divide by.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _f(model, X: np.ndarray, device="cpu") -> np.ndarray:
    return model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)


def fid_curves(model, x0: np.ndarray, attribution: np.ndarray, baseline: np.ndarray | None = None,
               device: str = "cpu"):
    """Return ``(fid_plus, fid_minus)``, each of shape ``(d + 1,)`` indexed by k = 0..d."""
    d = len(x0)
    b = np.zeros(d) if baseline is None else baseline
    order = np.argsort(-np.abs(attribution), kind="stable")
    # keep-masks: row k keeps the top-k features (others -> baseline)
    ranks = np.empty(d, dtype=int)
    ranks[order] = np.arange(d)
    keep_top = ranks[None, :] < np.arange(d + 1)[:, None]  # (d+1, d): feature kept iff rank < k
    X_keep = np.where(keep_top, x0[None, :], b[None, :])  # only top-k kept
    X_rem = np.where(keep_top, b[None, :], x0[None, :])  # top-k removed
    f0 = float(_f(model, x0[None, :], device)[0])
    s = 1.0 if f0 >= 0 else -1.0
    return s * (f0 - _f(model, X_rem, device)), s * (f0 - _f(model, X_keep, device))


def fid_summary(fid_plus: np.ndarray, fid_minus: np.ndarray, total_effect: float, eps: float = 1e-3) -> dict:
    """Area-under-curve summaries over k = 1..d-1 (raw and normalized by ``total_effect``)."""
    p, m = fid_plus[1:-1].mean(), fid_minus[1:-1].mean()
    out = {"fid_plus_auc": float(p), "fid_minus_auc": float(m)}
    if abs(total_effect) > eps:
        out["fid_plus_auc_norm"] = float(p / total_effect)
        out["fid_minus_auc_norm"] = float(m / total_effect)
    else:
        out["fid_plus_auc_norm"] = out["fid_minus_auc_norm"] = float("nan")
    return out


def total_effect(model, x0: np.ndarray, baseline: np.ndarray | None = None, device="cpu") -> float:
    """``s * (f(x0) - f(x'))``: the full drop in the logit from the data point to the baseline."""
    b = np.zeros_like(x0) if baseline is None else baseline
    f = _f(model, np.stack([x0, b]), device)
    s = 1.0 if f[0] >= 0 else -1.0
    return float(s * (f[0] - f[1]))
