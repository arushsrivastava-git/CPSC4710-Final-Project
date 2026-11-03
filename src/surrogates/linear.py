"""LIME-style linear surrogate: ridge regression of the target's logit on a neighborhood.

Our own implementation (rather than the ``lime`` package) so that it shares the neighborhood,
scaling and teacher signals with the nonlinear surrogate. Samples can be weighted (LIME uses a
proximity kernel); the local-linearity check uses uniform weights because the Gaussian
neighborhood already defines locality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearSurrogate:
    w: np.ndarray  # (d,) weights = the explanation / constant gradient of the surrogate
    b: float

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.w + self.b


def fit_linear_surrogate(Z, f, x0, sample_weight=None, alpha: float = 1e-3) -> LinearSurrogate:
    """Weighted ridge fit of ``f`` on ``Z - x0`` (centering makes ``b`` the fit value at x0).

    ``alpha`` is a small L2 penalty on the weights (not the intercept) for numerical stability.
    """
    X = Z - x0[None, :]
    sw = np.ones(len(f)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    sw_sum = sw.sum()
    xm = (sw[:, None] * X).sum(0) / sw_sum
    fm = (sw * f).sum() / sw_sum
    Xc, fc = X - xm, f - fm
    A = (Xc * sw[:, None]).T @ Xc / sw_sum + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, (Xc * sw[:, None]).T @ fc / sw_sum)
    b = fm - xm @ w  # intercept in the x0-centered coordinates
    # Express the model on the original Z: f ~ (Z - x0) @ w + b = Z @ w + (b - x0 @ w)
    return LinearSurrogate(w=w, b=float(b - x0 @ w))


def fidelity(surr_pred: np.ndarray, f: np.ndarray) -> float:
    """R^2 of surrogate predictions against the target logits (1 = perfect, <=0 = no better than the mean)."""
    ss_res = float(((f - surr_pred) ** 2).sum())
    ss_tot = float(((f - f.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def mean_gradient_cosine(w: np.ndarray, G: np.ndarray) -> float:
    """Mean cosine similarity between the linear surrogate's (constant) gradient ``w`` and the
    target gradients ``G`` (n, d) at each neighborhood sample."""
    num = G @ w
    den = np.linalg.norm(G, axis=1) * np.linalg.norm(w)
    return float(np.mean(num / np.maximum(den, 1e-30)))
