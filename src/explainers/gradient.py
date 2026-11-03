"""Gradient-based explainers for any differentiable scalar-output model (target or surrogate).

All functions take a model mapping ``(B, d) -> (B,)`` (the positive-class logit) and one input
``x`` of shape ``(d,)`` (numpy, standardized space) and return a per-feature attribution
``(d,)`` as float64. The model must be in eval mode.

Conventions (see docs/DESIGN_DECISIONS.md):
* The IG reference point ``x'`` (``baseline``) defaults to zeros = the training mean in
  standardized space.
* Attributions are ranked by absolute value when scored (``src/metrics/faithfulness.py``).
"""
from __future__ import annotations

import numpy as np
import torch


def _grad(model, X: torch.Tensor) -> torch.Tensor:
    X = X.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(model(X).sum(), X)
    return g


def integrated_gradients(model, x: np.ndarray, baseline: np.ndarray | None = None, steps: int = 100,
                         device: str = "cpu") -> np.ndarray:
    """Integrated Gradients (Sundararajan et al. 2017), trapezoidal rule over ``steps`` intervals.

    ``IG_i = (x_i - x'_i) * integral_0^1 dF(x' + a (x - x')) / dx_i da``. Completeness:
    ``sum(IG) ~ F(x) - F(x')``; check it with :func:`ig_completeness_error`.
    """
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    b_t = torch.zeros_like(x_t) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32, device=device)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[:, None]
    path = b_t[None, :] + alphas * (x_t - b_t)[None, :]
    g = _grad(model, path).double()  # (steps+1, d)
    avg = (g[:-1] + g[1:]).mean(0) / 2.0  # trapezoidal rule: mean over intervals of the endpoint average
    return ((x_t - b_t).double() * avg).cpu().numpy()


def ig_completeness_error(model, x, attribution, baseline=None, device="cpu") -> float:
    """``|sum(IG) - (F(x) - F(x'))|``: how far IG is from satisfying completeness."""
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)[None, :]
    b_t = torch.zeros_like(x_t) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32, device=device)[None, :]
    with torch.no_grad():
        delta = float(model(x_t) - model(b_t))
    return abs(float(np.sum(attribution)) - delta)


def smoothgrad(model, x: np.ndarray, n: int = 100, sigma: float = 0.25, times_input: bool = False,
               baseline: np.ndarray | None = None, seed: int = 0, device: str = "cpu") -> np.ndarray:
    """SmoothGrad (Smilkov et al. 2017): mean gradient over ``n`` copies ``x + N(0, sigma^2 I)``.

    ``sigma`` is in standardized units. With ``times_input=True`` the result is multiplied by
    ``(x - x')`` (gradient x input style), which puts it on IG's scale.
    """
    rng = np.random.default_rng(seed)
    X = x[None, :] + sigma * rng.standard_normal((n, x.shape[0]))
    g = _grad(model, torch.as_tensor(X, dtype=torch.float32, device=device)).double().mean(0).cpu().numpy()
    if times_input:
        b = np.zeros_like(x) if baseline is None else baseline
        g = g * (x - b)
    return g


def gradient_times_input(model, x: np.ndarray, baseline: np.ndarray | None = None, device: str = "cpu") -> np.ndarray:
    """Plain ``grad F(x) * (x - x')`` (sanity-check baseline)."""
    g = _grad(model, torch.as_tensor(x, dtype=torch.float32, device=device)[None, :]).double()[0].cpu().numpy()
    b = np.zeros_like(x) if baseline is None else baseline
    return g * (x - b)
