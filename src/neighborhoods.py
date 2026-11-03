"""Neighborhood sampling and teacher (target-model) signals.

A neighborhood of a test point ``x0`` is an isotropic Gaussian cloud in **standardized
feature space**: ``z = x0 + sigma * eps``, ``eps ~ N(0, I)``. ``sigma`` is in units of
training-set standard deviations (so sigma=1 perturbs each feature by about one std).

NOTE: in d dimensions the cloud's radius is about ``sigma * sqrt(d)``, so the same sigma is a
much larger neighborhood for Spambase (d=57) than for MAGIC (d=10). Compare sigma across
datasets with care.
"""
from __future__ import annotations

import numpy as np
import torch


def sample_neighborhood(x0: np.ndarray, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` Gaussian perturbations of ``x0`` (shape ``(d,)``) -> array ``(n, d)``."""
    return x0[None, :] + sigma * rng.standard_normal((n, x0.shape[0]))


def teacher_signals(model: torch.nn.Module, Z: np.ndarray, device: str = "cpu", batch: int = 4096):
    """Target logits ``f(z)`` and input gradients ``grad f(z)`` for each row of ``Z``.

    Returns ``(f, g)`` as float64 arrays of shapes ``(n,)`` and ``(n, d)``. The model must be
    in eval mode (dropout off, batch-norm in running-stat mode), which makes every sample's
    output independent of the others, so the batched gradient of ``sum(f)`` is exactly the
    per-sample gradient.
    """
    model.eval()
    fs, gs = [], []
    for i in range(0, len(Z), batch):
        z = torch.as_tensor(Z[i : i + batch], dtype=torch.float32, device=device).requires_grad_(True)
        out = model(z)
        (g,) = torch.autograd.grad(out.sum(), z)
        fs.append(out.detach().cpu().numpy())
        gs.append(g.cpu().numpy())
    return np.concatenate(fs).astype(np.float64), np.concatenate(gs).astype(np.float64)
