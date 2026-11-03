import numpy as np
import torch

from src.neighborhoods import sample_neighborhood, teacher_signals
from src.surrogates.linear import fidelity, fit_linear_surrogate, mean_gradient_cosine


class LinearModel(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forward(self, x):
        return x @ self.w + 0.5


def test_neighborhood_is_deterministic_and_centered():
    x0 = np.arange(5.0)
    a = sample_neighborhood(x0, 0.5, 20000, np.random.default_rng(0))
    b = sample_neighborhood(x0, 0.5, 20000, np.random.default_rng(0))
    assert np.array_equal(a, b)
    assert np.allclose(a.mean(0), x0, atol=0.02) and np.allclose(a.std(0), 0.5, atol=0.02)


def test_teacher_gradient_of_linear_model_is_its_weights():
    w = [1.0, -2.0, 3.0]
    f, g = teacher_signals(LinearModel(w), np.random.default_rng(0).normal(size=(10, 3)))
    assert np.allclose(g, np.array(w)[None, :], atol=1e-6)


def test_linear_surrogate_recovers_linear_target_perfectly():
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=4)
    w = np.array([1.0, -2.0, 0.5, 3.0])
    Z = sample_neighborhood(x0, 0.5, 4000, rng)
    f = Z @ w + 0.7
    s = fit_linear_surrogate(Z, f, x0, alpha=1e-9)
    assert np.allclose(s.w, w, atol=1e-5)
    assert fidelity(s.predict(Z), f) > 1 - 1e-9
    assert mean_gradient_cosine(s.w, np.tile(w, (50, 1))) > 1 - 1e-9


def test_linear_surrogate_fits_nonlinear_target_poorly():
    rng = np.random.default_rng(2)
    x0 = np.zeros(3)
    Z = sample_neighborhood(x0, 1.0, 4000, rng)
    f = Z[:, 0] * Z[:, 1] + Z[:, 2] ** 2  # pure interaction + curvature, no linear part at x0
    s = fit_linear_surrogate(Z, f, x0)
    assert fidelity(s.predict(Z), f) < 0.1
