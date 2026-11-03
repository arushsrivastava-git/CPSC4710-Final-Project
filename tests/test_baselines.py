import numpy as np
import torch

from src.explainers.gradient import (gradient_times_input, ig_completeness_error, integrated_gradients,
                                     smoothgrad)
from src.metrics.faithfulness import fid_curves, fid_summary, total_effect
from src.models.tab_resnet import TabResNet


class Linear(torch.nn.Module):
    def __init__(self, w, b=0.3):
        super().__init__()
        self.w, self.b = torch.tensor(w, dtype=torch.float32), b

    def forward(self, x):
        return x @ self.w + self.b


W = [3.0, -2.0, 0.5, 0.0]
X0 = np.array([1.0, 1.0, 1.0, 1.0])


def test_ig_on_linear_model_is_w_times_x_and_complete():
    m = Linear(W)
    a = integrated_gradients(m, X0, steps=10)
    assert np.allclose(a, np.array(W) * X0, atol=1e-5)
    assert ig_completeness_error(m, X0, a) < 1e-5


def test_ig_completeness_on_resnet():
    torch.manual_seed(0)
    m = TabResNet(6, d_main=32, n_blocks=3).eval()
    x = np.random.default_rng(0).normal(size=6)
    err = ig_completeness_error(m, x, integrated_gradients(m, x, steps=200))
    assert err < 2e-3
    # more steps -> no worse
    assert ig_completeness_error(m, x, integrated_gradients(m, x, steps=400)) <= err + 1e-6


def test_smoothgrad_and_grad_x_input_on_linear_model():
    m = Linear(W)
    assert np.allclose(smoothgrad(m, X0, n=50, sigma=0.5), W, atol=1e-5)  # constant gradient
    assert np.allclose(gradient_times_input(m, X0), np.array(W) * X0, atol=1e-5)


def test_fid_ground_truth_ranking_beats_random_on_linear_model():
    m = Linear(W)
    good = np.abs(np.array(W))            # true importance
    bad = -np.abs(np.array(W)) + 10       # reversed ranking
    fp_g, fm_g = fid_curves(m, X0, good)
    fp_b, fm_b = fid_curves(m, X0, bad)
    assert fp_g[1:-1].mean() > fp_b[1:-1].mean()   # removing truly important features hurts more
    assert fm_g[1:-1].mean() < fm_b[1:-1].mean()   # keeping them preserves the prediction better
    # endpoints: k=0 changes nothing; k=d on remove-all equals the full effect
    assert fp_g[0] == 0 and fm_g[-1] == 0
    assert np.isclose(fp_g[-1], total_effect(m, X0)) and np.isclose(fm_g[0], total_effect(m, X0))


def test_fid_sign_flip_for_negative_prediction():
    m = Linear([-1.0, 0.0], b=-0.5)  # predicts negative logit at x0
    fp, _ = fid_curves(m, np.array([1.0, 1.0]), np.array([1.0, 0.0]))
    assert fp[1] > 0  # removing the important feature moves the logit toward 0 -> positive drop toward the predicted side
    assert np.isfinite(fid_summary(fp, fp, 1.0)["fid_plus_auc"])
