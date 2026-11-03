import torch

from src.models.tab_resnet import TabResNet


def test_output_shape_is_one_logit_per_sample():
    m = TabResNet(30).eval()
    assert m(torch.randn(7, 30)).shape == (7,)


def test_input_gradient_matches_finite_differences():
    # Gradient correctness is what IG / SmoothGrad / Sobolev training all rely on.
    torch.manual_seed(0)
    m = TabResNet(10, d_main=32, n_blocks=3).double().eval()
    x = torch.randn(4, 10, dtype=torch.double, requires_grad=True)
    g, = torch.autograd.grad(m(x).sum(), x)
    eps = 1e-6
    for j in range(10):
        e = torch.zeros(10, dtype=torch.double)
        e[j] = eps
        fd = (m(x.detach() + e) - m(x.detach() - e)) / (2 * eps)
        assert torch.allclose(g[:, j], fd, rtol=1e-4, atol=1e-6)


def test_eval_mode_is_deterministic():
    m = TabResNet(5, dropout_first=0.5).eval()
    x = torch.randn(3, 5)
    assert torch.equal(m(x), m(x))
