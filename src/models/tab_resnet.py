"""Tabular ResNet target model.

Follows the ResNet baseline of Gorishniy et al. (2021), "Revisiting Deep Learning
Models for Tabular Data" (rtdl), re-implemented in plain PyTorch:

    x -> Linear(d_in, d_main)
      -> N x [ x + Dropout(Linear(Dropout(act(Linear(Norm(x)))))) ]
      -> Norm -> act -> Linear(d_main, 1)

Deviations from the reference, made on purpose for this project (documented in
docs/DESIGN_DECISIONS.md):

* ``activation`` defaults to GELU instead of ReLU. The surrogate pipeline relies on
  input gradients (Integrated Gradients, SmoothGrad, Sobolev matching); ReLU nets have
  piecewise-constant gradients, which are poor gradient teachers.
* The model outputs one **logit** (positive-class logit) of shape ``(B,)``. This is the
  scalar that is explained and distilled throughout the project.

The model is differentiable end to end with respect to its (standardized) input.
Always call ``model.eval()`` before computing gradients/explanations so that dropout
is off and batch-norm uses its running statistics.
"""
from __future__ import annotations

import torch
from torch import nn

_ACTS = {"gelu": nn.GELU, "relu": nn.ReLU, "softplus": nn.Softplus, "tanh": nn.Tanh}


def _norm(kind: str, d: int) -> nn.Module:
    if kind == "batchnorm":
        return nn.BatchNorm1d(d)
    if kind == "layernorm":
        return nn.LayerNorm(d)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown normalization {kind!r}")


class ResBlock(nn.Module):
    def __init__(self, d_main, d_hidden, act, norm, dropout_first, dropout_second):
        super().__init__()
        self.norm = _norm(norm, d_main)
        self.lin1 = nn.Linear(d_main, d_hidden)
        self.act = _ACTS[act]()
        self.drop1 = nn.Dropout(dropout_first)
        self.lin2 = nn.Linear(d_hidden, d_main)
        self.drop2 = nn.Dropout(dropout_second)

    def forward(self, x):
        h = self.drop1(self.act(self.lin1(self.norm(x))))
        return x + self.drop2(self.lin2(h))


class TabResNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_main: int = 128,
        n_blocks: int = 4,
        d_hidden_factor: float = 2.0,
        activation: str = "gelu",
        normalization: str = "batchnorm",
        dropout_first: float = 0.1,
        dropout_second: float = 0.0,
    ):
        super().__init__()
        d_hidden = int(d_main * d_hidden_factor)
        self.first = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList(
            ResBlock(d_main, d_hidden, activation, normalization, dropout_first, dropout_second)
            for _ in range(n_blocks)
        )
        self.head_norm = _norm(normalization, d_main)
        self.head_act = _ACTS[activation]()
        self.head = nn.Linear(d_main, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.head_act(self.head_norm(x))).squeeze(-1)


# Keys of the config's `model:` section that are constructor arguments.
ARCH_KEYS = (
    "d_main", "n_blocks", "d_hidden_factor", "activation", "normalization",
    "dropout_first", "dropout_second",
)


def build_model(d_in: int, model_cfg: dict) -> TabResNet:
    kwargs = {k: model_cfg[k] for k in ARCH_KEYS if k in model_cfg}
    return TabResNet(d_in, **kwargs)


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint written by ``src.training.train_target``.

    Returns ``(model_in_eval_mode, checkpoint_dict)``. The checkpoint dict also holds
    the scaler statistics, feature names and the split indices.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(len(ckpt["feature_names"]), ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
