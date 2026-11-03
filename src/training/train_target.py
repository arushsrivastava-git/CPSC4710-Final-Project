"""Train and evaluate one target model (Tabular ResNet) for one seed.

Training recipe: AdamW, mini-batch BCE-with-logits, early stopping on validation loss
(best weights restored). Metrics are reported on train / val / test.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn

from src.data.datasets import DataBundle
from src.models.tab_resnet import build_model
from src.utils.repro import seed_everything


def _tensors(b: DataBundle, device):
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return {
        k: (t(getattr(b, f"X_{k}"), torch.float32), t(getattr(b, f"y_{k}"), torch.float32))
        for k in ("train", "val", "test")
    }


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i : i + batch]) for i in range(0, len(X), batch)])


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    logits = predict_logits(model, X).cpu().numpy().astype(np.float64)
    yy = y.cpu().numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    return {
        "accuracy": float(accuracy_score(yy, p > 0.5)),
        "auc": float(roc_auc_score(yy, p)),
        "log_loss": float(log_loss(yy, np.clip(p, 1e-12, 1 - 1e-12))),
    }


def train_one(cfg: dict, data: DataBundle, seed: int, device: str = "cpu"):
    """Train one model. Returns ``(model, history, metrics, info)``.

    ``cfg`` is the full experiment config; the ``model`` and ``training`` sections are
    used here.
    """
    seed_everything(seed)
    tc = cfg["training"]
    dev = torch.device(device)
    T = _tensors(data, dev)
    model = build_model(data.n_features, cfg["model"]).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr, ytr = T["train"]
    Xva, yva = T["val"]
    gen = torch.Generator().manual_seed(seed)  # controls mini-batch order

    best = {"val_loss": float("inf"), "epoch": -1, "state": None}
    history, bad_epochs = [], 0
    t0 = time.time()
    for epoch in range(tc["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen).to(dev)
        run_loss, n_seen = 0.0, 0
        for i in range(0, len(Xtr), tc["batch_size"]):
            ix = perm[i : i + tc["batch_size"]]
            if len(ix) < 2:  # batch-norm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xtr[ix]), ytr[ix])
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(ix)
            n_seen += len(ix)
        with torch.no_grad():
            model.eval()
            val_loss = loss_fn(model(Xva), yva).item()
        history.append({"epoch": epoch, "train_loss": run_loss / n_seen, "val_loss": val_loss})
        if val_loss < best["val_loss"] - 1e-6:
            best.update(val_loss=val_loss, epoch=epoch, state=copy.deepcopy(model.state_dict()))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= tc["patience"]:
                break
    train_seconds = time.time() - t0

    model.load_state_dict(best["state"])
    model.eval()
    metrics = {k: evaluate(model, *T[k]) for k in ("train", "val", "test")}
    info = {
        "seed": seed,
        "n_params": n_params,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "best_val_loss": best["val_loss"],
        "train_seconds": round(train_seconds, 2),
        "device": device,
    }
    return model, history, metrics, info
