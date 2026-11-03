"""Stored checkpoints must reload and reproduce the logged test metrics exactly."""
import json

import numpy as np
import pytest
import torch

from src.data.datasets import prepare_data
from src.models.tab_resnet import load_checkpoint
from src.training.train_target import evaluate
from src.utils.repro import PROJECT_ROOT

RUN = PROJECT_ROOT / "runs" / "target_breast_cancer_resnet" / "seed_0"


@pytest.mark.skipif(not (RUN / "model.pt").exists(), reason="run train_target.py first")
def test_checkpoint_reproduces_logged_metrics_and_data():
    model, ck = load_checkpoint(RUN / "model.pt")
    b = prepare_data("breast_cancer", split_seed=0)
    # stored split + scaler match what prepare_data rebuilds
    assert np.array_equal(ck["idx_test"], b.idx_test)
    assert np.allclose(ck["scaler_mean"], b.scaler.mean)
    got = evaluate(model, torch.tensor(b.X_test, dtype=torch.float32),
                   torch.tensor(b.y_test, dtype=torch.float32))
    logged = json.load(open(RUN / "metrics.json"))["metrics"]["test"]
    for k in got:
        assert got[k] == pytest.approx(logged[k], abs=1e-6)
