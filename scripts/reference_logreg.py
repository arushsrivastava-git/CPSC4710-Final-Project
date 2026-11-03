"""Linear reference: logistic regression on the same splits/preprocessing as the target.

Purpose: context for the report. If a linear model is about as accurate as the ResNet, the
task is close to linear and a linear surrogate may already explain the target well (see the
complexity check in the README). This is a *global* comparison only; the local check
(linear surrogate vs. target in a neighborhood) comes with the baselines.

Usage: .venv/bin/python scripts/reference_logreg.py
Writes runs/reference_logreg/results.json and appends to docs/EXPERIMENT_LOG.md notes manually.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from src.data.datasets import prepare_data  # noqa: E402
from src.utils.repro import PROJECT_ROOT, environment_info, write_json  # noqa: E402

# dataset -> (log1p, max_samples); must match the target configs in configs/
SETTINGS = {
    "breast_cancer": (False, None),
    "spambase": (True, None),
    "magic": (False, None),
    "covtype_binary": (False, 20000),
}
res = {}
for name, (log1p, max_samples) in SETTINGS.items():
    d = prepare_data(name, split_seed=0, log1p=log1p, max_samples=max_samples)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):  # picked on the validation split
        m = LogisticRegression(C=C, max_iter=5000).fit(d.X_train, d.y_train)
        v = roc_auc_score(d.y_val, m.predict_proba(d.X_val)[:, 1])
        if best is None or v > best[0]:
            best = (v, C, m)
    _, C, m = best
    p = m.predict_proba(d.X_test)[:, 1]
    res[name] = {"C": C, "log1p": log1p, "test_accuracy": accuracy_score(d.y_test, p > 0.5),
                 "test_auc": roc_auc_score(d.y_test, p), "n_test": len(d.y_test),
                 "max_samples": max_samples}
    print(name, res[name])
write_json({"results": res, "environment": environment_info()}, PROJECT_ROOT / "runs/reference_logreg/results.json")
