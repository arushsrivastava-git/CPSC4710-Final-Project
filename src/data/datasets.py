"""Dataset loading, stratified splitting and standardization.

Design rules (see docs/DATA.md for the reasoning behind each):

* Label convention: **1 = the "positive" class we explain** (malignant for Breast
  Cancer, spam for Spambase). Note this flips scikit-learn's Breast Cancer coding,
  where 1 = benign.
* The split seed is independent of the model seed, so every model seed is trained
  and evaluated on the *same* train/val/test partition, and the test points we later
  explain are fixed.
* The scaler is fit on the training split only and stored with the split, so no
  validation/test statistics leak into preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases"
SPAMBASE_URL = f"{UCI_BASE}/spambase/spambase.data"

_SPAM_WORDS = (
    "make address all 3d our over remove internet order mail receive will people report "
    "addresses free business email you credit your font 000 money hp hpl george 650 lab "
    "labs telnet 857 data 415 85 technology 1999 parts pm direct cs meeting original "
    "project re edu table conference"
).split()
_SPAM_CHARS = ["semicolon", "paren", "bracket", "exclamation", "dollar", "hash"]
SPAMBASE_FEATURES = (
    [f"word_freq_{w}" for w in _SPAM_WORDS]
    + [f"char_freq_{c}" for c in _SPAM_CHARS]
    + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
)


MAGIC_FEATURES = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
]
COVTYPE_FEATURES = [  # the 10 continuous columns; the 44 one-hot columns are dropped (see docs/DATA.md)
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am",
    "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
]


def _subsample(X, y, max_samples, seed):
    """Random subsample of at most ``max_samples`` rows, preserving the class ratio."""
    if max_samples is None or max_samples >= len(y):
        return X, y
    keep, _ = train_test_split(
        np.arange(len(y)), train_size=max_samples, stratify=y, random_state=seed
    )
    keep = np.sort(keep)
    return X[keep], y[keep]


def _need(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint} (see docs/DATA.md). No automatic download is performed.")


def load_raw(
    name: str,
    data_dir: str | Path | None = None,
    max_samples: int | None = None,
    subsample_seed: int = 0,
):
    """Return ``(X, y, feature_names)`` with ``y`` in {0, 1}, 1 = positive class.

    ``max_samples`` (large datasets only) draws a class-stratified random subsample with a
    fixed seed; the subsample is part of the dataset definition and is recorded in the config.
    """
    root = Path(data_dir or PROJECT_ROOT / "data")
    if name == "magic":
        path = root / "magic04.data"
        _need(path, f"Download magic04.data from {UCI_BASE}/magic/")
        raw = np.genfromtxt(path, delimiter=",", dtype=str)
        X = raw[:, :10].astype(np.float64)
        y = (raw[:, 10] == "g").astype(np.int64)  # 1 = gamma (signal), 0 = hadron (background)
        return (*_subsample(X, y, max_samples, subsample_seed), list(MAGIC_FEATURES))
    if name == "covtype_binary":
        path = root / "covtype.data.gz"
        _need(path, f"Download covtype.data.gz from {UCI_BASE}/covtype/")
        arr = np.loadtxt(path, delimiter=",")  # numpy reads .gz transparently
        cls = arr[:, 54].astype(np.int64)
        arr = arr[(cls == 1) | (cls == 2)]  # Spruce/Fir vs Lodgepole Pine (the two largest classes)
        X = arr[:, :10].astype(np.float64)
        y = (arr[:, 54] == 1).astype(np.int64)  # 1 = Spruce/Fir, 0 = Lodgepole Pine
        return (*_subsample(X, y, max_samples, subsample_seed), list(COVTYPE_FEATURES))
    if name == "breast_cancer":
        d = load_breast_cancer()
        # sklearn: 0 = malignant, 1 = benign. Flip so 1 = malignant (positive class).
        y = 1 - d.target.astype(np.int64)
        return d.data.astype(np.float64), y, list(d.feature_names)
    if name == "spambase":
        path = Path(data_dir or PROJECT_ROOT / "data") / "spambase.data"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download {SPAMBASE_URL} and place it there "
                "(see docs/DATA.md). No automatic download is performed."
            )
        arr = np.loadtxt(path, delimiter=",")
        assert arr.shape[1] == 58, f"expected 58 columns, got {arr.shape[1]}"
        return arr[:, :57].astype(np.float64), arr[:, 57].astype(np.int64), list(SPAMBASE_FEATURES)
    raise ValueError(f"unknown dataset {name!r}")


@dataclass
class Standardizer:
    """Per-feature (x - mean) / std with statistics from the training split."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        std = X.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)  # constant feature: leave unscaled
        return cls(mean=X.mean(axis=0), std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean


@dataclass
class DataBundle:
    """Everything downstream code needs about one prepared dataset."""

    name: str
    feature_names: list[str]
    X_train: np.ndarray  # standardized
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray  # indices into the raw dataset
    idx_val: np.ndarray
    idx_test: np.ndarray
    scaler: Standardizer
    log1p: bool

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def prepare_data(
    name: str,
    split_seed: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    log1p: bool = False,
    data_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> DataBundle:
    """Load, (optionally) log-transform, stratified-split and standardize a dataset.

    ``log1p`` applies ``log(1 + x)`` to every feature *before* standardization; only
    valid for non-negative features (true for Spambase, not for Breast Cancer).
    """
    if max_samples is not None:
        X, y, names = load_raw(name, data_dir, max_samples=max_samples, subsample_seed=split_seed)
    else:
        X, y, names = load_raw(name, data_dir)
    if log1p:
        if (X < 0).any():
            raise ValueError("log1p requested but dataset has negative feature values")
        X = np.log1p(X)

    idx = np.arange(len(y))
    idx_trval, idx_test = train_test_split(
        idx, test_size=test_frac, stratify=y, random_state=split_seed
    )
    rel_val = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=rel_val, stratify=y[idx_trval], random_state=split_seed
    )

    scaler = Standardizer.fit(X[idx_train])  # training statistics only
    return DataBundle(
        name=name,
        feature_names=names,
        X_train=scaler.transform(X[idx_train]),
        y_train=y[idx_train],
        X_val=scaler.transform(X[idx_val]),
        y_val=y[idx_val],
        X_test=scaler.transform(X[idx_test]),
        y_test=y[idx_test],
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        scaler=scaler,
        log1p=log1p,
    )


def split_summary(b: DataBundle) -> dict:
    """Sizes and positive rates per split (for the run record / report tables)."""
    out = {}
    for k in ("train", "val", "test"):
        y = getattr(b, f"y_{k}")
        out[k] = {"n": int(len(y)), "positive_rate": float(y.mean())}
    out["n_features"] = b.n_features
    return out
