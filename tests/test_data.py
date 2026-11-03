import numpy as np
import pytest

from src.data.datasets import Standardizer, prepare_data, load_raw


def test_breast_cancer_shapes_and_label_flip():
    X, y, names = load_raw("breast_cancer")
    assert X.shape == (569, 30) and len(names) == 30
    # positive class = malignant = 212 samples (sklearn's own coding has 212 zeros)
    assert y.sum() == 212


def test_splits_are_disjoint_complete_and_stratified():
    b = prepare_data("breast_cancer", split_seed=0)
    all_idx = np.concatenate([b.idx_train, b.idx_val, b.idx_test])
    assert len(np.unique(all_idx)) == len(all_idx) == 569
    rate = 212 / 569
    for y in (b.y_train, b.y_val, b.y_test):
        assert abs(y.mean() - rate) < 0.03


def test_scaler_uses_training_statistics_only():
    b = prepare_data("breast_cancer", split_seed=0)
    assert np.allclose(b.X_train.mean(0), 0, atol=1e-8)
    assert np.allclose(b.X_train.std(0), 1, atol=1e-8)
    # val/test are NOT exactly standardized -> they did not influence the scaler
    assert not np.allclose(b.X_test.mean(0), 0, atol=1e-8)


def test_split_is_deterministic():
    a, b = prepare_data("breast_cancer", 0), prepare_data("breast_cancer", 0)
    assert np.array_equal(a.idx_test, b.idx_test)
    assert not np.array_equal(a.idx_test, prepare_data("breast_cancer", 1).idx_test)


def test_standardizer_roundtrip_and_constant_feature():
    X = np.random.default_rng(0).normal(size=(50, 3))
    X[:, 1] = 5.0
    s = Standardizer.fit(X)
    assert np.allclose(s.inverse(s.transform(X)), X)
    assert np.isfinite(s.transform(X)).all()


def test_spambase_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/DATA.md"):
        load_raw("spambase", data_dir=tmp_path)


def test_magic_and_covtype_loaders():
    import pytest
    from src.utils.repro import PROJECT_ROOT
    if not (PROJECT_ROOT / "data" / "magic04.data").exists():
        pytest.skip("data files not downloaded")
    X, y, names = load_raw("magic")
    assert X.shape == (19020, 10) and len(names) == 10 and y.sum() == 12332  # 12,332 gamma
    X, y, names = load_raw("covtype_binary", max_samples=5000, subsample_seed=0)
    assert X.shape == (5000, 10) and set(np.unique(y)) == {0, 1}
    assert abs(y.mean() - 211840 / 495141) < 0.01  # class ratio preserved by the subsample
    X2, y2, _ = load_raw("covtype_binary", max_samples=5000, subsample_seed=0)
    assert np.array_equal(X, X2)  # deterministic
