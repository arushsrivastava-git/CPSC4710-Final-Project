# Environment

- Machine: Apple M4, 16 GB RAM, macOS (Darwin 25.5.0).
- Python 3.14.7, PyTorch 2.11.0 (MPS available), NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.8.0.
- A virtual environment lives in `.venv/` (git-ignored). It was created with
  `python3 -m venv --system-site-packages .venv` so it reuses the already-installed
  torch/numpy/pandas/scikit-learn, and `pyyaml`, `pytest` and `matplotlib` were added into it.
  The system Python is externally managed (PEP 668), so nothing is installed globally.
- Run everything with `.venv/bin/python` (e.g. `.venv/bin/python -m pytest`).
- Every run writes `runs/<run>/environment.json` with exact versions, git commit, a dirty-tree
  flag, and the config hash, so reports can cite what actually ran.
- Training uses CPU: at this model and data size it is as fast as MPS and avoids MPS
  non-determinism. Training one seed on Breast Cancer takes about 0.5 to 1.5 seconds.
- Determinism: all seeds are fixed (`seed_everything`, plus a seeded generator for mini-batch
  order). CPU results are expected to reproduce across reruns on the same machine and library
  versions; this has not been verified beyond the checkpoint round-trip test.
