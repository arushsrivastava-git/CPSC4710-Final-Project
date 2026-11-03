# Data

Both datasets are tabular binary classification. Code: `src/data/datasets.py`.

## Breast Cancer Wisconsin (Diagnostic)
- Source: bundled with scikit-learn (`load_breast_cancer`), so no download is needed.
- 569 samples, 30 continuous features, no missing values; 357 benign / 212 malignant.
- **Label convention: 1 = malignant (the positive class we explain).** scikit-learn codes
  malignant as 0, so labels are flipped in `load_raw`. A test checks that the positive count is 212.

## Spambase
- Source: UCI Machine Learning Repository. Code never downloads it. It was fetched once
  (2026-11-03, with `curl`) from `https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/`
  as `spambase.data` and `spambase.names` into `data/` (git-ignored). To recreate, download both
  files from that URL into `data/`.
- File check: `spambase.data` has 4,601 rows and SHA-256
  `b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12`.
- 4,601 samples, 57 features (48 word frequencies, 6 character frequencies, 3 capital-run
  statistics), last column is the label (1 = spam, the positive class); about 39.4% spam.
- Feature names are hard-coded in `SPAMBASE_FEATURES` in the order of the UCI `.names` file.
  **Verified** on 2026-11-03: all 48 word names match `spambase.names` exactly, and the six
  character features (`; ( [ ! $ #`) and three capital-run features match in order (the code
  spells the symbols out, e.g. `char_freq_exclamation`).
- Features are heavy-tailed and non-negative, so `configs/target_spambase_resnet.yaml` sets
  `log1p: true` (apply log(1+x) before standardizing). This is a project decision; see
  `docs/DESIGN_DECISIONS.md`.

## MAGIC Gamma Telescope
- Source: UCI (`.../machine-learning-databases/magic/magic04.data` and `magic04.names`), fetched with `curl`
  on 2026-11-03 into git-ignored `data/`. SHA-256 of `magic04.data`:
  `e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a`.
- 19,020 samples, 10 continuous features (shower-image parameters: fLength, fWidth, fSize, fConc,
  fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist). No missing values. 12,332 gamma / 6,688 hadron.
- **Label convention: 1 = gamma (signal), 0 = hadron (background).** About 64.8% positive.
- Some features are negative (e.g. fAsym), so no log transform. Split: 13,314 / 2,853 / 2,853.

## Covertype (binary, subsampled)
- Source: UCI (`.../machine-learning-databases/covtype/covtype.data.gz` and `covtype.info`), fetched with
  `curl` on 2026-11-03. SHA-256 of `covtype.data.gz`:
  `614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771`. 581,012 rows, 54 features + class (1 to 7).
- **Derived task (a project decision, not part of the original dataset):** keep only classes 1 (Spruce/Fir,
  211,840 rows) and 2 (Lodgepole Pine, 283,301 rows), the two largest, giving 495,141 rows.
  **Label: 1 = Spruce/Fir, 0 = Lodgepole Pine.** Which class is "positive" is arbitrary.
- **Features: only the 10 continuous columns** (Elevation, Aspect, Slope, the three distance-to-hydrology/
  roadways/fire features, and three hillshade indices). The 4 wilderness-area and 40 soil-type one-hot columns
  are dropped because perturbing binary indicators with Gaussian noise creates off-manifold neighborhoods
  and makes feature-level explanations hard to read. Dropping them may lower accuracy.
- **Subsample:** 20,000 rows, class-stratified, seeded by `split_seed`, for training speed. Split:
  14,000 / 3,000 / 3,000 (about 42.8% positive in each).

## Splits and scaling (identical procedure for all datasets)
1. Stratified split 70% train / 15% validation / 15% test using `split_seed` (default 0).
   Breast Cancer gives 397 / 86 / 86 samples.
2. Optional `log1p`, then standardization with mean/std **computed on the training split only**.
3. The split seed is separate from model seeds, so all seeds share the same test points.
4. Split indices (rows of the raw dataset), scaler mean/std and feature names are stored in every
   checkpoint and in `runs/<run>/splits.npz`, so any later stage can rebuild the exact test set.

All downstream work (neighborhoods, surrogates, explanations) lives in this **standardized**
space; to read attributions in original units, use `Standardizer.inverse`.

Spambase split with `split_seed=0`: 3,220 train / 690 val / 691 test (about 39.4% spam in each).

## Report notes
- The Breast Cancer test set has only 86 samples, so one error is about 1.2 accuracy points.
  Report confidence intervals and avoid over-reading small accuracy differences.
- The EDA from the proposal (class balance, distributions, correlations, near-duplicate feature
  pairs) is **not done yet** (plan step 6).
