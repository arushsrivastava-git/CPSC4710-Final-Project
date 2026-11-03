# Design decisions

Each entry: the decision, why, and what it affects in the report. Items marked **OPEN** are
undecided.

| # | Decision | Reason | Affects |
|---|---|---|---|
| D1 | **Primary target = Tabular ResNet** | Project premise: the method only matters if the target is complex. | Methods section |
| D2 | Tabular ResNet implemented in plain PyTorch (`src/models/tab_resnet.py`), following Gorishniy et al. 2021 | Avoids dependency problems with Python 3.14; keeps the model fully controllable and differentiable. | Methods, limitations |
| D3 | **GELU activation instead of the reference ReLU** | IG, SmoothGrad and Sobolev matching need informative input gradients; ReLU nets have piecewise-constant gradients. | Deviation from reference; mention in report |
| D4 | Model outputs a single **positive-class logit** `(B,)` | Probabilities saturate, flattening gradients near confident predictions. This logit is the scalar explained and distilled everywhere. | All later stages |
| D5 | Label 1 = malignant / spam | Explanations are about the "positive" class. Flips sklearn's Breast Cancer coding. | Interpreting attributions |
| D6 | Split 70/15/15, stratified, `split_seed=0`, separate from model seeds | Same test points for every seed and method. | Evaluation |
| D7 | Scaler fit on training split only | No leakage. | Methods |
| D8 | Spambase uses `log1p` before standardizing | Heavy-tailed features. Not yet validated against the no-log alternative (the linear reference was also run only with log1p). **OPEN** until EDA is done. | Preprocessing section |
| D9 | Architecture fixed (d_main=128, 4 blocks, hidden factor 2, 269k parameters on 30 inputs), **not tuned** | Complexity is the goal, and all models reach near-ceiling validation scores on this small dataset. The README plan's hyperparameter search (step 8) was **skipped for now**. | Report must say the model was not tuned |
| D10 | Early stopping on validation loss, best weights restored | Standard; prevents overfitting on 397 training samples. | Methods |
| D11 | Mini-batch AdamW, lr 1e-3, weight decay 1e-4, batch 64 (Breast Cancer) / 128 (Spambase), patience 30 | Reasonable defaults, not searched. | Methods |
| D12 | CPU training | Equal speed, simpler determinism. | Reproducibility |
| D13 | Spambase fetched once with `curl` into git-ignored `data/`, not by project code | Initially declined, then explicitly approved by the user on 2026-11-03. Checksum recorded in `docs/DATA.md`. | Reproducibility notes |
| D14 | **OPEN:** whether to check the target is "complex enough" (README step 17 gate) | Needs the linear baseline, not built yet. | Whether the target needs to be larger |
| D15 | **RISK, found 2026-11-03:** a plain logistic regression matches or beats the ResNet on Breast Cancer and Spambase (`runs/reference_logreg/`) | Those two tasks look close to linear globally, so the premise "the target is complex" is not established for them. **Partly resolved by D16.** | Dataset choice |
| D16 | Add **MAGIC** and **binary Covertype (10 continuous features, 20k subsample)** as nonlinear datasets | The ResNet beats logistic regression by about 9 to 10 accuracy points on both (see log), so a linear surrogate has room to fail. This is a scope change from the proposal, which names only Breast Cancer and Spambase. **Needs group agreement.** | Datasets section |
| D17 | Covertype: drop the 44 one-hot columns, keep 10 continuous | Gaussian perturbations of indicator features are off-manifold and unreadable. May reduce accuracy. | Datasets, limitations |
| D18 | Covertype: 20,000-row subsample, seeded, class-stratified | Training speed (full set is 495k rows). Not tested whether the full set changes conclusions. | Limitations |
| D19 | Higgs not used | About 2.6 GB download; MAGIC and Covertype already cover the need. | Datasets |
| D20 | **RESOLVED 2026-11-03 (with caveats):** the local check supports D16 | `runs/local_linearity_check/`: at sigma >= 0.25 the linear surrogate fits the MAGIC and Covertype ResNets poorly (median R^2 about 0.6, gradient cosine about 0.5), but fits Breast Cancer and Spambase well (R^2 0.89 to 0.99, cosine 0.92 to 1.0). **Use MAGIC and Covertype as the main datasets**; keep Breast Cancer and Spambase as low-nonlinearity controls. | Datasets, framing |
| D21 | Local check settings: uniform-weight ridge (alpha 0.001), 5,000 fit and 5,000 independent held-out samples, 100 random test points (all 86 for Breast Cancer), seed-0 checkpoint only, sigma in {0.1, 0.25, 0.5, 1, 2} | Simple and cheap. Thresholds R^2 0.9 and cosine 0.8 are tentative and untuned. | Local-check caveats |
