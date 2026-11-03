# Experiment log

One row is appended automatically by `scripts/train_target.py` for each run. Add hand-written
notes (why a run was done, what was learned) in the last column or in the sections below.
`*` after a git commit means the working tree had uncommitted changes when the run started.

| when | run | config hash | git | seeds | test accuracy | test AUC | notes |
|---|---|---|---|---|---|---|---|
| 2026-11-03 15:09 | `target_breast_cancer_resnet` | `4e4b16fe` | `69f89d8` | 3 | 0.9419 ± 0.0000 | 0.9877 ± 0.0039 | train target (breast_cancer); see runs/target_breast_cancer_resnet/ |

## Notes

### 2026-11-03 — `target_breast_cancer_resnet` (first target model)
- Tabular ResNet, 269,057 parameters, 3 seeds, fixed split (397 / 86 / 86).
- Test accuracy 94.19% (all 3 seeds identical: 81/86 correct), test AUC 0.988 ± 0.004,
  test log-loss 0.141 ± 0.020. Train accuracy 99.1%, val accuracy 99.6%.
- Observations for the report: (1) the gap between val (near-perfect) and test (94.2%) is
  within noise for 86-sample splits, since one error moves accuracy by about 1.2 points;
  (2) best epochs varied widely (9 to 80), so early stopping on an 86-sample validation set
  is noisy; (3) training takes under 1.5 s per seed.
- Not done yet: hyperparameter search, the "target is complex enough" check, Spambase run.
| 2026-11-03 15:17 | `target_spambase_resnet` | `2cef5482` | `69f89d8*` | 3 | 0.9503 ± 0.0036 | 0.9815 ± 0.0010 | train target (spambase); see runs/target_spambase_resnet/ |

### 2026-11-03 — `target_spambase_resnet`
- Same architecture (272,513 parameters), 3 seeds, split 3,220 / 690 / 691, `log1p` then standardize.
- Test accuracy 95.03% ± 0.36, test AUC 0.9815 ± 0.0010, test log-loss 0.164 ± 0.006.
  Train accuracy 96.4%, val accuracy 94.3%.
- Best epochs were 3, 6 and 11: the model starts overfitting almost immediately, so training
  was cut off early (train accuracy only 96.4%). Not tuned; a smaller learning rate, more
  dropout or more weight decay might help. Worth stating as a limitation.
- Per-seed test accuracy was 0.9537, 0.9508 and 0.9465 (659, 657 and 654 of 691 correct), so
  seeds differ by about 5 test samples.

### 2026-11-03 — linear reference (`scripts/reference_logreg.py`)
Logistic regression, same splits and preprocessing, C chosen on the validation split.

| dataset | ResNet test acc | ResNet test AUC | logistic reg. test acc | logistic reg. test AUC |
|---|---|---|---|---|
| Breast Cancer (86 test) | 94.19% | 0.988 | **97.67%** | 0.986 |
| Spambase (691 test) | 95.03% | 0.982 | 95.08% | 0.978 |

- **Finding:** the linear model is as good as the ResNet (better on Breast Cancer, tied on
  Spambase). Globally, both tasks look nearly linear, which weakens the "complex target"
  premise (design decision D15).
- This is a global comparison. What matters for the project is how well a linear surrogate
  matches the ResNet *locally*, which is the planned complexity check (README step 17).
  Do that next before building the nonlinear surrogate.
- Caveat: the ResNet was not tuned, so it may simply be under-trained rather than the task being linear.
| 2026-11-03 15:22 | `target_magic_resnet` | `010f9a86` | `69f89d8*` | 3 | 0.8762 ± 0.0025 | 0.9323 ± 0.0010 | train target (magic); see runs/target_magic_resnet/ |
| 2026-11-03 15:24 | `target_covtype_binary_resnet` | `a2ea09d6` | `69f89d8*` | 3 | 0.8552 ± 0.0072 | 0.9296 ± 0.0039 | train target (covtype_binary); see runs/target_covtype_binary_resnet/ |

### 2026-11-03 — nonlinear datasets: MAGIC and binary Covertype
Same ResNet architecture and training recipe as before (not tuned), 3 seeds each; the linear reference
is the same logistic regression (`scripts/reference_logreg.py`, now covering four datasets).

| dataset | n test | ResNet test acc | ResNet test AUC | logistic reg. test acc | logistic reg. test AUC | acc gap |
|---|---|---|---|---|---|---|
| Breast Cancer | 86 | 94.19% ± 0.00 | 0.988 | 97.67% | 0.986 | -3.5 |
| Spambase | 691 | 95.03% ± 0.36 | 0.982 | 95.08% | 0.978 | -0.1 |
| **MAGIC** | 2,853 | **87.62% ± 0.25** | 0.932 | 78.41% | 0.841 | **+9.2** |
| **Covertype (binary)** | 3,000 | **85.52% ± 0.72** | 0.930 | 76.03% | 0.823 | **+9.5** |

- **Finding:** on MAGIC and Covertype the ResNet clearly beats a linear model (about 9 to 10 accuracy points,
  about 0.09 to 0.11 AUC), unlike on Breast Cancer and Spambase. These tasks have real nonlinear structure,
  so they are better testbeds for the project's premise.
- ResNet details: 266,497 parameters, MAGIC mean best epoch 65.7 and about 26 s per seed; Covertype mean best
  epoch 83.3 and about 34 s per seed. Covertype train accuracy is 92.9% against 85.5% test, so some overfitting remains.
- **Caveats:** (1) a global accuracy gap does not prove that the ResNet is nonlinear in a small neighborhood,
  which is what matters for the surrogates, so the local check (README step 17) is still required; (2) linear
  C was chosen from only four values on the validation split; (3) the ResNet is untuned, so the gap could grow
  or shrink with tuning; (4) Covertype uses only continuous features and a 20k subsample (D17, D18).
- Commands: `scripts/train_target.py --config configs/target_magic_resnet.yaml` (hash `010f9a86`) and
  `configs/target_covtype_binary_resnet.yaml` (hash `a2ea09d6`).

### 2026-11-03 — local linearity check (README step 17)
Run: `scripts/local_linearity_check.py --config configs/local_linearity_check.yaml` (config hash `82315860`).
Full table: `runs/local_linearity_check/summary.md`; per-point data: `per_point.csv`; figure: `local_linearity.png`.
Setup: seed-0 checkpoint of each target; per test point and sigma, a linear (ridge) surrogate is fit to the
ResNet's logit on 5,000 Gaussian neighborhood samples and scored on 5,000 independent ones. Test points:
100 random per dataset (all 86 for Breast Cancer). Time: about 20 to 40 s per dataset.

Median R^2 of the linear surrogate against the ResNet logit (median mean-gradient cosine in brackets):

| dataset | sigma 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| Breast Cancer | 0.990 (0.999) | 0.990 (0.996) | 0.959 (0.986) | 0.907 (0.961) | 0.909 (0.934) |
| Spambase | 0.989 (0.999) | 0.988 (0.995) | 0.956 (0.983) | 0.916 (0.953) | 0.890 (0.916) |
| **MAGIC** | 0.837 (0.870) | **0.608 (0.587)** | **0.600 (0.486)** | **0.620 (0.488)** | **0.685 (0.530)** |
| **Covertype** | 0.887 (0.908) | **0.684 (0.673)** | **0.586 (0.526)** | **0.587 (0.506)** | **0.591 (0.547)** |

- **Result:** the ResNet is locally close to linear on Breast Cancer and Spambase (the linear surrogate
  reaches R^2 above 0.95 up to sigma 0.5, and 0% of test points fall below R^2 0.9 except 12% for Breast
  Cancer at sigma 0.5), but strongly nonlinear on MAGIC and Covertype. For MAGIC and Covertype at
  sigma >= 0.5, the linear surrogate explains only about 60% of the logit variance, its gradient points
  in a very different direction from the ResNet's (cosine about 0.5), and 90 to 100% of test points fail the
  R^2 < 0.9 threshold. Class agreement drops to about 0.77 to 0.85 there.
- **Interpretation for the project:** MAGIC and Covertype give a linear LIME-style surrogate real room to
  fail, which is what a nonlinear surrogate is meant to fix. Breast Cancer and Spambase are useful as
  controls where the new method should not be expected to help much.
- **Caveats to state in the report:**
  1. Seed-0 checkpoint only; other seeds and untuned models could differ.
  2. sigma is in standardized units, and radius grows as sigma*sqrt(d): sigma=1 is a much bigger
     neighborhood for Spambase (d=57) than for MAGIC (d=10), so compare datasets at equal sigma with care.
  3. Large sigma (1 to 2) leaves the data distribution, so those neighborhoods are off-manifold and the ResNet
     is evaluated where it was never trained. The more trustworthy range is sigma 0.1 to 0.5.
  4. On MAGIC at sigma 0.1, 57% of points already fail R^2 < 0.9 with a wide IQR (0.57 to 0.96), so some of
     the ResNet's nonlinearity is very local. This could partly reflect a rough (non-smooth) learned function;
     not investigated. If the ResNet is rough, Sobolev matching may be harder.
  5. R^2 and cosine thresholds are tentative; ridge alpha 0.001 was not tuned.
  6. "Gradient cosine at x0" falls toward 0 for MAGIC at sigma 2 because the linear fit averages over a
     huge region while the gradient at x0 is local, so it measures different things at large sigma.

### 2026-11-03 — baseline explainers scored with fid+/fid- (README steps 14 to 17, 24)
Run: `scripts/run_baselines.py --config configs/baselines.yaml` (config hash `62bcbd58`); about 25 s per dataset
(under 2 minutes in total) on the laptop CPU. Full table: `runs/baselines/summary.md`; per-point data:
`per_point.csv`; attributions: `attributions.npz`. Same 100 random test points per dataset as the local check
(all 86 for Breast Cancer), seed-0 checkpoints, x' = 0.

Methods: IG and SmoothGrad on the target (direct baselines), SmoothGrad x input, gradient x input, LIME-style
linear surrogate at sigma 0.25 and 0.5, and a random ranking. Scores are against the target.

Raw-logit means (fid+ higher is better, fid- lower is better):

| dataset | IG fid+ / fid- | LIME-linear (sigma 0.5) fid+ / fid- | random fid+ / fid- | Spearman of LIME-linear vs IG |
|---|---|---|---|---|
| Breast Cancer | 3.95 / 0.38 | 3.83 / 0.44 | 1.90 / 1.85 | 0.90 |
| Spambase | 3.70 / 0.14 | 3.63 / 0.16 | 1.82 / 1.76 | 0.91 |
| **MAGIC** | 3.90 / 0.54 | 3.54 / 0.94 | 2.36 / 2.41 | 0.65 |
| **Covertype** | 4.35 / 0.83 | 4.30 / 1.00 | 2.89 / 2.62 | 0.64 |

- **Sanity checks passed:** IG completeness error is at most 0.0076 logits (mean at most 0.0007); every real
  method is far better than random on both metrics; unit tests confirm the metrics prefer the true ranking on a
  linear model with known weights.
- **Near-linear controls (Breast Cancer, Spambase):** the LIME-style linear surrogate is nearly
  indistinguishable from IG on the target (fid+ within about 0.1 logit, fid- within about 0.06; rank agreement
  about 0.9). Nothing here for a nonlinear surrogate to improve, as expected.
- **Nonlinear datasets (MAGIC, Covertype):** the linear surrogate's explanations are less faithful than IG on
  the target, mainly on **fid-** (sufficiency): MAGIC 0.94 to 0.95 vs 0.54, Covertype 1.00 to 1.15 vs 0.83.
  fid+ is about the same (differences are inside the noise at this sample size). Rank agreement with IG falls
  to about 0.65. This is the gap a nonlinear surrogate is meant to close. **It is a gap relative to IG on the
  target, not yet a result about the nonlinear surrogate.**
- SmoothGrad (plain mean gradient) is the weakest real method, since it ignores the (x - x') scaling; its
  `x input` version is close to IG.
- **Caveats:** (1) 100 points, one checkpoint, no confidence intervals or significance tests yet, so small
  differences should not be interpreted; (2) fid+/fid- use the same reference x' as IG, which may favor IG (D28);
  (3) the normalized fid columns are unreliable (D25), so use raw values and medians; (4) the linear
  baseline has not been cross-checked against the `lime` package; (5) SmoothGrad parameters are untuned.

### 2026-11-03 — CORRECTION and full baseline analysis (report/BASELINE_REPORT.md)
Ran the same baseline pipeline and local-fit check on ResNet seeds 1 and 2 (`configs/baselines_seed{1,2}.yaml`,
`configs/local_linearity_check_seed{1,2}.yaml`) and a full analysis (`scripts/analyze_baselines.py`, ~85 s).
- **Correction:** the entry above ("baseline explainers scored with fid+/fid-") concluded that on MAGIC and
  Covertype the linear surrogate is less faithful than IG, mainly on fid-. That was seed 0 only. Across seeds the
  fid- gap is MAGIC +0.41/-0.10/+0.03 and Covertype +0.18/-0.17/+0.07 (sigma 0.5); only MAGIC seed 0 is
  significant. The claim is withdrawn (D29).
- **Replicated:** the poor local linear fit on MAGIC and Covertype holds on all three seeds (D30).
- **New findings:** (1) rank agreement between LIME-linear and IG after removing the |x| factor is 0.88/0.90 on
  Breast Cancer/Spambase but 0.58/0.42 on MAGIC/Covertype; on MAGIC the linear sign agrees with IG on only 54% of
  IG's top-5 features (chance 52%); (2) infidelity separates the methods more than fid+/fid- (MAGIC IG 11.1 vs
  linear 43.6 to 49.6), seed 0 only; (3) the ordering IG >= linear is unchanged with an opposite-class-mean removal
  baseline; (4) the randomization test passes only after removing the |x| factor (D31).
- Full tables, figures and discussion: `report/BASELINE_REPORT.md`.
