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
