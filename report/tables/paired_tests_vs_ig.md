| dataset | metric | method vs IG | mean diff [95% CI] | points where method is better | p (Wilcoxon) | p (Holm, 6 tests) |
|---|---|---|---|---|---|---|
| Breast Cancer | fid+ | SmoothGrad (target) | -0.48 [-0.61, -0.36] | 0.209 | 1.36e-09 | 6.78e-09 |
| Breast Cancer | fid+ | SmoothGrad x input (target) | -0.17 [-0.24, -0.10] | 0.279 | 6.34e-06 | 1.27e-05 |
| Breast Cancer | fid+ | Gradient x input (target) | -0.19 [-0.26, -0.12] | 0.267 | 1.81e-06 | 7.25e-06 |
| Breast Cancer | fid+ | LIME-linear σ=0.25 | -0.17 [-0.24, -0.11] | 0.267 | 3.56e-06 | 1.07e-05 |
| Breast Cancer | fid+ | LIME-linear σ=0.5 | -0.13 [-0.19, -0.06] | 0.337 | 0.000137 | 0.000137 |
| Breast Cancer | fid+ | Random | -2.06 [-2.29, -1.82] | 0.0581 | 2.53e-15 | 1.52e-14 |
| Breast Cancer | fid- | SmoothGrad (target) | 0.24 [0.14, 0.32] | 0.221 | 1.06e-06 | 5.28e-06 |
| Breast Cancer | fid- | SmoothGrad x input (target) | 0.08 [0.04, 0.12] | 0.291 | 0.000126 | 0.000252 |
| Breast Cancer | fid- | Gradient x input (target) | 0.09 [0.05, 0.13] | 0.291 | 3.54e-05 | 0.000141 |
| Breast Cancer | fid- | LIME-linear σ=0.25 | 0.08 [0.04, 0.12] | 0.267 | 6.5e-05 | 0.000195 |
| Breast Cancer | fid- | LIME-linear σ=0.5 | 0.06 [0.03, 0.10] | 0.302 | 0.000404 | 0.000404 |
| Breast Cancer | fid- | Random | 1.47 [1.29, 1.65] | 0.0698 | 3.45e-15 | 2.07e-14 |
| Spambase | fid+ | SmoothGrad (target) | -0.83 [-1.00, -0.68] | 0.17 | 2.99e-14 | 1.5e-13 |
| Spambase | fid+ | SmoothGrad x input (target) | -0.09 [-0.15, -0.03] | 0.42 | 0.00595 | 0.0157 |
| Spambase | fid+ | Gradient x input (target) | -0.09 [-0.15, -0.03] | 0.38 | 0.00251 | 0.01 |
| Spambase | fid+ | LIME-linear σ=0.25 | -0.08 [-0.14, -0.02] | 0.4 | 0.00524 | 0.0157 |
| Spambase | fid+ | LIME-linear σ=0.5 | -0.07 [-0.12, -0.02] | 0.41 | 0.0064 | 0.0157 |
| Spambase | fid+ | Random | -1.88 [-2.14, -1.62] | 0.09 | 8.01e-17 | 4.81e-16 |
| Spambase | fid- | SmoothGrad (target) | 0.56 [0.46, 0.67] | 0.12 | 9.97e-15 | 4.98e-14 |
| Spambase | fid- | SmoothGrad x input (target) | 0.04 [0.02, 0.06] | 0.31 | 1.81e-05 | 4.43e-05 |
| Spambase | fid- | Gradient x input (target) | 0.04 [0.03, 0.06] | 0.3 | 3.45e-06 | 1.38e-05 |
| Spambase | fid- | LIME-linear σ=0.25 | 0.04 [0.02, 0.06] | 0.29 | 1.48e-05 | 4.43e-05 |
| Spambase | fid- | LIME-linear σ=0.5 | 0.03 [0.01, 0.04] | 0.39 | 0.000716 | 0.000716 |
| Spambase | fid- | Random | 1.62 [1.41, 1.82] | 0.05 | 3.98e-17 | 2.39e-16 |
| MAGIC | fid+ | SmoothGrad (target) | -0.59 [-0.92, -0.27] | 0.32 | 0.000863 | 0.00432 |
| MAGIC | fid+ | SmoothGrad x input (target) | -0.18 [-0.48, 0.11] | 0.46 | 0.274 | 0.548 |
| MAGIC | fid+ | Gradient x input (target) | -0.48 [-0.84, -0.11] | 0.37 | 0.00541 | 0.0216 |
| MAGIC | fid+ | LIME-linear σ=0.25 | -0.17 [-0.48, 0.14] | 0.47 | 0.288 | 0.548 |
| MAGIC | fid+ | LIME-linear σ=0.5 | -0.36 [-0.71, -0.02] | 0.44 | 0.0333 | 0.0999 |
| MAGIC | fid+ | Random | -1.55 [-2.09, -0.99] | 0.28 | 3.48e-07 | 2.09e-06 |
| MAGIC | fid- | SmoothGrad (target) | 0.71 [0.38, 1.02] | 0.29 | 6.77e-06 | 2.71e-05 |
| MAGIC | fid- | SmoothGrad x input (target) | 0.25 [-0.01, 0.50] | 0.4 | 0.00925 | 0.00973 |
| MAGIC | fid- | Gradient x input (target) | 0.66 [0.41, 0.93] | 0.29 | 1.31e-06 | 6.57e-06 |
| MAGIC | fid- | LIME-linear σ=0.25 | 0.42 [0.17, 0.66] | 0.34 | 0.000664 | 0.00199 |
| MAGIC | fid- | LIME-linear σ=0.5 | 0.41 [0.15, 0.67] | 0.44 | 0.00486 | 0.00973 |
| MAGIC | fid- | Random | 1.87 [1.35, 2.39] | 0.24 | 1.11e-09 | 6.66e-09 |
| Covertype | fid+ | SmoothGrad (target) | -0.53 [-0.88, -0.21] | 0.39 | 0.00383 | 0.0192 |
| Covertype | fid+ | SmoothGrad x input (target) | -0.27 [-0.51, -0.03] | 0.44 | 0.048 | 0.144 |
| Covertype | fid+ | Gradient x input (target) | -0.29 [-0.52, -0.08] | 0.42 | 0.0231 | 0.0923 |
| Covertype | fid+ | LIME-linear σ=0.25 | -0.19 [-0.47, 0.09] | 0.43 | 0.0525 | 0.144 |
| Covertype | fid+ | LIME-linear σ=0.5 | -0.06 [-0.28, 0.16] | 0.46 | 0.573 | 0.573 |
| Covertype | fid+ | Random | -1.46 [-1.97, -0.98] | 0.32 | 5.96e-07 | 3.58e-06 |
| Covertype | fid- | SmoothGrad (target) | 0.70 [0.38, 1.04] | 0.35 | 0.000171 | 0.000856 |
| Covertype | fid- | SmoothGrad x input (target) | 0.25 [0.01, 0.50] | 0.4 | 0.0638 | 0.128 |
| Covertype | fid- | Gradient x input (target) | 0.43 [0.17, 0.71] | 0.36 | 0.00291 | 0.0116 |
| Covertype | fid- | LIME-linear σ=0.25 | 0.33 [0.08, 0.59] | 0.41 | 0.0268 | 0.0804 |
| Covertype | fid- | LIME-linear σ=0.5 | 0.18 [-0.08, 0.43] | 0.45 | 0.268 | 0.268 |
| Covertype | fid- | Random | 1.79 [1.33, 2.27] | 0.23 | 3.13e-10 | 1.88e-09 |
