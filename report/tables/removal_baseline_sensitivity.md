| dataset | method | fid+ (mean baseline) | fid+ (opposite-class mean) | fid- (mean baseline) | fid- (opposite-class mean) |
|---|---|---|---|---|---|
| Breast Cancer | IG (target) | 3.95 | 8.77 | 0.38 | 2.09 |
| Breast Cancer | SmoothGrad (target) | 3.47 | 8.44 | 0.62 | 2.09 |
| Breast Cancer | SmoothGrad x input (target) | 3.78 | 8.54 | 0.46 | 2.30 |
| Breast Cancer | Gradient x input (target) | 3.77 | 8.52 | 0.47 | 2.32 |
| Breast Cancer | LIME-linear σ=0.25 | 3.78 | 8.54 | 0.46 | 2.29 |
| Breast Cancer | LIME-linear σ=0.5 | 3.83 | 8.60 | 0.44 | 2.24 |
| Breast Cancer | Random | 1.90 | 5.62 | 1.85 | 5.50 |
| Spambase | IG (target) | 3.70 | 7.49 | 0.14 | 1.01 |
| Spambase | SmoothGrad (target) | 2.87 | 6.48 | 0.70 | 1.74 |
| Spambase | SmoothGrad x input (target) | 3.61 | 7.24 | 0.18 | 1.14 |
| Spambase | Gradient x input (target) | 3.61 | 7.23 | 0.18 | 1.16 |
| Spambase | LIME-linear σ=0.25 | 3.62 | 7.24 | 0.18 | 1.15 |
| Spambase | LIME-linear σ=0.5 | 3.63 | 7.28 | 0.16 | 1.11 |
| Spambase | Random | 1.82 | 4.27 | 1.76 | 4.36 |
| MAGIC | IG (target) | 3.90 | 4.49 | 0.54 | 0.91 |
| MAGIC | SmoothGrad (target) | 3.32 | 3.96 | 1.25 | 1.65 |
| MAGIC | SmoothGrad x input (target) | 3.73 | 4.27 | 0.79 | 1.25 |
| MAGIC | Gradient x input (target) | 3.42 | 3.94 | 1.20 | 1.70 |
| MAGIC | LIME-linear σ=0.25 | 3.74 | 4.29 | 0.95 | 1.45 |
| MAGIC | LIME-linear σ=0.5 | 3.54 | 4.09 | 0.94 | 1.42 |
| MAGIC | Random | 2.36 | 2.84 | 2.41 | 3.02 |
| Covertype | IG (target) | 4.35 | 5.22 | 0.83 | 1.24 |
| Covertype | SmoothGrad (target) | 3.82 | 5.19 | 1.53 | 1.80 |
| Covertype | SmoothGrad x input (target) | 4.09 | 5.09 | 1.07 | 1.47 |
| Covertype | Gradient x input (target) | 4.06 | 4.76 | 1.26 | 1.75 |
| Covertype | LIME-linear σ=0.25 | 4.17 | 5.19 | 1.15 | 1.50 |
| Covertype | LIME-linear σ=0.5 | 4.30 | 5.24 | 1.00 | 1.27 |
| Covertype | Random | 2.89 | 3.62 | 2.62 | 3.43 |
