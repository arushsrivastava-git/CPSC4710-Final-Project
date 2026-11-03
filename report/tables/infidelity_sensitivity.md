| dataset | method | infidelity (mean) | infidelity, optimally scaled | max-sensitivity (mean) | max-sensitivity (median) |
|---|---|---|---|---|---|
| Breast Cancer | IG (target) | 1.09 | 0.156 | 0.174 | 0.17 |
| Breast Cancer | SmoothGrad (target) | 23.5 | 0.984 | n/a | n/a |
| Breast Cancer | SmoothGrad x input (target) | 0.668 | 0.198 | 0.196 | 0.187 |
| Breast Cancer | Gradient x input (target) | 0.743 | 0.207 | 0.203 | 0.194 |
| Breast Cancer | LIME-linear σ=0.25 | 0.686 | 0.198 | 0.196 | 0.187 |
| Breast Cancer | LIME-linear σ=0.5 | 0.519 | 0.178 | 0.185 | 0.18 |
| Breast Cancer | Random | 20 | 3.02 | n/a | n/a |
| Spambase | IG (target) | 0.427 | 0.138 | 0.195 | 0.198 |
| Spambase | SmoothGrad (target) | 9.12 | 4.03 | n/a | n/a |
| Spambase | SmoothGrad x input (target) | 0.707 | 0.228 | 0.214 | 0.219 |
| Spambase | Gradient x input (target) | 0.779 | 0.244 | 0.219 | 0.217 |
| Spambase | LIME-linear σ=0.25 | 0.745 | 0.228 | 0.214 | 0.216 |
| Spambase | LIME-linear σ=0.5 | 0.597 | 0.199 | 0.205 | 0.209 |
| Spambase | Random | 37.8 | 4.23 | n/a | n/a |
| MAGIC | IG (target) | 11.1 | 8.76 | 0.52 | 0.381 |
| MAGIC | SmoothGrad (target) | 65.4 | 15.7 | n/a | n/a |
| MAGIC | SmoothGrad x input (target) | 44.9 | 14.4 | 1.12 | 0.728 |
| MAGIC | Gradient x input (target) | 37.7 | 13.3 | 1.98 | 1.02 |
| MAGIC | LIME-linear σ=0.25 | 43.6 | 14.4 | 1.1 | 0.707 |
| MAGIC | LIME-linear σ=0.5 | 49.6 | 14.6 | 0.562 | 0.479 |
| MAGIC | Random | 28.2 | 17.5 | n/a | n/a |
| Covertype | IG (target) | 6.99 | 5.83 | 0.543 | 0.504 |
| Covertype | SmoothGrad (target) | 35.1 | 18.5 | n/a | n/a |
| Covertype | SmoothGrad x input (target) | 16 | 8.8 | 0.722 | 0.685 |
| Covertype | Gradient x input (target) | 32.6 | 9.73 | 1.06 | 0.97 |
| Covertype | LIME-linear σ=0.25 | 15.5 | 8.83 | 0.708 | 0.621 |
| Covertype | LIME-linear σ=0.5 | 10.1 | 7.2 | 0.393 | 0.359 |
| Covertype | Random | 30 | 16.7 | n/a | n/a |
