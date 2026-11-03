# baselines_seed2

config hash `7f27279a` · checkpoint seed 2 · x' = 0 (training mean)

Means over test points. fid+ higher = better, fid- lower = better. `norm` = divided by the point's total effect `s*(f(x)-f(x'))`; points with |total effect| < 1e-3 are excluded from the normalized means. **Prefer the raw-logit columns (and medians): the normalized means are inflated by points whose total effect is near zero or negative.**

| dataset | method | n | fid+ raw mean | fid+ raw median | fid- raw mean | fid- raw median | fid+ (norm, mean) | fid- (norm, mean) | Spearman vs IG |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | ig | 86 | 3.27 | 3.25 | 0.36 | 0.38 | 0.78 | 0.08 | 1.00 |
| breast_cancer | smoothgrad | 86 | 2.54 | 2.58 | 0.80 | 0.66 | 0.60 | 0.21 | 0.53 |
| breast_cancer | smoothgrad_x_input | 86 | 3.01 | 2.92 | 0.49 | 0.49 | 0.74 | 0.10 | 0.86 |
| breast_cancer | grad_x_input | 86 | 2.99 | 2.91 | 0.50 | 0.49 | 0.74 | 0.10 | 0.86 |
| breast_cancer | lime_linear@0.25 | 86 | 3.01 | 2.94 | 0.49 | 0.49 | 0.74 | 0.10 | 0.86 |
| breast_cancer | lime_linear@0.5 | 86 | 3.06 | 2.99 | 0.47 | 0.48 | 0.75 | 0.09 | 0.89 |
| breast_cancer | random | 86 | 1.59 | 1.47 | 1.63 | 1.50 | 0.43 | 0.36 | -0.00 |
| spambase | ig | 100 | 4.48 | 4.31 | 0.00 | -0.10 | 0.94 | -0.04 | 1.00 |
| spambase | smoothgrad | 100 | 3.39 | 3.31 | 1.11 | 0.83 | 0.65 | 0.28 | 0.66 |
| spambase | smoothgrad_x_input | 100 | 4.32 | 4.17 | 0.14 | 0.05 | 0.90 | -0.00 | 0.86 |
| spambase | grad_x_input | 100 | 4.29 | 4.29 | 0.18 | 0.08 | 0.89 | 0.01 | 0.85 |
| spambase | lime_linear@0.25 | 100 | 4.32 | 4.25 | 0.14 | 0.05 | 0.90 | 0.00 | 0.86 |
| spambase | lime_linear@0.5 | 100 | 4.36 | 4.20 | 0.10 | 0.02 | 0.90 | -0.02 | 0.89 |
| spambase | random | 100 | 2.19 | 1.98 | 2.23 | 1.82 | 0.48 | 0.42 | -0.02 |
| magic | ig | 100 | 3.66 | 4.53 | 0.76 | 0.79 | 1.26 | 0.56 | 1.00 |
| magic | smoothgrad | 100 | 3.10 | 3.57 | 1.25 | 1.39 | 0.96 | 2.13 | 0.45 |
| magic | smoothgrad_x_input | 100 | 3.49 | 4.23 | 0.93 | 0.94 | 0.70 | 0.58 | 0.66 |
| magic | grad_x_input | 100 | 3.32 | 3.86 | 1.00 | 1.07 | 0.77 | 0.50 | 0.67 |
| magic | lime_linear@0.25 | 100 | 3.50 | 4.19 | 0.91 | 0.85 | 0.70 | 0.47 | 0.67 |
| magic | lime_linear@0.5 | 100 | 3.72 | 4.27 | 0.78 | 0.75 | 0.88 | 0.97 | 0.65 |
| magic | random | 100 | 2.08 | 2.42 | 2.58 | 2.38 | 2.44 | 0.79 | -0.01 |
| covtype_binary | ig | 100 | 3.28 | 3.33 | 0.72 | 0.48 | 1.19 | 0.06 | 1.00 |
| covtype_binary | smoothgrad | 100 | 3.09 | 2.99 | 1.17 | 1.06 | 1.52 | 0.27 | 0.40 |
| covtype_binary | smoothgrad_x_input | 100 | 3.19 | 3.06 | 0.87 | 0.72 | 1.35 | 0.13 | 0.68 |
| covtype_binary | grad_x_input | 100 | 3.19 | 2.84 | 0.85 | 0.72 | 1.57 | -0.24 | 0.68 |
| covtype_binary | lime_linear@0.25 | 100 | 3.25 | 3.20 | 0.84 | 0.55 | 1.57 | 0.17 | 0.67 |
| covtype_binary | lime_linear@0.5 | 100 | 3.19 | 2.95 | 0.80 | 0.79 | 1.27 | 0.08 | 0.60 |
| covtype_binary | random | 100 | 2.10 | 1.58 | 2.15 | 1.84 | 0.29 | 1.38 | 0.04 |

IG completeness error (|sum IG - (f(x) - f(x'))|, logit units):

| dataset | mean | max |
|---|---|---|
| breast_cancer | 0.0000 | 0.0003 |
| covtype_binary | 0.0002 | 0.0010 |
| magic | 0.0005 | 0.0035 |
| spambase | 0.0001 | 0.0012 |
