# baselines_seed1

config hash `d221b2eb` · checkpoint seed 1 · x' = 0 (training mean)

Means over test points. fid+ higher = better, fid- lower = better. `norm` = divided by the point's total effect `s*(f(x)-f(x'))`; points with |total effect| < 1e-3 are excluded from the normalized means. **Prefer the raw-logit columns (and medians): the normalized means are inflated by points whose total effect is near zero or negative.**

| dataset | method | n | fid+ raw mean | fid+ raw median | fid- raw mean | fid- raw median | fid+ (norm, mean) | fid- (norm, mean) | Spearman vs IG |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | ig | 86 | 5.60 | 5.73 | 0.78 | 0.78 | 0.91 | 0.40 | 1.00 |
| breast_cancer | smoothgrad | 86 | 4.84 | 4.75 | 1.28 | 1.15 | 1.03 | 0.30 | 0.59 |
| breast_cancer | smoothgrad_x_input | 86 | 5.39 | 5.29 | 0.87 | 0.85 | 1.08 | 0.46 | 0.89 |
| breast_cancer | grad_x_input | 86 | 5.36 | 5.22 | 0.89 | 0.89 | 0.81 | 0.59 | 0.88 |
| breast_cancer | lime_linear@0.25 | 86 | 5.40 | 5.24 | 0.86 | 0.86 | 1.08 | 0.46 | 0.89 |
| breast_cancer | lime_linear@0.5 | 86 | 5.48 | 5.46 | 0.82 | 0.89 | 1.15 | 0.36 | 0.91 |
| breast_cancer | random | 86 | 3.01 | 2.98 | 3.18 | 3.16 | -0.32 | 1.63 | -0.01 |
| spambase | ig | 100 | 4.49 | 3.90 | 0.16 | 0.10 | 0.92 | 0.03 | 1.00 |
| spambase | smoothgrad | 100 | 3.44 | 3.14 | 0.99 | 0.84 | 0.73 | 0.18 | 0.61 |
| spambase | smoothgrad_x_input | 100 | 4.28 | 3.81 | 0.23 | 0.15 | 0.89 | 0.05 | 0.85 |
| spambase | grad_x_input | 100 | 4.27 | 3.75 | 0.24 | 0.15 | 0.89 | 0.05 | 0.84 |
| spambase | lime_linear@0.25 | 100 | 4.26 | 3.79 | 0.24 | 0.16 | 0.89 | 0.05 | 0.85 |
| spambase | lime_linear@0.5 | 100 | 4.31 | 3.77 | 0.22 | 0.15 | 0.89 | 0.05 | 0.87 |
| spambase | random | 100 | 2.21 | 1.77 | 2.21 | 1.75 | 0.41 | 0.46 | -0.02 |
| magic | ig | 100 | 3.38 | 4.00 | 0.72 | 0.51 | 1.19 | 0.18 | 1.00 |
| magic | smoothgrad | 100 | 3.01 | 3.81 | 0.91 | 0.98 | 1.12 | 0.25 | 0.50 |
| magic | smoothgrad_x_input | 100 | 3.35 | 4.16 | 0.67 | 0.57 | 1.11 | 0.21 | 0.69 |
| magic | grad_x_input | 100 | 3.24 | 3.74 | 0.88 | 0.96 | 1.00 | 0.21 | 0.73 |
| magic | lime_linear@0.25 | 100 | 3.42 | 4.29 | 0.78 | 0.61 | 1.15 | 0.24 | 0.69 |
| magic | lime_linear@0.5 | 100 | 3.33 | 4.38 | 0.62 | 0.66 | 1.26 | 0.11 | 0.69 |
| magic | random | 100 | 1.92 | 2.09 | 2.33 | 2.01 | 0.76 | 0.61 | -0.03 |
| covtype_binary | ig | 100 | 4.35 | 3.51 | 1.12 | 0.81 | 0.99 | -0.27 | 1.00 |
| covtype_binary | smoothgrad | 100 | 4.00 | 3.09 | 1.41 | 1.14 | 0.95 | -0.27 | 0.33 |
| covtype_binary | smoothgrad_x_input | 100 | 4.22 | 3.19 | 1.02 | 0.85 | 0.77 | -0.09 | 0.62 |
| covtype_binary | grad_x_input | 100 | 4.21 | 3.38 | 1.24 | 1.08 | 0.72 | 0.02 | 0.63 |
| covtype_binary | lime_linear@0.25 | 100 | 4.24 | 3.48 | 1.17 | 0.94 | 0.72 | -0.01 | 0.62 |
| covtype_binary | lime_linear@0.5 | 100 | 4.21 | 3.26 | 0.95 | 0.94 | 0.84 | -0.21 | 0.58 |
| covtype_binary | random | 100 | 2.78 | 1.99 | 2.92 | 2.35 | 0.40 | 0.20 | 0.04 |

IG completeness error (|sum IG - (f(x) - f(x'))|, logit units):

| dataset | mean | max |
|---|---|---|
| breast_cancer | 0.0001 | 0.0004 |
| covtype_binary | 0.0004 | 0.0017 |
| magic | 0.0005 | 0.0039 |
| spambase | 0.0001 | 0.0013 |
