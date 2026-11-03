# baselines

config hash `62bcbd58` · checkpoint seed 0 · x' = 0 (training mean)

Means over test points. fid+ higher = better, fid- lower = better. `norm` = divided by the point's total effect `s*(f(x)-f(x'))`; points with |total effect| < 1e-3 are excluded from the normalized means. **Prefer the raw-logit columns (and medians): the normalized means are inflated by points whose total effect is near zero or negative.**

| dataset | method | n | fid+ raw mean | fid+ raw median | fid- raw mean | fid- raw median | fid+ (norm, mean) | fid- (norm, mean) | Spearman vs IG |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | ig | 86 | 3.95 | 4.23 | 0.38 | 0.35 | 0.89 | 0.07 | 1.00 |
| breast_cancer | smoothgrad | 86 | 3.47 | 3.68 | 0.62 | 0.63 | 0.66 | 0.27 | 0.61 |
| breast_cancer | smoothgrad_x_input | 86 | 3.78 | 4.11 | 0.46 | 0.46 | 0.90 | 0.05 | 0.88 |
| breast_cancer | grad_x_input | 86 | 3.77 | 4.04 | 0.47 | 0.47 | 0.89 | 0.06 | 0.87 |
| breast_cancer | lime_linear@0.25 | 86 | 3.78 | 4.10 | 0.46 | 0.47 | 0.90 | 0.05 | 0.88 |
| breast_cancer | lime_linear@0.5 | 86 | 3.83 | 4.15 | 0.44 | 0.44 | 0.89 | 0.06 | 0.90 |
| breast_cancer | random | 86 | 1.90 | 1.97 | 1.85 | 1.83 | 0.57 | 0.32 | 0.02 |
| spambase | ig | 100 | 3.70 | 3.86 | 0.14 | 0.07 | 0.92 | 0.04 | 1.00 |
| spambase | smoothgrad | 100 | 2.87 | 3.01 | 0.70 | 0.58 | 0.73 | 0.18 | 0.69 |
| spambase | smoothgrad_x_input | 100 | 3.61 | 3.67 | 0.18 | 0.10 | 0.90 | 0.06 | 0.90 |
| spambase | grad_x_input | 100 | 3.61 | 3.60 | 0.18 | 0.10 | 0.90 | 0.06 | 0.89 |
| spambase | lime_linear@0.25 | 100 | 3.62 | 3.67 | 0.18 | 0.11 | 0.90 | 0.06 | 0.90 |
| spambase | lime_linear@0.5 | 100 | 3.63 | 3.69 | 0.16 | 0.11 | 0.90 | 0.06 | 0.91 |
| spambase | random | 100 | 1.82 | 1.89 | 1.76 | 1.61 | 0.47 | 0.44 | -0.02 |
| magic | ig | 100 | 3.90 | 4.28 | 0.54 | 0.62 | 1.43 | 0.09 | 1.00 |
| magic | smoothgrad | 100 | 3.32 | 3.87 | 1.25 | 1.67 | 1.34 | 0.74 | 0.45 |
| magic | smoothgrad_x_input | 100 | 3.73 | 4.38 | 0.79 | 0.76 | 1.36 | 0.55 | 0.66 |
| magic | grad_x_input | 100 | 3.42 | 4.01 | 1.20 | 1.11 | 1.40 | 0.47 | 0.64 |
| magic | lime_linear@0.25 | 100 | 3.74 | 4.36 | 0.95 | 1.00 | 1.23 | 0.49 | 0.66 |
| magic | lime_linear@0.5 | 100 | 3.54 | 4.27 | 0.94 | 0.79 | 1.47 | 0.31 | 0.65 |
| magic | random | 100 | 2.36 | 2.59 | 2.41 | 2.17 | 0.56 | 0.69 | -0.01 |
| covtype_binary | ig | 100 | 4.35 | 3.42 | 0.83 | 0.72 | 0.96 | 1.16 | 1.00 |
| covtype_binary | smoothgrad | 100 | 3.82 | 2.97 | 1.53 | 1.38 | 1.08 | 1.12 | 0.37 |
| covtype_binary | smoothgrad_x_input | 100 | 4.09 | 2.79 | 1.07 | 0.90 | 1.16 | 1.10 | 0.67 |
| covtype_binary | grad_x_input | 100 | 4.06 | 3.07 | 1.26 | 1.11 | 0.99 | 0.84 | 0.68 |
| covtype_binary | lime_linear@0.25 | 100 | 4.17 | 2.93 | 1.15 | 1.05 | 1.30 | 1.15 | 0.68 |
| covtype_binary | lime_linear@0.5 | 100 | 4.30 | 3.39 | 1.00 | 0.92 | 1.09 | 1.09 | 0.64 |
| covtype_binary | random | 100 | 2.89 | 2.43 | 2.62 | 2.23 | 0.84 | 1.53 | 0.02 |

IG completeness error (|sum IG - (f(x) - f(x'))|, logit units):

| dataset | mean | max |
|---|---|---|
| breast_cancer | 0.0001 | 0.0005 |
| covtype_binary | 0.0003 | 0.0013 |
| magic | 0.0007 | 0.0076 |
| spambase | 0.0001 | 0.0007 |
