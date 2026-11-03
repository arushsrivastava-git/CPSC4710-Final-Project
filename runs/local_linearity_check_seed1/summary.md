# local_linearity_check_seed1

config hash `2f067aab` · checkpoint seed 1 · 5000 fit / 5000 held-out samples per neighborhood · ridge alpha 0.001

`R2 fail` = fraction of test points with R^2 < 0.9; `cos fail` = fraction with mean gradient cosine < 0.8. Lower R^2 and cosine mean the linear surrogate fits the ResNet worse, which is what the project needs.

| dataset | sigma | points | median R^2 (IQR) | median grad cos | median grad cos at x0 | class agreement | R2 fail | cos fail |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0.1 | 86 | 0.989 (0.987 to 0.991) | 0.999 | 1.000 | 1.000 | 0% | 0% |
| breast_cancer | 0.25 | 86 | 0.985 (0.976 to 0.993) | 0.992 | 1.000 | 1.000 | 0% | 0% |
| breast_cancer | 0.5 | 86 | 0.956 (0.933 to 0.978) | 0.973 | 0.998 | 0.998 | 6% | 0% |
| breast_cancer | 1.0 | 86 | 0.913 (0.884 to 0.935) | 0.930 | 0.985 | 0.964 | 40% | 0% |
| breast_cancer | 2.0 | 86 | 0.884 (0.865 to 0.902) | 0.892 | 0.949 | 0.927 | 73% | 0% |
| spambase | 0.1 | 100 | 0.990 (0.984 to 0.991) | 0.999 | 1.000 | 1.000 | 0% | 0% |
| spambase | 0.25 | 100 | 0.989 (0.956 to 0.994) | 0.993 | 1.000 | 1.000 | 2% | 0% |
| spambase | 0.5 | 100 | 0.955 (0.889 to 0.976) | 0.972 | 0.998 | 0.996 | 29% | 0% |
| spambase | 1.0 | 100 | 0.872 (0.798 to 0.926) | 0.917 | 0.986 | 0.953 | 62% | 0% |
| spambase | 2.0 | 100 | 0.790 (0.757 to 0.849) | 0.856 | 0.946 | 0.901 | 95% | 0% |
| magic | 0.1 | 100 | 0.879 (0.629 to 0.956) | 0.896 | 0.995 | 0.993 | 54% | 37% |
| magic | 0.25 | 100 | 0.596 (0.451 to 0.821) | 0.603 | 0.906 | 0.882 | 84% | 71% |
| magic | 0.5 | 100 | 0.564 (0.464 to 0.667) | 0.468 | 0.621 | 0.771 | 96% | 89% |
| magic | 1.0 | 100 | 0.604 (0.490 to 0.672) | 0.470 | 0.217 | 0.793 | 100% | 100% |
| magic | 2.0 | 100 | 0.679 (0.559 to 0.712) | 0.524 | -0.119 | 0.821 | 100% | 100% |
| covtype_binary | 0.1 | 100 | 0.905 (0.828 to 0.942) | 0.923 | 0.996 | 0.998 | 46% | 14% |
| covtype_binary | 0.25 | 100 | 0.682 (0.518 to 0.811) | 0.664 | 0.920 | 0.921 | 93% | 73% |
| covtype_binary | 0.5 | 100 | 0.560 (0.427 to 0.683) | 0.494 | 0.713 | 0.836 | 99% | 95% |
| covtype_binary | 1.0 | 100 | 0.582 (0.524 to 0.637) | 0.493 | 0.526 | 0.796 | 100% | 100% |
| covtype_binary | 2.0 | 100 | 0.605 (0.574 to 0.645) | 0.537 | 0.462 | 0.804 | 100% | 100% |
