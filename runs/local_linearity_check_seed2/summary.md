# local_linearity_check_seed2

config hash `521dbca6` · checkpoint seed 2 · 5000 fit / 5000 held-out samples per neighborhood · ridge alpha 0.001

`R2 fail` = fraction of test points with R^2 < 0.9; `cos fail` = fraction with mean gradient cosine < 0.8. Lower R^2 and cosine mean the linear surrogate fits the ResNet worse, which is what the project needs.

| dataset | sigma | points | median R^2 (IQR) | median grad cos | median grad cos at x0 | class agreement | R2 fail | cos fail |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0.1 | 86 | 0.990 (0.987 to 0.991) | 0.999 | 1.000 | 1.000 | 0% | 0% |
| breast_cancer | 0.25 | 86 | 0.991 (0.979 to 0.997) | 0.996 | 1.000 | 1.000 | 0% | 0% |
| breast_cancer | 0.5 | 86 | 0.965 (0.915 to 0.989) | 0.987 | 0.998 | 1.000 | 16% | 0% |
| breast_cancer | 1.0 | 86 | 0.909 (0.860 to 0.943) | 0.964 | 0.989 | 0.982 | 44% | 0% |
| breast_cancer | 2.0 | 86 | 0.894 (0.874 to 0.912) | 0.943 | 0.955 | 0.953 | 57% | 0% |
| spambase | 0.1 | 100 | 0.988 (0.984 to 0.990) | 0.998 | 1.000 | 1.000 | 0% | 0% |
| spambase | 0.25 | 100 | 0.976 (0.959 to 0.990) | 0.988 | 1.000 | 1.000 | 1% | 0% |
| spambase | 0.5 | 100 | 0.927 (0.887 to 0.962) | 0.958 | 0.996 | 0.988 | 29% | 0% |
| spambase | 1.0 | 100 | 0.844 (0.789 to 0.900) | 0.882 | 0.979 | 0.936 | 75% | 3% |
| spambase | 2.0 | 100 | 0.767 (0.739 to 0.813) | 0.814 | 0.927 | 0.892 | 98% | 37% |
| magic | 0.1 | 100 | 0.821 (0.563 to 0.961) | 0.867 | 0.993 | 0.992 | 63% | 40% |
| magic | 0.25 | 100 | 0.595 (0.468 to 0.809) | 0.580 | 0.880 | 0.885 | 85% | 75% |
| magic | 0.5 | 100 | 0.588 (0.476 to 0.656) | 0.472 | 0.551 | 0.760 | 98% | 89% |
| magic | 1.0 | 100 | 0.632 (0.520 to 0.691) | 0.495 | 0.260 | 0.792 | 100% | 99% |
| magic | 2.0 | 100 | 0.698 (0.569 to 0.729) | 0.556 | -0.132 | 0.820 | 100% | 100% |
| covtype_binary | 0.1 | 100 | 0.923 (0.859 to 0.958) | 0.940 | 0.997 | 1.000 | 42% | 12% |
| covtype_binary | 0.25 | 100 | 0.716 (0.597 to 0.841) | 0.711 | 0.948 | 0.942 | 91% | 66% |
| covtype_binary | 0.5 | 100 | 0.590 (0.468 to 0.685) | 0.536 | 0.794 | 0.851 | 100% | 96% |
| covtype_binary | 1.0 | 100 | 0.556 (0.496 to 0.616) | 0.473 | 0.567 | 0.797 | 100% | 100% |
| covtype_binary | 2.0 | 100 | 0.575 (0.539 to 0.629) | 0.519 | 0.438 | 0.793 | 100% | 100% |
