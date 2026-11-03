# local_linearity_check

config hash `82315860` · checkpoint seed 0 · 5000 fit / 5000 held-out samples per neighborhood · ridge alpha 0.001

`R2 fail` = fraction of test points with R^2 < 0.9; `cos fail` = fraction with mean gradient cosine < 0.8. Lower R^2 and cosine mean the linear surrogate fits the ResNet worse, which is what the project needs.

| dataset | sigma | points | median R^2 (IQR) | median grad cos | median grad cos at x0 | class agreement | R2 fail | cos fail |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | 0.1 | 86 | 0.990 (0.987 to 0.991) | 0.999 | 1.000 | 1.000 | 0% | 0% |
| breast_cancer | 0.25 | 86 | 0.990 (0.970 to 0.995) | 0.996 | 1.000 | 1.000 | 0% | 0% |
| breast_cancer | 0.5 | 86 | 0.959 (0.917 to 0.979) | 0.986 | 0.998 | 0.999 | 12% | 0% |
| breast_cancer | 1.0 | 86 | 0.907 (0.884 to 0.932) | 0.961 | 0.989 | 0.965 | 44% | 0% |
| breast_cancer | 2.0 | 86 | 0.909 (0.896 to 0.914) | 0.934 | 0.969 | 0.946 | 27% | 0% |
| spambase | 0.1 | 100 | 0.989 (0.987 to 0.991) | 0.999 | 1.000 | 1.000 | 0% | 0% |
| spambase | 0.25 | 100 | 0.988 (0.976 to 0.996) | 0.995 | 1.000 | 1.000 | 0% | 0% |
| spambase | 0.5 | 100 | 0.956 (0.938 to 0.983) | 0.983 | 0.999 | 0.995 | 0% | 0% |
| spambase | 1.0 | 100 | 0.916 (0.898 to 0.935) | 0.953 | 0.993 | 0.958 | 30% | 0% |
| spambase | 2.0 | 100 | 0.890 (0.879 to 0.900) | 0.916 | 0.973 | 0.933 | 75% | 0% |
| magic | 0.1 | 100 | 0.837 (0.570 to 0.961) | 0.870 | 0.996 | 0.992 | 57% | 44% |
| magic | 0.25 | 100 | 0.608 (0.457 to 0.805) | 0.587 | 0.911 | 0.870 | 85% | 72% |
| magic | 0.5 | 100 | 0.600 (0.484 to 0.691) | 0.486 | 0.697 | 0.772 | 99% | 90% |
| magic | 1.0 | 100 | 0.620 (0.488 to 0.685) | 0.488 | 0.365 | 0.810 | 100% | 99% |
| magic | 2.0 | 100 | 0.685 (0.584 to 0.717) | 0.530 | -0.036 | 0.841 | 100% | 100% |
| covtype_binary | 0.1 | 100 | 0.887 (0.819 to 0.946) | 0.908 | 0.997 | 0.999 | 53% | 13% |
| covtype_binary | 0.25 | 100 | 0.684 (0.520 to 0.802) | 0.673 | 0.941 | 0.934 | 94% | 76% |
| covtype_binary | 0.5 | 100 | 0.586 (0.407 to 0.681) | 0.526 | 0.765 | 0.847 | 100% | 95% |
| covtype_binary | 1.0 | 100 | 0.587 (0.537 to 0.647) | 0.506 | 0.537 | 0.803 | 100% | 100% |
| covtype_binary | 2.0 | 100 | 0.591 (0.537 to 0.637) | 0.547 | 0.383 | 0.810 | 100% | 100% |
