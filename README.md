# Can a Nonlinear Surrogate Model Provide More Faithful Explanations of a White-Box Model?

CPSC 4710 course project — Arush Srivastava, Anay Seth, Lucas Papamitsakis (Yale University)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Motivation and Research Questions](#2-motivation-and-research-questions)
3. [Background](#3-background)
4. [Datasets](#4-datasets)
5. [Method](#5-method)
6. [Evaluation](#6-evaluation)
7. [Experimental Design](#7-experimental-design)
8. [Repository Layout](#8-repository-layout)
9. [Step-by-Step Implementation Plan](#9-step-by-step-implementation-plan)
10. [Risks and Mitigations](#10-risks-and-mitigations)
11. [Open Decisions](#11-open-decisions)
12. [References](#12-references)

---

## 1. Overview

Post-hoc explainability methods such as LIME explain a complicated model by fitting a **simple surrogate** (usually a sparse linear regression) to the model's behavior in a small neighborhood around one input, then reading the surrogate's weights as the explanation. The weakness is that a linear surrogate often cannot match a complicated model even locally, so the explanation can describe the surrogate rather than the model.

This project tests the opposite design:

> Replace the linear surrogate with a more expressive **nonlinear surrogate (an MLP)**, train it so it matches **both the model's outputs and its input gradients** (Sobolev training), and then run **gradient-based explainers** (Integrated Gradients, SmoothGrad) on the surrogate. Do those explanations track the original model's behavior better than (a) a LIME-style linear surrogate and (b) the same explainer applied directly to the original model?

We study this on two tabular binary-classification datasets (Breast Cancer Wisconsin Diagnostic, Spambase) because their features are named and interpretable, which makes feature-level explanations easy to inspect.

The deliverable is an empirical answer to the trade-off between **surrogate complexity, faithfulness, and interpretability**, including the conditions under which a nonlinear surrogate gives a real advantage and the conditions under which it does not.

### Terminology

| Term | Meaning in this project |
|---|---|
| **Target model** (a.k.a. "white-box"/"larger" model) | The trained classifier we want to explain. Trained on the full dataset (MLP, Tabular ResNet, TabM). |
| **Surrogate** | A small model trained to imitate the target model **inside a neighborhood** of one test point. |
| **Neighborhood** | A distribution of perturbed points around a test input `x0`. |
| **Distillation** | Training the surrogate on the target's outputs over neighborhood samples. |
| **Sobolev training** | Distillation whose loss also matches the target's input gradients. |
| **Explanation / attribution** | A per-feature importance vector for one prediction. |
| **Fidelity** | How well the surrogate matches the target (outputs and gradients). |
| **Faithfulness** | How well an explanation reflects what actually drives the target's prediction (fid⁺ / fid⁻). |

---

## 2. Motivation and Research Questions

Deep models are accurate but opaque. Explainability methods are only useful if the explanation is *faithful* to the model. LIME-style surrogates trade faithfulness for simplicity. A nonlinear surrogate can capture feature interactions and curvature inside the neighborhood, which may increase faithfulness, but it needs a post-hoc explainer of its own, and that explainer is only trustworthy if the surrogate is itself faithful.

**Primary research question.** Does a sufficiently faithful nonlinear surrogate improve explanation quality over a LIME-style linear surrogate and over explaining the target directly?

**Secondary research questions.**

- **RQ1 (surrogate fidelity).** Does adding gradient matching (Sobolev loss) improve gradient fidelity relative to output-only distillation, and at what cost in output fidelity and compute?
- **RQ2 (explanation faithfulness).** Do IG/SmoothGrad explanations computed on the surrogate score better on fid⁺ and fid⁻ than explanations from the baselines?
- **RQ3 (complexity).** How do surrogate depth, width, and regularization affect fidelity and faithfulness? Is there a point of diminishing returns?
- **RQ4 (dataset effects).** Do conclusions hold across a small, low-dimensional dataset (Breast Cancer, 569 × 30) and a larger, higher-dimensional one (Spambase, 4,601 × 57)?
- **RQ5 (correlated features).** Highly correlated features (e.g. mean radius vs. mean perimeter, r ≈ 0.998) can split attribution mass arbitrarily. Do the methods differ in how they distribute it?
- **RQ6 (explainer/baseline sensitivity).** How sensitive are the results to the IG baseline, neighborhood scale, and gradient noise?

**Success criteria (from the proposal).**

1. The surrogate is high quality within the local region, in **both outputs and gradients**.
2. Surrogate-based explanations achieve better **fid⁺ and fid⁻** than the baselines.

A negative result (the surrogate does not help, or helps only under stated conditions) is an acceptable outcome as long as it is measured carefully.

---

## 3. Background

### 3.1 LIME

For input `x0`, LIME samples perturbations `z` around `x0`, queries the target `f(z)`, weights samples by proximity to `x0`, and fits a sparse linear model `g(z) = w·z + b`. The weights `w` are the explanation. Our **linear baseline** reproduces this on tabular data (see §5.4).

### 3.2 Integrated Gradients (IG)

For a model `F`, input `x`, and baseline `x'`:

```
IG_i(x) = (x_i − x'_i) · ∫₀¹ ∂F(x' + α(x − x')) / ∂x_i dα
```

approximated with a Riemann sum over `m` steps. IG satisfies *completeness*: attributions sum to `F(x) − F(x')`. That property gives us a built-in sanity check (§6.5). IG depends on gradients along a whole path, so surrogate gradient quality over that **path**, not only at `x0`, is what matters.

### 3.3 SmoothGrad

Averages the gradient (or gradient × input) over `n` noisy copies `x + N(0, σ²)`. It reduces gradient noise, which is relevant because Sobolev-trained surrogates can have noisy gradients.

### 3.4 Sobolev Training

Czarnecki et al. (2017) train a student to match a teacher's values **and derivatives**:

```
L = ℓ(f_s(x), f_t(x)) + λ · ℓ(∇ₓ f_s(x), ∇ₓ f_t(x))
```

Because gradient-based explainers consume exactly these derivatives, matching them is the most direct way to make the surrogate's explanation agree with the target's. The cost is computing double backward (second-order) passes, and noisy teacher gradients can destabilize training.

### 3.5 Fidelity vs. faithfulness vs. sensitivity

Following Yeh et al. (2019), explanation quality has more than one axis. We report **fid⁺ / fid⁻** (faithfulness) as the primary metrics and **infidelity and max-sensitivity** as supporting metrics.

---

## 4. Datasets

Both are from the UCI Machine Learning Repository and available through `sklearn.datasets.load_breast_cancer` (Breast Cancer) and `fetch_openml`/UCI download (Spambase).

| | Breast Cancer Wisconsin (Diagnostic) | Spambase |
|---|---|---|
| Samples | 569 | 4,601 |
| Features | 30 (cell-nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, … ) | 57 (word freq., char freq., capital-run-length stats) |
| Classes | benign 357 (62.7%) / malignant 212 (37.3%) | non-spam 60.6% / spam 39.4% |
| Missing values | none | none |
| Challenge | very small, strongly correlated features | higher dimension, heavy-tailed / sparse features |

**Preprocessing.**

- Stratified train / validation / test split, with a fixed seed. The test set is used for explanation evaluation only; the validation set is used for target-model selection.
- Standardize features with **mean and std computed on the training split only**.
- Spambase's heavy-tailed features (e.g. capital-run-length) may be log-transformed *before* standardization. Decide once, apply to both target and surrogates, and document it.
- Keep a mapping from feature index to feature name so explanations can be read.

**Exploratory data analysis (to run before modeling).** Class balance, per-feature distributions, pairwise relationships, correlation matrices, and the list of near-duplicate feature pairs (|r| > 0.95). The correlated-pairs list is reused in the RQ5 analysis.

---

## 5. Method

### 5.1 Pipeline

```
 data ──► train target model f ──► pick test point x0
                                      │
                       sample neighborhood N(x0)
                                      │
            ┌─────────────────────────┼──────────────────────────┐
            ▼                         ▼                          ▼
   (A) IG/SmoothGrad on f     (B) LIME-style linear      (C) distill MLP surrogate g
       (direct baseline)          surrogate → weights        (output-only OR Sobolev)
                                                                  │
                                                      IG/SmoothGrad on g
            └─────────────────────────┬──────────────────────────┘
                                      ▼
              surrogate-fidelity gate ─► fid⁺/fid⁻ vs. target f ─► compare
```

### 5.2 Target models

Neural classifiers that are differentiable end to end, so that the "direct" baseline (IG on `f`) is well defined and Sobolev training has a gradient to match:

- **MLP** (primary)
- **Tabular ResNet**
- **TabM**

The target outputs a **logit** (or probability for the positive class). Decide once which scalar is being explained and matched. Recommended: the **positive-class logit**, since probabilities saturate and make gradients vanish near confident predictions.

Each target is tuned on the validation set, trained with several seeds, and reported with test accuracy / AUC. A target that is not accurate is not worth explaining.

### 5.3 Neighborhoods

For a test point `x0` (in standardized space), define the neighborhood as Gaussian perturbations:

```
z = x0 + σ · ε,   ε ~ N(0, I)    (σ in standardized units; swept, e.g. {0.1, 0.25, 0.5, 1.0})
```

Variants worth considering: perturb with per-feature empirical std, or sample from the training data's k nearest neighbors. Whichever is chosen, **the same neighborhood distribution is used for distillation, LIME-style fitting, and evaluation**, otherwise comparisons are not apples to apples. Use separate train / held-out neighborhood samples for the surrogate (held-out samples are used to measure fidelity).

Sample count `n` per neighborhood (e.g. 2,000–10,000) is a hyperparameter; the surrogate gets more data than LIME's customary 5,000 only if we are explicit about it in the comparison (see §10, "unfair budget").

### 5.4 Baselines

1. **Direct baseline.** IG (and SmoothGrad) applied to the target `f` at `x0`. This is the explanation we ask the surrogate to improve on.
2. **LIME-style linear surrogate.** Fit a proximity-weighted ridge/linear regression `g_lin` to `f` on the neighborhood; the explanation is `w ⊙ (x0 − x')` or simply `w`, chosen so that it is on the same scale as the attributions it is compared against. We implement this ourselves (rather than using the `lime` package) so it shares the exact neighborhood, scaling, and kernel with our surrogate. Optionally cross-check against the `lime` package.
3. **(Optional)** Random-attribution and gradient × input baselines as sanity lower bounds.

### 5.5 Nonlinear surrogate

A small MLP `g_θ: ℝᵈ → ℝ` fit on the neighborhood.

- Smooth activations (**softplus / GELU / tanh**), not ReLU. ReLU networks are piecewise linear, so their second derivatives are zero and gradient matching is poorly behaved.
- Ablation axes: number of layers, hidden width, weight decay / dropout, activation.
- Train on `(z, f(z), ∇f(z))` triples; early-stop on held-out neighborhood samples.

### 5.6 Training objectives (the "training variations")

| Variant | Loss |
|---|---|
| **V0: output distillation** | `MSE(g(z), f(z))` on logits |
| **V1: Sobolev** | `MSE(g(z), f(z)) + λ·MSE(∇g(z), ∇f(z))` |
| **V2: Sobolev, cosine gradient term** | `MSE(g, f) + λ·(1 − cos(∇g, ∇f))` (insensitive to gradient magnitude) |
| **V3: distribution-matching terms** | additional term(s) comparing output distributions over the neighborhood (e.g. MMD / KL on predicted probabilities). **Not yet decided**, see §11. |

Sweep `λ` (e.g. {0.01, 0.1, 1, 10}); gradient and output losses can differ by orders of magnitude, so consider normalizing each term by its teacher-side variance.

Teacher gradients `∇f(z)` are computed once per neighborhood sample with autograd and cached.

### 5.7 Explainers (the "explanation variations")

- Integrated Gradients (required)
- SmoothGrad (required)
- Optionally non-gradient methods (e.g. permutation / KernelSHAP-style) on the surrogate

Each is run on the target (direct baseline) and on each surrogate. Required choices to fix and report: IG baseline `x'` (zeros in standardized space = the training mean; also try the training-set median and a mean of the opposite class), IG step count `m` (≥ 50, check completeness error), SmoothGrad `n` and `σ`.

---

## 6. Evaluation

Evaluation is staged: a surrogate that fails the fidelity gate is **not** used for downstream faithfulness claims.

### 6.1 Stage 1 — Surrogate fidelity (gate)

Measured on **held-out** neighborhood samples (and along the IG path from `x'` to `x0`):

- **Output fidelity:** MSE / R² between `g(z)` and `f(z)`; agreement of predicted class.
- **Gradient fidelity:** mean cosine similarity between `∇g(z)` and `∇f(z)`; relative L2 error of gradients.
- **Local-only vs. path fidelity:** fidelity at `x0` can be high while the path to the baseline is poor. Report both.

A pre-registered threshold (e.g. R² ≥ 0.9 and mean gradient cosine ≥ 0.8; tune after seeing the pilot) decides whether a surrogate "passes". Report the pass rate per configuration.

### 6.2 Stage 2 — Explanation faithfulness (fid⁺ / fid⁻)

Given an attribution vector and a target model `f`, with `k` the number of features kept or removed:

- **fid⁺ (necessity / comprehensiveness):** drop in `f`'s output when the top-`k` attributed features are **removed** (replaced by baseline values). Larger is better.
- **fid⁻ (sufficiency):** drop in `f`'s output when **only** the top-`k` features are **kept** (all others set to baseline). Smaller is better.

Note on sign convention: the proposal says the surrogate explanations should have "better" fid⁺ and fid⁻. Under the definitions above, better means **higher fid⁺ and lower fid⁻**. Fix the convention in code and in the report. Always evaluate the effect on **the target `f`**, never on the surrogate, since the question is faithfulness to the original model.

Report as curves over `k` and summarized by area under the curve. The deletion/insertion curves are the visualization of the same quantity.

### 6.3 Stage 3 — Agreement with baseline expectations

A diagnostic rather than a score: rank correlation (Spearman/Kendall), top-`k` overlap, and sign agreement between surrogate-based explanations and (a) direct-baseline explanations, (b) LIME-style explanations. This tells us whether a change in faithfulness comes with a change in the explanation itself, and guards against the case where all gradient methods do badly on a hard point (so "winning" is uninformative).

### 6.4 Supporting metrics (Yeh et al., 2019)

- **Infidelity:** `E_I[(Iᵀ·Φ − (f(x) − f(x − I)))²]` over random perturbations `I`.
- **Max-sensitivity:** worst-case change in the explanation under small input perturbations.

### 6.5 Sanity checks

- IG **completeness**: `Σ IG_i ≈ F(x) − F(x')`; report the error for both `f` and `g`.
- Randomization test: explanations from a randomly re-initialized target should score poorly.
- Seed stability: variance of every metric across seeds.

### 6.6 Statistics

Evaluate on many test points (e.g. all test points on Breast Cancer, a few hundred sampled on Spambase), repeat across seeds, report **mean ± 95% CI** (bootstrap), and use paired tests (Wilcoxon signed-rank) when comparing methods on the same test points.

### 6.7 Visualizations

- Side-by-side attribution maps (bar charts per feature) for the baselines and each surrogate variation.
- Side-by-side deletion and insertion curves.
- 2-D embeddings (PCA / UMAP) of neighborhood samples colored by `|g(z) − f(z)|`, showing where the surrogate disagrees with the target.
- Fidelity vs. complexity (depth/width) and fidelity vs. `λ` plots.
- Attribution heatmaps over test points for correlated feature pairs (RQ5).

---

## 7. Experimental Design

**Main grid (per dataset).**

| Factor | Levels |
|---|---|
| Target model | MLP, Tabular ResNet, TabM |
| Neighborhood σ | 0.1, 0.25, 0.5, 1.0 |
| Surrogate | LIME-style linear; MLP × {V0, V1, V2, (V3)} |
| Surrogate size | depth {1, 2, 3} × width {32, 64, 128}; with/without weight decay |
| `λ` | 0.01, 0.1, 1, 10 |
| Explainer | IG, SmoothGrad (× IG baseline choice) |
| Seeds | ≥ 3 |

This is too large to run fully crossed. Plan: run the **full pipeline on the primary target (MLP), σ = 0.5, one surrogate size** first; then ablate **one factor at a time** from that default; run the other targets on the default only. Record every run's config and metrics in a results table (CSV/Parquet) so the analysis is reproducible.

**Ablation questions** map to the RQs: V0 vs. V1 (RQ1), size sweeps (RQ3), dataset comparison (RQ4), correlated-feature analysis (RQ5), σ / baseline / `n` sweeps (RQ6).

---

## 8. Repository Layout

Proposed structure (create as work proceeds):

```
.
├── README.md
├── CPSC_4710_Project_Proposal.pdf
├── pyproject.toml / requirements.txt
├── configs/                  # YAML configs: data, target model, surrogate, explainer, eval
├── data/                     # raw + processed (git-ignored; download script provided)
├── src/
│   ├── data/                 # loading, splitting, scaling, EDA helpers
│   ├── models/               # target models (MLP, TabResNet, TabM), surrogate MLP
│   ├── neighborhoods.py      # sampling around x0
│   ├── surrogates/           # linear (LIME-style) and nonlinear training (V0–V3)
│   ├── explainers/           # IG, SmoothGrad, wrappers for the baselines
│   ├── metrics/              # fidelity, fid+/fid−, infidelity, sensitivity, agreement
│   └── utils/                # seeding, logging, io
├── scripts/                  # train_target.py, run_surrogate.py, evaluate.py, make_figures.py
├── notebooks/                # EDA and result exploration
├── tests/                    # unit tests (see plan)
├── results/                  # metrics tables, figures (git-ignored or LFS)
└── report/                   # final write-up
```

---

## 9. Step-by-Step Implementation Plan

Phases are ordered by dependency. Each step lists the **deliverable** and a **done-when** check. Do not start a phase until the previous phase's checks pass.

### Phase 0 — Environment and repo setup

1. **Create the environment.** Python ≥ 3.10; PyTorch, NumPy, pandas, scikit-learn, matplotlib/seaborn, PyYAML, tqdm, pytest. Add Captum (optional; for cross-checking our own IG/SmoothGrad), `lime` (optional cross-check), UMAP, and a TabM/Tabular-ResNet implementation (e.g. the `rtdl` / `tabm` packages, or a minimal re-implementation).
   *Done when:* `pip install -r requirements.txt` works from scratch and `pytest` runs (even with zero tests).
2. **Create the layout from §8**, a `.gitignore` (data, results, caches, checkpoints), and a seeding utility that sets Python/NumPy/PyTorch seeds.
3. **Config system.** One YAML per experiment; a loader that validates keys. Every script takes `--config` and writes the resolved config next to its outputs.
   *Done when:* a dummy script reads a config and writes it to `results/<run_id>/config.yaml`.

### Phase 1 — Data and EDA

4. **Loaders** for Breast Cancer (sklearn) and Spambase (UCI/OpenML), returning `X`, `y`, `feature_names`.
5. **Splitting + scaling.** Stratified train/val/test with fixed seed; `StandardScaler` fit on train only; save the scaler and the splits' indices.
   *Done when:* a test confirms no test statistics leaked into the scaler and class ratios are preserved in each split.
6. **EDA notebook.** Class balance, distributions, correlation matrices, near-duplicate pair list (|r| > 0.95), and the Spambase skew analysis, which decides whether to log-transform (decision §4).
   *Done when:* figures saved to `results/eda/`, and the preprocessing decision is written into the config.

### Phase 2 — Target models

7. **Implement target architectures**: MLP, Tabular ResNet, TabM, each exposing `forward(x) → logit` and fully differentiable w.r.t. `x`.
8. **Train + tune** each on both datasets with a small validation-set search (width, depth, dropout, weight decay, lr), early stopping, ≥ 3 seeds. Save checkpoints and test accuracy / AUC.
   *Done when:* every target reaches sensible accuracy (Breast Cancer typically ≥ 95%, Spambase ≥ 93%) and the numbers are logged in a table.
9. **Gradient sanity.** Verify `∇ₓ f` against finite differences on random inputs (a unit test).
   *Done when:* relative error ≲ 1e-3 in float64.
10. **Choose the explained scalar** (positive-class logit recommended) and record it in the config.

### Phase 3 — Neighborhoods and teacher signals

11. **Neighborhood sampler** `sample_neighborhood(x0, sigma, n, rng)` supporting Gaussian-isotropic first; leave hooks for per-feature-std and kNN variants.
12. **Teacher data builder.** For each `(x0, σ)`: sample `n` train points and `n_heldout` held-out points, compute `f(z)` and `∇f(z)` in one batched autograd call, and cache as tensors.
    *Done when:* a test shows shapes `(n, d)`, `(n,)`, `(n, d)`, deterministic under a fixed seed.
13. **Proximity kernel** (exponential kernel on distance to `x0`) used identically by LIME-style fitting and, optionally, as sample weights in distillation.

### Phase 4 — Baselines

14. **IG implementation.** Batched Riemann-sum IG over `m` steps for any differentiable scalar-output model; configurable baseline `x'`.
    *Done when:* completeness error `|ΣIG − (F(x) − F(x'))|` is small (< 1e-2 of the output range for `m = 100`), and results match Captum on a few inputs.
15. **SmoothGrad implementation** (gradient and gradient × input variants), with configurable `n` and `σ`.
16. **LIME-style linear surrogate.** Weighted ridge regression on the neighborhood (optionally sparse via K-LASSO feature selection); output weights as the attribution, plus its own R² for the gate. Cross-check qualitatively against the `lime` package on a handful of points.
17. **Direct baseline wrapper**: IG/SmoothGrad on the target `f` itself.
    *Done when:* all three baseline explainers produce an attribution vector of shape `(d,)` for any `(f, x0)`, with tests.

### Phase 5 — Nonlinear surrogate and Sobolev training

18. **Surrogate MLP** with configurable depth/width/activation (smooth activations), weight decay/dropout.
19. **Training loop for V0** (output-only), with held-out early stopping and logging of output and gradient fidelity per epoch. This is the first working end-to-end surrogate.
    *Done when:* on one test point the surrogate reaches high output R² on held-out samples.
20. **V1 Sobolev loss.** Use `torch.autograd.grad(..., create_graph=True)` so the gradient term can be backpropagated. Normalize terms by teacher variance; implement `λ` as a config value. Add gradient clipping and an optional `λ` warm-up/ramp to handle noisy teacher gradients.
    *Done when:* gradient cosine similarity on held-out samples is higher than V0 on the same point; unit test checks the Sobolev loss is differentiable and equals zero when `g = f`.
21. **V2 cosine-gradient loss**, and a placeholder interface for V3 distribution-matching terms (implemented once §11 is decided).
22. **Surrogate fidelity metrics** (§6.1) including path fidelity along the IG path, and the pass/fail gate.
23. **Pilot sweep** on a handful of test points to pick sane defaults for `n`, learning rate, epochs, `λ`, and the gate thresholds. Freeze these defaults in `configs/` *before* the full evaluation so they are not tuned on the evaluation results.

### Phase 6 — Faithfulness evaluation

24. **fid⁺ / fid⁻.** Given `(f, x0, attribution, baseline, k)`, compute both and return curves over `k = 1..d`; implement the sign convention from §6.2 and document it in the docstring.
    *Done when:* tests pass on a toy model where the true important features are known (a linear model with known weights: ground-truth ranking gives optimal fid⁺/fid⁻; a random ranking is worse).
25. **Infidelity and max-sensitivity** (Yeh et al.), plus rank-agreement metrics (Spearman, Kendall, top-`k` overlap, sign agreement).
26. **Sanity checks**: IG completeness for `f` and `g`; a randomized-model test; a seed-stability test.
27. **Orchestrator** `evaluate.py`: for each test point → build neighborhood → fit LIME-style + each surrogate → gate on fidelity → explain with each method → score everything → append one row per `(test point, method)` to a results table with the full config hash.
    *Done when:* a single end-to-end run on Breast Cancer / MLP target completes and writes a results file.

### Phase 7 — Experiments

28. **Main run** on Breast Cancer, MLP target, default settings, all test points, ≥ 3 seeds.
29. **Spambase main run** on a sampled subset of test points (e.g. 200–500), same settings.
30. **Ablations**, one factor at a time from the defaults: training variant (V0/V1/V2/V3), `λ`, surrogate depth/width/regularization, σ, `n`, IG baseline, explainer (IG vs. SmoothGrad), target architecture.
31. **Correlated-feature study (RQ5).** Using the near-duplicate pairs from EDA, measure how each method splits attribution between correlated features and how stable that split is across seeds and σ.
32. **Failure analysis.** Inspect points where the surrogate fails the gate or loses to the baselines; use the embedding plots (§6.7) to see where the neighborhood disagrees.

### Phase 8 — Analysis, figures, and report

33. **Aggregate** results: mean ± 95% bootstrap CI per method, paired Wilcoxon tests against each baseline, and effect sizes.
34. **Figures**: attribution maps, deletion/insertion curves, embedding disagreement plots, fidelity-vs-complexity, fidelity-vs-λ, and the correlated-feature heatmaps (§6.7).
35. **Answer each research question** (§2) with a number and a figure, stating where the nonlinear surrogate helps, where it doesn't, and the compute cost of Sobolev training.
36. **Write-up**: problem, relevance, datasets, method, metrics, results, limitations (small Breast Cancer sample size, one neighborhood family, tabular only, local-only claims), and future work.
37. **Reproducibility pass**: a one-command script (or Makefile) that regenerates every figure and table from configs and seeds; pin dependency versions; record hardware and wall-clock for Sobolev vs. V0.

### Cross-cutting engineering rules

- **Seed everything**, log the seed in every result row.
- **Never evaluate on the training neighborhood**; fidelity is always on held-out samples.
- **Use float64** for gradient checks and completeness tests; float32 for bulk runs.
- **Test first on a toy problem** (a known analytic function, or a linear target) before trusting any metric.
- **One config = one run**; results tables must carry the config hash.
- **Do not tune on the final evaluation points**; use the pilot (step 23) for hyperparameters.

---

## 10. Risks and Mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Sobolev training is slow (second-order autograd) | Per-test-point training × many test points × many configs | Small surrogates; batch neighborhoods; cap epochs; run Sobolev on a subset for ablations; report cost honestly |
| Noisy teacher gradients | Gradient term can destabilize training | Normalize loss terms; gradient clipping; λ ramp; SmoothGrad-style smoothed teacher gradients as an option |
| ReLU target models have piecewise-constant gradients | Gradients are poor teachers | Prefer smooth activations in targets, or smooth the teacher |
| High output fidelity but poor gradient fidelity | IG would be wrong despite a "good" surrogate | Gate on both; also measure path fidelity |
| Unfair compute/sample budget between methods | A surrogate that sees more samples or tuning looks better by construction | Fix `n`, the neighborhood, and the tuning budget across methods; report cost |
| Surrogate "wins" only because all methods are bad on hard points | Uninformative comparison | Agreement metrics (§6.3), randomized-model and sanity checks |
| Correlated features split attribution arbitrarily | Rankings differ without either being wrong | Report explicitly (RQ5); also evaluate grouped-feature removal for near-duplicates |
| fid⁺/fid⁻ depend on how "removal" is defined | Baseline replacement creates off-manifold inputs | Use the same baseline everywhere; check a second baseline choice |
| Small Breast Cancer dataset | High variance | Seeds + CIs; k-fold over splits if needed |
| Results are local-only | Claims about the global model would be unjustified | State this limitation; evaluate many test points |

---

## 11. Open Decisions

These were left open in the proposal and need a group decision early:

1. **Distribution-comparison loss terms (V3).** Which term(s), if any, e.g. MMD, KL on predicted probabilities, or Wasserstein on outputs. Default: skip V3 until V0/V1/V2 results are in.
2. **Explained scalar.** Positive-class logit (recommended) vs. probability.
3. **Neighborhood family.** Isotropic Gaussian (default) vs. per-feature-std vs. kNN-from-training-data.
4. **Sparse LIME or ridge.** K-LASSO feature selection or plain ridge for the linear baseline.
5. **IG baseline.** Training mean (default) vs. median vs. class-conditional.
6. **Spambase preprocessing.** Whether to log-transform heavy-tailed features.
7. **Non-gradient explainers** (e.g. KernelSHAP) as an extra comparison: in scope or not.
8. **Which targets are mandatory** if time is short (MLP first; ResNet/TabM are extensions).

---

## 12. References

1. Czarnecki, Osindero, Jaderberg, Świrszcz, Pascanu (2017). *Sobolev Training for Neural Networks.* NeurIPS.
2. Tsay (2021). *Sobolev trained neural network surrogate models for optimization.* Computers & Chemical Engineering 153:107419.
3. Ribeiro, Singh, Guestrin (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME). KDD.
4. Sundararajan, Taly, Yan (2017). *Axiomatic Attribution for Deep Networks* (Integrated Gradients). ICML.
5. Yeh, Hsieh, Suggala, Inouye, Ravikumar (2019). *On the (In)fidelity and Sensitivity of Explanations.* NeurIPS.
6. Smilkov, Thorat, Kim, Viégas, Wattenberg (2017). *SmoothGrad: removing noise by adding noise.*
