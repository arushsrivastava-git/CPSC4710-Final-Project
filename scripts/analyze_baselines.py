"""Full analysis of the baseline run: statistics, extra metrics, sanity checks and figures.

Reads  runs/baselines/ (per_point.csv, attributions.npz), runs/baselines_seed{1,2}/ (if present),
       runs/local_linearity_check/, the trained checkpoints, and configs/baseline_analysis.yaml.
Writes report/figures/*.png and report/tables/*.{md,csv}; scripts/build_report.py then assembles
report/BASELINE_REPORT.md from them, so every number in the report comes from this script.

Usage: .venv/bin/python scripts/analyze_baselines.py --config configs/baseline_analysis.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import prepare_data  # noqa: E402
from src.explainers.gradient import gradient_times_input, integrated_gradients, smoothgrad  # noqa: E402
from src.metrics.faithfulness import fid_curves, fid_summary, total_effect  # noqa: E402
from src.models.tab_resnet import build_model, load_checkpoint  # noqa: E402
from src.neighborhoods import sample_neighborhood, teacher_signals  # noqa: E402
from src.surrogates.linear import fit_linear_surrogate  # noqa: E402
from src.utils.repro import PROJECT_ROOT, config_hash, environment_info, load_config, write_json  # noqa: E402

METHODS = ["ig", "smoothgrad", "smoothgrad_x_input", "grad_x_input", "lime_linear@0.25", "lime_linear@0.5", "random"]
LABEL = {"ig": "IG (target)", "smoothgrad": "SmoothGrad (target)", "smoothgrad_x_input": "SmoothGrad x input (target)",
         "grad_x_input": "Gradient x input (target)", "lime_linear@0.25": "LIME-linear σ=0.25",
         "lime_linear@0.5": "LIME-linear σ=0.5", "random": "Random"}
COLOR = {"ig": "#1f77b4", "smoothgrad": "#9ecae1", "smoothgrad_x_input": "#2ca02c", "grad_x_input": "#98df8a",
         "lime_linear@0.25": "#ff7f0e", "lime_linear@0.5": "#d62728", "random": "#7f7f7f"}
DSNAME = {"breast_cancer": "Breast Cancer", "spambase": "Spambase", "magic": "MAGIC", "covtype_binary": "Covertype"}
RNG = np.random.default_rng(12345)

OUT = PROJECT_ROOT / "report"
FIG, TAB = OUT / "figures", OUT / "tables"


# ----------------------------------------------------------------------------- helpers
def boot_ci(x, stat=np.mean, B=10000, rng=RNG):
    """Percentile bootstrap 95% CI of ``stat`` over points (resampling points with replacement)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    idx = rng.integers(0, len(x), size=(B, len(x)))
    s = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def fmt_ci(m, lo, hi, nd=2):
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def save_table(df: pd.DataFrame, name: str, floatfmt="%.3f"):
    df.to_csv(TAB / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [("n/a" if np.isnan(v) else floatfmt % v) if isinstance(v, (float, np.floating)) and not isinstance(v, bool) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n")


def load_ctx(ds, cfg, att):
    run = cfg["targets"][ds]
    model, ck = load_checkpoint(PROJECT_ROOT / "runs" / run / f"seed_{cfg['checkpoint_seed']}" / "model.pt")
    tcfg = load_config(PROJECT_ROOT / "runs" / run / "config.yaml")["data"]
    data = prepare_data(tcfg["name"], split_seed=tcfg["split_seed"], val_frac=tcfg["val_frac"],
                        test_frac=tcfg["test_frac"], log1p=tcfg.get("log1p", False), max_samples=tcfg.get("max_samples"))
    assert np.array_equal(ck["idx_test"], data.idx_test)
    pts = att[f"{ds}/points"]
    A = {m: att[f"{ds}/{m}"] for m in METHODS}
    return model, ck, data, pts, A


@torch.no_grad()
def fvals(model, X):
    return model(torch.as_tensor(np.atleast_2d(X), dtype=torch.float32)).numpy().astype(np.float64)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    acfg = load_config(ap.parse_args().config)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    bcfg = load_config(PROJECT_ROOT / "runs" / acfg["baselines_run"] / "config.yaml")
    att = np.load(PROJECT_ROOT / "runs" / acfg["baselines_run"] / "attributions.npz")
    pp = pd.read_csv(PROJECT_ROOT / "runs" / acfg["baselines_run"] / "per_point.csv")
    datasets = list(bcfg["targets"])
    results = {}

    # --- per-dataset context + per-point curves and metrics -----------------------------------
    curves, rows_main, rows_agree, rows_inf, rows_sens, rows_alt, rows_corr, rows_rand = {}, [], [], [], [], [], [], []
    rows_free = []
    pair_spearman = {}
    ctxs = {}
    for ds in datasets:
        model, ck, data, pts, A = load_ctx(ds, bcfg, att)
        ctxs[ds] = (model, data, pts, A)
        d, n = data.n_features, len(pts)
        X0 = data.X_test[pts]
        f0 = fvals(model, X0)
        s = np.where(f0 >= 0, 1.0, -1.0)

        # 1) fid curves (zeros baseline), averaged later for plots
        cp = {m: np.zeros((n, d + 1)) for m in METHODS}
        cm = {m: np.zeros((n, d + 1)) for m in METHODS}
        for m in METHODS:
            for i in range(n):
                cp[m][i], cm[m][i] = fid_curves(model, X0[i], A[m][i])
        curves[ds] = (cp, cm)

        # 2) agreement vs IG + pairwise Spearman matrix
        ig = A["ig"]
        topk = acfg["topk"]
        for m in METHODS:
            sp, kt, ov, sg = [], [], {k: [] for k in topk}, []
            for i in range(n):
                a, b = np.abs(A[m][i]), np.abs(ig[i])
                sp.append(1.0 if m == "ig" else spearmanr(a, b).statistic)
                kt.append(1.0 if m == "ig" else kendalltau(a, b).statistic)
                for k in topk:
                    ov[k].append(len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k)
                top5 = np.argsort(-b)[:5]
                sg.append(float(np.mean(np.sign(A[m][i][top5]) == np.sign(ig[i][top5]))))
            row = {"dataset": ds, "method": m, "spearman": boot_ci(sp), "kendall": boot_ci(kt), "sign_top5": boot_ci(sg)}
            for k in topk:
                row[f"top{k}_overlap"] = boot_ci(ov[k])
            rows_agree.append(row)
        M = np.zeros((len(METHODS), len(METHODS)))
        for a_i, ma in enumerate(METHODS):
            for b_i, mb in enumerate(METHODS):
                M[a_i, b_i] = np.mean([spearmanr(np.abs(A[ma][i]), np.abs(A[mb][i])).statistic for i in range(n)])
        pair_spearman[ds] = M

        # 3) infidelity (mask-based, Yeh et al. style) with common random masks per point
        for i, pi in enumerate(pts):
            rng = np.random.default_rng([acfg["rng_seed"], int(pi), 7])
            masks = rng.random((acfg["infidelity_masks"], d)) < 0.5  # True = feature set to baseline (0)
            df_true = f0[i] - fvals(model, np.where(masks, 0.0, X0[i]))
            for m in METHODS:
                pred = masks @ A[m][i]
                raw = float(np.mean((pred - df_true) ** 2))
                beta = float(pred @ df_true / max(pred @ pred, 1e-12))
                rows_inf.append({"dataset": ds, "method": m, "point": int(pi), "infidelity": raw,
                                 "infidelity_opt": float(np.mean((beta * pred - df_true) ** 2)),
                                 "scale_beta": beta, "effect_var": float(np.var(df_true))})

        # 4) max-sensitivity on a subset of points (relative L2 change of the attribution)
        def attr_fn(m, x, pi):
            if m == "ig":
                return integrated_gradients(model, x, steps=bcfg["ig_steps"])
            if m == "smoothgrad_x_input":
                return smoothgrad(model, x, n=bcfg["smoothgrad_n"], sigma=bcfg["smoothgrad_sigma"], times_input=True, seed=int(pi))
            if m == "grad_x_input":
                return gradient_times_input(model, x)
            sg_ = float(m.split("@")[1])
            rng = np.random.default_rng([bcfg["rng_seed"], int(pi), int(round(sg_ * 1000))])
            Z = sample_neighborhood(x, sg_, bcfg["lime_n_fit"], rng)  # common random numbers across perturbed x
            fz, _ = teacher_signals(model, Z)
            return fit_linear_surrogate(Z, fz, x, alpha=bcfg["ridge_alpha"]).w * x

        sens_methods = ["ig", "smoothgrad_x_input", "grad_x_input", "lime_linear@0.25", "lime_linear@0.5"]
        for i in range(min(acfg["sens_points"], n)):
            pi = pts[i]
            rr = np.random.default_rng([acfg["rng_seed"], int(pi), 11])
            R = acfg["sens_sigma"] * rr.standard_normal((acfg["sens_draws"], d))
            for m in sens_methods:
                base = A[m][i]
                ch = [np.linalg.norm(attr_fn(m, X0[i] + r, pi) - base) / max(np.linalg.norm(base), 1e-12) for r in R]
                rows_sens.append({"dataset": ds, "method": m, "point": int(pi), "max_sens": float(np.max(ch)), "mean_sens": float(np.mean(ch))})

        # 5) alternative removal baseline: mean of the class opposite to the prediction (IG still uses 0)
        mu = {c: data.X_train[data.y_train == c].mean(0) for c in (0, 1)}
        for i in range(n):
            b_alt = mu[0] if f0[i] >= 0 else mu[1]
            te = total_effect(model, X0[i], b_alt)
            for m in METHODS:
                fp, fm = fid_curves(model, X0[i], A[m][i], baseline=b_alt)
                rows_alt.append({"dataset": ds, "method": m, "point": int(pts[i]), **fid_summary(fp, fm, te)})

        # 6) randomization sanity test: untrained model with the same architecture
        torch.manual_seed(999)
        rand_model = build_model(d, ck["model_cfg"]).eval()
        sp_r = []
        for i in range(n):
            a_r = integrated_gradients(rand_model, X0[i], steps=bcfg["ig_steps"])
            sp_r.append(spearmanr(np.abs(a_r), np.abs(ig[i])).statistic)
        rows_rand.append({"dataset": ds, "spearman_random_model_vs_trained_ig": boot_ci(sp_r)})

        # 6b) input-magnitude artifact: attributions that contain the factor (x - x') rank features partly by |x|.
        # Compare with "input-free" versions (divide by x0, i.e. rank the per-unit-input sensitivity) on features with |x0| > 0.05.
        r_x_ig, r_x_rand, r_free_rand, r_free_lime_sg, r_free_ig_sg, r_free_lime_ig = [], [], [], [], [], []
        for i in range(n):
            x_i = X0[i]
            ok = np.abs(x_i) > 0.05
            if ok.sum() < 5:
                continue
            a_r = integrated_gradients(rand_model, X0[i], steps=bcfg["ig_steps"])
            r_x_ig.append(spearmanr(np.abs(x_i), np.abs(ig[i])).statistic)
            r_x_rand.append(spearmanr(np.abs(x_i), np.abs(a_r)).statistic)
            r_free_rand.append(spearmanr(np.abs(a_r[ok] / x_i[ok]), np.abs(ig[i][ok] / x_i[ok])).statistic)
            w05 = A["lime_linear@0.5"][i][ok] / x_i[ok]
            sgv = np.abs(A["smoothgrad"][i][ok])
            igf = np.abs(ig[i][ok] / x_i[ok])
            r_free_lime_sg.append(spearmanr(np.abs(w05), sgv).statistic)
            r_free_ig_sg.append(spearmanr(igf, sgv).statistic)
            r_free_lime_ig.append(spearmanr(np.abs(w05), igf).statistic)
        rows_free.append({"dataset": DSNAME[ds],
                          "Spearman(|x|, |IG|) trained": fmt_ci(*boot_ci(r_x_ig)),
                          "Spearman(|x|, |IG|) untrained model": fmt_ci(*boot_ci(r_x_rand)),
                          "input-free: untrained vs trained IG": fmt_ci(*boot_ci(r_free_rand)),
                          "input-free: LIME-linear σ=0.5 vs SmoothGrad": fmt_ci(*boot_ci(r_free_lime_sg)),
                          "input-free: IG vs SmoothGrad": fmt_ci(*boot_ci(r_free_ig_sg)),
                          "input-free: LIME-linear σ=0.5 vs IG": fmt_ci(*boot_ci(r_free_lime_ig))})

        # 7) correlated features (RQ5): near-duplicate pairs in the training data
        C = np.corrcoef(data.X_train.T)
        for i_ in range(d):
            for j_ in range(i_ + 1, d):
                if abs(C[i_, j_]) >= acfg["corr_threshold"]:
                    row = {"dataset": ds, "pair": f"{data.feature_names[i_]} / {data.feature_names[j_]}", "r": float(C[i_, j_])}
                    order_ig = np.abs(A["ig"][:, i_]) > np.abs(A["ig"][:, j_])
                    for m in ["ig", "smoothgrad_x_input", "grad_x_input", "lime_linear@0.25", "lime_linear@0.5"]:
                        ai, aj = np.abs(A[m][:, i_]), np.abs(A[m][:, j_])
                        row[f"share_{m}"] = float(np.mean(ai / np.maximum(ai + aj, 1e-12)))
                        row[f"order_disagree_vs_ig_{m}"] = float(np.mean((ai > aj) != order_ig))
                    rows_corr.append(row)
        print(f"[analysis] {ds} done", flush=True)

    # ============================================================================ tables
    # Main faithfulness table with bootstrap CIs
    ppm = pp.copy()
    rows = []
    for ds in datasets:
        for m in METHODS:
            sub = ppm[(ppm.dataset == ds) & (ppm.method == m)]
            fp, fm = sub.fid_plus_auc.values, sub.fid_minus_auc.values
            rows.append({"dataset": DSNAME[ds], "method": LABEL[m], "n": len(sub),
                         "fid+ mean [95% CI]": fmt_ci(*boot_ci(fp)), "fid+ median": float(np.median(fp)),
                         "fid- mean [95% CI]": fmt_ci(*boot_ci(fm)), "fid- median": float(np.median(fm))})
    save_table(pd.DataFrame(rows), "faithfulness_main", "%.2f")

    # Paired tests vs IG
    rows = []
    for ds in datasets:
        for metric, better in (("fid_plus_auc", "higher"), ("fid_minus_auc", "lower")):
            ref = ppm[(ppm.dataset == ds) & (ppm.method == "ig")].sort_values("point")[metric].values
            comp = [m for m in METHODS if m != "ig"]
            ps, recs = [], []
            for m in comp:
                v = ppm[(ppm.dataset == ds) & (ppm.method == m)].sort_values("point")[metric].values
                diff = v - ref
                try:
                    p = float(wilcoxon(diff).pvalue)
                except ValueError:
                    p = 1.0
                mean, lo, hi = boot_ci(diff)
                ps.append(p)
                recs.append({"dataset": DSNAME[ds], "metric": metric.replace("_auc", "").replace("fid_plus", "fid+").replace("fid_minus", "fid-"),
                             "method vs IG": LABEL[m], "mean diff [95% CI]": fmt_ci(mean, lo, hi, 2),
                             "points where method is better": float(np.mean(diff > 0) if better == "higher" else np.mean(diff < 0)),
                             "p (Wilcoxon)": p})
            for r_, pa in zip(recs, holm(ps)):
                r_["p (Holm, 6 tests)"] = float(pa)
                rows.append(r_)
    save_table(pd.DataFrame(rows), "paired_tests_vs_ig", "%.3g")
    results["paired"] = rows

    # Alternative removal baseline
    alt = pd.DataFrame(rows_alt)
    rows = []
    for ds in datasets:
        for m in METHODS:
            sub = alt[(alt.dataset == ds) & (alt.method == m)]
            base = ppm[(ppm.dataset == ds) & (ppm.method == m)]
            rows.append({"dataset": DSNAME[ds], "method": LABEL[m],
                         "fid+ (mean baseline)": float(base.fid_plus_auc.mean()), "fid+ (opposite-class mean)": float(sub.fid_plus_auc.mean()),
                         "fid- (mean baseline)": float(base.fid_minus_auc.mean()), "fid- (opposite-class mean)": float(sub.fid_minus_auc.mean())})
    save_table(pd.DataFrame(rows), "removal_baseline_sensitivity", "%.2f")

    # Agreement
    ag = pd.DataFrame(rows_agree)
    rows = []
    for _, r in ag.iterrows():
        if r.method == "ig":
            continue
        row = {"dataset": DSNAME[r.dataset], "method": LABEL[r.method], "Spearman": fmt_ci(*r.spearman), "Kendall": fmt_ci(*r.kendall)}
        for k in acfg["topk"]:
            row[f"top-{k} overlap"] = fmt_ci(*r[f"top{k}_overlap"])
        row["sign agreement (IG top-5)"] = fmt_ci(*r.sign_top5)
        rows.append(row)
    save_table(pd.DataFrame(rows), "agreement_with_ig", "%.2f")

    # Infidelity + sensitivity
    inf = pd.DataFrame(rows_inf)
    sens = pd.DataFrame(rows_sens)
    rows = []
    for ds in datasets:
        for m in METHODS:
            si = inf[(inf.dataset == ds) & (inf.method == m)]
            ss = sens[(sens.dataset == ds) & (sens.method == m)]
            rows.append({"dataset": DSNAME[ds], "method": LABEL[m], "infidelity (mean)": float(si.infidelity.mean()),
                         "infidelity, optimally scaled": float(si.infidelity_opt.mean()),
                         "max-sensitivity (mean)": float(ss.max_sens.mean()) if len(ss) else float("nan"),
                         "max-sensitivity (median)": float(ss.max_sens.median()) if len(ss) else float("nan")})
    save_table(pd.DataFrame(rows), "infidelity_sensitivity", "%.3g")

    # Surrogate fidelity of the linear baseline (from the local linearity check)
    ll = pd.read_csv(PROJECT_ROOT / "runs" / acfg["local_check_run"] / "summary.csv")
    ll = ll[ll.sigma.isin([0.25, 0.5])]
    rows = [{"dataset": DSNAME[r.dataset], "sigma": r.sigma, "median R²": f"{r.r2_median:.3f} (IQR {r.r2_q25:.2f} to {r.r2_q75:.2f})",
             "median gradient cosine": r.grad_cos_median, "class agreement": r.class_agreement_median,
             "points with R² < 0.9": f"{r.frac_r2_below_thr:.0%}"} for _, r in ll.iterrows()]
    save_table(pd.DataFrame(rows), "linear_surrogate_fidelity", "%.3f")

    # Sanity: IG completeness + randomization test
    ce = ppm[ppm.method == "ig"].groupby("dataset").ig_completeness_error.agg(["mean", "max"]).reset_index()
    rr_ = pd.DataFrame([{"dataset": r["dataset"], **dict(zip(["mean", "lo", "hi"], r["spearman_random_model_vs_trained_ig"]))} for r in rows_rand])
    san = ce.merge(rr_, on="dataset")
    san = pd.DataFrame({"dataset": [DSNAME[x] for x in san.dataset], "IG completeness error, mean (logits)": san["mean_x"],
                        "IG completeness error, max (logits)": san["max"],
                        "Spearman: IG of untrained model vs IG of trained model": [fmt_ci(a, b, c) for a, b, c in zip(san["mean_y"], san.lo, san.hi)]})
    save_table(san, "sanity_checks", "%.4f")

    save_table(pd.DataFrame(rows_free), "input_magnitude_artifact", "%.2f")

    # Per-seed paired comparison of LIME-linear vs IG (does the gap replicate across trained models?)
    rows = []
    for ds in datasets:
        for r in [acfg["baselines_run"]] + acfg["seed_runs"]:
            d_ = pd.read_csv(PROJECT_ROOT / "runs" / r / "per_point.csv")
            d_ = d_[d_.dataset == ds]
            seed_id = {acfg["baselines_run"]: 0, **{x: i + 1 for i, x in enumerate(acfg["seed_runs"])}}[r]
            for sg_ in ("lime_linear@0.25", "lime_linear@0.5"):
                for metric, name in (("fid_minus_auc", "fid-"), ("fid_plus_auc", "fid+")):
                    a = d_[d_.method == sg_].sort_values("point")[metric].values
                    b = d_[d_.method == "ig"].sort_values("point")[metric].values
                    diff = a - b
                    try:
                        pv = float(wilcoxon(diff).pvalue)
                    except ValueError:
                        pv = 1.0
                    mean, lo, hi = boot_ci(diff)
                    rows.append({"dataset": DSNAME[ds], "ResNet seed": seed_id, "method": LABEL[sg_], "metric": name,
                                 "mean diff vs IG [95% CI]": fmt_ci(mean, lo, hi), "p (Wilcoxon, uncorrected)": pv})
    save_table(pd.DataFrame(rows), "per_seed_paired_tests", "%.3g")

    # Local linear fit per seed
    rows = []
    for ds in datasets:
        for si, r in enumerate([acfg["local_check_run"]] + [acfg["local_check_run"] + f"_seed{k}" for k in (1, 2)]):
            pth = PROJECT_ROOT / "runs" / r / "summary.csv"
            if not pth.exists():
                continue
            d_ = pd.read_csv(pth)
            d_ = d_[(d_.dataset == ds) & d_.sigma.isin([0.25, 0.5])]
            rows.append({"dataset": DSNAME[ds], "ResNet seed": si,
                         "median R² σ=0.25": float(d_[np.isclose(d_.sigma, 0.25)].r2_median.iloc[0]),
                         "median R² σ=0.5": float(d_[np.isclose(d_.sigma, 0.5)].r2_median.iloc[0]),
                         "median grad cos σ=0.5": float(d_[np.isclose(d_.sigma, 0.5)].grad_cos_median.iloc[0]),
                         "points R² < 0.9 (σ=0.5)": f"{float(d_[np.isclose(d_.sigma, 0.5)].frac_r2_below_thr.iloc[0]):.0%}"})
    save_table(pd.DataFrame(rows), "linear_fit_per_seed", "%.3f")

    # Correlated features
    if rows_corr:
        cdf = pd.DataFrame(rows_corr)
        cdf["dataset"] = cdf.dataset.map(DSNAME)
        save_table(cdf, "correlated_features", "%.2f")
    else:
        pd.DataFrame().to_csv(TAB / "correlated_features.csv")
        (TAB / "correlated_features.md").write_text("(no feature pairs above the threshold)\n")

    # Seed stability
    seed_runs = [acfg["baselines_run"]] + [r for r in acfg["seed_runs"] if (PROJECT_ROOT / "runs" / r / "per_point.csv").exists()]
    rows, order_rows = [], []
    for ds in datasets:
        for m in METHODS:
            vals = {"fid+": [], "fid-": []}
            for r in seed_runs:
                d_ = pd.read_csv(PROJECT_ROOT / "runs" / r / "per_point.csv")
                d_ = d_[(d_.dataset == ds) & (d_.method == m)]
                vals["fid+"].append(d_.fid_plus_auc.mean())
                vals["fid-"].append(d_.fid_minus_auc.mean())
            rows.append({"dataset": DSNAME[ds], "method": LABEL[m], "seeds": len(seed_runs),
                         "fid+ per seed": " / ".join(f"{v:.2f}" for v in vals["fid+"]), "fid+ sd": float(np.std(vals["fid+"], ddof=1)) if len(seed_runs) > 1 else float("nan"),
                         "fid- per seed": " / ".join(f"{v:.2f}" for v in vals["fid-"]), "fid- sd": float(np.std(vals["fid-"], ddof=1)) if len(seed_runs) > 1 else float("nan")})
        gaps = []
        for r in seed_runs:
            d_ = pd.read_csv(PROJECT_ROOT / "runs" / r / "per_point.csv")
            d_ = d_[d_.dataset == ds]
            gaps.append(d_[d_.method == "lime_linear@0.5"].fid_minus_auc.mean() - d_[d_.method == "ig"].fid_minus_auc.mean())
        order_rows.append({"dataset": DSNAME[ds], "fid- gap (LIME-linear σ=0.5 minus IG) per seed": " / ".join(f"{g:+.2f}" for g in gaps),
                           "gap > 0 in every seed": bool(all(g > 0 for g in gaps))})
    save_table(pd.DataFrame(rows), "seed_stability", "%.2f")
    save_table(pd.DataFrame(order_rows), "seed_stability_gap", "%.2f")

    # ============================================================================ figures
    # Fig 1: deletion / insertion (fid+ / fid-) curves
    show = ["ig", "smoothgrad", "smoothgrad_x_input", "lime_linear@0.25", "lime_linear@0.5", "random"]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharex=False)
    for j, ds in enumerate(datasets):
        cp, cm = curves[ds]
        d = cp["ig"].shape[1] - 1
        x = np.arange(d + 1) / d
        for m in show:
            for row_i, cc in enumerate((cp, cm)):
                mu = cc[m].mean(0)
                se = cc[m].std(0, ddof=1) / np.sqrt(len(cc[m]))
                ax[row_i, j].plot(x, mu, color=COLOR[m], label=LABEL[m], lw=1.8, ls="--" if m == "random" else "-")
                ax[row_i, j].fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=COLOR[m], alpha=0.12)
        ax[0, j].set_title(f"{DSNAME[ds]} (d={d})")
        ax[1, j].set_xlabel("fraction of features k/d")
    ax[0, 0].set_ylabel("fid+ (deletion): logit drop when\ntop-k features are REMOVED  (higher = better)")
    ax[1, 0].set_ylabel("fid- (insertion): logit drop when only\ntop-k features are KEPT  (lower = better)")
    ax[0, 0].legend(fontsize=7)
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle("Deletion and insertion curves, mean over test points (band = 95% CI of the mean)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_deletion_insertion_curves.png", dpi=150)
    plt.close(fig)

    # Fig 2: summary bars with bootstrap CIs
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    for j, ds in enumerate(datasets):
        for row_i, metric in enumerate(("fid_plus_auc", "fid_minus_auc")):
            for k_, m in enumerate(METHODS):
                v = ppm[(ppm.dataset == ds) & (ppm.method == m)][metric].values
                mu, lo, hi = boot_ci(v)
                ax[row_i, j].bar(k_, mu, color=COLOR[m], yerr=[[mu - lo], [hi - mu]], capsize=3)
            ax[row_i, j].set_xticks(range(len(METHODS)))
            ax[row_i, j].set_xticklabels([LABEL[m].replace(" (target)", "") for m in METHODS], rotation=60, ha="right", fontsize=7)
            ax[row_i, j].grid(axis="y", alpha=0.3)
        ax[0, j].set_title(DSNAME[ds])
    ax[0, 0].set_ylabel("fid+ AUC (logits), higher = better")
    ax[1, 0].set_ylabel("fid- AUC (logits), lower = better")
    fig.suptitle("Faithfulness summary: mean over test points with 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_faithfulness_bars.png", dpi=150)
    plt.close(fig)

    # Fig 3/4: attribution maps for a typical and a worst-disagreement point
    sp05 = pp[pp.method == "lime_linear@0.5"]
    map_methods = ["ig", "smoothgrad_x_input", "grad_x_input", "lime_linear@0.25", "lime_linear@0.5"]
    chosen = {}
    for kind in ("typical", "worst"):
        fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
        for j, ds in enumerate(datasets):
            model, data, pts, A = ctxs[ds]
            sub = sp05[sp05.dataset == ds].sort_values("point").reset_index(drop=True)
            vals = sub.spearman_vs_ig.values
            idx = int(np.argmin(np.abs(vals - np.median(vals)))) if kind == "typical" else int(np.argmin(vals))
            chosen[(kind, ds)] = (int(pts[idx]), float(vals[idx]))
            top = np.argsort(-np.abs(A["ig"][idx]))[:8][::-1]
            h = 0.15
            for k_, m in enumerate(map_methods):
                a = A[m][idx] / max(np.abs(A[m][idx]).sum(), 1e-12)
                ax[j].barh(np.arange(len(top)) + (k_ - 2) * h, a[top], height=h, color=COLOR[m], label=LABEL[m])
            ax[j].set_yticks(np.arange(len(top)))
            ax[j].set_yticklabels([data.feature_names[t] for t in top], fontsize=8)
            ax[j].axvline(0, color="k", lw=0.6)
            ax[j].set_title(f"{DSNAME[ds]}, test point {pts[idx]}\n(Spearman LIME-linear σ=0.5 vs IG = {vals[idx]:.2f})", fontsize=9)
            ax[j].set_xlabel("signed share of total |attribution|")
        ax[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"Attribution maps, {kind} test point per dataset (top 8 features by |IG|)")
        fig.tight_layout()
        fig.savefig(FIG / f"fig3_attribution_maps_{kind}.png", dpi=150)
        plt.close(fig)

    # Fig 5: neighborhood embeddings colored by linear-surrogate disagreement
    fig, ax = plt.subplots(2, 4, figsize=(19, 8))
    for j, ds in enumerate(datasets):
        model, data, pts, A = ctxs[ds]
        pi, _ = chosen[("typical", ds)]
        x0 = data.X_test[pi]
        sg_ = acfg["embedding_sigma"]
        rng = np.random.default_rng([acfg["rng_seed"], int(pi), 5])
        Z = sample_neighborhood(x0, sg_, acfg["embedding_n"], rng)
        fz, _ = teacher_signals(model, Z)
        w = fit_linear_surrogate(Z, fz, x0, alpha=bcfg["ridge_alpha"])
        err = np.abs(w.predict(Z) - fz)
        # 2-D view of the neighborhood. PCA of an isotropic Gaussian cloud has no preferred axes, so use
        # informative axes instead: axis 1 = the linear surrogate's gradient direction (what the explanation says),
        # axis 2 = the direction (orthogonal to axis 1) along which the surrogate's error is concentrated
        # (top eigenvector of the error-weighted second-moment matrix of the neighborhood offsets).
        D = Z - x0
        u1 = w.w / max(np.linalg.norm(w.w), 1e-12)
        Dp = D - np.outer(D @ u1, u1)
        Mx = (Dp * err[:, None]).T @ Dp / max(err.sum(), 1e-12)
        u2 = np.linalg.eigh(Mx)[1][:, -1]
        u2 = u2 - (u2 @ u1) * u1
        u2 /= max(np.linalg.norm(u2), 1e-12)
        P = np.stack([D @ u1, D @ u2], axis=1)
        p0 = np.zeros(2)
        for row_i, (c, lab, cmap) in enumerate(((err, "|linear surrogate - target| (logits)", "magma"), (fz, "target logit f(z)", "coolwarm"))):
            sc = ax[row_i, j].scatter(P[:, 0], P[:, 1], c=c, s=5, cmap=cmap, vmax=np.percentile(c, 99) if row_i == 0 else None)
            ax[row_i, j].scatter(*p0, marker="*", s=220, color="lime", edgecolor="k", zorder=5, label="x0")
            plt.colorbar(sc, ax=ax[row_i, j], label=lab)
            ax[row_i, j].set_xlabel("offset along linear-surrogate gradient direction")
        ax[0, j].set_title(f"{DSNAME[ds]} (σ={sg_}, point {pi})")
    ax[0, 0].set_ylabel("offset along main error direction")
    ax[1, 0].set_ylabel("offset along main error direction")
    ax[0, 0].legend(loc="upper left")
    fig.suptitle("Neighborhood samples colored by where the LIME-style linear surrogate disagrees with the target (axes: surrogate gradient direction, main error direction)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_neighborhood_disagreement.png", dpi=150)
    plt.close(fig)

    # Fig 6: per-point linear-fit quality vs fid- gap
    llp = pd.read_csv(PROJECT_ROOT / "runs" / acfg["local_check_run"] / "per_point.csv")
    llp = llp[np.isclose(llp.sigma, 0.5)][["dataset", "point", "r2"]]
    gap = ppm[ppm.method == "lime_linear@0.5"][["dataset", "point", "fid_minus_auc"]].merge(
        ppm[ppm.method == "ig"][["dataset", "point", "fid_minus_auc"]], on=["dataset", "point"], suffixes=("_lime", "_ig"))
    gap["fid_minus_gap"] = gap.fid_minus_auc_lime - gap.fid_minus_auc_ig
    gj = gap.merge(llp, on=["dataset", "point"])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    rows = []
    for j, ds in enumerate(datasets):
        g_ = gj[gj.dataset == ds]
        ax[j].scatter(g_.r2, g_.fid_minus_gap, s=14, color="#d62728", alpha=0.7)
        ax[j].axhline(0, color="k", lw=0.6)
        rho = spearmanr(g_.r2, g_.fid_minus_gap)
        ax[j].set_title(f"{DSNAME[ds]}: Spearman {rho.statistic:.2f}")
        ax[j].set_xlabel("local R² of linear surrogate (σ=0.5)")
        ax[j].grid(alpha=0.3)
        rows.append({"dataset": DSNAME[ds], "n": len(g_), "Spearman(R², fid- gap)": float(rho.statistic), "p": float(rho.pvalue)})
    rho = spearmanr(gj.r2, gj.fid_minus_gap)
    rows.append({"dataset": "pooled", "n": len(gj), "Spearman(R², fid- gap)": float(rho.statistic), "p": float(rho.pvalue)})
    ax[0].set_ylabel("fid- gap: LIME-linear minus IG\n(positive = linear is worse)")
    fig.suptitle("Does a worse local linear fit mean a less faithful LIME-style explanation?")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_fit_quality_vs_faithfulness_gap.png", dpi=150)
    plt.close(fig)
    save_table(pd.DataFrame(rows), "fit_vs_gap_correlation", "%.3g")

    # Fig 7: method-by-method Spearman agreement heatmaps
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    names = [LABEL[m].replace(" (target)", "") for m in METHODS]
    for j, ds in enumerate(datasets):
        im = ax[j].imshow(pair_spearman[ds], vmin=-0.1, vmax=1, cmap="viridis")
        ax[j].set_xticks(range(len(METHODS)))
        ax[j].set_yticks(range(len(METHODS)))
        ax[j].set_xticklabels(names, rotation=70, ha="right", fontsize=7)
        ax[j].set_yticklabels(names if j == 0 else [], fontsize=7)
        for a_ in range(len(METHODS)):
            for b_ in range(len(METHODS)):
                ax[j].text(b_, a_, f"{pair_spearman[ds][a_, b_]:.2f}", ha="center", va="center", fontsize=6,
                           color="w" if pair_spearman[ds][a_, b_] < 0.6 else "k")
        ax[j].set_title(DSNAME[ds])
    fig.suptitle("Mean Spearman rank correlation of |attribution| between methods")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_method_agreement_heatmaps.png", dpi=150)
    plt.close(fig)

    # Copy the local-fit figure from the earlier check
    shutil.copy(PROJECT_ROOT / "runs" / acfg["local_check_run"] / "local_linearity.png", FIG / "fig7_linear_surrogate_fidelity.png")

    write_json({"config": acfg, "config_hash": config_hash(acfg), "environment": environment_info(),
                "chosen_example_points": {f"{k[0]}/{k[1]}": v for k, v in chosen.items()}}, OUT / "analysis_record.json")
    print("[analysis] finished; figures in", FIG, "tables in", TAB)


if __name__ == "__main__":
    main()
