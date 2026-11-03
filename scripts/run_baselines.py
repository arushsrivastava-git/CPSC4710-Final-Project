"""Run the baseline explainers and score them with fid+/fid- (against the target).

Methods, per test point:
  ig                    Integrated Gradients on the target (direct baseline)
  smoothgrad            SmoothGrad (mean gradient) on the target (direct baseline)
  smoothgrad_x_input    SmoothGrad x (x - x'), on IG's scale
  grad_x_input          plain gradient x (x - x') (sanity baseline)
  lime_linear@<sigma>   LIME-style linear surrogate; attribution = w * (x - x')
  random                random feature ranking (sanity lower bound)

Also records IG completeness error and the Spearman rank agreement of every method's |attribution|
with IG's. Attributions are saved so later stages can reuse them.

Usage: .venv/bin/python scripts/run_baselines.py --config configs/baselines.yaml
Outputs in runs/<run_name>/: config.yaml, environment.json, per_point.csv, attributions.npz,
summary.csv, summary.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import prepare_data  # noqa: E402
from src.explainers.gradient import (  # noqa: E402
    gradient_times_input, ig_completeness_error, integrated_gradients, smoothgrad,
)
from src.metrics.faithfulness import fid_curves, fid_summary, total_effect  # noqa: E402
from src.models.tab_resnet import load_checkpoint  # noqa: E402
from src.neighborhoods import sample_neighborhood, teacher_signals  # noqa: E402
from src.surrogates.linear import fit_linear_surrogate  # noqa: E402
from src.utils.repro import (  # noqa: E402
    PROJECT_ROOT, config_hash, environment_info, load_config, write_json, write_yaml,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = PROJECT_ROOT / "runs" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg, out / "config.yaml")
    write_json(environment_info() | {"config_hash": config_hash(cfg)}, out / "environment.json")

    rows, store = [], {}
    for ds, run in cfg["targets"].items():
        model, ck = load_checkpoint(PROJECT_ROOT / "runs" / run / f"seed_{cfg['checkpoint_seed']}" / "model.pt")
        tcfg = load_config(PROJECT_ROOT / "runs" / run / "config.yaml")["data"]
        data = prepare_data(tcfg["name"], split_seed=tcfg["split_seed"], val_frac=tcfg["val_frac"],
                            test_frac=tcfg["test_frac"], log1p=tcfg.get("log1p", False),
                            max_samples=tcfg.get("max_samples"))
        assert np.array_equal(ck["idx_test"], data.idx_test), f"{ds}: split mismatch"
        n_test = len(data.X_test)
        pts = np.sort(np.random.default_rng(cfg["rng_seed"]).choice(n_test, size=min(cfg["n_points"], n_test), replace=False))
        d = data.n_features
        t0 = time.time()
        attrs_ds = {}
        for pi in pts:
            x0 = data.X_test[pi]
            te = total_effect(model, x0)
            att = {
                "ig": integrated_gradients(model, x0, steps=cfg["ig_steps"]),
                "smoothgrad": smoothgrad(model, x0, n=cfg["smoothgrad_n"], sigma=cfg["smoothgrad_sigma"], seed=int(pi)),
                "smoothgrad_x_input": smoothgrad(model, x0, n=cfg["smoothgrad_n"], sigma=cfg["smoothgrad_sigma"],
                                                 times_input=True, seed=int(pi)),
                "grad_x_input": gradient_times_input(model, x0),
                "random": np.random.default_rng([cfg["rng_seed"], int(pi)]).standard_normal(d),
            }
            comp_err = ig_completeness_error(model, x0, att["ig"])
            for sg in cfg["lime_sigmas"]:
                rng = np.random.default_rng([cfg["rng_seed"], int(pi), int(round(sg * 1000))])
                Z = sample_neighborhood(x0, sg, cfg["lime_n_fit"], rng)
                f, _ = teacher_signals(model, Z)
                w = fit_linear_surrogate(Z, f, x0, alpha=cfg["ridge_alpha"]).w
                att[f"lime_linear@{sg}"] = w * x0  # w * (x - x'), x' = 0
            for name, a in att.items():
                fp, fm = fid_curves(model, x0, a)
                rho = spearmanr(np.abs(a), np.abs(att["ig"])).statistic if name != "ig" else 1.0
                rows.append({"dataset": ds, "point": int(pi), "method": name, "total_effect": te,
                             "ig_completeness_error": comp_err, "spearman_vs_ig": rho,
                             **fid_summary(fp, fm, te)})
                attrs_ds.setdefault(name, []).append(a)
        for name, lst in attrs_ds.items():
            store[f"{ds}/{name}"] = np.stack(lst)
        store[f"{ds}/points"] = pts
        print(f"[{ds}] {len(pts)} points in {time.time() - t0:.0f}s", flush=True)

    np.savez(out / "attributions.npz", **store)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_point.csv", index=False)
    order = list(cfg["targets"])
    meth_order = ["ig", "smoothgrad", "smoothgrad_x_input", "grad_x_input"] + [f"lime_linear@{s}" for s in cfg["lime_sigmas"]] + ["random"]
    summ = df.groupby(["dataset", "method"]).agg(
        n=("point", "size"),
        fid_plus_norm=("fid_plus_auc_norm", "mean"), fid_minus_norm=("fid_minus_auc_norm", "mean"),
        fid_plus_raw=("fid_plus_auc", "mean"), fid_minus_raw=("fid_minus_auc", "mean"),
        fid_plus_raw_med=("fid_plus_auc", "median"), fid_minus_raw_med=("fid_minus_auc", "median"),
        spearman_vs_ig=("spearman_vs_ig", "mean"),
    ).reset_index()
    summ["dataset"] = pd.Categorical(summ["dataset"], order, ordered=True)
    summ["method"] = pd.Categorical(summ["method"], meth_order, ordered=True)
    summ = summ.sort_values(["dataset", "method"]).reset_index(drop=True)
    summ.to_csv(out / "summary.csv", index=False)
    ce = df[df.method == "ig"].groupby("dataset")["ig_completeness_error"].agg(["mean", "max"])
    nan_frac = df.groupby("dataset")["fid_plus_auc_norm"].apply(lambda s: float(s.isna().mean()))

    md = [f"# {cfg['run_name']}", "", f"config hash `{config_hash(cfg)}` · checkpoint seed {cfg['checkpoint_seed']} · x' = 0 (training mean)",
          "", "Means over test points. fid+ higher = better, fid- lower = better. `norm` = divided by the point's total effect "
          "`s*(f(x)-f(x'))`; points with |total effect| < 1e-3 are excluded from the normalized means. **Prefer the raw-logit "
          "columns (and medians): the normalized means are inflated by points whose total effect is near zero or negative.**", "",
          "| dataset | method | n | fid+ raw mean | fid+ raw median | fid- raw mean | fid- raw median | fid+ (norm, mean) | fid- (norm, mean) | Spearman vs IG |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        md.append(f"| {r.dataset} | {r.method} | {r.n} | {r.fid_plus_raw:.2f} | {r.fid_plus_raw_med:.2f} | "
                  f"{r.fid_minus_raw:.2f} | {r.fid_minus_raw_med:.2f} | {r.fid_plus_norm:.2f} | {r.fid_minus_norm:.2f} | {r.spearman_vs_ig:.2f} |")
    md += ["", "IG completeness error (|sum IG - (f(x) - f(x'))|, logit units):", "",
           "| dataset | mean | max |", "|---|---|---|"] + [f"| {i} | {r['mean']:.4f} | {r['max']:.4f} |" for i, r in ce.iterrows()]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
