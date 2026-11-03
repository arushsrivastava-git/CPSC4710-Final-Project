"""Local linearity check: linear (LIME-style) surrogate vs. each target ResNet.

For every dataset, a sample of test points, and every sigma:
  1. sample a Gaussian neighborhood around x0 (fit set) and an independent one (eval set),
  2. query the target ResNet for logits and input gradients,
  3. fit the linear surrogate on the fit set,
  4. score it on the eval set: R^2 against the logits, and mean cosine between the surrogate's
     constant gradient and the target's gradients.

Low R^2 / low cosine = the target is locally nonlinear = a linear surrogate has room to fail
= the project's premise holds for that dataset and sigma.

Usage: .venv/bin/python scripts/local_linearity_check.py --config configs/local_linearity_check.yaml
Outputs in runs/<run_name>/: config.yaml, environment.json, per_point.csv, summary.csv,
summary.md, local_linearity.png.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import prepare_data  # noqa: E402
from src.models.tab_resnet import load_checkpoint  # noqa: E402
from src.neighborhoods import sample_neighborhood, teacher_signals  # noqa: E402
from src.surrogates.linear import fidelity, fit_linear_surrogate, mean_gradient_cosine  # noqa: E402
from src.utils.repro import (  # noqa: E402
    PROJECT_ROOT, config_hash, environment_info, load_config, write_json, write_yaml,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = PROJECT_ROOT / "runs" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg, out / "config.yaml")
    write_json(environment_info() | {"config_hash": config_hash(cfg)}, out / "environment.json")

    rows = []
    for ds, run in cfg["targets"].items():
        model, ck = load_checkpoint(PROJECT_ROOT / "runs" / run / f"seed_{cfg['checkpoint_seed']}" / "model.pt")
        tcfg = load_config(PROJECT_ROOT / "runs" / run / "config.yaml")["data"]
        data = prepare_data(
            tcfg["name"], split_seed=tcfg["split_seed"], val_frac=tcfg["val_frac"],
            test_frac=tcfg["test_frac"], log1p=tcfg.get("log1p", False), max_samples=tcfg.get("max_samples"),
        )
        # the stored split must be the one we rebuilt, otherwise we would explain the wrong points
        assert np.array_equal(ck["idx_test"], data.idx_test), f"{ds}: split mismatch"
        n_test = len(data.X_test)
        rng_pts = np.random.default_rng(cfg["rng_seed"])
        pts = np.sort(rng_pts.choice(n_test, size=min(cfg["n_points"], n_test), replace=False))
        t0 = time.time()
        for pi in pts:
            x0 = data.X_test[pi]
            _, g0 = teacher_signals(model, x0[None, :])
            for sigma in cfg["sigmas"]:
                rng = np.random.default_rng([cfg["rng_seed"], int(pi), int(round(sigma * 1000))])
                Zfit = sample_neighborhood(x0, sigma, cfg["n_fit"], rng)
                Zev = sample_neighborhood(x0, sigma, cfg["n_eval"], rng)
                ffit, _ = teacher_signals(model, Zfit)
                fev, gev = teacher_signals(model, Zev)
                s = fit_linear_surrogate(Zfit, ffit, x0, alpha=cfg["ridge_alpha"])
                r2 = fidelity(s.predict(Zev), fev)
                cos_mean = mean_gradient_cosine(s.w, gev)
                cos_x0 = float(g0[0] @ s.w / max(np.linalg.norm(g0[0]) * np.linalg.norm(s.w), 1e-30))
                # does the surrogate predict the same class as the target on the eval samples?
                agree = float(np.mean((s.predict(Zev) > 0) == (fev > 0)))
                rows.append({
                    "dataset": ds, "point": int(pi), "sigma": sigma, "r2": r2,
                    "grad_cos_mean": cos_mean, "grad_cos_x0": cos_x0,
                    "class_agreement": agree, "target_logit_std": float(fev.std()),
                })
        print(f"[{ds}] {len(pts)} points x {len(cfg['sigmas'])} sigmas in {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_point.csv", index=False)

    r2t, ct = cfg["r2_threshold"], cfg["cosine_threshold"]
    g = df.groupby(["dataset", "sigma"])
    summ = g.agg(
        n_points=("r2", "size"),
        r2_median=("r2", "median"), r2_q25=("r2", lambda x: x.quantile(0.25)), r2_q75=("r2", lambda x: x.quantile(0.75)),
        grad_cos_median=("grad_cos_mean", "median"),
        grad_cos_x0_median=("grad_cos_x0", "median"),
        class_agreement_median=("class_agreement", "median"),
        frac_r2_below_thr=("r2", lambda x: float((x < r2t).mean())),
        frac_cos_below_thr=("grad_cos_mean", lambda x: float((x < ct).mean())),
    ).reset_index()
    order = list(cfg["targets"])
    summ["dataset"] = pd.Categorical(summ["dataset"], order, ordered=True)
    summ = summ.sort_values(["dataset", "sigma"]).reset_index(drop=True)
    summ.to_csv(out / "summary.csv", index=False)

    md = [f"# {cfg['run_name']}", "",
          f"config hash `{config_hash(cfg)}` · checkpoint seed {cfg['checkpoint_seed']} · "
          f"{cfg['n_fit']} fit / {cfg['n_eval']} held-out samples per neighborhood · ridge alpha {cfg['ridge_alpha']}", "",
          f"`R2 fail` = fraction of test points with R^2 < {r2t}; `cos fail` = fraction with mean gradient cosine < {ct}. "
          "Lower R^2 and cosine mean the linear surrogate fits the ResNet worse, which is what the project needs.", "",
          "| dataset | sigma | points | median R^2 (IQR) | median grad cos | median grad cos at x0 | class agreement | R2 fail | cos fail |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        md.append(f"| {r.dataset} | {r.sigma} | {r.n_points} | {r.r2_median:.3f} ({r.r2_q25:.3f} to {r.r2_q75:.3f}) | "
                  f"{r.grad_cos_median:.3f} | {r.grad_cos_x0_median:.3f} | {r.class_agreement_median:.3f} | "
                  f"{r.frac_r2_below_thr:.0%} | {r.frac_cos_below_thr:.0%} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for ds in order:
        s = summ[summ.dataset == ds]
        ax[0].plot(s.sigma, s.r2_median, marker="o", label=ds)
        ax[0].fill_between(s.sigma, s.r2_q25, s.r2_q75, alpha=0.15)
        ax[1].plot(s.sigma, s.grad_cos_median, marker="o", label=ds)
    ax[0].axhline(r2t, color="k", ls=":", lw=1); ax[1].axhline(ct, color="k", ls=":", lw=1)
    for a, t in zip(ax, ["Linear surrogate R$^2$ vs target logit", "Mean gradient cosine (linear vs target)"]):
        a.set_xscale("log"); a.set_xlabel("neighborhood sigma (standardized units)"); a.set_title(t); a.grid(alpha=0.3)
    ax[0].set_ylabel("median over test points (band = IQR)"); ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "local_linearity.png", dpi=150)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
