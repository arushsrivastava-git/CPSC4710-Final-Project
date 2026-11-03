"""Train the target model over several seeds and store checkpoints + a full run record.

Usage:
    .venv/bin/python scripts/train_target.py --config configs/target_breast_cancer_resnet.yaml

Output directory ``runs/<run_name>/`` (all of it is small and committed to git):
    config.yaml        resolved config (what was run)
    environment.json   versions, git commit, config hash, timestamp
    data_summary.json  split sizes / positive rates / feature names
    splits.npz         train/val/test indices (raw-dataset row numbers)
    seed_<k>/model.pt          checkpoint (state_dict, arch cfg, scaler, feature names, splits)
    seed_<k>/history.csv       per-epoch train/val loss
    seed_<k>/metrics.json      accuracy/AUC/log-loss on train/val/test + training info
    summary.json / summary.md  mean +- std over seeds (paste-ready for the report)
And one line is appended to docs/EXPERIMENT_LOG.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import prepare_data, split_summary  # noqa: E402
from src.training.train_target import train_one  # noqa: E402
from src.utils.repro import (  # noqa: E402
    PROJECT_ROOT, config_hash, environment_info, load_config, write_json, write_yaml,
)

LOG = PROJECT_ROOT / "docs" / "EXPERIMENT_LOG.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="override config device (cpu|mps)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg["training"].get("device", "cpu")
    chash = config_hash(cfg)
    out = PROJECT_ROOT / "runs" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    data = prepare_data(
        dcfg["name"], split_seed=dcfg["split_seed"], val_frac=dcfg["val_frac"],
        test_frac=dcfg["test_frac"], log1p=dcfg.get("log1p", False),
        max_samples=dcfg.get("max_samples"),
    )
    summ = split_summary(data)
    print(f"[data] {data.name}: {summ}")

    write_yaml(cfg, out / "config.yaml")
    env = environment_info() | {"config_hash": chash, "device": device, "config_path": args.config}
    write_json(env, out / "environment.json")
    write_json(summ | {"feature_names": data.feature_names}, out / "data_summary.json")
    np.savez(out / "splits.npz", train=data.idx_train, val=data.idx_val, test=data.idx_test)

    rows = []
    for seed in cfg["training"]["seeds"]:
        model, hist, metrics, info = train_one(cfg, data, seed, device)
        sd = out / f"seed_{seed}"
        sd.mkdir(exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_cfg": cfg["model"],
                "feature_names": data.feature_names,
                "dataset": data.name,
                "scaler_mean": data.scaler.mean, "scaler_std": data.scaler.std,
                "log1p": data.log1p,
                "idx_train": data.idx_train, "idx_val": data.idx_val, "idx_test": data.idx_test,
                "seed": seed, "config_hash": chash,
            },
            sd / "model.pt",
        )
        pd.DataFrame(hist).to_csv(sd / "history.csv", index=False)
        write_json({"metrics": metrics, "info": info}, sd / "metrics.json")
        t = metrics["test"]
        print(f"[seed {seed}] test acc={t['accuracy']:.4f} auc={t['auc']:.4f} "
              f"logloss={t['log_loss']:.4f} | best_epoch={info['best_epoch']} "
              f"params={info['n_params']:,} time={info['train_seconds']}s")
        rows.append({"seed": seed, **{f"{s}_{m}": v for s, d in metrics.items() for m, v in d.items()}, **info})

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed_results.csv", index=False)
    agg = {}
    for col in df.columns:
        if col.startswith(("train_", "val_", "test_")) and df[col].dtype.kind == "f":
            agg[col] = {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=1)) if len(df) > 1 else 0.0}
    write_json({"config_hash": chash, "n_seeds": len(df), "aggregate": agg}, out / "summary.json")

    md = [f"# {cfg['run_name']}", "", f"config hash `{chash}` · {len(df)} seeds · device `{device}`", "",
          "| split | accuracy | AUC | log-loss |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        f = lambda m: f"{agg[f'{s}_{m}']['mean']:.4f} ± {agg[f'{s}_{m}']['std']:.4f}"
        md.append(f"| {s} | {f('accuracy')} | {f('auc')} | {f('log_loss')} |")
    md += ["", f"Parameters: {int(df['n_params'][0]):,} · mean best epoch: {df['best_epoch'].mean():.1f} "
               f"· mean train time/seed: {df['train_seconds'].mean():.1f}s"]
    (out / "summary.md").write_text("\n".join(md) + "\n")

    with open(LOG, "a") as f:
        t = agg
        f.write(
            f"| {datetime.now():%Y-%m-%d %H:%M} | `{cfg['run_name']}` | `{chash}` | "
            f"`{env['git_commit']}{'*' if env['git_dirty'] else ''}` | {len(df)} | "
            f"{t['test_accuracy']['mean']:.4f} ± {t['test_accuracy']['std']:.4f} | "
            f"{t['test_auc']['mean']:.4f} ± {t['test_auc']['std']:.4f} | "
            f"train target ({data.name}); see runs/{cfg['run_name']}/ |\n"
        )
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
