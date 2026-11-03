"""Assemble report/BASELINE_REPORT.md from report/template.md and the generated tables.

``{{table:<name>}}`` is replaced by report/tables/<name>.md; ``{{corr_compact}}`` by a short
selection of the correlated-feature table; ``{{runs_table}}`` by the run directories with their
config hashes and git commits (read from each run's environment.json).

Usage: .venv/bin/python scripts/build_report.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT, TAB = ROOT / "report", ROOT / "report" / "tables"


def corr_compact() -> str:
    df = pd.read_csv(TAB / "correlated_features.csv")
    keep = df[df.pair.isin([
        "mean radius / mean perimeter", "mean radius / mean area", "mean perimeter / mean area",
        "mean perimeter / worst area", "worst radius / worst perimeter", "mean concavity / mean concave points",
        "word_freq_857 / word_freq_415", "fConc / fConc1"])]
    cols = ["dataset", "pair", "r", "share_ig", "share_smoothgrad_x_input", "share_lime_linear@0.5",
            "order_disagree_vs_ig_smoothgrad_x_input", "order_disagree_vs_ig_lime_linear@0.5"]
    names = ["dataset", "pair (first / second)", "r", "share of first, IG", "share, SmoothGrad×input",
             "share, LIME-linear σ=0.5", "order differs from IG, SmoothGrad×input", "order differs from IG, LIME-linear σ=0.5"]
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join("---" for _ in names) + "|"]
    for _, r in keep.iterrows():
        lines.append("| " + " | ".join([r["dataset"], r["pair"], f"{r['r']:.2f}"] + [f"{r[c]:.2f}" for c in cols[3:]]) + " |")
    return "\n".join(lines)


def runs_table() -> str:
    runs = ["target_breast_cancer_resnet", "target_spambase_resnet", "target_magic_resnet", "target_covtype_binary_resnet",
            "local_linearity_check", "local_linearity_check_seed1", "local_linearity_check_seed2",
            "baselines", "baselines_seed1", "baselines_seed2"]
    lines = ["| run directory | config hash | git commit when run | uncommitted changes |", "|---|---|---|---|"]
    for r in runs:
        p = ROOT / "runs" / r / "environment.json"
        if p.exists():
            e = json.loads(p.read_text())
            if "environment" in e:  # reference_logreg stores it nested
                e = e["environment"]
            lines.append(f"| `runs/{r}/` | `{e.get('config_hash', 'n/a')}` | `{e.get('git_commit', 'n/a')}` | {'yes' if e.get('git_dirty') else 'no'} |")
        else:
            lines.append(f"| `runs/{r}/` | n/a | n/a | n/a |")
    rec = json.loads((REPORT / "analysis_record.json").read_text())
    lines.append(f"| `report/` (analysis) | `{rec['config_hash']}` | `{rec['environment']['git_commit']}` | {'yes' if rec['environment']['git_dirty'] else 'no'} |")
    lines.append("")
    lines.append("*\"Uncommitted changes: yes\" means tracked files differed from the commit when the run started (typically docs or scripts edited in the same session); the config hash identifies the configuration regardless.*")
    return "\n".join(lines)


def main():
    text = (REPORT / "template.md").read_text()
    missing = []

    def table(m):
        p = TAB / f"{m.group(1)}.md"
        if not p.exists():
            missing.append(m.group(1))
            return f"(missing table {m.group(1)})"
        return p.read_text().rstrip("\n")

    text = re.sub(r"\{\{table:([a-z_0-9]+)\}\}", table, text)
    text = text.replace("{{corr_compact}}", corr_compact()).replace("{{runs_table}}", runs_table())
    left = re.findall(r"\{\{[^}]*\}\}", text)
    (REPORT / "BASELINE_REPORT.md").write_text(text)
    print(f"wrote report/BASELINE_REPORT.md ({len(text.splitlines())} lines); missing={missing}; unreplaced={left}")
    if missing or left:
        sys.exit(1)


if __name__ == "__main__":
    main()
