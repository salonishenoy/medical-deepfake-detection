"""Aggregate runs/*/metrics.json into one table (mean +/- std across seeds).

python -m src.summarize_runs                 # prints the table
python -m src.summarize_runs --threshold default_0.5
Writes results/team_b_summary.csv (lightweight; safe to share).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

GROUP = ["arch", "modality", "task", "train_method", "pretrained", "split"]
COLS = ["auc", "balanced_accuracy", "recall", "specificity", "f1", "fn"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--threshold", default="youden_val", help="youden_val, default_0.5, or recall0.95_val")
    p.add_argument("--out", default="results/team_b_summary.csv")
    args = p.parse_args()

    rows = []
    for f in sorted(Path(args.runs).glob("*/metrics.json")):
        m = json.loads(f.read_text())
        a = m["args"]
        for split, s in m["splits"].items():
            at = s[args.threshold]
            rows.append({
                "arch": a["arch"], "modality": a["modality"], "task": a["task"],
                "train_method": a["train_method"] if a["modality"] == "CT" else "SD",
                "pretrained": a["pretrained"], "split": split, "seed": a["seed"],
                "auc": s["auc"], **{k: at[k] for k in COLS[1:]},
            })
    if not rows:
        raise SystemExit(f"no metrics.json found under {args.runs}/")

    df = pd.DataFrame(rows)
    agg = df.groupby(GROUP)[COLS].agg(["mean", "std"])
    seeds = df.groupby(GROUP)["seed"].nunique().rename("n_seeds")
    table = pd.DataFrame(index=agg.index)
    for c in COLS:
        table[c] = [f"{m:.3f} ± {0 if pd.isna(s) else s:.3f}" for m, s in zip(agg[(c, "mean")], agg[(c, "std")])]
    table = table.join(seeds).reset_index()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"threshold: {args.threshold}\n")
    print(table.to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
