#!/usr/bin/env python3
"""
plot_recall.py — plot recall vs churn round for each delete model.

Reads the per-model CSVs written by recall_experiment.sh (one file per model,
columns: recall,search_OPS,delete_OPS,insert_OPS; one row per round) and plots
recall vs round, one line per model. The repair models should hold recall
roughly flat; weaker ones decay as the graph churns.

Usage:
    python3 plot_recall.py [recall_results_dir] [-o out.png]
"""
import csv, glob, os, re, sys, argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "0": "violent",
    "1": "pintopout (DwFC)",
    "2": "Wolverine (search)",
    "3": "WolverinePro (two-hop)",
    "4": "WolverineProMax (approx)",
    "5": "refactor",
}
COLORS = {"0": "#7f7f7f", "1": "#9467bd", "2": "#1f77b4",
          "3": "#2ca02c", "4": "#ff7f0e", "5": "#8c564b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="recall_results",
                    help="directory of recall_model*.csv (default: recall_results)")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default="Repair quality: recall vs churn rounds")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "recall_model*.csv")))
    if not files:
        sys.exit(f"No recall_model*.csv in {args.dir}")
    out = args.out or os.path.join(args.dir, "recall.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in files:
        m = re.search(r"recall_model(\d+)\.csv", os.path.basename(path))
        model = m.group(1) if m else "?"
        recalls = []
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    recalls.append(float(row["recall"]))
                except (KeyError, ValueError):
                    pass
        if not recalls:
            continue
        ax.plot(range(len(recalls)), recalls, marker=".",
                color=COLORS.get(model), label=NAMES.get(model, f"model {model}"))
        print(f"model {model} ({NAMES.get(model, model)}): "
              f"recall {recalls[0]:.4f} -> {recalls[-1]:.4f} over {len(recalls)} rounds")

    ax.set(xlabel="churn round (cumulative delete+re-add)", ylabel="recall@10",
           title=args.title)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
