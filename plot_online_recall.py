#!/usr/bin/env python3
"""
plot_online_recall.py — plot the ONLINE recall trajectory per delete model.

Reads the interleaved-benchmark CSVs written by online_recall.sh (one per model)
and plots recall vs completed_ops from the "validation" rows — i.e. recall under
live concurrency, one line per repair model.

Usage: python3 plot_online_recall.py [online_recall_results_dir] [-o out.png]
"""
import csv, glob, os, re, sys, argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"2": "search (Wolverine)", "3": "two-hop (Pro)", "4": "approx (ProMax)"}
COLORS = {"2": "#1f77b4", "3": "#2ca02c", "4": "#ff7f0e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="online_recall_results")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default="Online repair quality: recall under concurrent churn")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "online_recall_model*.csv")))
    if not files:
        sys.exit(f"No online_recall_model*.csv in {args.dir}")
    out = args.out or os.path.join(args.dir, "online_recall.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for path in files:
        mm = re.search(r"online_recall_model(\d+)\.csv", os.path.basename(path))
        model = mm.group(1) if mm else "?"
        xs, ys = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("section") != "validation":
                    continue
                try:
                    xs.append(float(row["completed_ops"])); ys.append(float(row["recall"]))
                except (KeyError, ValueError):
                    pass
        if not xs:
            continue
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
        ax.plot(xs, ys, marker="o", color=COLORS.get(model), label=NAMES.get(model, f"model {model}"))
        print(f"model {model} ({NAMES.get(model, model)}): recall {ys[0]:.4f} -> {ys[-1]:.4f}")

    ax.set(xlabel="completed operations (concurrent churn)", ylabel="recall@10", title=args.title)
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
