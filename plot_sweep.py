#!/usr/bin/env python3
"""
plot_sweep.py — turn sweep.sh output into slide-ready plots.

Reads a CSV produced by sweep.sh with columns:
    impl,threads,search_ops,search_throughput,insert_throughput,
    delete_throughput,recall,invalid_labels

Produces a two-panel PNG:
    (left)  search throughput vs threads, one line per implementation
    (right) parallel speedup vs threads (relative to each impl's 1-thread run),
            with an ideal-linear reference line

Usage:
    python3 plot_sweep.py [sweep_results.csv] [-o out.png]
    # defaults: newest sweep_results*.csv  ->  <csv-stem>.png

No pandas required (stdlib csv + matplotlib only).
"""
import csv
import glob
import os
import sys
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt

LABELS = {"fine": "fine-grained", "coarse": "coarse (global lock)"}
COLORS = {"fine": "#1f77b4", "coarse": "#d62728"}


def newest_csv():
    cands = sorted(glob.glob("sweep_results*.csv"), key=os.path.getmtime)
    if not cands:
        sys.exit("No CSV given and no sweep_results*.csv found in cwd.")
    return cands[-1]


def load(path):
    # data[impl] = list of (threads, search_tp, insert_tp, delete_tp, recall)
    data = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["impl"]].append((
                int(row["threads"]),
                float(row["search_throughput"]),
                float(row["insert_throughput"]),
                float(row["delete_throughput"]),
                float(row["recall"]),
            ))
    for impl in data:
        data[impl].sort(key=lambda r: r[0])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="sweep CSV (default: newest sweep_results*.csv)")
    ap.add_argument("-o", "--out", help="output PNG (default: <csv-stem>.png)")
    ap.add_argument("--title", default="Concurrent HNSW: fine-grained vs coarse locking")
    args = ap.parse_args()

    path = args.csv or newest_csv()
    out = args.out or (os.path.splitext(path)[0] + ".png")
    data = load(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- panel 1: throughput vs threads ----
    for impl, rows in data.items():
        t = [r[0] for r in rows]
        tp = [r[1] for r in rows]
        ax1.plot(t, tp, marker="o", color=COLORS.get(impl), label=LABELS.get(impl, impl))
    ax1.set_xlabel("worker threads")
    ax1.set_ylabel("search throughput (ops/s)")
    ax1.set_title("Throughput")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    # ---- panel 2: speedup vs threads ----
    max_t = 1
    for impl, rows in data.items():
        base = next((r[1] for r in rows if r[0] == 1), rows[0][1])
        t = [r[0] for r in rows]
        sp = [r[1] / base if base else 0 for r in rows]
        max_t = max(max_t, max(t))
        ax2.plot(t, sp, marker="o", color=COLORS.get(impl), label=LABELS.get(impl, impl))
    ax2.plot([1, max_t], [1, max_t], ls="--", color="gray", alpha=0.6, label="ideal linear")
    ax2.set_xlabel("worker threads")
    ax2.set_ylabel("speedup vs 1 thread")
    ax2.set_title("Parallel speedup")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # small text summary for the console / slide notes
    for impl, rows in data.items():
        peak = max(rows, key=lambda r: r[1])
        rec = [r[4] for r in rows]
        print(f"{impl:7s}: peak search {peak[1]:.0f} ops/s @ {peak[0]} threads; "
              f"recall {min(rec):.4f}-{max(rec):.4f}")


if __name__ == "__main__":
    main()
