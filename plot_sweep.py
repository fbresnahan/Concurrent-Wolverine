#!/usr/bin/env python3
"""
plot_sweep.py — turn a thread-sweep CSV (from sweep.sh) into slide-ready plots.

Aggregates repeated runs (median, with min/max error bars) and produces:
    (1) search throughput vs threads
    (2) parallel speedup vs threads (vs each impl's 1-thread median) + ideal line
    (3) search p99 latency vs threads      [only if latency columns are present]

Backward compatible with the older sweep CSV (no run/mix/latency columns).

Usage:
    python3 plot_sweep.py [sweep_results.csv] [-o out.png]
"""
import csv, glob, os, sys, argparse
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"fine": "fine-grained", "coarse": "coarse (global lock)"}
COLORS = {"fine": "#1f77b4", "coarse": "#d62728"}


def newest_csv():
    c = sorted(glob.glob("sweep_results*.csv"), key=os.path.getmtime)
    if not c:
        sys.exit("No CSV given and no sweep_results*.csv found.")
    return c[-1]


def fnum(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def load(path):
    # raw[impl][threads] = {"tp": [...], "p99": [...], "recall": [...]}
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    has_lat = False
    with open(path) as f:
        for row in csv.DictReader(f):
            impl = row["impl"]; t = int(row["threads"])
            raw[impl][t]["tp"].append(fnum(row, "search_throughput"))
            raw[impl][t]["recall"].append(fnum(row, "recall"))
            if "search_p99_ms" in row:
                p99 = fnum(row, "search_p99_ms")
                raw[impl][t]["p99"].append(p99)
                has_lat = has_lat or p99 > 0
    # aggregate -> agg[impl] = sorted list of (threads, tp_med, tp_lo, tp_hi, p99_med, recall_min, recall_max)
    agg = {}
    for impl, by_t in raw.items():
        rows = []
        for t in sorted(by_t):
            tp = by_t[t]["tp"]; p99 = by_t[t]["p99"]; rec = by_t[t]["recall"]
            rows.append((
                t, median(tp), min(tp), max(tp),
                median(p99) if p99 else 0.0,
                min(rec) if rec else 0.0, max(rec) if rec else 0.0,
            ))
        agg[impl] = rows
    return agg, has_lat


def errbars(rows, med_i, lo_i, hi_i):
    med = [r[med_i] for r in rows]
    lo = [r[med_i] - r[lo_i] for r in rows]
    hi = [r[hi_i] - r[med_i] for r in rows]
    return med, [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default="Concurrent HNSW: fine-grained vs coarse locking")
    args = ap.parse_args()

    path = args.csv or newest_csv()
    out = args.out or (os.path.splitext(path)[0] + ".png")
    agg, has_lat = load(path)

    npanels = 3 if has_lat else 2
    fig, axes = plt.subplots(1, npanels, figsize=(6 * npanels, 5))
    ax_tp, ax_sp = axes[0], axes[1]

    # panel 1: throughput vs threads (median + min/max bars)
    for impl, rows in agg.items():
        t = [r[0] for r in rows]
        med, yerr = errbars(rows, 1, 2, 3)
        ax_tp.errorbar(t, med, yerr=yerr, marker="o", capsize=3,
                       color=COLORS.get(impl), label=LABELS.get(impl, impl))
    ax_tp.set(xlabel="worker threads", ylabel="search throughput (ops/s)", title="Throughput")
    ax_tp.set_xscale("log", base=2); ax_tp.grid(True, which="both", ls=":", alpha=0.5); ax_tp.legend()

    # panel 2: speedup vs threads
    max_t = 1
    for impl, rows in agg.items():
        base = next((r[1] for r in rows if r[0] == 1), rows[0][1])
        t = [r[0] for r in rows]; sp = [r[1] / base if base else 0 for r in rows]
        max_t = max(max_t, max(t))
        ax_sp.plot(t, sp, marker="o", color=COLORS.get(impl), label=LABELS.get(impl, impl))
    ax_sp.plot([1, max_t], [1, max_t], ls="--", color="gray", alpha=0.6, label="ideal linear")
    ax_sp.set(xlabel="worker threads", ylabel="speedup vs 1 thread", title="Parallel speedup")
    ax_sp.set_xscale("log", base=2); ax_sp.grid(True, which="both", ls=":", alpha=0.5); ax_sp.legend()

    # panel 3: p99 latency vs threads
    if has_lat:
        ax_lat = axes[2]
        for impl, rows in agg.items():
            t = [r[0] for r in rows]; p99 = [r[4] for r in rows]
            ax_lat.plot(t, p99, marker="o", color=COLORS.get(impl), label=LABELS.get(impl, impl))
        ax_lat.set(xlabel="worker threads", ylabel="search p99 latency (ms)", title="Tail latency (p99)")
        ax_lat.set_xscale("log", base=2); ax_lat.grid(True, which="both", ls=":", alpha=0.5); ax_lat.legend()

    fig.suptitle(args.title); fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for impl, rows in agg.items():
        peak = max(rows, key=lambda r: r[1])
        rmin = min(r[5] for r in rows); rmax = max(r[6] for r in rows)
        print(f"{impl:7s}: peak search {peak[1]:.0f} ops/s @ {peak[0]} threads; recall {rmin:.4f}-{rmax:.4f}")


if __name__ == "__main__":
    main()
