#!/usr/bin/env python3
"""
plot_mix.py — turn a workload-mix CSV (from mix_sweep.sh) into slide-ready plots.

x-axis is the write fraction (insert+delete share of the mix). Shows how the
two implementations behave as the workload shifts from read-heavy to
write-heavy, at a fixed thread count. Aggregates repeats (median + min/max bars).

    (1) search throughput vs write fraction
    (2) search p99 latency vs write fraction   [if latency columns present]

Usage:
    python3 plot_mix.py [mix_results.csv] [-o out.png]
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
    c = sorted(glob.glob("mix_results*.csv"), key=os.path.getmtime)
    if not c:
        sys.exit("No CSV given and no mix_results*.csv found.")
    return c[-1]


def fnum(row, key, d=0.0):
    try:
        return float(row.get(key, d) or d)
    except ValueError:
        return d


def write_fraction(mix):  # "sw/iw/dw" -> (iw+dw)/total
    sw, iw, dw = (float(x) for x in mix.split("/"))
    tot = sw + iw + dw
    return (iw + dw) / tot if tot else 0.0


def load(path):
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    has_lat = False
    with open(path) as f:
        for row in csv.DictReader(f):
            impl = row["impl"]; mix = row["mix"]
            raw[impl][mix]["tp"].append(fnum(row, "search_throughput"))
            raw[impl][mix]["recall"].append(fnum(row, "recall"))
            if "search_p99_ms" in row:
                p = fnum(row, "search_p99_ms"); raw[impl][mix]["p99"].append(p)
                has_lat = has_lat or p > 0
    agg = {}
    for impl, by_mix in raw.items():
        rows = []
        for mix in by_mix:
            tp = by_mix[mix]["tp"]; p99 = by_mix[mix]["p99"]
            rows.append((write_fraction(mix), mix, median(tp), min(tp), max(tp),
                         median(p99) if p99 else 0.0))
        rows.sort(key=lambda r: r[0])
        agg[impl] = rows
    return agg, has_lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default="Concurrent HNSW: throughput vs workload mix")
    args = ap.parse_args()

    path = args.csv or newest_csv()
    out = args.out or (os.path.splitext(path)[0] + ".png")
    agg, has_lat = load(path)

    npanels = 2 if has_lat else 1
    fig, axes = plt.subplots(1, npanels, figsize=(6 * npanels, 5), squeeze=False)
    ax_tp = axes[0][0]

    for impl, rows in agg.items():
        x = [r[0] * 100 for r in rows]
        med = [r[2] for r in rows]
        yerr = [[r[2] - r[3] for r in rows], [r[4] - r[2] for r in rows]]
        ax_tp.errorbar(x, med, yerr=yerr, marker="o", capsize=3,
                       color=COLORS.get(impl), label=LABELS.get(impl, impl))
    ax_tp.set(xlabel="write fraction (% insert+delete)", ylabel="search throughput (ops/s)",
              title="Throughput vs write mix")
    ax_tp.grid(True, ls=":", alpha=0.5); ax_tp.legend()

    if has_lat:
        ax_lat = axes[0][1]
        for impl, rows in agg.items():
            x = [r[0] * 100 for r in rows]; p99 = [r[5] for r in rows]
            ax_lat.plot(x, p99, marker="o", color=COLORS.get(impl), label=LABELS.get(impl, impl))
        ax_lat.set(xlabel="write fraction (% insert+delete)", ylabel="search p99 latency (ms)",
                   title="Tail latency vs write mix")
        ax_lat.grid(True, ls=":", alpha=0.5); ax_lat.legend()

    fig.suptitle(args.title); fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
