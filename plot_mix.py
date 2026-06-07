#!/usr/bin/env python3
"""
plot_mix.py — workload-mix CSV (from mix_sweep.sh) -> slide-ready plots.

x-axis = write fraction (insert+delete share). Each line is a series (impl, model),
so it serves both the locking comparison and the repair-method comparison.
Aggregates repeats (median + min/max bars). Two panels: search throughput and
delete throughput vs write fraction.

Usage: python3 plot_mix.py [mix_results.csv] [-o out.png]
"""
import csv, glob, os, sys, argparse
from collections import defaultdict
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMPL_LABEL = {"fine": "fine-grained", "coarse": "coarse (global lock)"}
IMPL_COLOR = {"fine": "#1f77b4", "coarse": "#d62728"}
MODEL_LABEL = {"2": "search (Wolverine)", "3": "two-hop (Pro)", "4": "approx (ProMax)"}
MODEL_COLOR = {"2": "#1f77b4", "3": "#2ca02c", "4": "#ff7f0e"}
CYCLE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]


def newest_csv():
    c = sorted(glob.glob("mix_results*.csv"), key=os.path.getmtime)
    if not c: sys.exit("No CSV given and no mix_results*.csv found.")
    return c[-1]


def fnum(row, key, d=0.0):
    try: return float(row.get(key, d) or d)
    except ValueError: return d


def wfrac(mix):
    sw, iw, dw = (float(x) for x in mix.split("/")); tot = sw + iw + dw
    return (iw + dw) / tot if tot else 0.0


def load(path):
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row.get("impl", "?"), row.get("model", "")); mix = row["mix"]
            raw[key][mix]["s_tp"].append(fnum(row, "search_throughput"))
            raw[key][mix]["d_tp"].append(fnum(row, "delete_throughput"))
    agg = {}
    for key, by_mix in raw.items():
        rows = []
        for mix in by_mix:
            s = by_mix[mix]["s_tp"]; d = by_mix[mix]["d_tp"]
            rows.append({"wf": wfrac(mix) * 100,
                         "s_tp": (median(s), min(s), max(s)),
                         "d_tp": (median(d), min(d), max(d))})
        rows.sort(key=lambda r: r["wf"]); agg[key] = rows
    return agg


def styling(keys):
    impls = sorted({k[0] for k in keys}); models = sorted({k[1] for k in keys})
    def label(k):
        impl, model = k
        if len(models) <= 1: return IMPL_LABEL.get(impl, impl)
        if len(impls) <= 1: return MODEL_LABEL.get(model, f"model {model}")
        return f"{IMPL_LABEL.get(impl, impl)} / {MODEL_LABEL.get(model, model)}"
    def color(k, i):
        impl, model = k
        if len(models) <= 1: return IMPL_COLOR.get(impl, CYCLE[i % len(CYCLE)])
        if len(impls) <= 1: return MODEL_COLOR.get(model, CYCLE[i % len(CYCLE)])
        return CYCLE[i % len(CYCLE)]
    return label, color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default="Concurrent HNSW delete — workload mix")
    args = ap.parse_args()
    path = args.csv or newest_csv()
    out = args.out or (os.path.splitext(path)[0] + ".png")
    agg = load(path)
    label_fn, color_fn = styling(list(agg.keys()))
    keys = sorted(agg.keys())

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13, 5))
    for i, k in enumerate(keys):
        rows = agg[k]; x = [r["wf"] for r in rows]; c = color_fn(k, i); lab = label_fn(k)
        for ax, metric in [(ax_s, "s_tp"), (ax_d, "d_tp")]:
            med = [r[metric][0] for r in rows]
            ye = [[r[metric][0] - r[metric][1] for r in rows], [r[metric][2] - r[metric][0] for r in rows]]
            ax.errorbar(x, med, yerr=ye, marker="o", capsize=3, color=c, label=lab)
    ax_s.set(xlabel="write fraction (%)", ylabel="search throughput (ops/s)", title="Search throughput vs write mix")
    ax_d.set(xlabel="write fraction (%)", ylabel="delete throughput (ops/s)", title="Delete throughput vs write mix")
    for ax in (ax_s, ax_d): ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=8)

    fig.suptitle(args.title); fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
