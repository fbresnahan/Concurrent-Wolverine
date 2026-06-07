#!/usr/bin/env python3
"""
plot_sweep.py — thread-sweep CSV (from sweep.sh) -> slide-ready plots.

Each line is a series identified by (impl, model). Works for both:
  * locking comparison    (IMPLS="fine coarse", one model)  -> lines = fine vs coarse
  * repair-method compare (IMPLS=fine, MODELS="2 3 4")       -> lines = search/two-hop/approx

Aggregates repeated runs (median; min/max error bars on throughput). Produces a
2x2 figure: search throughput, search speedup, delete throughput, delete p99.
Backward compatible with CSVs lacking the `model` column.

Usage: python3 plot_sweep.py [sweep_results.csv] [-o out.png]
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
    c = sorted(glob.glob("sweep_results*.csv"), key=os.path.getmtime)
    if not c: sys.exit("No CSV given and no sweep_results*.csv found.")
    return c[-1]


def fnum(row, key, d=0.0):
    try: return float(row.get(key, d) or d)
    except ValueError: return d


def load(path):
    # raw[(impl,model)][threads][metric] = [values]
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row.get("impl", "?"), row.get("model", ""))
            t = int(row["threads"])
            raw[key][t]["s_tp"].append(fnum(row, "search_throughput"))
            raw[key][t]["d_tp"].append(fnum(row, "delete_throughput"))
            raw[key][t]["s_p99"].append(fnum(row, "search_p99_ms"))
            raw[key][t]["d_p99"].append(fnum(row, "delete_p99_ms"))
            raw[key][t]["recall"].append(fnum(row, "recall"))
    agg = {}
    for key, by_t in raw.items():
        rows = []
        for t in sorted(by_t):
            m = by_t[t]
            rows.append({
                "t": t,
                "s_tp": (median(m["s_tp"]), min(m["s_tp"]), max(m["s_tp"])),
                "d_tp": (median(m["d_tp"]), min(m["d_tp"]), max(m["d_tp"])),
                "s_p99": median(m["s_p99"]),
                "d_p99": median(m["d_p99"]),
                "recall": (min(m["recall"]), max(m["recall"])),
            })
        agg[key] = rows
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


def errbar(rows, metric):
    med = [r[metric][0] for r in rows]
    lo = [r[metric][0] - r[metric][1] for r in rows]
    hi = [r[metric][2] - r[metric][0] for r in rows]
    return med, [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title", default="Concurrent HNSW delete — thread scaling")
    args = ap.parse_args()
    path = args.csv or newest_csv()
    out = args.out or (os.path.splitext(path)[0] + ".png")
    agg = load(path)
    label_fn, color_fn = styling(list(agg.keys()))
    keys = sorted(agg.keys())

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    (ax_stp, ax_sp), (ax_dtp, ax_dp99) = axes

    for i, k in enumerate(keys):
        rows = agg[k]; t = [r["t"] for r in rows]; c = color_fn(k, i); lab = label_fn(k)
        med, ye = errbar(rows, "s_tp")
        ax_stp.errorbar(t, med, yerr=ye, marker="o", capsize=3, color=c, label=lab)
        base = next((r["s_tp"][0] for r in rows if r["t"] == 1), rows[0]["s_tp"][0])
        ax_sp.plot(t, [r["s_tp"][0] / base if base else 0 for r in rows], marker="o", color=c, label=lab)
        med, ye = errbar(rows, "d_tp")
        ax_dtp.errorbar(t, med, yerr=ye, marker="o", capsize=3, color=c, label=lab)
        ax_dp99.plot(t, [r["d_p99"] for r in rows], marker="o", color=c, label=lab)

    max_t = max(r["t"] for rows in agg.values() for r in rows)
    ax_sp.plot([1, max_t], [1, max_t], ls="--", color="gray", alpha=0.6, label="ideal linear")

    for ax, ttl, yl in [(ax_stp, "Search throughput", "search ops/s"),
                        (ax_sp, "Search speedup", "speedup vs 1 thread"),
                        (ax_dtp, "Delete throughput", "delete ops/s"),
                        (ax_dp99, "Delete tail latency", "delete p99 (ms)")]:
        ax.set(xlabel="worker threads", ylabel=yl, title=ttl)
        ax.set_xscale("log", base=2); ax.grid(True, which="both", ls=":", alpha=0.5); ax.legend(fontsize=8)

    fig.suptitle(args.title); fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for k in keys:
        peak = max(agg[k], key=lambda r: r["s_tp"][0])
        rmin = min(r["recall"][0] for r in agg[k]); rmax = max(r["recall"][1] for r in agg[k])
        print(f"{label_fn(k):30s}: peak search {peak['s_tp'][0]:.0f} ops/s @ {peak['t']} thr; recall {rmin:.4f}-{rmax:.4f}")


if __name__ == "__main__":
    main()
