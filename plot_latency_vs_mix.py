#!/usr/bin/env python3
"""plot_latency_vs_mix.py — search latency vs write fraction, per delete model.

Reads the CSV from latency_vs_mix.sh and plots a chosen search-latency percentile
vs write fraction (median across repeats, min/max whiskers), one line per model.

    python3 plot_latency_vs_mix.py latency_vs_mix.csv -o latency_vs_mix.png
    python3 plot_latency_vs_mix.py latency_vs_mix.csv --metric search_p95_ms
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="latency_vs_mix.csv")
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: <csv>.png)")
    ap.add_argument("--metric", default="search_p99_ms",
                    help="column to plot (default search_p99_ms)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = defaultdict(lambda: defaultdict(list))
    names = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            names[model] = row.get("model_name", model)
            wf = float(row["write_frac"])
            try:
                series[model][wf].append(float(row[args.metric]))
            except (TypeError, ValueError, KeyError):
                pass

    label_map = {"2": "search-delete", "3": "two-hop", "4": "approx two-hop",
                 "6": "naive tombstone", "7": "naive reconstruction"}
    color_map = {"2": "tab:green", "3": "tab:blue", "4": "tab:orange",
                 "6": "tab:red", "7": "tab:purple"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in sorted(series, key=lambda m: int(m) if m.isdigit() else 0):
        wfs = sorted(series[model])
        med, lo, hi = [], [], []
        for wf in wfs:
            vals = sorted(series[model][wf])
            n = len(vals)
            m = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
            med.append(m)
            lo.append(m - vals[0])
            hi.append(vals[-1] - m)
        label = label_map.get(model, names.get(model, f"model {model}"))
        ax.errorbar(wfs, med, yerr=[lo, hi], marker="o", capsize=3,
                    color=color_map.get(model), label=label, linewidth=2)

    pretty = args.metric.replace("_ms", "").replace("search_", "search ").replace("_", " ")
    ax.set_xlabel("write fraction (% of ops that are insert+delete)")
    ax.set_ylabel(f"{pretty} latency (ms)")
    ax.set_title(args.title or f"Search {pretty} latency vs write fraction")
    ax.grid(True, alpha=0.3)
    ax.legend(title="delete model")

    out = args.out or (args.csv.rsplit(".", 1)[0] + ".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
