#!/usr/bin/env python3
"""plot_latency_timeseries.py — search latency OVER TIME, one line per delete model.

Reads the CSV from latency_timeseries.sh and plots a search-latency percentile
against elapsed time (default) or cumulative deletes, one line per model. Values
are the median across repeats per (model, bucket).

    python3 plot_latency_timeseries.py latency_timeseries.csv -o latency_timeseries.png
    python3 plot_latency_timeseries.py latency_timeseries.csv --x deletes
    python3 plot_latency_timeseries.py latency_timeseries.csv --metric search_p95_ms
"""
import argparse
import csv
from collections import defaultdict


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="latency_timeseries.csv")
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: <csv>.png)")
    ap.add_argument("--metric", default="search_p50_ms",
                    help="y column (search_p50_ms|search_p95_ms|search_mean_ms)")
    ap.add_argument("--x", default="time", choices=["time", "deletes"],
                    help="x axis: elapsed time (default) or cumulative deletes")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xcol = "t_mid_s" if args.x == "time" else "cum_deletes"

    # series[model][bucket] = {"x": [...], "y": [...]}  (one entry per repeat)
    series = defaultdict(lambda: defaultdict(lambda: {"x": [], "y": []}))
    names = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            model = row["model"]
            names[model] = row.get("model_name", model)
            try:
                b = int(row["bucket"])
                series[model][b]["x"].append(float(row[xcol]))
                series[model][b]["y"].append(float(row[args.metric]))
            except (TypeError, ValueError, KeyError):
                pass

    label_map = {"0": "violent", "2": "search-delete", "3": "two-hop",
                 "4": "approx two-hop", "6": "naive tombstone",
                 "7": "naive reconstruction"}
    color_map = {"0": "tab:brown", "2": "tab:green", "3": "tab:blue",
                 "4": "tab:orange", "6": "tab:red", "7": "tab:purple"}

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model in sorted(series, key=lambda m: int(m) if m.isdigit() else 0):
        buckets = sorted(series[model])
        xs = [_median(series[model][b]["x"]) for b in buckets]
        ys = [_median(series[model][b]["y"]) for b in buckets]
        label = label_map.get(model, names.get(model, f"model {model}"))
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=2,
                color=color_map.get(model), label=label)

    pct = args.metric.replace("_ms", "").replace("search_", "").replace("_", " ")
    ax.set_xlabel("elapsed time (s)" if args.x == "time" else "cumulative deletes")
    ax.set_ylabel(f"search {pct} latency (ms)")
    ax.set_title(args.title or f"Search {pct} latency over time")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(title="delete model")

    out = args.out or (args.csv.rsplit(".", 1)[0] + ".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
