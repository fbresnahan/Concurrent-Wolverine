#!/usr/bin/env bash
#
# latency_timeseries.sh — SEARCH LATENCY OVER TIME, per delete model.
#
# Runs one delete-heavy interleaved workload per model and captures the search
# latency in time buckets across the run (via the binary's --timeseries-file).
# Concatenates every run into one CSV for plotting:
#
#     x = elapsed time (s)   [or cumulative deletes]
#     y = search p50 latency (ms)
#     one line per model: two-hop (3), approx (4), naive tombstone (6)
#
# The point: as deletes accumulate, tombstoned nodes pile up in the graph and
# search latency drifts UP for naive tombstoning, while two-hop / approx remove
# dead nodes so their latency stays flat.
#
# Build first:  make interleaved      (or the local g++ one-liner)
# Run:          ./latency_timeseries.sh
# Plot:         python3 plot_latency_timeseries.py latency_timeseries.csv -o latency_timeseries.png
#
# Knobs (env): MODELS [3 4 6], THREADS [40], INITIAL_ACTIVE [10000],
#   TOTAL_OPS [300000], BUCKETS [50], REPEATS [3], EFS [100],
#   SEARCH_W/INSERT_W/DELETE_W [40/30/30], M/EFC/K/NEW_LINK, SEED, ATTEMPTS,
#   BIN, OUT, RUNDIR.

set -euo pipefail
BIN="${BIN:-./hnsw_Wolverine_interleaved_test}"
DATA="${DATA:-./datasets/sift_learn.fbin}"
QUERIES="${QUERIES:-./datasets/sift_query.fbin}"
MODELS="${MODELS:-3 4 6}"
THREADS="${THREADS:-40}"
INITIAL_ACTIVE="${INITIAL_ACTIVE:-10000}"
TOTAL_OPS="${TOTAL_OPS:-300000}"
BUCKETS="${BUCKETS:-50}"
REPEATS="${REPEATS:-3}"
EFS="${EFS:-100}"
SEARCH_W="${SEARCH_W:-40}"; INSERT_W="${INSERT_W:-30}"; DELETE_W="${DELETE_W:-30}"
M="${M:-16}"; EFC="${EFC:-200}"; K="${K:-10}"; NEW_LINK="${NEW_LINK:-16}"
SEED="${SEED:-100}"
ATTEMPTS="${ATTEMPTS:-3}"
OUT="${OUT:-latency_timeseries.csv}"
RUNDIR="${RUNDIR:-./latency_timeseries_runs}"

model_name() { case "$1" in 0) echo violent;; 2) echo search;; 3) echo two-hop;; 4) echo approx;; 6) echo naive-tombstone;; 7) echo naive-reconstruction;; *) echo "model$1";; esac; }

mkdir -p "$RUNDIR"
[ -x "$BIN" ] || { echo "Missing $BIN — run 'make interleaved'"; exit 1; }

# Combined CSV: model,model_name,run,bucket,t_mid_s,search_count,search_p50_ms,search_p95_ms,search_mean_ms,cum_deletes
echo "model,model_name,run,bucket,t_mid_s,search_count,search_p50_ms,search_p95_ms,search_mean_ms,cum_deletes" > "$OUT"
echo "Latency over time: models=[$MODELS] threads=$THREADS ops=$TOTAL_OPS buckets=$BUCKETS ef=$EFS mix=${SEARCH_W}/${INSERT_W}/${DELETE_W} repeats=$REPEATS"

for m in $MODELS; do
  for r in $(seq 1 "$REPEATS"); do
    tag="m${m}_r${r}"
    log="$RUNDIR/${tag}.log"
    ts="$RUNDIR/${tag}.ts.csv"
    seed=$(( SEED + r ))
    ok=0
    for attempt in $(seq 1 "$ATTEMPTS"); do
      if ${NUMACTL:-} "$BIN" --data "$DATA" --queries "$QUERIES" --results "$RUNDIR/${tag}.csv" \
          --timeseries-file "$ts" --timeseries-buckets "$BUCKETS" \
          --initial-active "$INITIAL_ACTIVE" --total-ops "$TOTAL_OPS" \
          --validation-interval 0 --validation-queries 100 \
          --k "$K" --M "$M" --ef-construction "$EFC" --ef-search "$EFS" \
          --worker-threads "$THREADS" \
          --search-weight "$SEARCH_W" --insert-weight "$INSERT_W" --delete-weight "$DELETE_W" \
          --delete-model "$m" --new-link-size "$NEW_LINK" --check-reverse-links 0 --seed "$seed" \
          > "$log" 2>&1; then
        ok=1; break
      fi
      echo "  · attempt $attempt failed: $tag (see $log) — retrying"
    done
    if [ "$ok" -ne 1 ]; then echo "  ! run failed after $ATTEMPTS attempts: $tag"; continue; fi
    # append the per-bucket rows (skip the timeseries header), injecting the run column
    awk -F, -v run="$r" 'NR>1 {print $1","$2","run","$3","$4","$5","$6","$7","$8","$9}' "$ts" >> "$OUT"
    last=$(tail -1 "$ts")
    printf "  %-16s r%s done  (final bucket: %s)\n" "$(model_name "$m")" "$r" "$(echo "$last" | cut -d, -f4-9)"
  done
done

echo "Wrote $OUT  ->  python3 plot_latency_timeseries.py $OUT -o ${OUT%.csv}.png"
