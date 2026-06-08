#!/usr/bin/env bash
#
# recall_vs_mix.sh — RECALL vs WRITE FRACTION, per delete model.
#
# Sweeps the write fraction of the interleaved workload and records the final
# recall (measured by the built-in validator: brute force over the LIVE active
# set) for each delete model. Produces the data for a single graph:
#
#     x = write fraction (% of ops that are insert+delete, split evenly)
#     y = recall
#     one line per model: two-hop (3), approx-two-hop (4), naive reconstruction (7)
#
# Naive reconstruction just wires a deleted node's neighbors to each other without
# restoring the monotonic search path; two-hop / approx pick path-aware edges. The
# graph shows whether path-aware repair buys recall over the naive patch as churn
# grows.
#
# IMPORTANT — ef_search sensitivity: with a generous ef_search (e.g. 200) ALL
# methods sit at ~1.0 recall and the graph is flat — the search budget hides every
# difference. To see the contrast you must use a MODERATE ef_search where baseline
# recall is below the ceiling (default EFS=40). All three are close, so use
# REPEATS>=5 for stable medians and min/max whiskers.
#
# Build first:  make interleaved      (or the local g++ one-liner)
# Run:          ./recall_vs_mix.sh
# Plot:         python3 plot_recall_vs_mix.py recall_vs_mix.csv -o recall_vs_mix.png
#
# Knobs (env): MODELS [3 4 7], WRITE_FRACS [0 20 40 60 80], THREADS [8],
#   INITIAL_ACTIVE [10000], TOTAL_OPS [150000], REPEATS [5],
#   VAL_QUERIES [500], M/EFC/EFS[40]/K/NEW_LINK, SEED, BIN, OUT.

set -euo pipefail
BIN="${BIN:-./hnsw_Wolverine_interleaved_test}"
DATA="${DATA:-./datasets/sift_learn.fbin}"
QUERIES="${QUERIES:-./datasets/sift_query.fbin}"
MODELS="${MODELS:-3 4 7}"
WRITE_FRACS="${WRITE_FRACS:-0 20 40 60 80}"
THREADS="${THREADS:-8}"
INITIAL_ACTIVE="${INITIAL_ACTIVE:-10000}"
TOTAL_OPS="${TOTAL_OPS:-150000}"
REPEATS="${REPEATS:-5}"
VAL_QUERIES="${VAL_QUERIES:-500}"
M="${M:-16}"; EFC="${EFC:-200}"; EFS="${EFS:-40}"; K="${K:-10}"; NEW_LINK="${NEW_LINK:-16}"
SEED="${SEED:-100}"
OUT="${OUT:-recall_vs_mix.csv}"
RUNDIR="${RUNDIR:-./recall_vs_mix_runs}"

model_name() { case "$1" in 2) echo search;; 3) echo two-hop;; 4) echo approx;; 6) echo naive-tombstone;; 7) echo naive-reconstruction;; *) echo "model$1";; esac; }

mkdir -p "$RUNDIR"
[ -x "$BIN" ] || { echo "Missing $BIN — run 'make interleaved'"; exit 1; }

echo "model,model_name,write_frac,search_w,insert_w,delete_w,run,recall,invalid_labels,active_count,threads" > "$OUT"
echo "Recall vs write fraction: models=[$MODELS] write_fracs=[$WRITE_FRACS] threads=$THREADS ops=$TOTAL_OPS repeats=$REPEATS"

for m in $MODELS; do
  for wf in $WRITE_FRACS; do
    # split the write fraction evenly between insert and delete; rest is search
    dw=$(( wf / 2 )); iw=$(( wf - dw )); sw=$(( 100 - wf ))
    for r in $(seq 1 "$REPEATS"); do
      tag="m${m}_wf${wf}_r${r}"
      log="$RUNDIR/${tag}.log"
      seed=$(( SEED + r ))
      if ! ${NUMACTL:-} "$BIN" --data "$DATA" --queries "$QUERIES" --results "$RUNDIR/${tag}.csv" \
          --initial-active "$INITIAL_ACTIVE" --total-ops "$TOTAL_OPS" \
          --validation-interval 0 --validation-queries "$VAL_QUERIES" \
          --k "$K" --M "$M" --ef-construction "$EFC" --ef-search "$EFS" \
          --worker-threads "$THREADS" \
          --search-weight "$sw" --insert-weight "$iw" --delete-weight "$dw" \
          --delete-model "$m" --new-link-size "$NEW_LINK" --check-reverse-links 0 --seed "$seed" \
          > "$log" 2>&1; then
        echo "  ! run failed: $tag (see $log)"; continue
      fi
      final=$(grep -m1 "Final validation" "$log" || true)
      recall=$(sed -nE 's/.*recall: ([0-9.]+).*/\1/p' <<<"$final")
      inval=$(sed -nE 's/.*invalid_labels: ([0-9]+).*/\1/p' <<<"$final")
      active=$(sed -nE 's/.*active_count: ([0-9]+).*/\1/p' <<<"$final")
      echo "${m},$(model_name "$m"),${wf},${sw},${iw},${dw},${r},${recall:-0},${inval:-0},${active:-0},${THREADS}" >> "$OUT"
      printf "  %-16s wf=%-3s r%s  recall=%-7s invalid=%-4s active=%s\n" \
        "$(model_name "$m")" "$wf" "$r" "${recall:-0}" "${inval:-0}" "${active:-0}"
    done
  done
done

echo "Wrote $OUT  ->  python3 plot_recall_vs_mix.py $OUT -o ${OUT%.csv}.png"
