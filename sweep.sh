#!/usr/bin/env bash
#
# sweep.sh — run the fine-grained build and the coarse-global-lock baseline across
# a list of thread counts and emit a single CSV for plotting.
#
# Build the two binaries first:   make compare
# Then run:                       ./sweep.sh
#
# Everything is configurable via environment variables (defaults in brackets):
#
#   THREADS         thread counts to sweep            [1 2 4 8 16 32]
#   FINE_BIN        fine-grained binary              [./hnsw_Wolverine_interleaved_test]
#   COARSE_BIN      coarse baseline binary           [./hnsw_Wolverine_interleaved_baseline]
#   DATA / QUERIES  dataset paths                    [./datasets/sift_*.fbin]
#   INITIAL_ACTIVE  points in the index at start      [60000]
#   TOTAL_OPS       total mixed ops per run           [200000]
#   SEARCH_W/INSERT_W/DELETE_W  op mix weights        [80/10/10]
#   DELETE_MODEL    3=two-hop, 4=approx two-hop       [3]
#   M/EFC/EFS/K/NEW_LINK/SEED  index + search params  [16/200/200/10/16/100]
#   OUT             output CSV                        [sweep_results.csv]
#
# IMPORTANT capacity constraint: each run allocates a fresh index sized to the
# dataset (sift_learn = 100,000 vectors). deletePointConcurrent does NOT recycle
# internal ids, so per-run inserts consume fresh slots. Keep:
#       (INSERT_W/100) * TOTAL_OPS  <  (max_elements - INITIAL_ACTIVE)
# or inserts will start failing near the end of a run. The defaults satisfy this
# for sift_learn (20,000 inserts vs 40,000 headroom).

set -euo pipefail

THREADS="${THREADS:-1 2 4 8 16 32}"
FINE_BIN="${FINE_BIN:-./hnsw_Wolverine_interleaved_test}"
COARSE_BIN="${COARSE_BIN:-./hnsw_Wolverine_interleaved_baseline}"
DATA="${DATA:-./datasets/sift_learn.fbin}"
QUERIES="${QUERIES:-./datasets/sift_query.fbin}"
INITIAL_ACTIVE="${INITIAL_ACTIVE:-60000}"
TOTAL_OPS="${TOTAL_OPS:-200000}"
SEARCH_W="${SEARCH_W:-80}"
INSERT_W="${INSERT_W:-10}"
DELETE_W="${DELETE_W:-10}"
DELETE_MODEL="${DELETE_MODEL:-3}"
M="${M:-16}"
EFC="${EFC:-200}"
EFS="${EFS:-200}"
K="${K:-10}"
NEW_LINK="${NEW_LINK:-16}"
SEED="${SEED:-100}"
OUT="${OUT:-sweep_results.csv}"
RUNDIR="${RUNDIR:-./sweep_runs}"

mkdir -p "$RUNDIR"
echo "impl,threads,search_ops,search_throughput,insert_throughput,delete_throughput,recall,invalid_labels" > "$OUT"

# Pull "throughput: <num>" from a specific "<Label> ops:" line.
field_throughput() { grep -m1 "$2" "$1" | grep -oE "throughput: [0-9.]+" | grep -oE "[0-9.]+" || echo 0; }

run_one() {
  local impl="$1" bin="$2" threads="$3"
  local log="$RUNDIR/${impl}_t${threads}.log"
  "$bin" \
    --data "$DATA" --queries "$QUERIES" \
    --results "$RUNDIR/${impl}_t${threads}.csv" \
    --initial-active "$INITIAL_ACTIVE" --total-ops "$TOTAL_OPS" \
    --validation-interval 0 --check-reverse-links 0 \
    --k "$K" --M "$M" --ef-construction "$EFC" --ef-search "$EFS" \
    --worker-threads "$threads" \
    --search-weight "$SEARCH_W" --insert-weight "$INSERT_W" --delete-weight "$DELETE_W" \
    --delete-model "$DELETE_MODEL" --new-link-size "$NEW_LINK" --seed "$SEED" \
    > "$log" 2>&1

  local s_ops s_tp i_tp d_tp recall inval
  s_ops=$(grep -m1 "Search ops:" "$log"  | grep -oE "Search ops: [0-9]+" | grep -oE "[0-9]+" || echo 0)
  s_tp=$(field_throughput "$log" "Search ops:")
  i_tp=$(field_throughput "$log" "Insert ops:")
  d_tp=$(field_throughput "$log" "Delete ops:")
  recall=$(grep -m1 "Final validation recall:" "$log" | grep -oE "recall: [0-9.]+" | grep -oE "[0-9.]+" || echo 0)
  inval=$(grep -m1 "Final validation" "$log" | grep -oE "invalid_labels: [0-9]+" | grep -oE "[0-9]+" || echo 0)

  echo "$impl,$threads,$s_ops,$s_tp,$i_tp,$d_tp,$recall,$inval" >> "$OUT"
  printf "  %-7s t=%-4s  search=%-12s ops/s   recall=%-7s invalid=%s\n" "$impl" "$threads" "$s_tp" "$recall" "$inval"
}

echo "Sweeping threads: $THREADS"
echo "initial_active=$INITIAL_ACTIVE total_ops=$TOTAL_OPS mix=$SEARCH_W/$INSERT_W/$DELETE_W delete_model=$DELETE_MODEL"
for t in $THREADS; do
  echo "== threads=$t =="
  [ -x "$FINE_BIN" ]   && run_one fine   "$FINE_BIN"   "$t" || echo "  (missing $FINE_BIN — run 'make compare')"
  [ -x "$COARSE_BIN" ] && run_one coarse "$COARSE_BIN" "$t" || echo "  (missing $COARSE_BIN — run 'make compare')"
done
echo
echo "Wrote $OUT"
