#!/usr/bin/env bash
#
# sweep.sh — thread-count scaling sweep of the fine-grained build vs the
# coarse-global-lock baseline, with repeats for error bars and latency capture.
#
# Build first:  make compare
# Run:          ./sweep.sh           (then: python3 plot_sweep.py sweep_results.csv)
#
# Environment knobs (defaults in brackets):
#   THREADS     thread counts to sweep            [1 2 4 8 16 32]
#   REPEATS     runs per (impl,threads) point     [3]   (-> median + min/max bars)
#   SEARCH_W/INSERT_W/DELETE_W  op mix             [80/10/10]
#   FINE_BIN / COARSE_BIN       binaries          [./hnsw_Wolverine_interleaved_*]
#   OUT         output CSV                         [sweep_results.csv]
#   + workload knobs from bench_lib.sh (INITIAL_ACTIVE, TOTAL_OPS, DELETE_MODEL, M, ...)
#
# Capacity note: keep (INSERT_W/100)*TOTAL_OPS < (max_elements - INITIAL_ACTIVE).

set -euo pipefail
source "$(dirname "$0")/bench_lib.sh"

FINE_BIN="${FINE_BIN:-./hnsw_Wolverine_interleaved_test}"
COARSE_BIN="${COARSE_BIN:-./hnsw_Wolverine_interleaved_baseline}"
THREADS="${THREADS:-1 2 4 8 16 32}"
REPEATS="${REPEATS:-3}"
SEARCH_W="${SEARCH_W:-80}"; INSERT_W="${INSERT_W:-10}"; DELETE_W="${DELETE_W:-10}"
SEED="${SEED:-100}"
OUT="${OUT:-sweep_results.csv}"

mkdir -p "$RUNDIR"
bench_header > "$OUT"

echo "Thread sweep: [$THREADS]  repeats=$REPEATS  mix=$SEARCH_W/$INSERT_W/$DELETE_W"
echo "workload: initial_active=$INITIAL_ACTIVE total_ops=$TOTAL_OPS delete_model=$DELETE_MODEL"
for t in $THREADS; do
  echo "== threads=$t =="
  for r in $(seq 1 "$REPEATS"); do
    s=$((SEED + r))
    [ -x "$FINE_BIN" ]   && bench_run fine   "$FINE_BIN"   "$t" "$SEARCH_W" "$INSERT_W" "$DELETE_W" "$r" "$s" || true
    [ -x "$COARSE_BIN" ] && bench_run coarse "$COARSE_BIN" "$t" "$SEARCH_W" "$INSERT_W" "$DELETE_W" "$r" "$s" || true
  done
done
echo
echo "Wrote $OUT"
