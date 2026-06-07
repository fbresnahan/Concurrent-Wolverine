#!/usr/bin/env bash
#
# mix_sweep.sh — sweep the read/write mix (search/insert/delete weights) at a
# FIXED thread count, to show where fine-grained locking wins big (read-heavy)
# and where the global reverse-link / metadata locks become the bottleneck
# (write-heavy).
#
# Build first:  make compare
# Run:          ./mix_sweep.sh      (then: python3 plot_mix.py mix_results.csv)
#
# Environment knobs (defaults in brackets):
#   MIX_THREADS   thread count for all mixes        [16]
#   MIXES         "sw iw dw" combos, '|'-separated   [100/0/0 .. 0/50/50]
#   REPEATS       runs per (impl,mix) point          [3]
#   FINE_BIN / COARSE_BIN, OUT, + bench_lib knobs
#
# Write-heavy mixes insert a lot, and deletePointConcurrent does NOT recycle
# internal ids, so we default to a SMALLER initial set + fewer ops to stay under
# the dataset capacity (sift_learn = 100k): 50% of 100k ops = 50k inserts, and
# 20k initial + 50k < 100k. Override INITIAL_ACTIVE/TOTAL_OPS if you change the
# heaviest mix.

set -euo pipefail
# write-safe defaults BEFORE sourcing the lib (lib uses :=, so these win)
: "${INITIAL_ACTIVE:=20000}"
: "${TOTAL_OPS:=100000}"
source "$(dirname "$0")/bench_lib.sh"

FINE_BIN="${FINE_BIN:-./hnsw_Wolverine_interleaved_test}"
COARSE_BIN="${COARSE_BIN:-./hnsw_Wolverine_interleaved_baseline}"
MIX_THREADS="${MIX_THREADS:-16}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-100}"
OUT="${OUT:-mix_results.csv}"
# Keep search >= 30% in every mix so search throughput stays meaningful; the
# write fraction still ranges 0% -> 70%.
MIXES="${MIXES:-100 0 0|90 5 5|70 15 15|50 25 25|30 35 35}"

mkdir -p "$RUNDIR"
bench_header > "$OUT"

echo "Mix sweep @ ${MIX_THREADS} threads  repeats=$REPEATS"
echo "workload: initial_active=$INITIAL_ACTIVE total_ops=$TOTAL_OPS delete_model=$DELETE_MODEL"
IFS='|' read -ra MIXARR <<< "$MIXES"
for mix in "${MIXARR[@]}"; do
  read -r sw iw dw <<< "$mix"
  echo "== mix=$sw/$iw/$dw =="
  for r in $(seq 1 "$REPEATS"); do
    s=$((SEED + r))
    [ -x "$FINE_BIN" ]   && bench_run fine   "$FINE_BIN"   "$MIX_THREADS" "$sw" "$iw" "$dw" "$r" "$s" || true
    [ -x "$COARSE_BIN" ] && bench_run coarse "$COARSE_BIN" "$MIX_THREADS" "$sw" "$iw" "$dw" "$r" "$s" || true
  done
done
echo
echo "Wrote $OUT"
