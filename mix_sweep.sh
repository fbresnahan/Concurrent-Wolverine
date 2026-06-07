#!/usr/bin/env bash
#
# mix_sweep.sh — sweep the read/write mix at a FIXED thread count, over locking
# implementations (IMPLS) and delete models (MODELS). Shows where fine-grained
# wins (read-heavy) vs where the global locks bottleneck (write-heavy), and how
# each repair model's write overhead grows.
#
# Build first:  make compare
# Run:          ./mix_sweep.sh      # then: python3 plot_mix.py mix_results.csv
#
#   Locking:        IMPLS="fine coarse" MODELS=3 ./mix_sweep.sh
#   Repair methods: IMPLS=fine MODELS="2 3 4" ./mix_sweep.sh
#
# Knobs: MIX_THREADS [16], MIXES [0%..70% writes], REPEATS [3], IMPLS, MODELS,
#        FINE_BIN/COARSE_BIN/OUT + bench_lib knobs.
#
# Write-safe capacity defaults (sift_learn = 100k; heaviest mix inserts a lot):
: "${INITIAL_ACTIVE:=20000}"
: "${TOTAL_OPS:=100000}"

set -euo pipefail
source "$(dirname "$0")/bench_lib.sh"

FINE_BIN="${FINE_BIN:-./hnsw_Wolverine_interleaved_test}"
COARSE_BIN="${COARSE_BIN:-./hnsw_Wolverine_interleaved_baseline}"
IMPLS="${IMPLS:-fine coarse}"
MODELS="${MODELS:-3}"
MIX_THREADS="${MIX_THREADS:-16}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-100}"
OUT="${OUT:-mix_results.csv}"
# Keep search >= 30% so search throughput stays meaningful; write fraction 0->70%.
MIXES="${MIXES:-100 0 0|90 5 5|70 15 15|50 25 25|30 35 35}"

bin_for() { case "$1" in fine) echo "$FINE_BIN";; coarse) echo "$COARSE_BIN";; *) echo "";; esac; }

mkdir -p "$RUNDIR"
bench_header > "$OUT"

echo "Mix sweep @ ${MIX_THREADS} threads: impls=[$IMPLS] models=[$MODELS] repeats=$REPEATS"
IFS='|' read -ra MIXARR <<< "$MIXES"
for model in $MODELS; do
  export DELETE_MODEL="$model"
  for mix in "${MIXARR[@]}"; do
    read -r sw iw dw <<< "$mix"
    echo "== model=$model mix=$sw/$iw/$dw =="
    for r in $(seq 1 "$REPEATS"); do
      s=$((SEED + r))
      for impl in $IMPLS; do
        bin=$(bin_for "$impl")
        if [ -x "$bin" ]; then bench_run "$impl" "$bin" "$MIX_THREADS" "$sw" "$iw" "$dw" "$r" "$s" || true
        else echo "  (missing $impl binary: $bin)"; fi
      done
    done
  done
done
echo
echo "Wrote $OUT"
