#!/usr/bin/env bash
#
# sweep.sh — thread-count scaling sweep, with repeats (error bars) and latency.
# Sweeps over locking implementations (IMPLS) AND delete models (MODELS).
#
# Build first:  make compare        # fine + coarse binaries
# Run:          ./sweep.sh          # then: python3 plot_sweep.py sweep_results.csv
#
# Two common configurations:
#   Locking comparison (fine vs coarse):   IMPLS="fine coarse" MODELS=3 ./sweep.sh
#   Repair-method comparison (fine only):  IMPLS=fine MODELS="2 3 4" ./sweep.sh
#
# Environment knobs (defaults in brackets):
#   IMPLS    locking impls to run        [fine coarse]   (fine|coarse)
#   MODELS   delete models to run        [3]             (2=search 3=two-hop 4=approx)
#   THREADS  thread counts               [1 2 4 8 16 32]
#   REPEATS  runs per data point         [3]
#   SEARCH_W/INSERT_W/DELETE_W  op mix    [80/10/10]
#   FINE_BIN / COARSE_BIN / OUT + bench_lib knobs (INITIAL_ACTIVE, TOTAL_OPS, ...)

set -euo pipefail
source "$(dirname "$0")/bench_lib.sh"

FINE_BIN="${FINE_BIN:-./hnsw_Wolverine_interleaved_test}"
COARSE_BIN="${COARSE_BIN:-./hnsw_Wolverine_interleaved_baseline}"
IMPLS="${IMPLS:-fine coarse}"
MODELS="${MODELS:-3}"
THREADS="${THREADS:-1 2 4 8 16 32}"
REPEATS="${REPEATS:-3}"
SEARCH_W="${SEARCH_W:-80}"; INSERT_W="${INSERT_W:-10}"; DELETE_W="${DELETE_W:-10}"
SEED="${SEED:-100}"
OUT="${OUT:-sweep_results.csv}"

bin_for() { case "$1" in fine) echo "$FINE_BIN";; coarse) echo "$COARSE_BIN";; *) echo "";; esac; }

mkdir -p "$RUNDIR"
bench_header > "$OUT"

echo "Thread sweep: impls=[$IMPLS] models=[$MODELS] threads=[$THREADS] repeats=$REPEATS mix=$SEARCH_W/$INSERT_W/$DELETE_W"
for model in $MODELS; do
  export DELETE_MODEL="$model"
  for t in $THREADS; do
    echo "== model=$model threads=$t =="
    for r in $(seq 1 "$REPEATS"); do
      s=$((SEED + r))
      for impl in $IMPLS; do
        bin=$(bin_for "$impl")
        if [ -x "$bin" ]; then bench_run "$impl" "$bin" "$t" "$SEARCH_W" "$INSERT_W" "$DELETE_W" "$r" "$s" || true
        else echo "  (missing $impl binary: $bin — run 'make compare')"; fi
      done
    done
  done
done
echo
echo "Wrote $OUT"
