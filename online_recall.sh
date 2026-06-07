#!/usr/bin/env bash
#
# online_recall.sh — ONLINE repair-quality experiment.
#
# Runs the interleaved benchmark per delete model and records the recall
# trajectory measured by its built-in validator (recall vs brute force over the
# LIVE active set, at checkpoints), while searches/inserts/deletes run
# concurrently. This is the apples-to-apples online quality comparison that the
# batch recall test could not provide — now possible because SEARCH_DELETE (2)
# runs online too.
#
# Build first:  make interleaved
# Run:          ./online_recall.sh   # then: python3 plot_online_recall.py online_recall_results
#
# Knobs: MODELS [2 3 4], THREADS [16], TOTAL_OPS [400000], VAL_INTERVAL [20000],
#        VAL_QUERIES [200], mix SEARCH_W/INSERT_W/DELETE_W [60/20/20],
#        INITIAL_ACTIVE [20000], BIN, OUTDIR.

set -euo pipefail
BIN="${BIN:-./hnsw_Wolverine_interleaved_test}"
DATA="${DATA:-./datasets/sift_learn.fbin}"
QUERIES="${QUERIES:-./datasets/sift_query.fbin}"
MODELS="${MODELS:-2 3 4}"
THREADS="${THREADS:-16}"
INITIAL_ACTIVE="${INITIAL_ACTIVE:-20000}"
TOTAL_OPS="${TOTAL_OPS:-400000}"
VAL_INTERVAL="${VAL_INTERVAL:-20000}"
VAL_QUERIES="${VAL_QUERIES:-200}"
SEARCH_W="${SEARCH_W:-60}"; INSERT_W="${INSERT_W:-20}"; DELETE_W="${DELETE_W:-20}"
M="${M:-16}"; EFC="${EFC:-200}"; EFS="${EFS:-200}"; K="${K:-10}"; NEW_LINK="${NEW_LINK:-16}"
SEED="${SEED:-100}"
OUTDIR="${OUTDIR:-online_recall_results}"

model_name() { case "$1" in 2) echo search;; 3) echo two-hop;; 4) echo approx;; *) echo "model$1";; esac; }
mkdir -p "$OUTDIR"
if [ ! -x "$BIN" ]; then echo "Missing $BIN — run 'make interleaved'"; exit 1; fi

echo "Online recall: models=[$MODELS] threads=$THREADS total_ops=$TOTAL_OPS mix=$SEARCH_W/$INSERT_W/$DELETE_W"
for m in $MODELS; do
  out="$OUTDIR/online_recall_model${m}.csv"
  echo "== model $m ($(model_name "$m")) -> $out =="
  ${NUMACTL:-} "$BIN" --data "$DATA" --queries "$QUERIES" --results "$out" \
    --initial-active "$INITIAL_ACTIVE" --total-ops "$TOTAL_OPS" \
    --validation-interval "$VAL_INTERVAL" --validation-queries "$VAL_QUERIES" \
    --k "$K" --M "$M" --ef-construction "$EFC" --ef-search "$EFS" \
    --worker-threads "$THREADS" \
    --search-weight "$SEARCH_W" --insert-weight "$INSERT_W" --delete-weight "$DELETE_W" \
    --delete-model "$m" --new-link-size "$NEW_LINK" --check-reverse-links 1 --seed "$SEED" \
    > "$OUTDIR/model${m}.log" 2>&1 \
    || { echo "  ! run failed (see $OUTDIR/model${m}.log)"; continue; }
  grep -E "validation@|Final validation" "$OUTDIR/model${m}.log" | tail -2
done
echo "Per-model CSVs (validation rows hold the recall trajectory) in $OUTDIR/"
