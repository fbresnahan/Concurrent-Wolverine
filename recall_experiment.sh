#!/usr/bin/env bash
#
# recall_experiment.sh — repair-quality experiment.
#
# Runs repeated delete+re-add churn rounds under each delete model and records
# recall per round, to show how well each model's repair preserves search
# accuracy as the graph churns. Quality only (orthogonal to locking throughput).
#
# Build first:  make recall          (produces ./hnsw_Wolverine_recall)
# Run:          ./recall_experiment.sh
# Plot:         python3 plot_recall.py recall_results
#
# Every model is run through the id-recycling batch path (the binary is built
# with -DRECALL_QUALITY), so repeated rounds don't exhaust the index and all
# models start from the SAME pristine saved graph -> a fair comparison.
#
# Environment knobs (defaults in brackets):
#   MODELS   delete models to compare   [1 2 3 4]
#            0=violent 1=pintopout(DwFC) 2=Wolverine(search)
#            3=WolverinePro(two-hop) 4=WolverineProMax(approx two-hop)
#   ROUNDS   churn rounds               [50]
#   DRATE    fraction deleted per round [0.01]
#   M/EF/CAND/THREADS                    [16/200/16/16]
#   BIN, DATA, QUERIES, GT, OUTDIR, INDEX

set -euo pipefail

BIN="${BIN:-./hnsw_Wolverine_recall}"
DATA="${DATA:-./datasets/sift_learn.fbin}"
QUERIES="${QUERIES:-./datasets/sift_query.fbin}"
GT="${GT:-./datasets/sift_query_learn_gt100}"
M="${M:-16}"; EF="${EF:-200}"; CAND="${CAND:-16}"; THREADS="${THREADS:-16}"
ROUNDS="${ROUNDS:-50}"; DRATE="${DRATE:-0.01}"
MODELS="${MODELS:-1 2 3 4}"
OUTDIR="${OUTDIR:-recall_results}"
INDEX="${INDEX:-./index/recall_M${M}_ef${EF}}"

mkdir -p "$OUTDIR" ./index
model_name() {
  case "$1" in
    0) echo violent;;  1) echo pintopout;;  2) echo wolverine_search;;
    3) echo wolverine_pro_twohop;;  4) echo wolverine_promax_approx;;  5) echo refactor;;
    *) echo "model$1";;
  esac
}

if [ ! -x "$BIN" ]; then echo "Missing $BIN — run 'make recall'"; exit 1; fi
echo "Recall experiment: models=[$MODELS] rounds=$ROUNDS drate=$DRATE M=$M ef=$EF"

for m in $MODELS; do
  out="$OUTDIR/recall_model${m}.csv"
  echo "== model $m ($(model_name "$m")) -> $out =="
  # args: M ef candLimit threads rounds drate model data query gt result index
  "$BIN" "$M" "$EF" "$CAND" "$THREADS" "$ROUNDS" "$DRATE" "$m" \
    "$DATA" "$QUERIES" "$GT" "$out" "$INDEX" \
    > "$OUTDIR/model${m}.log" 2>&1 \
    || { echo "  ! run failed (see $OUTDIR/model${m}.log)"; continue; }
  # quick peek: first and last recall in the CSV (col 1, skipping header)
  first=$(awk -F, 'NR==2{print $1}' "$out"); last=$(awk -F, 'END{print $1}' "$out")
  echo "  recall: start=$first  end=$last"
done
echo "Per-model CSVs in $OUTDIR/  (columns: recall,search_OPS,delete_OPS,insert_OPS; one row per round)"
