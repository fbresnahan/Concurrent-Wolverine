#!/usr/bin/env bash
#
# latency_vs_mix.sh — SEARCH LATENCY vs WRITE FRACTION, per delete model.
#
# Sweeps the write fraction of the interleaved workload and records search-side
# latency percentiles (p50/p95/p99) and throughput for each delete model.
# Produces the data for a single graph:
#
#     x = write fraction (% of ops that are insert+delete, split evenly)
#     y = search latency (default p99)
#     one line per model: two-hop (3), approx-two-hop (4), naive tombstone (6)
#
# The story: naive tombstoning never repairs, so its DELETES are nearly free, but
# dead nodes pile up in the graph and searches must wade through them — its search
# latency climbs. The repair models pay on the write side (longer lock holds) but
# keep the graph clean. Reader/writer lock contention pushes ALL of them up as the
# write fraction rises; the gap between them is the point.
#
# Build first:  make interleaved
# Run:          ./latency_vs_mix.sh
# Plot:         python3 plot_latency_vs_mix.py latency_vs_mix.csv -o latency_vs_mix.png
#
# Knobs (env): MODELS [3 4 6], WRITE_FRACS [0 20 40 60 80], THREADS [40],
#   INITIAL_ACTIVE [40000], TOTAL_OPS [400000], REPEATS [3],
#   M/EFC/EFS[200]/K/NEW_LINK, SEED, BIN, OUT.

set -euo pipefail
BIN="${BIN:-./hnsw_Wolverine_interleaved_test}"
DATA="${DATA:-./datasets/sift_learn.fbin}"
QUERIES="${QUERIES:-./datasets/sift_query.fbin}"
MODELS="${MODELS:-3 4 6}"
WRITE_FRACS="${WRITE_FRACS:-0 20 40 60 80}"
THREADS="${THREADS:-40}"
INITIAL_ACTIVE="${INITIAL_ACTIVE:-40000}"
TOTAL_OPS="${TOTAL_OPS:-400000}"
REPEATS="${REPEATS:-3}"
M="${M:-16}"; EFC="${EFC:-200}"; EFS="${EFS:-200}"; K="${K:-10}"; NEW_LINK="${NEW_LINK:-16}"
SEED="${SEED:-100}"
OUT="${OUT:-latency_vs_mix.csv}"
RUNDIR="${RUNDIR:-./latency_vs_mix_runs}"

model_name() { case "$1" in 2) echo search;; 3) echo two-hop;; 4) echo approx;; 6) echo naive-tombstone;; 7) echo naive-reconstruction;; *) echo "model$1";; esac; }

# pull throughput and a latency field (2=p50,3=p95,4=p99) out of an "ops" line
_tp()  { sed -nE 's/.*throughput: ([0-9.]+).*/\1/p' <<<"$1"; }
_lat() { local q; q=$(sed -nE 's#.*ms: ([0-9./]+).*#\1#p' <<<"$1"); cut -d/ -f"$2" <<<"$q"; }

mkdir -p "$RUNDIR"
[ -x "$BIN" ] || { echo "Missing $BIN — run 'make interleaved'"; exit 1; }

echo "model,model_name,write_frac,search_w,insert_w,delete_w,run,search_throughput,search_p50_ms,search_p95_ms,search_p99_ms,delete_p99_ms,recall" > "$OUT"
echo "Search latency vs write fraction: models=[$MODELS] fracs=[$WRITE_FRACS] threads=$THREADS ef=$EFS ops=$TOTAL_OPS repeats=$REPEATS"

for m in $MODELS; do
  for wf in $WRITE_FRACS; do
    dw=$(( wf / 2 )); iw=$(( wf - dw )); sw=$(( 100 - wf ))
    for r in $(seq 1 "$REPEATS"); do
      tag="m${m}_wf${wf}_r${r}"
      log="$RUNDIR/${tag}.log"
      seed=$(( SEED + r ))
      if ! ${NUMACTL:-} "$BIN" --data "$DATA" --queries "$QUERIES" --results "$RUNDIR/${tag}.csv" \
          --initial-active "$INITIAL_ACTIVE" --total-ops "$TOTAL_OPS" \
          --validation-interval 0 --validation-queries 200 \
          --k "$K" --M "$M" --ef-construction "$EFC" --ef-search "$EFS" \
          --worker-threads "$THREADS" \
          --search-weight "$sw" --insert-weight "$iw" --delete-weight "$dw" \
          --delete-model "$m" --new-link-size "$NEW_LINK" --check-reverse-links 0 --seed "$seed" \
          > "$log" 2>&1; then
        echo "  ! run failed: $tag (see $log)"; continue
      fi
      sline=$(grep -m1 "Search ops:" "$log" || true)
      dline=$(grep -m1 "Delete ops:" "$log" || true)
      final=$(grep -m1 "Final validation" "$log" || true)
      s_tp=$(_tp "$sline"); s50=$(_lat "$sline" 2); s95=$(_lat "$sline" 3); s99=$(_lat "$sline" 4)
      d99=$(_lat "$dline" 4)
      recall=$(sed -nE 's/.*recall: ([0-9.]+).*/\1/p' <<<"$final")
      echo "${m},$(model_name "$m"),${wf},${sw},${iw},${dw},${r},${s_tp:-0},${s50:-0},${s95:-0},${s99:-0},${d99:-0},${recall:-0}" >> "$OUT"
      printf "  %-16s wf=%-3s r%s  search_tp=%-9s p50=%-7s p95=%-7s p99=%-7s del_p99=%s\n" \
        "$(model_name "$m")" "$wf" "$r" "${s_tp:-0}" "${s50:-0}" "${s95:-0}" "${s99:-0}" "${d99:-0}"
    done
  done
done

echo "Wrote $OUT  ->  python3 plot_latency_vs_mix.py $OUT -o ${OUT%.csv}.png"
