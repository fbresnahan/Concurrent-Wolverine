#!/usr/bin/env bash
# bench_lib.sh — shared run/parse helpers for sweep.sh and mix_sweep.sh.
#
# Source this, then call bench_header (once, to start the CSV) and bench_run
# (per measurement). Workload knobs come from the environment; defaults below
# only apply if the caller hasn't already set them.

: "${DATA:=./datasets/sift_learn.fbin}"
: "${QUERIES:=./datasets/sift_query.fbin}"
: "${INITIAL_ACTIVE:=60000}"
: "${TOTAL_OPS:=200000}"
: "${DELETE_MODEL:=3}"        # 3=two-hop, 4=approx two-hop
: "${M:=16}"
: "${EFC:=200}"
: "${EFS:=200}"
: "${K:=10}"
: "${NEW_LINK:=16}"
: "${RUNDIR:=./bench_runs}"

bench_header() {
  echo "impl,model,threads,mix,run,search_throughput,insert_throughput,delete_throughput,search_p50_ms,search_p95_ms,search_p99_ms,insert_p99_ms,delete_p99_ms,recall,invalid_labels"
}

_tp()  { sed -nE 's/.*throughput: ([0-9.]+).*/\1/p' <<<"$1"; }            # throughput from a line
_lat() { local q; q=$(sed -nE 's#.*ms: ([0-9./]+).*#\1#p' <<<"$1"); cut -d/ -f"$2" <<<"$q"; }  # field 1..4 = mean/p50/p95/p99

# bench_run impl bin threads sw iw dw run seed   (appends one CSV row to $OUT)
bench_run() {
  local impl="$1" bin="$2" threads="$3" sw="$4" iw="$5" dw="$6" run="$7" seed="$8"
  local tag="${impl}_t${threads}_m${sw}-${iw}-${dw}_r${run}"
  local log="$RUNDIR/${tag}.log"

  if ! "$bin" \
      --data "$DATA" --queries "$QUERIES" --results "$RUNDIR/${tag}.csv" \
      --initial-active "$INITIAL_ACTIVE" --total-ops "$TOTAL_OPS" \
      --validation-interval 0 --check-reverse-links 0 \
      --k "$K" --M "$M" --ef-construction "$EFC" --ef-search "$EFS" \
      --worker-threads "$threads" \
      --search-weight "$sw" --insert-weight "$iw" --delete-weight "$dw" \
      --delete-model "$DELETE_MODEL" --new-link-size "$NEW_LINK" --seed "$seed" \
      > "$log" 2>&1; then
    echo "  ! run failed: $tag (see $log)"; return 1
  fi

  local sline iline dline final
  sline=$(grep -m1 "Search ops:" "$log" || true)
  iline=$(grep -m1 "Insert ops:" "$log" || true)
  dline=$(grep -m1 "Delete ops:" "$log" || true)
  final=$(grep -m1 "Final validation" "$log" || true)

  local s_tp i_tp d_tp s50 s95 s99 i99 d99 recall inval
  s_tp=$(_tp "$sline"); i_tp=$(_tp "$iline"); d_tp=$(_tp "$dline")
  s50=$(_lat "$sline" 2); s95=$(_lat "$sline" 3); s99=$(_lat "$sline" 4)
  i99=$(_lat "$iline" 4); d99=$(_lat "$dline" 4)
  recall=$(sed -nE 's/.*recall: ([0-9.]+).*/\1/p' <<<"$final")
  inval=$(sed -nE 's/.*invalid_labels: ([0-9]+).*/\1/p' <<<"$final")

  echo "${impl},${DELETE_MODEL},${threads},${sw}/${iw}/${dw},${run},${s_tp:-0},${i_tp:-0},${d_tp:-0},${s50:-0},${s95:-0},${s99:-0},${i99:-0},${d99:-0},${recall:-0},${inval:-0}" >> "$OUT"
  printf "  %-7s m%s t=%-3s mix=%-9s r%s  search=%-9s ops/s  del_tp=%-8s del_p99=%-7s recall=%s\n" \
    "$impl" "${DELETE_MODEL}" "$threads" "${sw}/${iw}/${dw}" "$run" "${s_tp:-0}" "${d_tp:-0}" "${d99:-0}" "${recall:-0}"
}
