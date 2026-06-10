#!/usr/bin/env bash
#
# workload_table.sh — QPS + median search latency for three named workloads.
#
# Runs the Wolverine fine-grained concurrent index (two-hop repair by default)
# under three realistic read/write mixes and produces a small table:
#
#   RAG / blob-storage search (read-heavy)   80/10/10
#   E-commerce semantic search               70/15/15
#   Real-time anomaly / fraud detection      50/25/25   (write-heavy)
#
# For each workload it reports:
#   search_qps      — queries (searches) per second           = "QPS"
#   search_p50_ms   — median search latency (ms)               = "median latency"
#   overall_ops_s   — total ops/sec (search+insert+delete)     (bonus)
#   recall          — sanity check
# averaged across REPEATS runs.
#
# Build first:  make interleaved   (or the local g++ one-liner)
# Run:          ./workload_table.sh
#
# Knobs (env): THREADS [40], REPEATS [5], DELETE_MODEL [3=two-hop],
#   INITIAL_ACTIVE [40000], TOTAL_OPS [200000], EFS [200], SEED, WORKLOADS,
#   BIN, OUT, TABLE, RUNDIR + bench_lib knobs (M/EFC/K/NEW_LINK).
#
# Capacity: deleted labels are NOT recycled to the reserve pool, so net inserts
# must fit in (max_elements - INITIAL_ACTIVE). Heaviest mix (50/25/25) inserts
# ~25% of TOTAL_OPS; defaults give 50k inserts <= 60k reserve.

# Write-safe capacity defaults (must be set before sourcing bench_lib).
: "${INITIAL_ACTIVE:=40000}"
: "${TOTAL_OPS:=200000}"

set -euo pipefail
source "$(dirname "$0")/bench_lib.sh"

BIN="${BIN:-./hnsw_Wolverine_interleaved_test}"
THREADS="${THREADS:-40}"
REPEATS="${REPEATS:-5}"
SEED="${SEED:-100}"
export DELETE_MODEL="${DELETE_MODEL:-3}"   # 3=two-hop (Wolverine flagship)
export EFS="${EFS:-200}"
OUT="${OUT:-workload_table.csv}"           # per-run rows (bench_lib format)
TABLE="${TABLE:-workload_summary.csv}"     # aggregated, one row per workload

# name:search:insert:delete   (label uses no spaces so it's a clean CSV key)
WORKLOADS="${WORKLOADS:-rag:80:10:10 ecommerce:70:15:15 fraud:50:25:25}"

mkdir -p "$RUNDIR"
[ -x "$BIN" ] || { echo "Missing $BIN — run 'make interleaved'"; exit 1; }
bench_header > "$OUT"

echo "Workload table @ ${THREADS} threads  model=$DELETE_MODEL  ef=$EFS  repeats=$REPEATS  ops=$TOTAL_OPS"
for w in $WORKLOADS; do
  IFS=: read -r name sw iw dw <<< "$w"
  echo "== $name  (${sw}/${iw}/${dw}) =="
  for r in $(seq 1 "$REPEATS"); do
    s=$((SEED + r))
    bench_run "$name" "$BIN" "$THREADS" "$sw" "$iw" "$dw" "$r" "$s" || true
  done
done

# ---- aggregate (average across repeats), preserving WORKLOADS order ----------
# bench_lib row cols: 1 impl(name) 2 model 3 threads 4 mix 5 run
#   6 search_tp 7 insert_tp 8 delete_tp 9 search_p50 10 s95 11 s99 ... 14 recall
echo "workload,mix,search_qps,overall_ops_s,search_p50_ms,recall" > "$TABLE"
for w in $WORKLOADS; do
  IFS=: read -r name sw iw dw <<< "$w"
  awk -F, -v key="$name" -v mix="${sw}/${iw}/${dw}" '
    $1==key { sq+=$6; ovr+=($6+$7+$8); p50+=$9; rec+=$14; n++ }
    END {
      if (n==0) { printf "%s,%s,0,0,0,0\n", key, mix; }
      else printf "%s,%s,%.0f,%.0f,%.3f,%.4f\n", key, mix, sq/n, ovr/n, p50/n, rec/n
    }' "$OUT" >> "$TABLE"
done

echo
echo "================ workload table ===================="
if command -v column >/dev/null 2>&1; then column -t -s, "$TABLE"; else cat "$TABLE"; fi
echo "==================================================="
echo "Per-run CSV: $OUT"
echo "Table CSV:   $TABLE"
