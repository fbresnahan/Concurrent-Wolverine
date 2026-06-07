# Experiment plan — Concurrent Wolverine

Test suite for the presentation. Two contributions to demonstrate:

1. **Fine-grained concurrent locking** for HNSW deletes (vs a coarse global lock) — scales; the global lock doesn't.
2. **Online concurrent repair for all three Wolverine repair models** — two-hop (Pro, 3), approx-two-hop (ProMax, 4), and now **search-path repair (base Wolverine, 2)** — with a quality/throughput trade-off.

…all **correct under concurrency** (ThreadSanitizer + live invariant checks).

## Setup to report
- Machine: CNSI Pod `batch` node — 2× Intel Xeon Gold 6148 (40 cores), 192 GB, AVX2 build (`-Ofast -mavx2`). AVX-512 optional (ratios unchanged).
- Dataset: SIFT learn (100k × 128) + 10k queries, ground truth gt100. recall@10.
- Methodology: `REPEATS=3` per point, fixed seeds, median + min/max error bars. CSVs + PNGs are produced by the scripts below.

## Experiment matrix

| ID | Question | Script | Lines compared | Key metrics |
|----|----------|--------|----------------|-------------|
| **E0** | Is it correct under concurrency? | TSan builds + `--check-reverse-links 1` | all 3 models | races, deadlocks, reverse_links_ok, invalid_labels, recall |
| **E1** | Does fine-grained locking scale (vs global lock)? | `sweep.sh` IMPLS="fine coarse" MODELS=3 | fine vs coarse | search throughput, speedup vs threads |
| **E2** | How do the repair models compare online (cost)? | `sweep.sh` IMPLS=fine MODELS="2 3 4" | search/two-hop/approx | delete throughput, delete p99, search throughput vs threads |
| **E3** | How does the mix (read→write) change things? | `mix_sweep.sh` (both modes) | fine vs coarse; and models | throughput vs write fraction |
| **E4** | Does search-repair preserve recall *online*? | `online_recall.sh` MODELS="2 3 4" | search/two-hop/approx | recall vs concurrent churn |
| **E5** | Recall vs throughput trade-off | derived from E2 + E4 | the 3 models | scatter (delete tput, recall) |

## E0 — Correctness (table slide)
Build the interleaved test with `-fsanitize=thread -O1` and run each model with `--check-reverse-links 1`. Report: 0 races, 0 deadlocks, `reverse_links_ok=1`, `invalid_labels=0`, recall sane — for models 2, 3, 4. (Already validated on the dev machine; re-run on the cluster for the record.)
```
clang++ -std=c++17 -g -O1 -fsanitize=thread -pthread -I. hnsw_Wolverine_interleaved_test.cpp -o itl_tsan
for M in 2 3 4; do ./itl_tsan --delete-model $M --check-reverse-links 1 \
  --initial-active 3000 --total-ops 8000 --worker-threads 8 \
  --search-weight 60 --insert-weight 20 --delete-weight 20 --validation-interval 4000 ; done
```

## E1 — Locking scalability (headline)
`sbatch run_comparison.job` (fine vs coarse, model 3). Produces `sweep_results_<jid>.{csv,png}` and `mix_results_<jid>.{csv,png}`.
- **Slide:** throughput + speedup vs threads — fine rises, coarse flat.

## E2 — Repair-method scaling (the SEARCH_DELETE result)
Part of `run_repair.job`, or:
```
make compare
IMPLS=fine MODELS="2 3 4" THREADS="1 2 4 8 16 32 40" REPEATS=3 \
  INITIAL_ACTIVE=40000 TOTAL_OPS=200000 OUT=repair_sweep.csv ./sweep.sh
python3 plot_sweep.py repair_sweep.csv
```
- **Slide:** delete throughput + delete p99 — approx > two-hop > search on throughput; search has the highest tail latency (full search per neighbor). All scale with threads.

## E3 — Workload-mix sensitivity
Locking cut (run_comparison.job already includes it): fine vs coarse, write 0→70%.
Repair cut:
```
IMPLS=fine MODELS="2 3 4" MIX_THREADS=40 REPEATS=3 OUT=repair_mix.csv ./mix_sweep.sh
python3 plot_mix.py repair_mix.csv
```
- **Slide:** throughput vs write fraction — fine/coarse gap shrinks as writes rise (global locks bottleneck); search-repair's cost grows fastest with writes.

## E4 — Online recall (quality, the key comparison)
```
make interleaved
MODELS="2 3 4" THREADS=40 TOTAL_OPS=600000 VAL_INTERVAL=30000 ./online_recall.sh
python3 plot_online_recall.py online_recall_results
```
- **Slide:** recall vs concurrent churn — does search-repair (2) hold recall above two-hop (3)/approx (4) under live deletion? This is measured **online**, not batch.

## E5 — Recall vs throughput trade-off (money slide)
Combine E2's delete throughput with E4's steady-state recall into one scatter: search = high recall / low throughput; approx = high throughput / lower recall; two-hop in between. (Plot by hand from the two CSVs, or ask for a helper.)

## Sizing & hygiene
- Capacity: deletes don't recycle ids, so per run keep `(insert_weight/100)*total_ops < (100000 - initial_active)`; `skipped` inserts late in a run are benign (pool exhaustion), `failures` should be 0.
- `model 2` (search) deletes are ~10–50× slower than two-hop, so use smaller `TOTAL_OPS` for E2/E4 to keep runtime bounded.
- Report medians of `REPEATS=3`; error bars are min/max.
- One-node exclusive allocation (`--cpus-per-task=40`) already isolates the cores.

## Jobs
- `run_comparison.job` — E1 + E3 (locking: fine vs coarse).
- `run_repair.job` — E2 + E3 (repair models) + E4 (online recall).
- `run_recall.job` — batch repair-quality (kept only as a cheap upper-bound screen; E4 supersedes it for the online story).
