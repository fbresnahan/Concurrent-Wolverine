#include "hnsw_Wolverine/hnswlib.h"
#include "hnsw_Wolverine/hnsw_Wolverine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace std;

namespace {

struct Config {
    string data_file_path = "./datasets/sift_learn.fbin";
    string query_file_path = "./datasets/sift_query.fbin";
    string result_file_path = "./interleaved_results.csv";
    size_t initial_active_count = 200000;
    size_t total_ops = 20000;
    size_t validation_interval = 5000;
    size_t validation_queries = 32;
    size_t k = 10;
    int M = 16;
    int ef_construction = 200;
    int ef_search = 200;
    int worker_threads = 16;
    int build_threads = 1;
    int search_weight = 80;
    int insert_weight = 10;
    int delete_weight = 10;
    int delete_model = TWOHOP_DELETE;
    int new_link_size = 16;
    bool check_reverse_links = false;
    unsigned int seed = 100;
    // Optional: write a per-bucket search-latency-over-time CSV (time vs latency).
    // Used to show tombstone search latency drifting up as deletes accumulate,
    // while two-hop / approx stay flat. Empty path disables it.
    string timeseries_file;
    size_t timeseries_buckets = 40;
};

struct ValidationRecord {
    string phase;
    size_t completed_ops = 0;
    size_t active_count = 0;
    double elapsed_sec = 0.0;
    double recall = 0.0;
    size_t invalid_labels = 0;
    bool reverse_links_ok = true;
    string reverse_links_error;
};

struct WorkerStats {
    vector<double> search_latencies_ms;
    vector<double> insert_latencies_ms;
    vector<double> delete_latencies_ms;
    // Completion time (seconds since benchmark_start) paired index-for-index with
    // the latency vectors above, for building latency-over-time series.
    vector<double> search_time_s;
    vector<double> delete_time_s;
    size_t search_ops = 0;
    size_t insert_ops = 0;
    size_t delete_ops = 0;
    size_t search_failures = 0;
    size_t insert_failures = 0;
    size_t delete_failures = 0;
    size_t skipped_inserts = 0;
    size_t skipped_deletes = 0;
};

struct SummaryStats {
    size_t count = 0;
    double throughput_ops = 0.0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
};

struct LabelPool {
    mutable mutex mu;
    vector<size_t> active_labels;
    vector<size_t> reserve_labels;
    size_t deleted_labels = 0;

    bool reserveDelete(mt19937 &rng, size_t &label) {
        lock_guard<mutex> lock(mu);
        if (active_labels.empty()) {
            return false;
        }
        uniform_int_distribution<size_t> pick(0, active_labels.size() - 1);
        size_t idx = pick(rng);
        label = active_labels[idx];
        active_labels[idx] = active_labels.back();
        active_labels.pop_back();
        return true;
    }

    void revertDelete(size_t label) {
        lock_guard<mutex> lock(mu);
        active_labels.emplace_back(label);
    }

    void commitDelete() {
        lock_guard<mutex> lock(mu);
        deleted_labels++;
    }

    bool reserveInsert(mt19937 &rng, size_t &label) {
        lock_guard<mutex> lock(mu);
        if (reserve_labels.empty()) {
            return false;
        }
        uniform_int_distribution<size_t> pick(0, reserve_labels.size() - 1);
        size_t idx = pick(rng);
        label = reserve_labels[idx];
        reserve_labels[idx] = reserve_labels.back();
        reserve_labels.pop_back();
        return true;
    }

    void revertInsert(size_t label) {
        lock_guard<mutex> lock(mu);
        reserve_labels.emplace_back(label);
    }

    void commitInsert(size_t label) {
        lock_guard<mutex> lock(mu);
        active_labels.emplace_back(label);
    }

    vector<size_t> snapshotActive() const {
        lock_guard<mutex> lock(mu);
        return active_labels;
    }

    size_t activeCount() const {
        lock_guard<mutex> lock(mu);
        return active_labels.size();
    }

    size_t reserveCount() const {
        lock_guard<mutex> lock(mu);
        return reserve_labels.size();
    }
};

struct ParsedArgs {
    bool show_help = false;
    Config config;
};

double elapsedMilliseconds(const chrono::steady_clock::time_point &start) {
    chrono::duration<double, std::milli> elapsed = chrono::steady_clock::now() - start;
    return elapsed.count();
}

double percentile(vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

SummaryStats summarizeLatencies(const vector<double> &latencies, double elapsed_sec) {
    SummaryStats summary;
    summary.count = latencies.size();
    summary.throughput_ops = elapsed_sec > 0.0 ? static_cast<double>(latencies.size()) / elapsed_sec : 0.0;
    if (!latencies.empty()) {
        summary.mean_ms = accumulate(latencies.begin(), latencies.end(), 0.0) /
            static_cast<double>(latencies.size());
    }
    summary.p50_ms = percentile(latencies, 0.50);
    summary.p95_ms = percentile(latencies, 0.95);
    summary.p99_ms = percentile(latencies, 0.99);
    return summary;
}

vector<hnswlib::labeltype> extractLabels(
    priority_queue<pair<float, hnswlib::labeltype>> results) {
    vector<hnswlib::labeltype> labels;
    labels.reserve(results.size());
    while (!results.empty()) {
        labels.emplace_back(results.top().second);
        results.pop();
    }
    return labels;
}

vector<hnswlib::labeltype> bruteForceTopK(
    const float *query,
    const float *data,
    int dim,
    const vector<size_t> &active_labels,
    size_t k,
    hnswlib::L2Space &space) {
    vector<hnswlib::labeltype> empty;
    if (active_labels.empty() || k == 0) {
        return empty;
    }

    auto dist_func = space.get_dist_func();
    void *dist_param = space.get_dist_func_param();
    priority_queue<pair<float, hnswlib::labeltype>> best;
    size_t effective_k = min(k, active_labels.size());
    for (size_t label : active_labels) {
        float distance = dist_func(query, data + label * dim, dist_param);
        if (best.size() < effective_k) {
            best.emplace(distance, static_cast<hnswlib::labeltype>(label));
            continue;
        }
        if (distance < best.top().first) {
            best.pop();
            best.emplace(distance, static_cast<hnswlib::labeltype>(label));
        }
    }
    return extractLabels(std::move(best));
}

size_t overlapCount(const vector<hnswlib::labeltype> &lhs, const vector<hnswlib::labeltype> &rhs) {
    unordered_set<hnswlib::labeltype> rhs_set(rhs.begin(), rhs.end());
    size_t overlap = 0;
    for (hnswlib::labeltype label : lhs) {
        if (rhs_set.find(label) != rhs_set.end()) {
            overlap++;
        }
    }
    return overlap;
}

template <typename dist_t>
optional<string> validateReverseLinks(hnswlib::HierarchicalNSW<dist_t> *alg_hnsw) {
    for (hnswlib::tableint source = 0; source < alg_hnsw->cur_element_count; source++) {
        if (alg_hnsw->isMarkedDeletedWithSharedLock(source)) {
            continue;
        }
        for (int level = 0; level <= alg_hnsw->element_levels_[source]; level++) {
            vector<hnswlib::tableint> forward = alg_hnsw->getConnectionsWithSharedLock(source, level);
            unordered_set<hnswlib::tableint> unique_forward;
            for (hnswlib::tableint target : forward) {
                if (!unique_forward.emplace(target).second) {
                    return "duplicate forward edge";
                }
                if (target >= alg_hnsw->cur_element_count) {
                    return "forward edge points outside index";
                }
                if (level > alg_hnsw->element_levels_[target]) {
                    return "forward edge points to missing level";
                }
                if (alg_hnsw->isMarkedDeletedWithSharedLock(target)) {
                    return "forward edge points to tombstoned node";
                }
                vector<hnswlib::tableint> reverse = alg_hnsw->getReverseConnectionsWithSharedLock(target, level);
                if (find(reverse.begin(), reverse.end(), source) == reverse.end()) {
                    return "missing reverse edge";
                }
            }

            vector<hnswlib::tableint> reverse = alg_hnsw->getReverseConnectionsWithSharedLock(source, level);
            unordered_set<hnswlib::tableint> unique_reverse;
            for (hnswlib::tableint inbound : reverse) {
                if (!unique_reverse.emplace(inbound).second) {
                    return "duplicate reverse edge";
                }
                if (inbound >= alg_hnsw->cur_element_count) {
                    return "reverse edge points outside index";
                }
                if (level > alg_hnsw->element_levels_[inbound]) {
                    return "reverse edge points to missing level";
                }
                vector<hnswlib::tableint> inbound_forward = alg_hnsw->getConnectionsWithSharedLock(inbound, level);
                if (find(inbound_forward.begin(), inbound_forward.end(), source) == inbound_forward.end()) {
                    return "stale reverse edge";
                }
            }
        }
    }
    return nullopt;
}

ValidationRecord runValidation(
    hnswlib::HierarchicalNSW<float> *alg_hnsw,
    hnswlib::L2Space &space,
    const Config &config,
    const LabelPool &label_pool,
    const float *data,
    int dim,
    const float *queries,
    int query_sum,
    int query_dim,
    size_t completed_ops,
    const chrono::steady_clock::time_point &benchmark_start,
    mt19937 &rng,
    const string &phase) {
    ValidationRecord record;
    record.phase = phase;
    record.completed_ops = completed_ops;
    record.elapsed_sec = chrono::duration<double>(chrono::steady_clock::now() - benchmark_start).count();
    vector<size_t> active_snapshot = label_pool.snapshotActive();
    record.active_count = active_snapshot.size();

    if (config.check_reverse_links) {
        optional<string> reverse_error = validateReverseLinks(alg_hnsw);
        record.reverse_links_ok = !reverse_error.has_value();
        if (reverse_error.has_value()) {
            record.reverse_links_error = *reverse_error;
        }
    }

    if (active_snapshot.empty()) {
        record.recall = 1.0;
        return record;
    }

    unordered_set<hnswlib::labeltype> active_set(active_snapshot.begin(), active_snapshot.end());
    size_t sampled_queries = min(config.validation_queries, static_cast<size_t>(query_sum));
    if (sampled_queries == 0) {
        record.recall = 1.0;
        return record;
    }

    uniform_int_distribution<int> query_pick(0, query_sum - 1);
    double overlap_sum = 0.0;
    for (size_t sample = 0; sample < sampled_queries; sample++) {
        int query_idx = query_pick(rng);
        const float *query_ptr = queries + static_cast<size_t>(query_idx) * query_dim;
        vector<hnswlib::labeltype> expected = bruteForceTopK(
            query_ptr,
            data,
            dim,
            active_snapshot,
            config.k,
            space);
        vector<hnswlib::labeltype> actual = extractLabels(alg_hnsw->searchKnn(query_ptr, config.k));
        for (hnswlib::labeltype label : actual) {
            if (active_set.find(label) == active_set.end()) {
                record.invalid_labels++;
            }
        }
        size_t effective_k = min(expected.size(), actual.size());
        if (effective_k == 0) {
            overlap_sum += 1.0;
            continue;
        }
        overlap_sum += static_cast<double>(overlapCount(actual, expected)) / static_cast<double>(expected.size());
    }
    record.recall = overlap_sum / static_cast<double>(sampled_queries);
    return record;
}

int weightedChoice(const Config &config, mt19937 &rng) {
    int total_weight = config.search_weight + config.insert_weight + config.delete_weight;
    if (total_weight <= 0) {
        return 0;
    }
    uniform_int_distribution<int> pick(0, total_weight - 1);
    int choice = pick(rng);
    if (choice < config.search_weight) {
        return 0;
    }
    choice -= config.search_weight;
    if (choice < config.insert_weight) {
        return 1;
    }
    return 2;
}

ParsedArgs parseArgs(int argc, char **argv) {
    ParsedArgs parsed;
    for (int idx = 1; idx < argc; idx++) {
        string key = argv[idx];
        if (key == "--help") {
            parsed.show_help = true;
            return parsed;
        }
        if (idx + 1 >= argc) {
            throw runtime_error("missing value for " + key);
        }
        string value = argv[++idx];
        if (key == "--data") {
            parsed.config.data_file_path = value;
        } else if (key == "--queries") {
            parsed.config.query_file_path = value;
        } else if (key == "--results") {
            parsed.config.result_file_path = value;
        } else if (key == "--initial-active") {
            parsed.config.initial_active_count = stoull(value);
        } else if (key == "--total-ops") {
            parsed.config.total_ops = stoull(value);
        } else if (key == "--validation-interval") {
            parsed.config.validation_interval = stoull(value);
        } else if (key == "--validation-queries") {
            parsed.config.validation_queries = stoull(value);
        } else if (key == "--k") {
            parsed.config.k = stoull(value);
        } else if (key == "--M") {
            parsed.config.M = stoi(value);
        } else if (key == "--ef-construction") {
            parsed.config.ef_construction = stoi(value);
        } else if (key == "--ef-search") {
            parsed.config.ef_search = stoi(value);
        } else if (key == "--worker-threads") {
            parsed.config.worker_threads = stoi(value);
        } else if (key == "--build-threads") {
            parsed.config.build_threads = stoi(value);
        } else if (key == "--search-weight") {
            parsed.config.search_weight = stoi(value);
        } else if (key == "--insert-weight") {
            parsed.config.insert_weight = stoi(value);
        } else if (key == "--delete-weight") {
            parsed.config.delete_weight = stoi(value);
        } else if (key == "--delete-model") {
            parsed.config.delete_model = stoi(value);
        } else if (key == "--new-link-size") {
            parsed.config.new_link_size = stoi(value);
        } else if (key == "--check-reverse-links") {
            parsed.config.check_reverse_links = (stoi(value) != 0);
        } else if (key == "--seed") {
            parsed.config.seed = static_cast<unsigned int>(stoul(value));
        } else if (key == "--timeseries-file") {
            parsed.config.timeseries_file = value;
        } else if (key == "--timeseries-buckets") {
            parsed.config.timeseries_buckets = stoull(value);
        } else {
            throw runtime_error("unknown argument: " + key);
        }
    }
    if (parsed.config.worker_threads <= 0) {
        parsed.config.worker_threads = static_cast<int>(thread::hardware_concurrency());
    }
    if (parsed.config.build_threads <= 0) {
        parsed.config.build_threads = parsed.config.worker_threads;
    }
    return parsed;
}

void printHelp(const char *program_name) {
    cout << "Usage: " << program_name << " [options]\n";
    cout << "  --data PATH\n";
    cout << "  --queries PATH\n";
    cout << "  --results PATH\n";
    cout << "  --initial-active N\n";
    cout << "  --total-ops N\n";
    cout << "  --validation-interval N\n";
    cout << "  --validation-queries N\n";
    cout << "  --k N\n";
    cout << "  --M N\n";
    cout << "  --ef-construction N\n";
    cout << "  --ef-search N\n";
    cout << "  --worker-threads N\n";
    cout << "  --build-threads N\n";
    cout << "  --search-weight N\n";
    cout << "  --insert-weight N\n";
    cout << "  --delete-weight N\n";
    cout << "  --delete-model N (2 SEARCH, 3 TWOHOP, 4 APPROX_TWOHOP, 6 NAIVE_TOMBSTONE, 7 NAIVE_RECONSTRUCTION)\n";
    cout << "  --new-link-size N\n";
    cout << "  --check-reverse-links 0|1\n";
    cout << "  --seed N\n";
    cout << "  --timeseries-file PATH (write per-bucket search-latency-over-time CSV)\n";
    cout << "  --timeseries-buckets N (default 40)\n";
}

string deleteModelName(int m) {
    switch (m) {
        case 0: return "violent";
        case 2: return "search";
        case 3: return "two-hop";
        case 4: return "approx";
        case 6: return "naive-tombstone";
        case 7: return "naive-reconstruction";
        default: return "model" + to_string(m);
    }
}

// Write a per-bucket "search latency over time" CSV: split the run's wall-clock
// duration into equal buckets, and for each bucket report the p50/p95/mean of the
// searches that completed in it, plus cumulative deletes so far. This makes the
// tombstone story visible: as deletes accumulate the tombstone graph grows and
// its search latency drifts up, while two-hop / approx (which remove dead nodes)
// stay flat.
void writeLatencyTimeseries(
    const Config &config,
    const vector<double> &search_time_s,
    const vector<double> &search_latencies_ms,
    const vector<double> &delete_time_s,
    double total_elapsed_sec) {
    if (config.timeseries_file.empty()) return;
    size_t B = config.timeseries_buckets ? config.timeseries_buckets : 1;
    double t_max = total_elapsed_sec > 0.0 ? total_elapsed_sec : 1e-9;
    double width = t_max / static_cast<double>(B);

    vector<vector<double>> buckets(B);
    for (size_t i = 0; i < search_time_s.size() && i < search_latencies_ms.size(); i++) {
        size_t b = static_cast<size_t>(search_time_s[i] / width);
        if (b >= B) b = B - 1;
        buckets[b].emplace_back(search_latencies_ms[i]);
    }
    vector<size_t> deletes_per_bucket(B, 0);
    for (double t : delete_time_s) {
        size_t b = static_cast<size_t>(t / width);
        if (b >= B) b = B - 1;
        deletes_per_bucket[b]++;
    }

    ofstream out(config.timeseries_file);
    out << "model,model_name,bucket,t_mid_s,search_count,search_p50_ms,search_p95_ms,search_mean_ms,cum_deletes\n";
    size_t cum_del = 0;
    for (size_t b = 0; b < B; b++) {
        cum_del += deletes_per_bucket[b];
        double t_mid = (static_cast<double>(b) + 0.5) * width;
        SummaryStats s = summarizeLatencies(buckets[b], 1.0);
        out << config.delete_model << ',' << deleteModelName(config.delete_model) << ','
            << b << ',' << fixed << setprecision(6) << t_mid << ','
            << buckets[b].size() << ',' << s.p50_ms << ',' << s.p95_ms << ',' << s.mean_ms << ','
            << cum_del << '\n';
    }
}

void writeResultsCsv(
    const Config &config,
    const vector<ValidationRecord> &records,
    double total_elapsed_sec,
    const SummaryStats &search_summary,
    const SummaryStats &insert_summary,
    const SummaryStats &delete_summary,
    const WorkerStats &totals,
    const ValidationRecord &final_record) {
    ofstream output(config.result_file_path);
    output << "section,phase,completed_ops,elapsed_sec,active_count,recall,invalid_labels,reverse_links_ok,reverse_links_error,"
           << "search_count,search_failures,search_skips,search_throughput,search_mean_ms,search_p50_ms,search_p95_ms,search_p99_ms,"
           << "insert_count,insert_failures,insert_skips,insert_throughput,insert_mean_ms,insert_p50_ms,insert_p95_ms,insert_p99_ms,"
           << "delete_count,delete_failures,delete_skips,delete_throughput,delete_mean_ms,delete_p50_ms,delete_p95_ms,delete_p99_ms\n";

    output << "summary,final," << config.total_ops << ',' << fixed << setprecision(6) << total_elapsed_sec
           << ',' << final_record.active_count << ',' << final_record.recall << ','
           << final_record.invalid_labels << ',' << (final_record.reverse_links_ok ? 1 : 0) << ','
           << final_record.reverse_links_error << ','
           << search_summary.count << ',' << totals.search_failures << ",0," << search_summary.throughput_ops << ','
           << search_summary.mean_ms << ',' << search_summary.p50_ms << ',' << search_summary.p95_ms << ',' << search_summary.p99_ms << ','
           << insert_summary.count << ',' << totals.insert_failures << ',' << totals.skipped_inserts << ',' << insert_summary.throughput_ops << ','
           << insert_summary.mean_ms << ',' << insert_summary.p50_ms << ',' << insert_summary.p95_ms << ',' << insert_summary.p99_ms << ','
           << delete_summary.count << ',' << totals.delete_failures << ',' << totals.skipped_deletes << ',' << delete_summary.throughput_ops << ','
           << delete_summary.mean_ms << ',' << delete_summary.p50_ms << ',' << delete_summary.p95_ms << ',' << delete_summary.p99_ms << '\n';

    for (const ValidationRecord &record : records) {
        output << "validation," << record.phase << ',' << record.completed_ops << ','
               << fixed << setprecision(6) << record.elapsed_sec << ','
               << record.active_count << ',' << record.recall << ',' << record.invalid_labels << ','
               << (record.reverse_links_ok ? 1 : 0) << ',' << record.reverse_links_error << ","
               << ",,,,,,,,,,,,,,,,,,,\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    ParsedArgs parsed = parseArgs(argc, argv);
    if (parsed.show_help) {
        printHelp(argv[0]);
        return 0;
    }

    Config config = parsed.config;
    if (config.delete_model != SEARCH_DELETE &&
        config.delete_model != TWOHOP_DELETE &&
        config.delete_model != APPROXIMATE_TWOHOP_DELETE &&
        config.delete_model != NAIVE_TOMBSTONE_DELETE &&
        config.delete_model != NAIVE_RECONSTRUCTION_DELETE) {
        throw runtime_error("interleaved benchmark supports SEARCH_DELETE(2), TWOHOP_DELETE(3), APPROXIMATE_TWOHOP_DELETE(4), NAIVE_TOMBSTONE(6), NAIVE_RECONSTRUCTION(7)");
    }

    int32_t dim = 0;
    int32_t max_elements = 0;
    float *data = nullptr;
    int32_t query_sum = 0;
    int32_t query_dim = 0;
    float *querys = nullptr;

    readInitData<float>(dim, max_elements, data, config.data_file_path);
    readQuerys<float>(query_sum, query_dim, querys, config.query_file_path);
    if (query_dim != dim) {
        throw runtime_error("query dimension does not match data dimension");
    }
    if (max_elements <= 1) {
        throw runtime_error("dataset is too small");
    }

    if (config.initial_active_count == 0) {
        config.initial_active_count = 1;
    }
    if (config.initial_active_count >= static_cast<size_t>(max_elements)) {
        config.initial_active_count = static_cast<size_t>(max_elements) - 1;
    }
    if (config.new_link_size <= 0) {
        config.new_link_size = config.M;
    }

    cout << "-----------------------------------------------------------------------------------\n";
    cout << "Interleaved benchmark\n";
#ifdef COARSE_GLOBAL_LOCK
    cout << "locking_mode: COARSE_GLOBAL_LOCK (single global mutex baseline)\n";
#else
    cout << "locking_mode: fine-grained\n";
#endif
    cout << "M: " << config.M
         << " ef_construction: " << config.ef_construction
         << " ef_search: " << config.ef_search
         << " K: " << config.k
         << " initial_active_count: " << config.initial_active_count
         << " total_ops: " << config.total_ops
         << " worker_threads: " << config.worker_threads
         << " build_threads: " << config.build_threads
         << " weights(search/insert/delete): " << config.search_weight << '/'
         << config.insert_weight << '/' << config.delete_weight
         << " delete_model: " << config.delete_model
         << " newLinkSize: " << config.new_link_size << '\n';
    cout << "data_file_path: " << config.data_file_path
         << " query_file_path: " << config.query_file_path
         << " result_file_path: " << config.result_file_path << '\n';
    cout << "-----------------------------------------------------------------------------------\n";

    hnswlib::L2Space space(dim);
    auto *alg_hnsw = new hnswlib::HierarchicalNSW<float>(
        &space,
        static_cast<size_t>(max_elements),
        static_cast<size_t>(config.M),
        static_cast<size_t>(config.ef_construction),
        config.seed);
    alg_hnsw->setEf(config.ef_search);

    mt19937 bootstrap_rng(config.seed);
    vector<size_t> all_labels(static_cast<size_t>(max_elements));
    iota(all_labels.begin(), all_labels.end(), 0);
    shuffle(all_labels.begin(), all_labels.end(), bootstrap_rng);

    LabelPool label_pool;
    label_pool.active_labels.assign(all_labels.begin(), all_labels.begin() + config.initial_active_count);
    label_pool.reserve_labels.assign(all_labels.begin() + config.initial_active_count, all_labels.end());

    cout << "Building initial index with " << config.initial_active_count << " points\n";
    ParallelFor(0, config.initial_active_count, config.build_threads, [&](size_t row, size_t) {
        size_t label = label_pool.active_labels[row];
        alg_hnsw->addPoint(static_cast<void *>(data + static_cast<size_t>(dim) * label), label);
    });

    vector<WorkerStats> worker_stats(config.worker_threads);
    vector<ValidationRecord> validation_records;
    mutex validation_mutex;
    atomic<size_t> issued_ops{0};
    atomic<size_t> completed_ops{0};
    atomic<int> active_mutators{0};
    atomic<bool> pause_mutators{false};
    atomic<bool> stop_validator{false};

    chrono::steady_clock::time_point benchmark_start = chrono::steady_clock::now();

    thread validator([&]() {
        mt19937 validator_rng(config.seed + 99991U);
        size_t next_checkpoint = config.validation_interval;
        while (!stop_validator.load()) {
            if (config.validation_interval == 0) {
                this_thread::sleep_for(chrono::milliseconds(50));
                continue;
            }
            size_t finished = completed_ops.load();
            if (finished < next_checkpoint) {
                this_thread::sleep_for(chrono::milliseconds(50));
                continue;
            }
            pause_mutators.store(true);
            while (active_mutators.load() != 0) {
                this_thread::yield();
            }
            ValidationRecord record = runValidation(
                alg_hnsw,
                space,
                config,
                label_pool,
                data,
                dim,
                querys,
                query_sum,
                query_dim,
                finished,
                benchmark_start,
                validator_rng,
                "periodic");
            {
                lock_guard<mutex> lock(validation_mutex);
                validation_records.emplace_back(record);
            }
            cout << "validation@" << finished
                 << " recall=" << fixed << setprecision(4) << record.recall
                 << " invalid_labels=" << record.invalid_labels
                 << " active=" << record.active_count;
            if (config.check_reverse_links) {
                cout << " reverse_links_ok=" << (record.reverse_links_ok ? 1 : 0);
            }
            cout << '\n';
            pause_mutators.store(false);
            next_checkpoint += config.validation_interval;
        }
    });

    auto run_search = [&](WorkerStats &stats, mt19937 &rng) {
        uniform_int_distribution<int> pick_query(0, query_sum - 1);
        int query_idx = pick_query(rng);
        auto start = chrono::steady_clock::now();
        try {
            (void)alg_hnsw->searchKnn(
                static_cast<void *>(querys + static_cast<size_t>(query_idx) * query_dim),
                config.k);
            stats.search_ops++;
            stats.search_latencies_ms.emplace_back(elapsedMilliseconds(start));
            stats.search_time_s.emplace_back(
                chrono::duration<double>(chrono::steady_clock::now() - benchmark_start).count());
        } catch (...) {
            stats.search_failures++;
        }
    };

    auto run_insert = [&](WorkerStats &stats, mt19937 &rng) {
        while (pause_mutators.load()) {
            this_thread::yield();
        }
        active_mutators.fetch_add(1);
        size_t label = 0;
        if (!label_pool.reserveInsert(rng, label)) {
            stats.skipped_inserts++;
            active_mutators.fetch_sub(1);
            return;
        }
        auto start = chrono::steady_clock::now();
        try {
            alg_hnsw->addPoint(static_cast<void *>(data + static_cast<size_t>(dim) * label), label);
            label_pool.commitInsert(label);
            stats.insert_ops++;
            stats.insert_latencies_ms.emplace_back(elapsedMilliseconds(start));
        } catch (...) {
            label_pool.revertInsert(label);
            stats.insert_failures++;
        }
        active_mutators.fetch_sub(1);
    };

    auto run_delete = [&](WorkerStats &stats, mt19937 &rng) {
        while (pause_mutators.load()) {
            this_thread::yield();
        }
        active_mutators.fetch_add(1);
        size_t label = 0;
        if (!label_pool.reserveDelete(rng, label)) {
            stats.skipped_deletes++;
            active_mutators.fetch_sub(1);
            return;
        }
        auto start = chrono::steady_clock::now();
        try {
            alg_hnsw->deletePointConcurrent(label, config.delete_model, config.new_link_size);
            label_pool.commitDelete();
            stats.delete_ops++;
            stats.delete_latencies_ms.emplace_back(elapsedMilliseconds(start));
            stats.delete_time_s.emplace_back(
                chrono::duration<double>(chrono::steady_clock::now() - benchmark_start).count());
        } catch (...) {
            label_pool.revertDelete(label);
            stats.delete_failures++;
        }
        active_mutators.fetch_sub(1);
    };

    vector<thread> workers;
    workers.reserve(config.worker_threads);
    for (int thread_id = 0; thread_id < config.worker_threads; thread_id++) {
        workers.emplace_back([&, thread_id]() {
            mt19937 rng(config.seed + 104729U * static_cast<unsigned int>(thread_id + 1));
            WorkerStats &stats = worker_stats[thread_id];
            while (true) {
                size_t op_idx = issued_ops.fetch_add(1);
                if (op_idx >= config.total_ops) {
                    break;
                }
                int op = weightedChoice(config, rng);
                if (op == 0) {
                    run_search(stats, rng);
                } else if (op == 1) {
                    run_insert(stats, rng);
                } else {
                    run_delete(stats, rng);
                }
                completed_ops.fetch_add(1);
            }
        });
    }

    for (thread &worker : workers) {
        worker.join();
    }
    stop_validator.store(true);
    validator.join();

    pause_mutators.store(true);
    while (active_mutators.load() != 0) {
        this_thread::yield();
    }
    mt19937 final_validation_rng(config.seed + 424242U);
    ValidationRecord final_record = runValidation(
        alg_hnsw,
        space,
        config,
        label_pool,
        data,
        dim,
        querys,
        query_sum,
        query_dim,
        completed_ops.load(),
        benchmark_start,
        final_validation_rng,
        "final");
    {
        lock_guard<mutex> lock(validation_mutex);
        validation_records.emplace_back(final_record);
    }
    pause_mutators.store(false);

    double total_elapsed_sec = chrono::duration<double>(chrono::steady_clock::now() - benchmark_start).count();
    WorkerStats totals;
    for (WorkerStats &stats : worker_stats) {
        totals.search_ops += stats.search_ops;
        totals.insert_ops += stats.insert_ops;
        totals.delete_ops += stats.delete_ops;
        totals.search_failures += stats.search_failures;
        totals.insert_failures += stats.insert_failures;
        totals.delete_failures += stats.delete_failures;
        totals.skipped_inserts += stats.skipped_inserts;
        totals.skipped_deletes += stats.skipped_deletes;
        totals.search_latencies_ms.insert(
            totals.search_latencies_ms.end(),
            stats.search_latencies_ms.begin(),
            stats.search_latencies_ms.end());
        totals.insert_latencies_ms.insert(
            totals.insert_latencies_ms.end(),
            stats.insert_latencies_ms.begin(),
            stats.insert_latencies_ms.end());
        totals.delete_latencies_ms.insert(
            totals.delete_latencies_ms.end(),
            stats.delete_latencies_ms.begin(),
            stats.delete_latencies_ms.end());
        totals.search_time_s.insert(
            totals.search_time_s.end(),
            stats.search_time_s.begin(),
            stats.search_time_s.end());
        totals.delete_time_s.insert(
            totals.delete_time_s.end(),
            stats.delete_time_s.begin(),
            stats.delete_time_s.end());
    }

    SummaryStats search_summary = summarizeLatencies(totals.search_latencies_ms, total_elapsed_sec);
    SummaryStats insert_summary = summarizeLatencies(totals.insert_latencies_ms, total_elapsed_sec);
    SummaryStats delete_summary = summarizeLatencies(totals.delete_latencies_ms, total_elapsed_sec);

    cout << fixed << setprecision(4);
    cout << "Final validation recall: " << final_record.recall
         << " invalid_labels: " << final_record.invalid_labels
         << " active_count: " << final_record.active_count;
    if (config.check_reverse_links) {
        cout << " reverse_links_ok: " << (final_record.reverse_links_ok ? 1 : 0);
    }
    cout << '\n';
    cout << "Search ops: " << totals.search_ops
         << " throughput: " << search_summary.throughput_ops
         << " mean/p50/p95/p99 ms: " << search_summary.mean_ms << '/' << search_summary.p50_ms << '/'
         << search_summary.p95_ms << '/' << search_summary.p99_ms << '\n';
    cout << "Insert ops: " << totals.insert_ops
         << " throughput: " << insert_summary.throughput_ops
         << " mean/p50/p95/p99 ms: " << insert_summary.mean_ms << '/' << insert_summary.p50_ms << '/'
         << insert_summary.p95_ms << '/' << insert_summary.p99_ms
         << " failures: " << totals.insert_failures
         << " skipped: " << totals.skipped_inserts << '\n';
    cout << "Delete ops: " << totals.delete_ops
         << " throughput: " << delete_summary.throughput_ops
         << " mean/p50/p95/p99 ms: " << delete_summary.mean_ms << '/' << delete_summary.p50_ms << '/'
         << delete_summary.p95_ms << '/' << delete_summary.p99_ms
         << " failures: " << totals.delete_failures
         << " skipped: " << totals.skipped_deletes << '\n';

    writeResultsCsv(
        config,
        validation_records,
        total_elapsed_sec,
        search_summary,
        insert_summary,
        delete_summary,
        totals,
        final_record);

    writeLatencyTimeseries(
        config,
        totals.search_time_s,
        totals.search_latencies_ms,
        totals.delete_time_s,
        total_elapsed_sec);

    delete alg_hnsw;
    delete[] data;
    delete[] querys;
    return 0;
}
