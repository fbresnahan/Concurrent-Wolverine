#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <algorithm>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <omp.h>
#include <sys/time.h>

using namespace std;

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    mutable std::shared_mutex metadata_lock_;
    mutable std::vector<std::shared_mutex> link_list_locks_;
    // Reverse adjacency is guarded by a fixed bank of striped locks (keyed by the
    // TARGET node id), so concurrent writers updating different nodes' reverse
    // lists don't all serialize on one global mutex. Each add/remove/get touches
    // exactly one target, so a single stripe is held at a time (leaf lock).
    static const size_t REVERSE_LOCK_STRIPES = 4096;
    mutable std::vector<std::shared_mutex> reverse_links_locks_ =
        std::vector<std::shared_mutex>(REVERSE_LOCK_STRIPES);

#ifdef COARSE_GLOBAL_LOCK
    // Coarse-grained baseline: a single global mutex that serializes every public
    // operation (search/insert/delete). Build with -DCOARSE_GLOBAL_LOCK to compare
    // this "one big lock" design against the fine-grained scheme. The per-node and
    // reverse-link locks below remain in place but are always uncontended in this mode.
    mutable std::mutex global_op_lock_;
#endif

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element
    std::vector<std::vector<std::vector<tableint>>> reverse_link_lists_;

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::shared_mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    mutable std::mutex level_generator_lock_;  // guards level_generator_ against concurrent inserts
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    std::mutex deleted_internalId_lock;  // lock for deleted_internalId
    std::queue<tableint> deleted_internalId;
    // std::unordered_set<tableint> deleted_internalId;    

    bool *deleteFlags;

    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }

    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
        
    }

    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            reverse_link_lists_(max_elements, std::vector<std::vector<tableint>>(1)),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
        deleteFlags=new bool[max_elements_];
        memset(deleteFlags,0,max_elements_);
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
        reverse_link_lists_.clear();

        free(deleteFlags);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }

    struct SearchMetadataSnapshot {
        tableint enterpoint_node;
        int maxlevel;
    };

    SearchMetadataSnapshot getSearchMetadataSnapshot() const {
        std::shared_lock<std::shared_mutex> lock(metadata_lock_);
        return {enterpoint_node_, maxlevel_};
    }

    tableint greedySearchUpperLayers(const void *query_data, SearchMetadataSnapshot metadata) const {
        tableint currObj = metadata.enterpoint_node;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);

        for (int level = metadata.maxlevel; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::vector<tableint> neighbors = getConnectionsWithSharedLock(currObj, level);
#ifdef COLLECT_METRICS
                // These are global atomics; on the hot search path they cause
                // cross-core/socket cache-line contention. Off by default.
                metric_hops++;
                metric_distance_computations += neighbors.size();
#endif

                for (tableint cand : neighbors) {
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        return currObj;
    }

    std::vector<tableint> getConnectionsNoLock(tableint internalId, int level) const {
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }

    std::vector<tableint> getConnectionsWithSharedLock(tableint internalId, int level) const {
        std::shared_lock<std::shared_mutex> lock(link_list_locks_[internalId]);
        return getConnectionsNoLock(internalId, level);
    }

    void ensureReverseLevelNoLock(tableint internalId, int level) {
        if (reverse_link_lists_[internalId].size() <= static_cast<size_t>(level)) {
            reverse_link_lists_[internalId].resize(level + 1);
        }
    }

    std::vector<tableint> getReverseConnectionsNoLock(tableint internalId, int level) const {
        if (reverse_link_lists_[internalId].size() <= static_cast<size_t>(level)) {
            return {};
        }
        return reverse_link_lists_[internalId][level];
    }

    std::shared_mutex& reverseLockFor(tableint target) const {
        return reverse_links_locks_[target & (REVERSE_LOCK_STRIPES - 1)];
    }

    std::vector<tableint> getReverseConnectionsWithSharedLock(tableint internalId, int level) const {
        std::shared_lock<std::shared_mutex> lock(reverseLockFor(internalId));
        return getReverseConnectionsNoLock(internalId, level);
    }

    // *NoLock helpers assume the caller already holds reverseLockFor(target).
    void addReverseLinkNoLock(tableint target, int level, tableint source) {
        ensureReverseLevelNoLock(target, level);
        std::vector<tableint> &reverse_neighbors = reverse_link_lists_[target][level];
        if (find(reverse_neighbors.begin(), reverse_neighbors.end(), source) == reverse_neighbors.end()) {
            reverse_neighbors.emplace_back(source);
        }
    }

    void removeReverseLinkNoLock(tableint target, int level, tableint source) {
        if (reverse_link_lists_[target].size() <= static_cast<size_t>(level)) {
            return;
        }
        std::vector<tableint> &reverse_neighbors = reverse_link_lists_[target][level];
        reverse_neighbors.erase(
            std::remove(reverse_neighbors.begin(), reverse_neighbors.end(), source),
            reverse_neighbors.end());
    }

    // Self-locking variants: each takes the TARGET's stripe (one lock at a time).
    void addReverseLink(tableint target, int level, tableint source) {
        std::unique_lock<std::shared_mutex> lock(reverseLockFor(target));
        addReverseLinkNoLock(target, level, source);
    }

    void removeReverseLink(tableint target, int level, tableint source) {
        std::unique_lock<std::shared_mutex> lock(reverseLockFor(target));
        removeReverseLinkNoLock(target, level, source);
    }

    // Diff old->new and update each affected target's reverse list under its own
    // stripe lock. No global reverse lock is held.
    void syncReverseLinksForSource(
        tableint source,
        int level,
        const std::vector<tableint> &old_neighbors,
        const std::vector<tableint> &new_neighbors) {
        for (tableint old_neighbor : old_neighbors) {
            if (find(new_neighbors.begin(), new_neighbors.end(), old_neighbor) == new_neighbors.end()) {
                removeReverseLink(old_neighbor, level, source);
            }
        }
        for (tableint new_neighbor : new_neighbors) {
            if (find(old_neighbors.begin(), old_neighbors.end(), new_neighbor) == old_neighbors.end()) {
                addReverseLink(new_neighbor, level, source);
            }
        }
    }

    void overwriteConnectionsNoLock(tableint internalId, int level, const std::vector<tableint> &neighbors) {
        linklistsizeint *ll = get_linklist_at_level(internalId, level);
        size_t max_neighbors = level ? maxM_ : maxM0_;
        if (neighbors.size() > max_neighbors) {
            throw std::runtime_error("Too many neighbors for level");
        }

        tableint *data = (tableint *) (ll + 1);
        setListCount(ll, neighbors.size());
        for (size_t idx = 0; idx < neighbors.size(); idx++) {
            data[idx] = neighbors[idx];
        }
        for (size_t idx = neighbors.size(); idx < max_neighbors; idx++) {
            data[idx] = 0;
        }
    }

    void replaceConnectionsLocked(tableint internalId, int level, const std::vector<tableint> &neighbors) {
        std::vector<tableint> old_neighbors = getConnectionsNoLock(internalId, level);
        overwriteConnectionsNoLock(internalId, level, neighbors);
        syncReverseLinksForSource(internalId, level, old_neighbors, neighbors);
    }

    // Stop-the-world rebuild: take every reverse stripe (ascending order) before
    // reassigning the reverse adjacency wholesale.
    std::vector<std::unique_lock<std::shared_mutex>> lockAllReverseStripes() {
        std::vector<std::unique_lock<std::shared_mutex>> locks;
        locks.reserve(REVERSE_LOCK_STRIPES);
        for (size_t s = 0; s < REVERSE_LOCK_STRIPES; s++) {
            locks.emplace_back(reverse_links_locks_[s]);
        }
        return locks;
    }

    void rebuildReverseAdjacency() {
        std::vector<std::vector<std::vector<tableint>>> snapshots(cur_element_count);
        for (tableint internalId = 0; internalId < cur_element_count; internalId++) {
            snapshots[internalId].resize(element_levels_[internalId] + 1);
            for (int level = 0; level <= element_levels_[internalId]; level++) {
                snapshots[internalId][level] = getConnectionsWithSharedLock(internalId, level);
            }
        }

        std::vector<std::unique_lock<std::shared_mutex>> reverse_locks = lockAllReverseStripes();
        reverse_link_lists_.assign(max_elements_, std::vector<std::vector<tableint>>(1));
        for (tableint internalId = 0; internalId < cur_element_count; internalId++) {
            reverse_link_lists_[internalId].resize(element_levels_[internalId] + 1);
            for (int level = 0; level <= element_levels_[internalId]; level++) {
                for (tableint neighbor : snapshots[internalId][level]) {
                    addReverseLinkNoLock(neighbor, level, internalId);
                }
            }
        }
    }

    bool isMarkedDeletedWithSharedLock(tableint internalId) const {
        // The DELETE_MARK flag is a relaxed-atomic byte (see isMarkedDeleted), so
        // reading it needs no lock. This used to take a shared link-list lock per
        // call; on the search hot path (one call per visited candidate once any
        // tombstone exists) that lock dominated multi-thread throughput.
        return isMarkedDeleted(internalId);
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        std::lock_guard<std::mutex> lock(level_generator_lock_);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock<std::shared_mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    MYsearchBaseLayer(tableint ep_id, const void *data_point, int layer,int ef) {

        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock<std::shared_mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    // Read-only per-level greedy search using SHARED link-list locks. Mirrors
    // MYsearchBaseLayer (which takes UNIQUE locks at each node and would serialize
    // concurrent repair searches). Used by the online SEARCH_DELETE repair: it
    // runs lock-free of writers (only momentary shared reads), so many concurrent
    // deletes can search in parallel. Returns the (closest-first prunable) top_ef.
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchLayerShared(tableint ep_id, const void *data_point, int layer, size_t ef) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeletedWithSharedLock(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef) {
                break;
            }
            candidateSet.pop();
            tableint curNodeNum = curr_el_pair.second;

            std::vector<tableint> neighbors = getConnectionsWithSharedLock(curNodeNum, layer);
            for (tableint candidate_id : neighbors) {
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                dist_t dist1 = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);
                if (top_candidates.size() < ef || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
                    if (!isMarkedDeletedWithSharedLock(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);
                    if (top_candidates.size() > ef)
                        top_candidates.pop();
                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeletedWithSharedLock(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            std::vector<tableint> neighbors = getConnectionsWithSharedLock(current_node_id, 0);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += neighbors.size();
            }

            for (size_t j = 0; j < neighbors.size(); j++) {
                tableint candidate_id = neighbors[j];
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeletedWithSharedLock(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        vector<tableint>& visitedNodes,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeletedWithSharedLock(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;
        visitedNodes.emplace_back(ep_id);

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            std::vector<tableint> neighbors = getConnectionsWithSharedLock(current_node_id, 0);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += neighbors.size();
            }

            for (size_t j = 0; j < neighbors.size(); j++) {
                tableint candidate_id = neighbors[j];
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    visitedNodes.emplace_back(candidate_id);

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeletedWithSharedLock(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        #ifdef moreLink
        std::priority_queue<std::pair<dist_t, tableint>> nogood_closest;
        #endif
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query){
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
            #ifdef moreLink
            else{
                nogood_closest.emplace(curent_pair);
            }
            #endif
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
        #ifdef moreLink
        while(top_candidates.size()<M){
            top_candidates.emplace(-nogood_closest.top().first, nogood_closest.top().second);
            nogood_closest.pop();
        }
        #endif
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isDelete,
        bool ifcut) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        if(!isDelete||isDelete && ifcut){
            getNeighborsByHeuristic2(top_candidates, M_);
        }
        if (!isDelete && top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            if(top_candidates.top().second!=cur_c)
                selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.empty() ? cur_c : selectedNeighbors.back();

        if(!isDelete){
            std::vector<tableint> write_set = selectedNeighbors;
            write_set.emplace_back(cur_c);
            std::sort(write_set.begin(), write_set.end());
            write_set.erase(std::unique(write_set.begin(), write_set.end()), write_set.end());
            std::vector<std::unique_lock<std::shared_mutex>> write_locks = lockNodesUnique(write_set);

            std::vector<tableint> live_selected_neighbors;
            live_selected_neighbors.reserve(selectedNeighbors.size());
            for (tableint neighbor : selectedNeighbors) {
                if (neighbor == cur_c || level > element_levels_[neighbor] || isMarkedDeleted(neighbor)) {
                    continue;
                }
                live_selected_neighbors.emplace_back(neighbor);
            }
            next_closest_entry_point = live_selected_neighbors.empty() ? cur_c : live_selected_neighbors.back();

            std::unordered_map<tableint, std::vector<tableint>> old_neighbors_by_node;
            old_neighbors_by_node.reserve(write_set.size());
            for (tableint node_id : write_set) {
                if (level <= element_levels_[node_id]) {
                    old_neighbors_by_node.emplace(node_id, getConnectionsNoLock(node_id, level));
                }
            }

            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (getListCount(ll_cur) != 0) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, live_selected_neighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < live_selected_neighbors.size(); idx++) {
                if (data[idx])
                    throw std::runtime_error("Possible memory corruption");
                data[idx] = live_selected_neighbors[idx];
            }

            for (size_t idx = 0; idx < live_selected_neighbors.size(); idx++) {
                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(live_selected_neighbors[idx]);
                else
                    ll_other = get_linklist(live_selected_neighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (live_selected_neighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");

                tableint *data = (tableint *) (ll_other + 1);

                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(live_selected_neighbors[idx]),
                                                dist_func_param_);
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c || isMarkedDeleted(data[j])) {
                            continue;
                        }
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(live_selected_neighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }

            for (tableint node_id : write_set) {
                if (level > element_levels_[node_id]) {
                    continue;
                }
                syncReverseLinksForSource(
                    node_id,
                    level,
                    old_neighbors_by_node[node_id],
                    getConnectionsNoLock(node_id, level));
            }
            return next_closest_entry_point;
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::shared_mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);
            std::vector<tableint> old_neighbors(data, data + sz_link_list_other);

            bool is_cur_c_present = false;
            if (isDelete) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }

            std::vector<tableint> new_neighbors = getConnectionsNoLock(selectedNeighbors[idx], level);
            syncReverseLinksForSource(selectedNeighbors[idx], level, old_neighbors, new_neighbors);
        }
        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);
        reverse_link_lists_.resize(new_max_elements, std::vector<std::vector<tableint>>(1));

        std::vector<std::shared_mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::shared_mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        reverse_link_lists_.assign(max_elements_, std::vector<std::vector<tableint>>(1));
        for (size_t i = 0; i < cur_element_count; i++) {
            reverse_link_lists_[i].resize(element_levels_[i] + 1);
        }
        rebuildReverseAdjacency();

        input.close();

        deleteFlags=new bool[max_elements_];
        memset(deleteFlags,0,max_elements_);

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::shared_lock<std::shared_mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeletedWithSharedLock(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::shared_mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        label_lookup_.erase(label);
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        std::unique_lock<std::shared_mutex> node_lock(link_list_locks_[internalId]);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            __atomic_or_fetch(ll_cur, DELETE_MARK, __ATOMIC_RELAXED);
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::shared_mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        std::unique_lock<std::shared_mutex> node_lock(link_list_locks_[internalId]);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            __atomic_and_fetch(ll_cur, (unsigned char)~DELETE_MARK, __ATOMIC_RELAXED);
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return __atomic_load_n(ll_cur, __ATOMIC_RELAXED) & DELETE_MARK;
    }

    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
#ifdef COARSE_GLOBAL_LOCK
        std::lock_guard<std::mutex> coarse_lock(global_op_lock_);
#endif
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        addPoint(data_point, label, -1);
    }

    // Legacy name kept for existing batch repair code paths.
    // This now returns a read snapshot under a shared node lock.
    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        return getConnectionsWithSharedLock(internalId, level);
    }

    std::vector<tableint> getConnectionsNOTWithLock(tableint internalId, int level) {
        return getConnectionsNoLock(internalId, level);
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock<std::shared_mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                throw std::runtime_error("The label is exised.");
            }

            // std::unique_lock <std::mutex> lock_del_inId(deleted_internalId_lock);
            if (cur_element_count >= max_elements_&&deleted_internalId.empty()) {
                // std::cout<<"cur_element_count: "<<cur_element_count<<" max_elements_: "<<max_elements_<<std::endl;
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }
            if(!deleted_internalId.empty()){
                std::unique_lock <std::mutex> lock_del_inId(deleted_internalId_lock);
                cur_c=deleted_internalId.front();
                deleted_internalId.pop();
                deleteFlags[cur_c]=false;
                // cur_c=*deleted_internalId.begin();
                // deleted_internalId.erase(cur_c);
            }
            else{
                cur_c = cur_element_count;
                cur_element_count++;
            }
            label_lookup_[label] = cur_c;
        }

        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        std::unique_lock<std::shared_mutex> templock(metadata_lock_);
        {
            std::unique_lock<std::shared_mutex> lock_el(link_list_locks_[cur_c]);
            element_levels_[cur_c] = curlevel;

            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(cur_c)) + 2;
            __atomic_or_fetch(ll_cur, DELETE_MARK, __ATOMIC_RELAXED);
            num_deleted_ += 1;

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }
        }

        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;
        
        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        // Read-only descent: a shared lock is enough and lets
                        // concurrent searches proceed instead of being blocked.
                        std::shared_lock<std::shared_mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                // Use the shared-lock search (instead of searchBaseLayer's unique
                // locks) so inserts don't block concurrent queries during the
                // candidate-gathering traversal. Edges are still written under
                // unique locks in mutuallyConnectNewElement.
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchLayerShared(
                        currObj, data_point, level, ef_construction_);
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false,true);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        {
            std::unique_lock<std::shared_mutex> lock_el(link_list_locks_[cur_c]);
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(cur_c)) + 2;
            __atomic_and_fetch(ll_cur, (unsigned char)~DELETE_MARK, __ATOMIC_RELAXED);
            num_deleted_ -= 1;
        }
        return cur_c;
    }

    // Multithreaded executor
    // The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
    // An alternative is using #pragme omp parallel for or any other C++ threading
    template<class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        } else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                threads.push_back(std::thread([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if (id >= end) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        } catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                            * This will work even when current is the largest value that
                            * size_t can fit, because fetch_add returns the previous value
                            * before the increment (what will result in overflow
                            * and produce 0 instead of current + 1).
                            */
                            current = end;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }
    }

    #define VIOLENT_DELETE 0
    #define PINTOPOUT_DELETE 1
    #define SEARCH_DELETE 2
    #define TWOHOP_DELETE 3
    #define APPROXIMATE_TWOHOP_DELETE 4
    #define REFACTOR_DELETE 5
    // Soft delete only: tombstone the node and drop its label, but do NO graph
    // repair. The dead node stays in the graph as a router (its out-edges are
    // left intact) and is filtered from results via isMarkedDeleted. This is the
    // classic hnswlib "markDelete" behavior — our baseline for recall vs churn.
    #define NAIVE_TOMBSTONE_DELETE 6
    // Naive reconstruction: when X is deleted, connect X's neighbors directly to
    // each other (then drop X). It patches the hole but, unlike two-hop/approx,
    // makes no attempt to pick connections that restore the monotonic search
    // path — it just wires the immediate neighbors together. Online baseline for
    // recall vs a path-aware repair.
    #define NAIVE_RECONSTRUCTION_DELETE 7

    bool supportsConcurrentDelete(int deleteModel) const {
        return deleteModel == SEARCH_DELETE ||
               deleteModel == TWOHOP_DELETE ||
               deleteModel == APPROXIMATE_TWOHOP_DELETE ||
               deleteModel == NAIVE_TOMBSTONE_DELETE ||
               deleteModel == NAIVE_RECONSTRUCTION_DELETE;
    }

    std::vector<std::unique_lock<std::shared_mutex>> lockNodesUnique(
        const std::vector<tableint> &node_ids) {
        std::vector<std::unique_lock<std::shared_mutex>> locks;
        locks.reserve(node_ids.size());
        for (tableint node_id : node_ids) {
            locks.emplace_back(link_list_locks_[node_id]);
        }
        return locks;
    }

    std::vector<tableint> collectConcurrentDeleteNeighborhood(
        tableint internalId,
        int deleteModel,
        std::vector<std::vector<tableint>> &affected_by_level) const {
        std::unordered_set<tableint> neighborhood;
        neighborhood.emplace(internalId);

        affected_by_level.assign(element_levels_[internalId] + 1, {});
        for (int level = element_levels_[internalId]; level >= 0; level--) {
            affected_by_level[level] = getReverseConnectionsWithSharedLock(internalId, level);
            std::vector<tableint> deleted_neighbors = getConnectionsWithSharedLock(internalId, level);

            neighborhood.insert(affected_by_level[level].begin(), affected_by_level[level].end());
            neighborhood.insert(deleted_neighbors.begin(), deleted_neighbors.end());

            for (tableint affected : affected_by_level[level]) {
                if (affected >= cur_element_count || level > element_levels_[affected]) {
                    continue;
                }
                std::vector<tableint> one_hop = getConnectionsWithSharedLock(affected, level);
                neighborhood.insert(one_hop.begin(), one_hop.end());
                for (tableint one_hop_node : one_hop) {
                    if (one_hop_node >= cur_element_count || level > element_levels_[one_hop_node]) {
                        continue;
                    }
                    std::vector<tableint> two_hop = getConnectionsWithSharedLock(one_hop_node, level);
                    neighborhood.insert(two_hop.begin(), two_hop.end());
                }
            }

            if (deleteModel == APPROXIMATE_TWOHOP_DELETE) {
                for (tableint one_hop_node : deleted_neighbors) {
                    if (one_hop_node >= cur_element_count || level > element_levels_[one_hop_node]) {
                        continue;
                    }
                    std::vector<tableint> two_hop = getConnectionsWithSharedLock(one_hop_node, level);
                    neighborhood.insert(two_hop.begin(), two_hop.end());
                }
            }
        }

        std::vector<tableint> neighborhood_ids(neighborhood.begin(), neighborhood.end());
        std::sort(neighborhood_ids.begin(), neighborhood_ids.end());
        return neighborhood_ids;
    }

    std::vector<tableint> buildTwoHopRepairNeighborsLocked(
        tableint source,
        tableint deletedId,
        int level,
        int newLinkSize,
        const std::unordered_set<tableint> &locked_nodes) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        size_t candidate_limit = std::max(static_cast<size_t>(newLinkSize) * 5, Mcurmax);
        std::unordered_set<tableint> candidate_ids;
        std::vector<tableint> current_neighbors = getConnectionsNoLock(source, level);
        std::vector<tableint> live_neighbors;

        for (tableint neighbor : current_neighbors) {
            if (neighbor == deletedId || neighbor == source || isMarkedDeleted(neighbor)) {
                continue;
            }
            candidate_ids.emplace(neighbor);
            live_neighbors.emplace_back(neighbor);
        }

        for (tableint one_hop : live_neighbors) {
            if (!locked_nodes.count(one_hop) || level > element_levels_[one_hop]) {
                continue;
            }
            std::vector<tableint> two_hop_neighbors = getConnectionsNoLock(one_hop, level);
            for (tableint two_hop : two_hop_neighbors) {
                if (two_hop == deletedId || two_hop == source || isMarkedDeleted(two_hop)) {
                    continue;
                }
                candidate_ids.emplace(two_hop);
                if (candidate_ids.size() >= candidate_limit) {
                    break;
                }
            }
            if (candidate_ids.size() >= candidate_limit) {
                break;
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
        for (tableint candidate : candidate_ids) {
            candidates.emplace(
                fstdistfunc_(getDataByInternalId(source), getDataByInternalId(candidate), dist_func_param_),
                candidate);
        }

        getNeighborsByHeuristic2(candidates, Mcurmax);
        std::vector<tableint> new_neighbors;
        while (!candidates.empty()) {
            new_neighbors.emplace_back(candidates.top().second);
            candidates.pop();
        }
        return new_neighbors;
    }

    // Naive reconstruction repair for one affected in-neighbor `source` of the
    // deleted node. The candidate pool is just (a) source's own surviving
    // neighbors with the edge to X dropped, and (b) X's direct neighbors — i.e.
    // "connect X's neighbors to each other." No 2-hop expansion and, crucially,
    // NO diversity heuristic: we keep the Mcurmax closest candidates by raw
    // distance (contrast buildTwoHop/Approx, which call getNeighborsByHeuristic2
    // to prune for path coverage). Keeping the nearest-by-distance set leaves
    // clustered/redundant edges and cannot rebuild the monotonic search path —
    // deliberately the dumb baseline.
    std::vector<tableint> buildNaiveReconstructionNeighborsLocked(
        tableint source,
        tableint deletedId,
        int level,
        int newLinkSize,
        const std::unordered_set<tableint> &locked_nodes) {
        (void) newLinkSize; (void) locked_nodes;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        std::unordered_set<tableint> candidate_ids;

        for (tableint neighbor : getConnectionsNoLock(source, level)) {
            if (neighbor != deletedId && neighbor != source && !isMarkedDeleted(neighbor)) {
                candidate_ids.emplace(neighbor);
            }
        }
        // X's own neighbors: wire them to source ("neighbors to each other").
        for (tableint sibling : getConnectionsNoLock(deletedId, level)) {
            if (sibling != source && sibling != deletedId && !isMarkedDeleted(sibling)) {
                candidate_ids.emplace(sibling);
            }
        }

        std::vector<std::pair<dist_t, tableint>> scored;
        scored.reserve(candidate_ids.size());
        for (tableint candidate : candidate_ids) {
            scored.emplace_back(
                fstdistfunc_(getDataByInternalId(source), getDataByInternalId(candidate), dist_func_param_),
                candidate);
        }
        // Dumb selection: nearest Mcurmax by distance, no diversity pruning.
        std::sort(scored.begin(), scored.end(),
                  [](const std::pair<dist_t, tableint> &a, const std::pair<dist_t, tableint> &b) {
                      return a.first < b.first;
                  });
        std::vector<tableint> new_neighbors;
        for (size_t i = 0; i < scored.size() && i < Mcurmax; i++) {
            new_neighbors.emplace_back(scored[i].second);
        }
        return new_neighbors;
    }

    std::vector<tableint> buildApproximateTwoHopRepairNeighborsLocked(
        tableint source,
        tableint deletedId,
        int level,
        int newLinkSize,
        const std::unordered_set<tableint> &locked_nodes) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        size_t candidate_limit = std::max(static_cast<size_t>(newLinkSize) * 2, Mcurmax);
        std::unordered_set<tableint> candidate_ids;
        std::vector<tableint> current_neighbors = getConnectionsNoLock(source, level);
        dist_t source_to_deleted = fstdistfunc_(getDataByInternalId(source), getDataByInternalId(deletedId), dist_func_param_);

        for (tableint neighbor : current_neighbors) {
            if (neighbor != deletedId && neighbor != source && !isMarkedDeleted(neighbor)) {
                candidate_ids.emplace(neighbor);
            }
        }

        std::vector<tableint> deleted_neighbors = getConnectionsNoLock(deletedId, level);
        for (tableint one_hop : deleted_neighbors) {
            if (one_hop == source || one_hop == deletedId || isMarkedDeleted(one_hop)) {
                continue;
            }

            dist_t source_to_one_hop =
                fstdistfunc_(getDataByInternalId(source), getDataByInternalId(one_hop), dist_func_param_);
            if (source_to_one_hop >= source_to_deleted) {
                continue;
            }

            candidate_ids.emplace(one_hop);
            if (!locked_nodes.count(one_hop) || level > element_levels_[one_hop]) {
                continue;
            }

            std::vector<tableint> two_hop_neighbors = getConnectionsNoLock(one_hop, level);
            for (tableint two_hop : two_hop_neighbors) {
                if (two_hop == source || two_hop == deletedId || isMarkedDeleted(two_hop)) {
                    continue;
                }

                dist_t deleted_to_two_hop =
                    fstdistfunc_(getDataByInternalId(deletedId), getDataByInternalId(two_hop), dist_func_param_);
                dist_t source_to_two_hop =
                    fstdistfunc_(getDataByInternalId(source), getDataByInternalId(two_hop), dist_func_param_);

                if (deleted_to_two_hop > source_to_deleted &&
                    source_to_two_hop < source_to_deleted &&
                    source_to_two_hop + source_to_deleted > deleted_to_two_hop) {
                    candidate_ids.emplace(two_hop);
                    if (candidate_ids.size() >= candidate_limit) {
                        break;
                    }
                }
            }
            if (candidate_ids.size() >= candidate_limit) {
                break;
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
        for (tableint candidate : candidate_ids) {
            candidates.emplace(
                fstdistfunc_(getDataByInternalId(source), getDataByInternalId(candidate), dist_func_param_),
                candidate);
        }

        getNeighborsByHeuristic2(candidates, Mcurmax);
        std::vector<tableint> new_neighbors;
        while (!candidates.empty()) {
            new_neighbors.emplace_back(candidates.top().second);
            candidates.pop();
        }
        return new_neighbors;
    }

    void updateEntrypointForConcurrentDeleteLocked(
        tableint deletedId,
        const std::vector<tableint> &locked_nodes) {
        if (enterpoint_node_ != deletedId) {
            return;
        }

        tableint replacement = deletedId;
        int replacement_level = maxlevel_;
        for (tableint candidate : locked_nodes) {
            if (candidate == deletedId || isMarkedDeleted(candidate)) {
                continue;
            }
            if (replacement == deletedId || element_levels_[candidate] > replacement_level) {
                replacement = candidate;
                replacement_level = element_levels_[candidate];
            }
        }

        if (replacement != deletedId) {
            enterpoint_node_ = replacement;
            maxlevel_ = replacement_level;
        }
    }

    // Proposed edge changes for an online SEARCH_DELETE, computed lock-free in
    // Phase 1 and applied under bounded ordered locks in Phase 2.
    struct SearchRepairPlan {
        // removal_by_level[level] = in-neighbors of the deleted node (they drop the edge to it)
        std::vector<std::vector<tableint>> removal_by_level;
        // additions[c][level] = list of (dist, thePoint): edges c -> thePoint to add
        std::unordered_map<tableint, std::vector<std::vector<std::pair<dist_t, tableint>>>> additions;
        std::vector<tableint> write_set;  // sorted, unique: every node whose forward list may change
    };

    // Phase 1 (no write locks): replicate the batch SEARCH_DELETE repair search.
    // For each out-neighbor `thePoint` of the deleted node, search from thePoint
    // and propose edges {candidate -> thePoint} so thePoint stays reachable. The
    // search set is global, but the WRITE set is bounded by deg(D)*newLinkSize.
    SearchRepairPlan buildSearchRepairPlan(tableint internalId, int newLinkSize) {
        SearchRepairPlan plan;
        int top_level = element_levels_[internalId];
        plan.removal_by_level.assign(top_level + 1, {});
        std::unordered_set<tableint> write_set;
        write_set.insert(internalId);

        for (int level = top_level; level >= 0; level--) {
            std::vector<tableint> out_neighbors = getConnectionsWithSharedLock(internalId, level);
            std::vector<tableint> in_neighbors = getReverseConnectionsWithSharedLock(internalId, level);
            plan.removal_by_level[level] = in_neighbors;
            write_set.insert(in_neighbors.begin(), in_neighbors.end());

            for (tableint thePoint : out_neighbors) {
                if (thePoint == internalId || thePoint >= cur_element_count) continue;
                if (level > element_levels_[thePoint] || isMarkedDeletedWithSharedLock(thePoint)) continue;
                // thePoint receives new in-edges; lock it in Phase 2 so the
                // candidate->thePoint edges can't target a concurrently-deleted node.
                write_set.insert(thePoint);

                auto cand = searchLayerShared(thePoint, getDataByInternalId(thePoint), level, ef_construction_);
                getNeighborsByHeuristic2(cand, newLinkSize);
                while (!cand.empty()) {
                    dist_t d = cand.top().first;
                    tableint c = cand.top().second;
                    cand.pop();
                    if (c == thePoint || c == internalId || c >= cur_element_count) continue;
                    if (level > element_levels_[c] || isMarkedDeletedWithSharedLock(c)) continue;
                    auto &perLevel = plan.additions[c];
                    if (perLevel.size() <= static_cast<size_t>(level)) perLevel.resize(level + 1);
                    perLevel[level].emplace_back(d, thePoint);
                    write_set.insert(c);
                }
            }
        }

        plan.write_set.assign(write_set.begin(), write_set.end());
        std::sort(plan.write_set.begin(), plan.write_set.end());
        return plan;
    }

    // Phase 2: lock the bounded write set in id order, re-validate against the
    // current graph, apply removals + additions, and clear the tombstone's edges.
    void commitSearchRepair(tableint internalId, SearchRepairPlan &plan, bool update_entrypoint) {
        std::unique_lock<std::shared_mutex> metadata_lock(metadata_lock_, std::defer_lock);
        if (update_entrypoint) {
            metadata_lock.lock();
        }
        std::vector<std::unique_lock<std::shared_mutex>> node_locks = lockNodesUnique(plan.write_set);

        if (update_entrypoint) {
            updateEntrypointForConcurrentDeleteLocked(internalId, plan.write_set);
        }

        // 2a. Removal: drop the deleted node from each in-neighbor's forward list.
        for (int level = 0; level < static_cast<int>(plan.removal_by_level.size()); level++) {
            for (tableint X : plan.removal_by_level[level]) {
                if (X == internalId || X >= cur_element_count) continue;
                if (level > element_levels_[X] || isMarkedDeleted(X)) continue;
                std::vector<tableint> cur = getConnectionsNoLock(X, level);
                std::vector<tableint> updated;
                updated.reserve(cur.size());
                for (tableint n : cur) {
                    if (n != internalId) updated.emplace_back(n);
                }
                if (updated.size() != cur.size()) {
                    replaceConnectionsLocked(X, level, updated);
                }
            }
        }

        // 2b. Addition: apply candidate -> thePoint edges into each candidate's
        // forward list, re-validated and degree-capped (mirrors mulLink).
        for (auto &kv : plan.additions) {
            tableint c = kv.first;
            if (c == internalId || c >= cur_element_count || isMarkedDeleted(c)) continue;
            std::vector<std::vector<std::pair<dist_t, tableint>>> &perLevel = kv.second;
            for (int level = 0; level < static_cast<int>(perLevel.size()); level++) {
                if (perLevel[level].empty() || level > element_levels_[c]) continue;
                size_t Mcurmax = level ? maxM_ : maxM0_;
                std::vector<tableint> cur = getConnectionsNoLock(c, level);
                std::unordered_set<tableint> present(cur.begin(), cur.end());

                std::vector<tableint> to_add;
                for (std::pair<dist_t, tableint> &edge : perLevel[level]) {
                    tableint thePoint = edge.second;
                    if (thePoint == c || thePoint == internalId || thePoint >= cur_element_count) continue;
                    if (level > element_levels_[thePoint] || isMarkedDeleted(thePoint)) continue;
                    if (present.count(thePoint)) continue;
                    present.insert(thePoint);
                    to_add.emplace_back(thePoint);
                }
                if (to_add.empty()) continue;

                std::vector<tableint> updated;
                if (cur.size() + to_add.size() <= Mcurmax) {
                    updated = cur;
                    updated.insert(updated.end(), to_add.begin(), to_add.end());
                } else {
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cand;
                    for (tableint n : cur) {
                        cand.emplace(fstdistfunc_(getDataByInternalId(c), getDataByInternalId(n), dist_func_param_), n);
                    }
                    for (tableint n : to_add) {
                        cand.emplace(fstdistfunc_(getDataByInternalId(c), getDataByInternalId(n), dist_func_param_), n);
                    }
                    getNeighborsByHeuristic2(cand, Mcurmax);
                    while (!cand.empty()) {
                        updated.emplace_back(cand.top().second);
                        cand.pop();
                    }
                }
                replaceConnectionsLocked(c, level, updated);
            }
        }

        // 2c. Clear the tombstone's own forward edges (and its reverse contributions).
        for (int level = element_levels_[internalId]; level >= 0; level--) {
            std::vector<tableint> empty;
            replaceConnectionsLocked(internalId, level, empty);
        }
    }

    void deletePointConcurrent(labeltype label, int deleteModel, int newLinkSize) {
#ifdef COARSE_GLOBAL_LOCK
        std::lock_guard<std::mutex> coarse_lock(global_op_lock_);
#endif
        if (!supportsConcurrentDelete(deleteModel)) {
            throw std::runtime_error("Concurrent delete supports SEARCH_DELETE, TWOHOP_DELETE, APPROXIMATE_TWOHOP_DELETE");
        }

        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));
        tableint internalId = 0;
        bool update_entrypoint = false;
        {
            std::unique_lock<std::shared_mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            internalId = search->second;
            update_entrypoint = (getSearchMetadataSnapshot().enterpoint_node == internalId);
        }

        markDeletedInternal(internalId);
        {
            std::unique_lock<std::shared_mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label);
        }

        if (deleteModel == NAIVE_TOMBSTONE_DELETE) {
            // Soft delete only: the node is already tombstoned (markDeletedInternal
            // above) and its label removed. Leave its edges in place so it keeps
            // routing; search filters it out. No repair, so no neighborhood locks.
            // The entry point is intentionally NOT updated — a deleted node may
            // remain the entry router, exactly as in stock hnswlib markDelete.
            return;
        }

        if (deleteModel == SEARCH_DELETE) {
            // Phase 1 (lock-free search) then Phase 2 (validated bounded commit).
            SearchRepairPlan plan = buildSearchRepairPlan(internalId, newLinkSize);
            commitSearchRepair(internalId, plan, update_entrypoint);
            return;
        }

        std::vector<std::vector<tableint>> affected_by_level;
        std::vector<tableint> neighborhood = collectConcurrentDeleteNeighborhood(internalId, deleteModel, affected_by_level);
        std::unordered_set<tableint> locked_node_set(neighborhood.begin(), neighborhood.end());

        std::unique_lock<std::shared_mutex> metadata_lock(metadata_lock_, std::defer_lock);
        if (update_entrypoint) {
            metadata_lock.lock();
        }
        std::vector<std::unique_lock<std::shared_mutex>> node_locks = lockNodesUnique(neighborhood);

        if (update_entrypoint) {
            updateEntrypointForConcurrentDeleteLocked(internalId, neighborhood);
        }

        for (int level = element_levels_[internalId]; level >= 0; level--) {
            for (tableint affected : affected_by_level[level]) {
                if (affected == internalId || affected >= cur_element_count) {
                    continue;
                }
                if (level > element_levels_[affected] || isMarkedDeleted(affected)) {
                    continue;
                }

                std::vector<tableint> repaired_neighbors;
                if (deleteModel == TWOHOP_DELETE) {
                    repaired_neighbors = buildTwoHopRepairNeighborsLocked(
                        affected, internalId, level, newLinkSize, locked_node_set);
                } else if (deleteModel == NAIVE_RECONSTRUCTION_DELETE) {
                    repaired_neighbors = buildNaiveReconstructionNeighborsLocked(
                        affected, internalId, level, newLinkSize, locked_node_set);
                } else {
                    repaired_neighbors = buildApproximateTwoHopRepairNeighborsLocked(
                        affected, internalId, level, newLinkSize, locked_node_set);
                }
                replaceConnectionsLocked(affected, level, repaired_neighbors);
            }
        }
    }

    void mulLink(tableint thePoint, int level,vector<pair<dist_t, tableint>> &cand){
        // cout<<cand.size()<<endl;
        size_t Mcurmax = level ? maxM_ : maxM0_;
        std::unique_lock<std::shared_mutex> lock(link_list_locks_[thePoint]);
        std::vector<tableint> old_neighbors = getConnectionsNoLock(thePoint, level);
        unsigned int *thePoint_data = get_linklist_at_level(thePoint, level);
        int thePoint_size = getListCount(thePoint_data);
        tableint *thePoint_datal = (tableint *) (thePoint_data + 1);
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        for(int i=0;i<cand.size();i++){
            if(cand[i].second!=thePoint && find(thePoint_datal,thePoint_datal+thePoint_size,cand[i].second)==thePoint_datal+thePoint_size){
                top_candidates.emplace(cand[i]);
            }
        }
        if(thePoint_size+top_candidates.size()<Mcurmax){
            while(!top_candidates.empty()){
                thePoint_datal[thePoint_size++]=top_candidates.top().second;
                top_candidates.pop();
            }
            setListCount(thePoint_data, thePoint_size);
        }
        else{
            for(int i=0;i<thePoint_size;i++){
                top_candidates.emplace(make_pair(fstdistfunc_(getDataByInternalId(thePoint_datal[i]), getDataByInternalId(thePoint), dist_func_param_),thePoint_datal[i]));
            }
            getNeighborsByHeuristic2(top_candidates, M_);
            thePoint_size=top_candidates.size();
            for(int i=0;i<thePoint_size;i++){
                thePoint_datal[i]=top_candidates.top().second;
                top_candidates.pop();
            }
            setListCount(thePoint_data, thePoint_size);
        }
        std::vector<tableint> new_neighbors = getConnectionsNoLock(thePoint, level);
        syncReverseLinksForSource(thePoint, level, old_neighbors, new_neighbors);
    }

    void patchDelete(vector<labeltype>deleteList,int deleteModel,int newLinkSize,int num_threads){
        vector<tableint> internalDeleteList;
        bool changeEp=false;       
        ParallelFor(0, deleteList.size(), num_threads, [&](size_t row, size_t threadId) {
            labeltype label=deleteList[row];
            std::unique_lock<std::shared_mutex> lock_table(label_lookup_lock); //get internalId
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                std::cout<<"delete element don`t exit!!! "<<label<<std::endl;
                throw "delete element don`t exit!!!";
            }
            internalDeleteList.emplace_back(search->second);
            deleteFlags[search->second]=true;
            if(search->second==enterpoint_node_){
                changeEp=true;
            }
        });
        for(labeltype label:deleteList){
            label_lookup_.erase(label);
        }
        patchDeleteInternalDeleteList(internalDeleteList,deleteModel,num_threads,newLinkSize,changeEp);
    }

    void patchDelete(labeltype deleteStart,size_t deleteLen,int deleteModel,int newLinkSize,int num_threads){
        vector<tableint> internalDeleteList;
        bool changeEp=false;
        // #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        // for(size_t row=deleteStart;row<deleteStart+deleteLen;row++){
        // }

        ParallelFor(deleteStart, deleteStart+deleteLen, num_threads, [&](size_t row, size_t threadId) {
            labeltype label=row;
            std::unique_lock<std::shared_mutex> lock_table(label_lookup_lock); //get internalId
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                std::cout<<"delete element don`t exit!!! "<<label<<std::endl;
                throw "delete element don`t exit!!!";
            }
            internalDeleteList.emplace_back(search->second);
            deleteFlags[search->second]=true;
            if(search->second==enterpoint_node_){
                changeEp=true;
            }
        });

        for(labeltype label=deleteStart;label<deleteStart+deleteLen;label++){
            label_lookup_.erase(label);
        }
        patchDeleteInternalDeleteList(internalDeleteList,deleteModel,num_threads,newLinkSize,changeEp);
    }

    void addCandToNewLink(unordered_map<tableint,vector<vector<pair<dist_t, tableint>>>>& newLink,std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>& top_candidates,int level,tableint thepoint){
        while (!top_candidates.empty()){
            tableint newInPoint=top_candidates.top().second;
            dist_t newInPointDist=top_candidates.top().first;
            if(newLink.find(newInPoint)==newLink.end()){
                newLink.insert(make_pair(newInPoint,vector<vector<pair<dist_t, tableint>>>(element_levels_[newInPoint]+1)));
            }
            newLink[newInPoint][level].emplace_back(make_pair(newInPointDist,thepoint));
            top_candidates.pop();
        }
    }

    inline void patchDeleteInternalDeleteList(vector<tableint> internalDeleteList,int deleteModel,int num_threads,int newLinkSize,bool changeEp){   
        if(changeEp){                   //update enterpoint_node
            std::cout<<"delete enterpoint_node_"<<std::endl;
            std::unique_lock<std::shared_mutex> templock(metadata_lock_);
            tableint internalId=enterpoint_node_;
            bool changeOver=false;
            for(int level=element_levels_[enterpoint_node_];level>=0&&!changeOver;level--){
                std::unique_lock<std::shared_mutex> lock(link_list_locks_[internalId]);
                unsigned int *data = get_linklist_at_level(internalId, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
                for(int i=0;i<size;i++){
                    if(find(internalDeleteList.begin(),internalDeleteList.end(),datal[i]) == internalDeleteList.end()){
                        enterpoint_node_=datal[i];
                        changeOver=true;
                        break;
                    }
                }
                maxlevel_=level;
            }
            if(!changeOver){
                cout<<"Change enterPoint failed!!!!!!!!!!!!!"<<endl;
                throw;
            }
        }
        // int* modifyNum= new int[num_threads];
        // for(int i=0;i<num_threads;i++){
        //     modifyNum[i]=0;
        // }
        ParallelFor(0, cur_element_count, num_threads, [&](size_t row, size_t threadId) {
            for(int level=element_levels_[row];level>=0;level--){   //update inpoint  
                std::unique_lock<std::shared_mutex> lock(link_list_locks_[row]);
                unsigned int *data = get_linklist_at_level(row, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
                vector<tableint> connectedDeletePoint;
                // bool modifyFlag=false;
                for(int i=0;i<size;i++){
                    if(deleteFlags[datal[i]]){
                        connectedDeletePoint.emplace_back(datal[i]);
                        datal[i--] = datal[--size];
                        // modifyFlag=true;
                    }
                }
                // if(level==0&&modifyFlag){
                //     modifyNum[threadId]++;
                // }
                if(deleteModel==PINTOPOUT_DELETE && !deleteFlags[row]){
                    size_t Mcurmax = level ? maxM_ : maxM0_;
                    unordered_set<tableint> cand_list;
                    for(int i=0;i<size;i++){
                        cand_list.emplace(datal[i]);
                    }
                    for(tableint DeletePoint : connectedDeletePoint){
                        vector<tableint> DeletePoint_datal=getConnectionsWithLock(DeletePoint,level);
                        for(tableint DeletePoint_link:DeletePoint_datal){
                            if(DeletePoint_link!=row && !deleteFlags[DeletePoint_link]){
                                cand_list.emplace(DeletePoint_link);
                            }
                        }
                    }
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
                    for(tableint cand : cand_list){
                        top_candidates.emplace(make_pair(fstdistfunc_(getDataByInternalId(cand), getDataByInternalId(row), dist_func_param_),cand));
                    }
                    getNeighborsByHeuristic2(top_candidates, Mcurmax);
                    size=top_candidates.size();
                    for(int i=0;i<size;i++){
                        datal[i]=top_candidates.top().second;
                        top_candidates.pop();
                    }
                }
                setListCount(data,size);
            }
        });   
        // int modifysum=0;
        // for(int i=0;i<num_threads;i++){
        //     modifysum+=modifyNum[i];
        // } 
        // cout<<endl<<"modifysum/cur_element_count: "<<modifysum<<"/"<<cur_element_count<<endl;
        if(deleteModel>=SEARCH_DELETE){
            ParallelFor(0, internalDeleteList.size(), num_threads, [&](size_t row, size_t threadId) {
                unordered_map<tableint,vector<vector<pair<dist_t, tableint>>>> newLink;
                tableint internalId=internalDeleteList[row];
                for(int level=element_levels_[internalId];level>=0;level--){
                    vector<tableint> internalId_datal=getConnectionsWithLock(internalId,level);
                    int internalId_size = internalId_datal.size();
                    if(deleteModel==SEARCH_DELETE){
                        for(int linkID=0;linkID<internalId_size;linkID++){
                            tableint thePoint=internalId_datal[linkID];
                            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
                            top_candidates = MYsearchBaseLayer(thePoint, getDataByInternalId(thePoint), level,ef_construction_);
                            getNeighborsByHeuristic2(top_candidates, newLinkSize);
                            addCandToNewLink(newLink,top_candidates,level,thePoint);
                        }
                    }
                    else if(deleteModel==TWOHOP_DELETE){
                        for(int linkID=0;linkID<internalId_size;linkID++){
                            tableint thePoint=internalId_datal[linkID];
                            unordered_set<tableint> predict_list;
                            vector<tableint> thePoint_oneHopList=getConnectionsWithLock(thePoint,level);
                            for(tableint oneHopPoint:thePoint_oneHopList){
                                vector<tableint> thePoint_twoHopList=getConnectionsWithLock(oneHopPoint,level);
                                for(tableint twoHopPoint:thePoint_twoHopList){
                                    predict_list.emplace(twoHopPoint);
                                    if(predict_list.size()>5*newLinkSize)break;
                                }
                                if(predict_list.size()>5*newLinkSize)break;
                            }
                            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
                            for(tableint predict:predict_list){
                                top_candidates.emplace(make_pair(fstdistfunc_(getDataByInternalId(thePoint), getDataByInternalId(predict), dist_func_param_),predict));
                            }
                            getNeighborsByHeuristic2(top_candidates, newLinkSize);
                            addCandToNewLink(newLink,top_candidates,level,thePoint);
                        }
                    }
                    else if(deleteModel==APPROXIMATE_TWOHOP_DELETE){
                        for(int linkID=0;linkID<internalId_size;linkID++){
                            tableint thePoint=internalId_datal[linkID];
                            tableint deletePoint=internalId;
                            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
                            unordered_set<tableint> predict_list;
                            dist_t thePoint_to_deletePoint_dist=fstdistfunc_(getDataByInternalId(thePoint), getDataByInternalId(deletePoint), dist_func_param_);
                            for(tableint oneHopPoint:internalId_datal){
                                // if(oneHopPoint==thePoint)continue;
                                dist_t thePoint_to_oneHop_dist=fstdistfunc_(getDataByInternalId(thePoint), getDataByInternalId(oneHopPoint), dist_func_param_);
                                if(thePoint_to_oneHop_dist<thePoint_to_deletePoint_dist){
                                    predict_list.emplace(oneHopPoint);
                                    vector<tableint> thePoint_twoHopList=getConnectionsWithLock(oneHopPoint,level);
                                    for(tableint twoHopPoint:thePoint_twoHopList){
                                        dist_t deletePoint_to_twoHop_dist=fstdistfunc_(getDataByInternalId(deletePoint), getDataByInternalId(twoHopPoint), dist_func_param_);
                                        dist_t thePoint_to_twoHop_dist=fstdistfunc_(getDataByInternalId(thePoint), getDataByInternalId(twoHopPoint), dist_func_param_);
                                        if(
                                        deletePoint_to_twoHop_dist>thePoint_to_deletePoint_dist &&
                                        thePoint_to_twoHop_dist<thePoint_to_deletePoint_dist && 
                                        thePoint_to_twoHop_dist+thePoint_to_deletePoint_dist>deletePoint_to_twoHop_dist
                                        ){
                                            predict_list.emplace(twoHopPoint);
                                            if(predict_list.size()>2*newLinkSize)break;
                                        }
                                    }
                                }
                                if(predict_list.size()>2*newLinkSize)break;
                            }
                            for(tableint predict:predict_list){
                                top_candidates.emplace(make_pair(fstdistfunc_(getDataByInternalId(thePoint), getDataByInternalId(predict), dist_func_param_),predict));
                            }
                            getNeighborsByHeuristic2(top_candidates, newLinkSize);                   
                            addCandToNewLink(newLink,top_candidates,level,internalId_datal[linkID]);
                        }
                    }
                }
                for(pair<const tableint,vector<vector<pair<dist_t, tableint>>>>& thePointNewLink:newLink){
                    tableint newInPoint=thePointNewLink.first;
                    for(int level=element_levels_[newInPoint];level>=0;level--){
                        if(thePointNewLink.second[level].size()>0){
                            mulLink(newInPoint,level,thePointNewLink.second[level]);
                        }
                    }
                }
            });
        }
        ParallelFor(0, internalDeleteList.size(), num_threads, [&](size_t row, size_t threadId) {
            tableint internalId=internalDeleteList[row];
            if(element_levels_[internalId]!=0){
                element_levels_[internalId]=0;
                free(linkLists_[internalId]);
                linkLists_[internalId]=nullptr;
            }
            unsigned int *internalId_LinkList = get_linklist0(internalId);
            setListCount(internalId_LinkList,0);
            std::unique_lock <std::mutex> deleteList_lock(deleted_internalId_lock);
            deleted_internalId.emplace(internalId);
        });
        rebuildReverseAdjacency();
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
#ifdef COARSE_GLOBAL_LOCK
        std::lock_guard<std::mutex> coarse_lock(global_op_lock_);
#endif
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        SearchMetadataSnapshot metadata = getSearchMetadataSnapshot();
        tableint currObj = greedySearchUpperLayers(query_data, metadata);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k,vector<tableint>& visitedNodes, BaseFilterFunctor* isIdAllowed = nullptr) const {
#ifdef COARSE_GLOBAL_LOCK
        std::lock_guard<std::mutex> coarse_lock(global_op_lock_);
#endif
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        SearchMetadataSnapshot metadata = getSearchMetadataSnapshot();
        tableint currObj = greedySearchUpperLayers(query_data, metadata);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k),visitedNodes, isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k),visitedNodes, isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        SearchMetadataSnapshot metadata = getSearchMetadataSnapshot();
        tableint currObj = greedySearchUpperLayers(query_data, metadata);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

    dist_t cluDis(const void *query_data,const void * b){
        return fstdistfunc_(query_data, b, dist_func_param_);
    }
};
}  // namespace hnswlib
