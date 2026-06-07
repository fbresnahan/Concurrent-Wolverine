CXX=g++
GRAPH=HNSW
DELETE_MODEL=DM_ATWOHOP
CXXFLAGS= -I. -std=c++17 -Ofast -pthread -fopenmp -D__AVX__ -mavx2 -D$(GRAPH)
TARGET=hnsw_Wolverine_test
INTERLEAVED_TARGET=hnsw_Wolverine_interleaved_test
BASELINE_TARGET=hnsw_Wolverine_interleaved_baseline
SRC=hnsw_Wolverine_test.cpp
INTERLEAVED_SRC=hnsw_Wolverine_interleaved_test.cpp

all: $(TARGET) $(INTERLEAVED_TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(INTERLEAVED_TARGET): $(INTERLEAVED_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

# Coarse-grained "one big lock" baseline, built from the same source with
# -DCOARSE_GLOBAL_LOCK. Use it to contrast against the fine-grained build.
$(BASELINE_TARGET): $(INTERLEAVED_SRC)
	$(CXX) $(CXXFLAGS) -DCOARSE_GLOBAL_LOCK $< -o $@

interleaved: $(INTERLEAVED_TARGET)

baseline: $(BASELINE_TARGET)

# Build both interleaved variants for head-to-head comparison.
compare: $(INTERLEAVED_TARGET) $(BASELINE_TARGET)

clean:
	rm -f $(TARGET) $(INTERLEAVED_TARGET) $(BASELINE_TARGET)
