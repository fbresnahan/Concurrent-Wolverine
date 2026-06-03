CXX=g++
GRAPH=HNSW
DELETE_MODEL=DM_ATWOHOP
CXXFLAGS= -I. -std=c++17 -Ofast -pthread -fopenmp -D__AVX__ -mavx2 -D$(GRAPH)
TARGET=hnsw_Wolverine_test
INTERLEAVED_TARGET=hnsw_Wolverine_interleaved_test
SRC=hnsw_Wolverine_test.cpp
INTERLEAVED_SRC=hnsw_Wolverine_interleaved_test.cpp

all: $(TARGET) $(INTERLEAVED_TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(INTERLEAVED_TARGET): $(INTERLEAVED_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

interleaved: $(INTERLEAVED_TARGET)

clean:
	rm -f $(TARGET) $(INTERLEAVED_TARGET)
