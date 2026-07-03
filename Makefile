NVCC        := nvcc
CUDA_ARCH   ?= 75
CXXSTD      := -std=c++14
NVCCFLAGS   := $(CXXSTD) -arch=sm_$(CUDA_ARCH) --ptxas-options=-v \
               -Xcompiler -O2

TARGET           := apsep_test
BENCH_TARGET     := bench_variants
PROFILE_TARGET   := profile_lookback
APPROACHES_TARGET := bench_approaches
SRC              := src/main.cu
BENCH_SRC        := src/bench_variants.cu
PROFILE_SRC      := src/profile_lookback.cu
APPROACHES_SRC   := src/bench_approaches.cu
INCLUDES         := -Isrc

.PHONY: all bench profile approaches clean

all: $(TARGET)

bench: $(BENCH_TARGET)

profile: $(PROFILE_TARGET)

approaches: $(APPROACHES_TARGET)

$(TARGET): $(SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRC)

$(BENCH_TARGET): $(BENCH_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(BENCH_SRC)

$(PROFILE_TARGET): $(PROFILE_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(PROFILE_SRC)

$(APPROACHES_TARGET): $(APPROACHES_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(APPROACHES_SRC)

clean:
	rm -f $(TARGET) $(BENCH_TARGET) $(PROFILE_TARGET) $(APPROACHES_TARGET)
