NVCC        := nvcc
CUDA_ARCH   ?= 75
CXXSTD      := -std=c++14
NVCCFLAGS   := $(CXXSTD) -arch=sm_$(CUDA_ARCH) --ptxas-options=-v \
               -Xcompiler -O2

TARGET       := apsep_test
BENCH_TARGET := bench_variants
SRC          := src/main.cu
BENCH_SRC    := src/bench_variants.cu
INCLUDES     := -Isrc

.PHONY: all bench clean

all: $(TARGET)

bench: $(BENCH_TARGET)

$(TARGET): $(SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRC)

$(BENCH_TARGET): $(BENCH_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(BENCH_SRC)

clean:
	rm -f $(TARGET) $(BENCH_TARGET)
