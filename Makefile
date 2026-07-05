NVCC        := nvcc
CUDA_ARCH   ?= 75
CXXSTD      := -std=c++14
NVCCFLAGS   := $(CXXSTD) -arch=sm_$(CUDA_ARCH) --ptxas-options=-v \
               -Xcompiler -O2

TARGET           := apsep_test
BENCH_TARGET     := bench_variants
PROFILE_TARGET   := profile_lookback
PROFILE_WSNT_TARGET := profile_wsnt
APPROACHES_TARGET := bench_approaches
SWEEP_TARGET     := bench_sweep
SRC              := src/main.cu
BENCH_SRC        := src/bench_variants.cu
PROFILE_SRC      := src/profile_lookback.cu
APPROACHES_SRC   := src/bench_approaches.cu
SWEEP_SRC        := src/bench_sweep.cu
INCLUDES         := -Isrc

.PHONY: all bench profile approaches bench_sweep profile_wsnt clean

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

$(SWEEP_TARGET): $(SWEEP_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SWEEP_SRC)

$(PROFILE_WSNT_TARGET): src/profile_wsnt.cu src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ src/profile_wsnt.cu

SPT_WORST_TARGET := bench_spt_worst
SPT_WORST_SRC    := src/bench_spt_worst.cu

SPT_PHASES_TARGET := bench_spt_phases
SPT_PHASES_SRC    := src/bench_spt_phases.cu

bench_spt_phases: $(SPT_PHASES_TARGET)

$(SPT_PHASES_TARGET): $(SPT_PHASES_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SPT_PHASES_SRC)

bench_spt_worst: $(SPT_WORST_TARGET)

$(SPT_WORST_TARGET): $(SPT_WORST_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SPT_WORST_SRC)

SPT_BLOCKED_TARGET := bench_spt_blocked
SPT_BLOCKED_SRC    := src/bench_spt_blocked.cu

bench_spt_blocked: $(SPT_BLOCKED_TARGET)

$(SPT_BLOCKED_TARGET): $(SPT_BLOCKED_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SPT_BLOCKED_SRC)

SPT_P3TEAM_TARGET := bench_spt_p3team
SPT_P3TEAM_SRC    := src/bench_spt_p3team.cu

bench_spt_p3team: $(SPT_P3TEAM_TARGET)

$(SPT_P3TEAM_TARGET): $(SPT_P3TEAM_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SPT_P3TEAM_SRC)

SPT_IPT8_TARGET := bench_spt_ipt8
SPT_IPT8_SRC    := src/bench_spt_ipt8.cu

bench_spt_ipt8: $(SPT_IPT8_TARGET)

$(SPT_IPT8_TARGET): $(SPT_IPT8_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SPT_IPT8_SRC)

SPT_WARPAUTO_TARGET := bench_spt_warpauto
SPT_WARPAUTO_SRC    := src/bench_spt_warpauto.cu

bench_spt_warpauto: $(SPT_WARPAUTO_TARGET)

$(SPT_WARPAUTO_TARGET): $(SPT_WARPAUTO_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SPT_WARPAUTO_SRC)

SPT_LE_TARGET := bench_spt_le
SPT_LE_SRC    := src/bench_spt_le.cu

bench_spt_le: $(SPT_LE_TARGET)

$(SPT_LE_TARGET): $(SPT_LE_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SPT_LE_SRC)

PROFILE_BN_TARGET := profile_bottleneck
PROFILE_BN_SRC    := src/profile_bottleneck.cu

profile_bottleneck: $(PROFILE_BN_TARGET)

$(PROFILE_BN_TARGET): $(PROFILE_BN_SRC) src/apsep.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(PROFILE_BN_SRC)

clean:
	rm -f $(TARGET) $(BENCH_TARGET) $(PROFILE_TARGET) $(PROFILE_WSNT_TARGET) \
	      $(APPROACHES_TARGET) $(SWEEP_TARGET) $(SPT_WORST_TARGET) \
	      $(SPT_PHASES_TARGET) $(SPT_BLOCKED_TARGET) $(SPT_P3TEAM_TARGET) \
	      $(SPT_IPT8_TARGET) $(SPT_WARPAUTO_TARGET) $(SPT_LE_TARGET) \
	      $(PROFILE_BN_TARGET)
