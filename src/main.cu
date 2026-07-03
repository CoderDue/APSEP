// Test and benchmark program for the single-pass APSEP kernel.
//
// Compiles a CPU reference implementation of the PSE problem and compares it
// against the GPU kernel for varying input sizes.  Also reports throughput.

#include "apsep.cuh"

#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <type_traits>

// ---------------------------------------------------------------------------
// CPU reference: nearest previous strictly-smaller element
// Returns a vector of indices (or -1).
// ---------------------------------------------------------------------------

static std::vector<int> cpuApsep(const std::vector<int>& arr) {
    int n = (int)arr.size();
    std::vector<int> result(n, -1);
    std::vector<int> stk;
    stk.reserve(n);
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && arr[stk.back()] >= arr[i])
            stk.pop_back();
        result[i] = stk.empty() ? -1 : stk.back();
        stk.push_back(i);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void printArray(const char* label, const std::vector<int>& v, int limit = 16) {
    printf("%s: [", label);
    for (int i = 0; i < (int)v.size() && i < limit; i++)
        printf(i ? ", %d" : "%d", v[i]);
    if ((int)v.size() > limit) printf(", ...");
    printf("]\n");
}

// ---------------------------------------------------------------------------
// Single correctness test case
// Returns true on pass.
// ---------------------------------------------------------------------------

static bool testCase(const std::vector<int>& h_in, bool verbose = false) {
    int n = (int)h_in.size();

    // CPU reference
    auto h_expected = cpuApsep(h_in);

    // GPU
    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    launchApsep<int>(d_in, d_out, n);

    std::vector<int> h_out(n);
    gpuAssert(cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_in);
    cudaFree(d_out);

    // Compare
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != h_expected[i]) {
            if (verbose) {
                printf("  MISMATCH at i=%d  val=%d  got=%d  expected=%d\n",
                       i, h_in[i], h_out[i], h_expected[i]);
            }
            ok = false;
        }
    }

    if (verbose) {
        if (!ok) {
            printArray("  input   ", h_in);
            printArray("  expected", h_expected);
            printArray("  got     ", h_out);
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Structured small tests
// ---------------------------------------------------------------------------

static bool runSmallTests() {
    bool all_ok = true;
    struct TC { const char* name; std::vector<int> data; };

    TC cases[] = {
        {"single element",           {42}},
        {"two ascending",            {1, 2}},
        {"two descending",           {2, 1}},
        {"all equal",                {5, 5, 5, 5, 5}},
        {"ascending sequence",       {1, 2, 3, 4, 5, 6, 7, 8}},
        {"descending sequence",      {8, 7, 6, 5, 4, 3, 2, 1}},
        {"alternating",              {3, 1, 4, 1, 5, 9, 2, 6}},
        {"exact block size (B=512)", {} /* filled below */},
        {"two full blocks",          {} /* filled below */},
        {"non-power-of-two length",  {} /* filled below */},
    };

    // B = 128 * 4 = 512
    constexpr int B = 128 * 4;

    auto& tc_exact = cases[7].data;
    tc_exact.resize(B);
    std::iota(tc_exact.begin(), tc_exact.end(), 0);

    auto& tc_two = cases[8].data;
    tc_two.resize(2 * B);
    for (int i = 0; i < 2 * B; i++) tc_two[i] = (i % 7) * 13;

    auto& tc_odd = cases[9].data;
    tc_odd.resize(B + 13);
    for (int i = 0; i < (int)tc_odd.size(); i++) tc_odd[i] = i % 17;

    for (auto& tc : cases) {
        bool ok = testCase(tc.data, /*verbose=*/true);
        printf("  %-40s %s\n", tc.name, ok ? "PASS" : "FAIL");
        all_ok = all_ok && ok;
    }
    return all_ok;
}

// ---------------------------------------------------------------------------
// Random stress test
// ---------------------------------------------------------------------------

static bool runStressTest(int num_trials, int max_n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> len_dist(1, max_n);
    std::uniform_int_distribution<int> val_dist(0, 1000);

    int failures = 0;
    for (int t = 0; t < num_trials; t++) {
        int n = len_dist(rng);
        std::vector<int> h_in(n);
        for (auto& v : h_in) v = val_dist(rng);
        if (!testCase(h_in)) {
            printf("  Stress FAIL n=%d trial=%d\n", n, t);
            testCase(h_in, /*verbose=*/true);
            failures++;
        }
    }
    return failures == 0;
}

// ---------------------------------------------------------------------------
// Benchmark statistics
// ---------------------------------------------------------------------------

static void computeDescriptors(const std::vector<float>& measurements,
                                size_t bytes) {
    size_t size = measurements.size();
    double sample_mean     = 0;
    double sample_variance = 0;
    double sample_gbps     = 0;
    double factor = (double)bytes / (1000.0 * (double)size);

    for (size_t i = 0; i < size; i++) {
        double diff = std::max(1e3 * (double)measurements[i], 0.5);
        sample_mean     += diff / (double)size;
        sample_variance += (diff * diff) / (double)size;
        sample_gbps     += factor / diff;
    }
    double sample_std = std::sqrt(sample_variance);
    double bound = (0.95 * sample_std) / std::sqrt((double)size);

    printf("%.0fμs (95%% CI: [%.1fμs, %.1fμs]); %.1f GB/s",
           sample_mean, sample_mean - bound, sample_mean + bound, sample_gbps);
}

// ---------------------------------------------------------------------------
// Baseline: plain device-to-device memcpy (measures achievable bandwidth)
// ---------------------------------------------------------------------------

__global__ void copyKernel(const int* __restrict__ src, int* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

static double baselineBandwidth(int n, int warmup = 5, int iters = 50) {
    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  (size_t)n * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)n * sizeof(int)));
    gpuAssert(cudaMemset(d_in, 0, (size_t)n * sizeof(int)));

    constexpr int BS = 256;
    int grid = (n + BS - 1) / BS;

    for (int i = 0; i < warmup; i++)
        copyKernel<<<grid, BS>>>(d_in, d_out, n);

    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    std::vector<float> measurements(iters);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(tstart);
        copyKernel<<<grid, BS>>>(d_in, d_out, n);
        cudaEventRecord(tstop);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&measurements[i], tstart, tstop);
    }

    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    cudaFree(d_in);
    cudaFree(d_out);

    size_t bytes = (size_t)n * 2 * sizeof(int);
    double mean_ms = 0;
    for (float m : measurements) mean_ms += m / iters;
    return (double)bytes / 1e9 / (mean_ms * 1e-3);
}

// ---------------------------------------------------------------------------
// Benchmark: one (IPT, K) configuration. Returns mean GB/s.
// ---------------------------------------------------------------------------

template <int IPT, int K>
static double benchmarkConfig(const int* d_in, int* d_out, int n,
                              double peak_gbps, double baseline_gbps,
                              int warmup = 5, int iters = 50) {
    constexpr int BLOCK_SIZE  = 128;
    constexpr int B           = BLOCK_SIZE * IPT;

    auto scratch = allocApsepScratch<int, BLOCK_SIZE, IPT, K>(n);

    for (int i = 0; i < warmup; i++) {
        runApsep<int, BLOCK_SIZE, IPT, K>(d_in, d_out, n, scratch);
        gpuAssert(cudaDeviceSynchronize());
    }

    std::vector<float> measurements(iters);
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    for (int i = 0; i < iters; i++) {
        cudaEventRecord(tstart);
        runApsep<int, BLOCK_SIZE, IPT, K>(d_in, d_out, n, scratch);
        cudaEventRecord(tstop);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&measurements[i], tstart, tstop);
    }

    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    freeApsepScratch<int, BLOCK_SIZE, IPT, K>(scratch);

    int    num_blocks    = (n + B - 1) / B;
    int    num_sb        = (num_blocks + K - 1) / K;

    size_t bs_sz  = sizeof(BlockState<int>);
    size_t sbs_sz = sizeof(SuperBlockState<int>);
    size_t bytes =
        (size_t)n * 2 * sizeof(int)
        + (size_t)num_blocks * 2 * B * sizeof(int)
        + (size_t)num_blocks * bs_sz
        + (size_t)num_sb * K * B * sizeof(int)
        + (size_t)num_sb * 2 * K * B * sizeof(int)
        + (size_t)num_sb * sbs_sz
        + (size_t)num_sb * (K - 1) * bs_sz;

    double mean_ms       = 0;
    for (float m : measurements) mean_ms += m / iters;
    double achieved_gbps = (double)bytes / 1e9 / (mean_ms * 1e-3);
    double serial_factor = baseline_gbps / achieved_gbps;

    printf("  IPT=%-2d  K=%-4d  B=%-4d  blocks=%-7d  superblocks=%-6d  ",
           IPT, K, B, num_blocks, num_sb);
    computeDescriptors(measurements, bytes);
    printf("  (%.0f%% of peak; serialization %.1fx)\n",
           achieved_gbps / peak_gbps * 100.0, serial_factor);

    return achieved_gbps;
}

// ---------------------------------------------------------------------------
// Sweep K and IPT using struct-based recursion
// ---------------------------------------------------------------------------

struct BestConfig { int ipt, k; double gbps; };

template <int IPT, int K, bool DONE = (K >= 64)>
struct SweepK {
    static BestConfig run(const int* d_in, int* d_out, int n,
                          double peak_gbps, double baseline_gbps,
                          BestConfig best) {
        double gbps = benchmarkConfig<IPT, K>(d_in, d_out, n, peak_gbps, baseline_gbps);
        if (gbps > best.gbps) best = {IPT, K, gbps};
        return SweepK<IPT, K * 2>::run(d_in, d_out, n, peak_gbps, baseline_gbps, best);
    }
};

template <int IPT, int K>
struct SweepK<IPT, K, true> {
    static BestConfig run(const int*, int*, int, double, double, BestConfig best) {
        return best;
    }
};

template <int IPT, bool DONE = (IPT >= 32)>
struct SweepIPT {
    static BestConfig run(const int* d_in, int* d_out, int n,
                          double peak_gbps, double baseline_gbps,
                          BestConfig best) {
        best = SweepK<IPT, 1>::run(d_in, d_out, n, peak_gbps, baseline_gbps, best);
        return SweepIPT<IPT * 2>::run(d_in, d_out, n, peak_gbps, baseline_gbps, best);
    }
};

template <int IPT>
struct SweepIPT<IPT, true> {
    static BestConfig run(const int*, int*, int, double, double, BestConfig best) {
        printf("  --> Best: IPT=%d  K=%d  (%.1f GB/s)\n", best.ipt, best.k, best.gbps);
        return best;
    }
};

// ---------------------------------------------------------------------------
// Large correctness test helper
// ---------------------------------------------------------------------------

static bool largeTest(const char* label, std::vector<int>& h_in) {
    int n = (int)h_in.size();
    printf("=== Large correctness test: %s (%zu MiB) ===\n",
           label, h_in.size() * sizeof(int) / (1024 * 1024));

    auto h_expected = cpuApsep(h_in);

    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  (size_t)n * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)n * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), (size_t)n * sizeof(int),
                         cudaMemcpyHostToDevice));

    launchApsep<int>(d_in, d_out, n);

    std::vector<int> h_out(n);
    gpuAssert(cudaMemcpy(h_out.data(), d_out, (size_t)n * sizeof(int),
                         cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != h_expected[i]) {
            if (mismatches < 5)
                printf("  MISMATCH at i=%d  val=%d  got=%d  expected=%d\n",
                       i, h_in[i], h_out[i], h_expected[i]);
            mismatches++;
        }
    }
    bool ok = (mismatches == 0);
    printf("  %s", ok ? "PASSED" : "FAILED");
    if (!ok) printf(" (%d mismatches)", mismatches);
    printf("\n\n");
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Print device info
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s  (SM %d.%d, %d SMs, %.0f MB shared/SM)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.sharedMemPerMultiprocessor / 1024.0 / 1024.0);
    printf("Kernel config: BLOCK_SIZE=128\n\n");

    // ---- Correctness ----
    printf("=== Small structured tests ===\n");
    bool ok1 = runSmallTests();
    printf("\n");

    printf("=== Stress tests (500 random arrays, n up to 4096) ===\n");
    bool ok2 = runStressTest(500, 4096, 42);
    printf("  %s\n\n", ok2 ? "All PASSED" : "Some FAILED");

    printf("=== Stress tests (100 large random arrays, n up to 1M) ===\n");
    bool ok3 = runStressTest(100, 1 << 20, 137);
    printf("  %s\n\n", ok3 ? "All PASSED" : "Some FAILED");

    // ---- Large correctness tests (100 MiB each) ----
    constexpr int N_LARGE = 100 * 1024 * 1024 / sizeof(int);
    bool ok_large = true;

    {
        std::vector<int> h(N_LARGE);
        std::mt19937 rng(0xDEAD);
        std::uniform_int_distribution<int> dist(0, 1000000);
        for (auto& v : h) v = dist(rng);
        ok_large = largeTest("uniform random [0, 1000000]", h) && ok_large;
    }

    {
        std::vector<int> h(N_LARGE, 1);
        h[0] = 0;
        ok_large = largeTest("mostly-ones [0, 1, 1, 1, ...]", h) && ok_large;
    }

    {
        std::vector<int> h(N_LARGE);
        std::mt19937 rng(0xBEEF);
        std::exponential_distribution<double> dist(1.0);
        for (auto& v : h) v = (int)(dist(rng) * 1000.0);
        ok_large = largeTest("exponential (lambda=1, scaled x1000)", h) && ok_large;
    }

    ok1 = ok1 && ok_large;

    // ---- Benchmark ----
    double peak_gbps = (double)prop.memoryClockRate * 1e3
                     * (double)prop.memoryBusWidth / 8.0
                     * 2.0
                     / 1e9;
    constexpr int N_BENCH = 500 * 1024 * 1024 / sizeof(int);
    double baseline_gbps = baselineBandwidth(N_BENCH);
    printf("=== IPT sweep benchmark (theoretical peak: %.1f GB/s, copy baseline: %.1f GB/s) ===\n",
           peak_gbps, baseline_gbps);

    std::vector<int> h_bench(N_BENCH);
    {
        std::mt19937 rng(0xBEEF);
        std::uniform_int_distribution<int> dist(0, N_BENCH);
        for (auto& v : h_bench) v = dist(rng);
    }
    int *d_bench_in = nullptr, *d_bench_out = nullptr;
    gpuAssert(cudaMalloc(&d_bench_in,  (size_t)N_BENCH * sizeof(int)));
    gpuAssert(cudaMalloc(&d_bench_out, (size_t)N_BENCH * sizeof(int)));
    gpuAssert(cudaMemcpy(d_bench_in, h_bench.data(), (size_t)N_BENCH * sizeof(int),
                         cudaMemcpyHostToDevice));

    printf("\n--- Full APSEP sweep (IPT x K) ---\n");
    SweepIPT<1>::run(d_bench_in, d_bench_out, N_BENCH, peak_gbps, baseline_gbps, {1, 1, 0.0});

    // ---- Stack look-back kernel ----
    printf("\n--- Stack look-back APSEP (correctness) ---\n"); fflush(stdout);
    {
        struct TC { const char* name; std::vector<int> data; };
        TC small_cases[] = {
            {"single",       {42}},
            {"two asc",      {1, 2}},
            {"two desc",     {2, 1}},
            {"all equal",    {3, 3, 3, 3}},
            {"ascending",    {1,2,3,4,5,6,7,8}},
            {"descending",   {8,7,6,5,4,3,2,1}},
            {"alternating",  {3,1,4,1,5,9,2,6}},
        };
        bool ok_stack = true;
        for (auto& tc : small_cases) {
            int n = (int)tc.data.size();
            auto expected = cpuApsep(tc.data);
            int *d_in, *d_out;
            gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
            gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
            gpuAssert(cudaMemcpy(d_in, tc.data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            launchStackApsep<int, 128, 2>(d_in, d_out, n);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_in); cudaFree(d_out);
            bool ok = (got == expected);
            printf("  %-20s %s\n", tc.name, ok ? "PASS" : "FAIL");
            if (!ok) {
                for (int i = 0; i < n; i++)
                    if (got[i] != expected[i])
                        printf("    i=%d val=%d got=%d expected=%d\n",
                               i, tc.data[i], got[i], expected[i]);
            }
            ok_stack = ok_stack && ok;
            fflush(stdout);
        }
        // Stress test
        {
            std::mt19937 rng(42);
            int nfail = 0;
            for (int t = 0; t < 200 && nfail == 0; t++) {
                int n = 1 + rng() % 4096;
                std::vector<int> h_in(n);
                for (auto& v : h_in) v = rng() % 1000;
                auto expected = cpuApsep(h_in);
                int *d_in, *d_out;
                gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
                gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
                gpuAssert(cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
                launchStackApsep<int, 128, 2>(d_in, d_out, n);
                std::vector<int> got(n);
                gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
                cudaFree(d_in); cudaFree(d_out);
                for (int i = 0; i < n; i++) {
                    if (got[i] != expected[i]) {
                        printf("  STRESS FAIL t=%d i=%d val=%d got=%d expected=%d\n",
                               t, i, h_in[i], got[i], expected[i]);
                        nfail++;
                        break;
                    }
                }
            }
            printf("  Stress 200x4096: %s\n", nfail == 0 ? "PASS" : "FAIL");
            ok_stack = ok_stack && (nfail == 0);
        }
        printf("  Stack correctness: %s\n\n", ok_stack ? "PASSED" : "FAILED");

        if (ok_stack) {
            printf("--- Stack look-back APSEP benchmark ---\n"); fflush(stdout);
            constexpr int BS = 128, IPT = 2;
            auto scratch = allocStackScratch<int, BS, IPT>(N_BENCH);
            int warmup = 5, iters = 30;
            for (int i = 0; i < warmup; i++)
                runStackApsep<int, BS, IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
            gpuAssert(cudaDeviceSynchronize());
            std::vector<float> meas(iters);
            cudaEvent_t t0, t1;
            gpuAssert(cudaEventCreate(&t0));
            gpuAssert(cudaEventCreate(&t1));
            for (int i = 0; i < iters; i++) {
                gpuAssert(cudaEventRecord(t0));
                runStackApsep<int, BS, IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                gpuAssert(cudaEventRecord(t1));
                gpuAssert(cudaEventSynchronize(t1));
                float ms; gpuAssert(cudaEventElapsedTime(&ms, t0, t1));
                meas[i] = ms;
            }
            gpuAssert(cudaEventDestroy(t0));
            gpuAssert(cudaEventDestroy(t1));
            freeStackScratch<int, BS, IPT>(scratch);
            constexpr int B_stk = BS * IPT;
            int nb = (N_BENCH + B_stk - 1) / B_stk;
            int avg_depth = 6;
            size_t bytes = (size_t)N_BENCH * 2 * sizeof(int)
                + (size_t)nb * (3 * sizeof(int)
                    + (size_t)avg_depth * 2 * sizeof(int));
            printf("  ");
            computeDescriptors(meas, bytes);
            printf("\n");
        }
    }

    // =========================================================================
    // Option A: Per-block look-back (no superblocks)
    // =========================================================================
    printf("\n--- Option A: Per-block look-back (correctness) ---\n"); fflush(stdout);
    {
        bool ok_a = true;
        // Small structured tests
        struct TC { const char* name; std::vector<int> data; };
        TC small_cases[] = {
            {"single",           {42}},
            {"two asc",          {1, 2}},
            {"two desc",         {2, 1}},
            {"all equal",        {3, 3, 3, 3}},
            {"ascending",        {1,2,3,4,5,6,7,8}},
            {"descending",       {8,7,6,5,4,3,2,1}},
            {"alternating",      {3,1,4,1,5,9,2,6}},
        };
        for (auto& tc : small_cases) {
            int n = (int)tc.data.size();
            auto expected = cpuApsep(tc.data);
            int *d_in, *d_out;
            gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
            gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
            gpuAssert(cudaMemcpy(d_in, tc.data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            launchPerBlockApsep<int, 128, 2>(d_in, d_out, n);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_in); cudaFree(d_out);
            bool ok = (got == expected);
            printf("  %-20s %s\n", tc.name, ok ? "PASS" : "FAIL");
            if (!ok) {
                for (int i = 0; i < n; i++)
                    if (got[i] != expected[i])
                        printf("    i=%d val=%d got=%d expected=%d\n",
                               i, tc.data[i], got[i], expected[i]);
            }
            ok_a = ok_a && ok;
        }
        // Stress
        {
            std::mt19937 rng(42);
            int nfail = 0;
            for (int t = 0; t < 500 && nfail == 0; t++) {
                int n = 1 + rng() % 8192;
                std::vector<int> h_in(n);
                for (auto& v : h_in) v = rng() % 1000;
                auto expected = cpuApsep(h_in);
                int *d_in, *d_out;
                gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
                gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
                gpuAssert(cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
                launchPerBlockApsep<int, 128, 2>(d_in, d_out, n);
                std::vector<int> got(n);
                gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
                cudaFree(d_in); cudaFree(d_out);
                for (int i = 0; i < n; i++) {
                    if (got[i] != expected[i]) {
                        printf("  STRESS FAIL t=%d n=%d i=%d val=%d got=%d expected=%d\n",
                               t, n, i, h_in[i], got[i], expected[i]);
                        nfail++;
                        break;
                    }
                }
            }
            printf("  Stress 500x8192: %s\n", nfail == 0 ? "PASS" : "FAIL");
            ok_a = ok_a && (nfail == 0);
        }
        printf("  Option A correctness: %s\n\n", ok_a ? "PASSED" : "FAILED");

        if (ok_a) {
            printf("--- Option A: Per-block look-back benchmark (IPT sweep) ---\n"); fflush(stdout);
            constexpr int BS = 128;
            int warmup = 5, iters = 50;
            {
                auto bench_a = [&](auto IPT_tag) {
                    constexpr int IPT = IPT_tag;
                    constexpr int B   = BS * IPT;
                    auto scratch = allocPerBlockScratch<int, BS, IPT>(N_BENCH);
                    for (int i = 0; i < warmup; i++) {
                        runPerBlockApsep<int, BS, IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                        gpuAssert(cudaDeviceSynchronize());
                    }
                    std::vector<float> meas(iters);
                    cudaEvent_t t0, t1;
                    gpuAssert(cudaEventCreate(&t0));
                    gpuAssert(cudaEventCreate(&t1));
                    for (int i = 0; i < iters; i++) {
                        gpuAssert(cudaEventRecord(t0));
                        runPerBlockApsep<int, BS, IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                        gpuAssert(cudaEventRecord(t1));
                        gpuAssert(cudaEventSynchronize(t1));
                        float ms; gpuAssert(cudaEventElapsedTime(&ms, t0, t1));
                        meas[i] = ms;
                    }
                    gpuAssert(cudaEventDestroy(t0));
                    gpuAssert(cudaEventDestroy(t1));
                    freePerBlockScratch<int, BS, IPT>(scratch);
                    int nb = (N_BENCH + B - 1) / B;
                    size_t bytes = (size_t)N_BENCH * 2 * sizeof(int)
                        + (size_t)nb * 2 * B * sizeof(int)
                        + (size_t)nb * sizeof(BlockState<int>);
                    printf("  IPT=%-2d  B=%-4d  blocks=%-7d  ", IPT, B, nb);
                    computeDescriptors(meas, bytes);
                    printf("\n");
                };
                bench_a(std::integral_constant<int,1>{});
                bench_a(std::integral_constant<int,2>{});
                bench_a(std::integral_constant<int,4>{});
                bench_a(std::integral_constant<int,8>{});
            }
        }
    }

    // =========================================================================
    // Option B: Warp-cooperative look-back (with superblocks)
    // =========================================================================
    printf("\n--- Option B: Warp-cooperative look-back (correctness) ---\n"); fflush(stdout);
    {
        bool ok_b = true;
        struct TC { const char* name; std::vector<int> data; };
        TC small_cases[] = {
            {"single",           {42}},
            {"two asc",          {1, 2}},
            {"two desc",         {2, 1}},
            {"all equal",        {3, 3, 3, 3}},
            {"ascending",        {1,2,3,4,5,6,7,8}},
            {"descending",       {8,7,6,5,4,3,2,1}},
            {"alternating",      {3,1,4,1,5,9,2,6}},
        };
        // Use K=1 for correctness tests: K>1 inherits the same pre-existing
        // intra-SB cross-block limitation as apsepKernel (elements in blocks
        // 1..K-1 of the first superblock can't look back across blocks).
        for (auto& tc : small_cases) {
            int n = (int)tc.data.size();
            auto expected = cpuApsep(tc.data);
            int *d_in, *d_out;
            gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
            gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
            gpuAssert(cudaMemcpy(d_in, tc.data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            launchWarpCoopApsep<int, 128, 2, 1>(d_in, d_out, n);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_in); cudaFree(d_out);
            bool ok = (got == expected);
            printf("  %-20s %s\n", tc.name, ok ? "PASS" : "FAIL");
            if (!ok) {
                for (int i = 0; i < n; i++)
                    if (got[i] != expected[i])
                        printf("    i=%d val=%d got=%d expected=%d\n",
                               i, tc.data[i], got[i], expected[i]);
            }
            ok_b = ok_b && ok;
        }
        // Stress: use K=1 to avoid pre-existing intra-SB cross-block issue
        {
            std::mt19937 rng(99);
            int nfail = 0;
            for (int t = 0; t < 500 && nfail == 0; t++) {
                int n = 1 + rng() % 8192;
                std::vector<int> h_in(n);
                for (auto& v : h_in) v = rng() % 1000;
                auto expected = cpuApsep(h_in);
                int *d_in, *d_out;
                gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
                gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
                gpuAssert(cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
                launchWarpCoopApsep<int, 128, 2, 1>(d_in, d_out, n);
                std::vector<int> got(n);
                gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
                cudaFree(d_in); cudaFree(d_out);
                for (int i = 0; i < n; i++) {
                    if (got[i] != expected[i]) {
                        printf("  STRESS FAIL t=%d n=%d i=%d val=%d got=%d expected=%d\n",
                               t, n, i, h_in[i], got[i], expected[i]);
                        nfail++;
                        break;
                    }
                }
            }
            printf("  Stress 500x8192 (K=1): %s\n", nfail == 0 ? "PASS" : "FAIL");
            ok_b = ok_b && (nfail == 0);
        }
        printf("  Option B correctness: %s\n\n", ok_b ? "PASSED" : "FAILED");

        if (ok_b) {
            printf("--- Option B: Warp-cooperative look-back benchmark (K sweep, IPT=2) ---\n"); fflush(stdout);
            constexpr int BS = 128, IPT = 2;
            int warmup = 5, iters = 50;
            auto bench_b = [&](auto K_tag) {
                constexpr int K  = K_tag;
                constexpr int B  = BS * IPT;
                constexpr int KB = K * B;
                auto scratch = allocWarpCoopScratch<int, BS, IPT, K>(N_BENCH);
                for (int i = 0; i < warmup; i++) {
                    runWarpCoopApsep<int, BS, IPT, K>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaDeviceSynchronize());
                }
                std::vector<float> meas(iters);
                cudaEvent_t t0, t1;
                gpuAssert(cudaEventCreate(&t0));
                gpuAssert(cudaEventCreate(&t1));
                for (int i = 0; i < iters; i++) {
                    gpuAssert(cudaEventRecord(t0));
                    runWarpCoopApsep<int, BS, IPT, K>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaEventRecord(t1));
                    gpuAssert(cudaEventSynchronize(t1));
                    float ms; gpuAssert(cudaEventElapsedTime(&ms, t0, t1));
                    meas[i] = ms;
                }
                gpuAssert(cudaEventDestroy(t0));
                gpuAssert(cudaEventDestroy(t1));
                int nb  = scratch.num_blocks;
                int nsb = scratch.num_superblocks;
                freeWarpCoopScratch<int, BS, IPT, K>(scratch);
                size_t bytes = (size_t)N_BENCH * 2 * sizeof(int)
                    + (size_t)nb  * 2 * B  * sizeof(int)
                    + (size_t)nb  * sizeof(BlockState<int>)
                    + (size_t)nsb * 2 * KB * sizeof(int)
                    + (size_t)nsb * sizeof(SuperBlockState<int>);
                printf("  K=%-4d  superblocks=%-6d  ", K, nsb);
                computeDescriptors(meas, bytes);
                printf("\n");
            };
            bench_b(std::integral_constant<int,1>{});
            bench_b(std::integral_constant<int,2>{});
            bench_b(std::integral_constant<int,4>{});
            bench_b(std::integral_constant<int,8>{});
            bench_b(std::integral_constant<int,16>{});
            bench_b(std::integral_constant<int,32>{});
        }
    }

    // =========================================================================
    // Helpers shared by V1/V2/V3
    // =========================================================================

    // Stress-test any launch function taking (d_in, d_out, n)
    auto stressTest = [&](auto launchFn, const char* name, int trials, int maxN,
                          unsigned seed) -> bool {
        std::mt19937 rng(seed);
        int nfail = 0;
        for (int t = 0; t < trials && nfail == 0; t++) {
            int n = 1 + rng() % maxN;
            std::vector<int> h_in(n);
            for (auto& v : h_in) v = rng() % 1000;
            auto expected = cpuApsep(h_in);
            int *d_in, *d_out;
            gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
            gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
            gpuAssert(cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            launchFn(d_in, d_out, n);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_in); cudaFree(d_out);
            for (int i = 0; i < n; i++) {
                if (got[i] != expected[i]) {
                    printf("  [%s] STRESS FAIL t=%d n=%d i=%d val=%d got=%d expected=%d\n",
                           name, t, n, i, h_in[i], got[i], expected[i]);
                    nfail++;
                    break;
                }
            }
        }
        return nfail == 0;
    };


    // =========================================================================
    // V1: Warp-per-element, per-block trees
    // =========================================================================
    printf("\n--- V1: Warp-per-element look-back (correctness) ---\n"); fflush(stdout);
    {
        bool ok_v1 = true;
        struct TC { const char* name; std::vector<int> data; };
        TC small_cases[] = {
            {"single", {42}}, {"two asc", {1,2}}, {"two desc", {2,1}},
            {"all equal", {3,3,3,3}}, {"ascending", {1,2,3,4,5,6,7,8}},
            {"descending", {8,7,6,5,4,3,2,1}}, {"alternating", {3,1,4,1,5,9,2,6}},
        };
        for (auto& tc : small_cases) {
            int n = (int)tc.data.size();
            auto expected = cpuApsep(tc.data);
            int *d_in, *d_out;
            gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
            gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
            gpuAssert(cudaMemcpy(d_in, tc.data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            launchV1Apsep<int, 128, 2>(d_in, d_out, n);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_in); cudaFree(d_out);
            bool ok = (got == expected);
            printf("  %-20s %s\n", tc.name, ok ? "PASS" : "FAIL");
            if (!ok) for (int i = 0; i < n; i++)
                if (got[i] != expected[i])
                    printf("    i=%d val=%d got=%d expected=%d\n", i, tc.data[i], got[i], expected[i]);
            ok_v1 = ok_v1 && ok;
        }
        ok_v1 = ok_v1 && stressTest(
            [](int* di, int* dou, int n){ launchV1Apsep<int,128,2>(di, dou, n); },
            "V1", 500, 8192, 11);
        printf("  Stress 500x8192: %s\n", ok_v1 ? "PASS" : "FAIL");
        printf("  V1 correctness: %s\n\n", ok_v1 ? "PASSED" : "FAILED");

        if (ok_v1) {
            printf("--- V1 benchmark (IPT sweep) ---\n"); fflush(stdout);
            int warmup = 5, iters = 50;
            auto bench = [&](auto IPT_tag) {
                constexpr int IPT = IPT_tag;
                constexpr int B   = 128 * IPT;
                auto scratch = allocV1Scratch<int, 128, IPT>(N_BENCH);
                for (int i = 0; i < warmup; i++) {
                    runV1Apsep<int,128,IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaDeviceSynchronize());
                }
                std::vector<float> meas(iters);
                cudaEvent_t t0, t1;
                gpuAssert(cudaEventCreate(&t0)); gpuAssert(cudaEventCreate(&t1));
                for (int i = 0; i < iters; i++) {
                    gpuAssert(cudaEventRecord(t0));
                    runV1Apsep<int,128,IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaEventRecord(t1));
                    gpuAssert(cudaEventSynchronize(t1));
                    float ms; gpuAssert(cudaEventElapsedTime(&ms, t0, t1)); meas[i] = ms;
                }
                gpuAssert(cudaEventDestroy(t0)); gpuAssert(cudaEventDestroy(t1));
                int nb = scratch.num_blocks;
                freeV1Scratch<int,128,IPT>(scratch);
                size_t bytes = (size_t)N_BENCH*2*sizeof(int)
                    + (size_t)nb*2*B*sizeof(int) + (size_t)nb*sizeof(BlockState<int>);
                printf("  IPT=%-2d  B=%-4d  blocks=%-7d  ", IPT, B, nb);
                computeDescriptors(meas, bytes); printf("\n");
            };
            bench(std::integral_constant<int,1>{});
            bench(std::integral_constant<int,2>{});
            bench(std::integral_constant<int,4>{});
            bench(std::integral_constant<int,8>{});
        }
    }

    // =========================================================================
    // V2: Full-block cooperative, one element at a time
    // =========================================================================
    printf("\n--- V2: Full-block serial look-back (correctness) ---\n"); fflush(stdout);
    {
        bool ok_v2 = true;
        struct TC { const char* name; std::vector<int> data; };
        TC small_cases[] = {
            {"single", {42}}, {"two asc", {1,2}}, {"two desc", {2,1}},
            {"all equal", {3,3,3,3}}, {"ascending", {1,2,3,4,5,6,7,8}},
            {"descending", {8,7,6,5,4,3,2,1}}, {"alternating", {3,1,4,1,5,9,2,6}},
        };
        for (auto& tc : small_cases) {
            int n = (int)tc.data.size();
            auto expected = cpuApsep(tc.data);
            int *d_in, *d_out;
            gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
            gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
            gpuAssert(cudaMemcpy(d_in, tc.data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            launchV2Apsep<int, 128, 2>(d_in, d_out, n);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_in); cudaFree(d_out);
            bool ok = (got == expected);
            printf("  %-20s %s\n", tc.name, ok ? "PASS" : "FAIL");
            if (!ok) for (int i = 0; i < n; i++)
                if (got[i] != expected[i])
                    printf("    i=%d val=%d got=%d expected=%d\n", i, tc.data[i], got[i], expected[i]);
            ok_v2 = ok_v2 && ok;
        }
        ok_v2 = ok_v2 && stressTest(
            [](int* di, int* dou, int n){ launchV2Apsep<int,128,2>(di, dou, n); },
            "V2", 500, 8192, 22);
        printf("  Stress 500x8192: %s\n", ok_v2 ? "PASS" : "FAIL");
        printf("  V2 correctness: %s\n\n", ok_v2 ? "PASSED" : "FAILED");

        if (ok_v2) {
            printf("--- V2 benchmark (IPT sweep) ---\n"); fflush(stdout);
            int warmup = 5, iters = 50;
            auto bench = [&](auto IPT_tag) {
                constexpr int IPT = IPT_tag;
                constexpr int B   = 128 * IPT;
                auto scratch = allocV2Scratch<int, 128, IPT>(N_BENCH);
                for (int i = 0; i < warmup; i++) {
                    runV2Apsep<int,128,IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaDeviceSynchronize());
                }
                std::vector<float> meas(iters);
                cudaEvent_t t0, t1;
                gpuAssert(cudaEventCreate(&t0)); gpuAssert(cudaEventCreate(&t1));
                for (int i = 0; i < iters; i++) {
                    gpuAssert(cudaEventRecord(t0));
                    runV2Apsep<int,128,IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaEventRecord(t1));
                    gpuAssert(cudaEventSynchronize(t1));
                    float ms; gpuAssert(cudaEventElapsedTime(&ms, t0, t1)); meas[i] = ms;
                }
                gpuAssert(cudaEventDestroy(t0)); gpuAssert(cudaEventDestroy(t1));
                int nb = scratch.num_blocks;
                freeV2Scratch<int,128,IPT>(scratch);
                size_t bytes = (size_t)N_BENCH*2*sizeof(int)
                    + (size_t)nb*2*B*sizeof(int) + (size_t)nb*sizeof(BlockState<int>);
                printf("  IPT=%-2d  B=%-4d  blocks=%-7d  ", IPT, B, nb);
                computeDescriptors(meas, bytes); printf("\n");
            };
            bench(std::integral_constant<int,1>{});
            bench(std::integral_constant<int,2>{});
            bench(std::integral_constant<int,4>{});
            bench(std::integral_constant<int,8>{});
        }
    }

    // =========================================================================
    // V3: Full-block batched look-back
    // =========================================================================
    printf("\n--- V3: Batched block look-back (correctness) ---\n"); fflush(stdout);
    {
        bool ok_v3 = true;
        struct TC { const char* name; std::vector<int> data; };
        TC small_cases[] = {
            {"single", {42}}, {"two asc", {1,2}}, {"two desc", {2,1}},
            {"all equal", {3,3,3,3}}, {"ascending", {1,2,3,4,5,6,7,8}},
            {"descending", {8,7,6,5,4,3,2,1}}, {"alternating", {3,1,4,1,5,9,2,6}},
        };
        for (auto& tc : small_cases) {
            int n = (int)tc.data.size();
            auto expected = cpuApsep(tc.data);
            int *d_in, *d_out;
            gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
            gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
            gpuAssert(cudaMemcpy(d_in, tc.data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            launchV3Apsep<int, 128, 2>(d_in, d_out, n);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_in); cudaFree(d_out);
            bool ok = (got == expected);
            printf("  %-20s %s\n", tc.name, ok ? "PASS" : "FAIL");
            if (!ok) for (int i = 0; i < n; i++)
                if (got[i] != expected[i])
                    printf("    i=%d val=%d got=%d expected=%d\n", i, tc.data[i], got[i], expected[i]);
            ok_v3 = ok_v3 && ok;
        }
        ok_v3 = ok_v3 && stressTest(
            [](int* di, int* dou, int n){ launchV3Apsep<int,128,2>(di, dou, n); },
            "V3", 500, 8192, 33);
        printf("  Stress 500x8192: %s\n", ok_v3 ? "PASS" : "FAIL");
        printf("  V3 correctness: %s\n\n", ok_v3 ? "PASSED" : "FAILED");

        if (ok_v3) {
            printf("--- V3 benchmark (IPT sweep) ---\n"); fflush(stdout);
            int warmup = 5, iters = 50;
            auto bench = [&](auto IPT_tag) {
                constexpr int IPT = IPT_tag;
                constexpr int B   = 128 * IPT;
                auto scratch = allocV3Scratch<int, 128, IPT>(N_BENCH);
                for (int i = 0; i < warmup; i++) {
                    runV3Apsep<int,128,IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaDeviceSynchronize());
                }
                std::vector<float> meas(iters);
                cudaEvent_t t0, t1;
                gpuAssert(cudaEventCreate(&t0)); gpuAssert(cudaEventCreate(&t1));
                for (int i = 0; i < iters; i++) {
                    gpuAssert(cudaEventRecord(t0));
                    runV3Apsep<int,128,IPT>(d_bench_in, d_bench_out, N_BENCH, scratch);
                    gpuAssert(cudaEventRecord(t1));
                    gpuAssert(cudaEventSynchronize(t1));
                    float ms; gpuAssert(cudaEventElapsedTime(&ms, t0, t1)); meas[i] = ms;
                }
                gpuAssert(cudaEventDestroy(t0)); gpuAssert(cudaEventDestroy(t1));
                int nb = scratch.num_blocks;
                freeV3Scratch<int,128,IPT>(scratch);
                size_t bytes = (size_t)N_BENCH*2*sizeof(int)
                    + (size_t)nb*2*B*sizeof(int) + (size_t)nb*sizeof(BlockState<int>);
                printf("  IPT=%-2d  B=%-4d  blocks=%-7d  ", IPT, B, nb);
                computeDescriptors(meas, bytes); printf("\n");
            };
            bench(std::integral_constant<int,1>{});
            bench(std::integral_constant<int,2>{});
            bench(std::integral_constant<int,4>{});
            bench(std::integral_constant<int,8>{});
        }
    }

    cudaFree(d_bench_in);
    cudaFree(d_bench_out);
    printf("\n");

    bool all_ok = ok1 && ok2 && ok3;
    printf("Overall: %s\n", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
