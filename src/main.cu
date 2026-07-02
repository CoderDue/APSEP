// Test and benchmark program for the single-pass APOSE kernel.
//
// Compiles a CPU reference implementation of the PSE problem and compares it
// against the GPU kernel for varying input sizes.  Also reports throughput.

#include "apose.cuh"

#include <vector>
#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// CPU reference: nearest previous strictly-smaller element
// Returns a vector of indices (or -1).
// ---------------------------------------------------------------------------

static std::vector<int> cpuApose(const std::vector<int>& arr) {
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
    auto h_expected = cpuApose(h_in);

    // GPU
    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    launchApose<int>(d_in, d_out, n);

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
        {"single element",            {42}},
        {"two ascending",             {1, 2}},
        {"two descending",            {2, 1}},
        {"all equal",                 {5, 5, 5, 5, 5}},
        {"ascending sequence",        {1, 2, 3, 4, 5, 6, 7, 8}},
        {"descending sequence",       {8, 7, 6, 5, 4, 3, 2, 1}},
        {"alternating",               {3, 1, 4, 1, 5, 9, 2, 6}},
        {"exact block size (B=512)",  {} /* filled below */},
        {"two full blocks",           {} /* filled below */},
        {"non-power-of-two length",   {} /* filled below */},
    };

    // Fill dynamic cases
    // B = 128 * 4 = 512
    constexpr int B = 128 * 4;

    auto& tc_exact = cases[7].data;
    tc_exact.resize(B);
    std::iota(tc_exact.begin(), tc_exact.end(), 0); // 0,1,2,...,B-1

    auto& tc_two = cases[8].data;
    tc_two.resize(2 * B);
    for (int i = 0; i < 2 * B; i++) tc_two[i] = (i % 7) * 13; // pseudo-random

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
// Benchmark: measure GB/s for a large random array
// ---------------------------------------------------------------------------

static void benchmark(int n, int warmup = 3, int iters = 10) {
    constexpr int BLOCK_SIZE = 128;
    constexpr int IPT        = 4;
    constexpr int B          = BLOCK_SIZE * IPT;

    std::vector<int> h_in(n);
    std::mt19937 rng(0xBEEF);
    std::uniform_int_distribution<int> dist(0, n);
    for (auto& v : h_in) v = dist(rng);

    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  (size_t)n * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)n * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), (size_t)n * sizeof(int),
                         cudaMemcpyHostToDevice));

    // Warm-up
    for (int i = 0; i < warmup; i++)
        launchApose<int, BLOCK_SIZE, IPT>(d_in, d_out, n);

    // Timed runs
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart);
    cudaEventCreate(&tstop);

    cudaEventRecord(tstart);
    for (int i = 0; i < iters; i++)
        launchApose<int, BLOCK_SIZE, IPT>(d_in, d_out, n);
    cudaEventRecord(tstop);
    cudaEventSynchronize(tstop);

    float ms = 0;
    cudaEventElapsedTime(&ms, tstart, tstop);
    ms /= iters;

    double bytes = (double)n * (sizeof(int) + sizeof(int)); // read + write
    double gbps  = bytes / (ms * 1e-3) / 1e9;

    int num_blocks = (n + B - 1) / B;
    printf("  n=%-10d  blocks=%-6d  %.2f ms  %.1f GB/s\n",
           n, num_blocks, ms, gbps);

    cudaEventDestroy(tstart);
    cudaEventDestroy(tstop);
    cudaFree(d_in);
    cudaFree(d_out);
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
    printf("Kernel config: BLOCK_SIZE=128, IPT=4, B=%d\n\n", 128 * 4);

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

    // ---- Benchmark ----
    printf("=== Benchmark ===\n");
    for (int exp = 18; exp <= 26; exp++)
        benchmark(1 << exp);
    printf("\n");

    bool all_ok = ok1 && ok2 && ok3;
    printf("Overall: %s\n", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
