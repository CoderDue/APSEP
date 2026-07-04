// Test and benchmark the two best APSEP kernels:
//   WSTL (WarpScanTreeLookup) — best on random input   (~65 GB/s, N=32M)
//   SPT  (SinglePassTree)     — best on worst-case input (~68 GB/s, N=32M)
//
// Sections:
//   1. Correctness: structured small cases + random stress for both kernels
//   2. Benchmark:   median GB/s on random, descending, and ascending N=32M input

#include "apsep.cuh"

#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

static std::vector<int> cpuApsep(const std::vector<int>& a) {
    int n = (int)a.size();
    std::vector<int> r(n, -1);
    std::vector<int> stk;
    stk.reserve(n);
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && a[stk.back()] >= a[i]) stk.pop_back();
        r[i] = stk.empty() ? -1 : stk.back();
        stk.push_back(i);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float medianMs(std::vector<float>& ms) {
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

// Run kernel fn, return median GB/s over iters timed runs (bytes = R+W bytes).
template <typename Fn>
static double benchKernel(Fn fn, long long bytes, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) { fn(); gpuAssert(cudaDeviceSynchronize()); }
    cudaEvent_t t0, t1;
    gpuAssert(cudaEventCreate(&t0));
    gpuAssert(cudaEventCreate(&t1));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; i++) {
        gpuAssert(cudaEventRecord(t0));
        fn();
        gpuAssert(cudaEventRecord(t1));
        gpuAssert(cudaEventSynchronize(t1));
        gpuAssert(cudaEventElapsedTime(&ms[i], t0, t1));
    }
    gpuAssert(cudaEventDestroy(t0));
    gpuAssert(cudaEventDestroy(t1));
    float med = medianMs(ms);
    return (double)bytes / (med * 1e-3) / 1e9;
}

// Run GPU kernel via launch-fn, compare against CPU reference. Returns true on pass.
template <typename LaunchFn>
static bool stressKernel(LaunchFn launch, const char* name, int trials, int maxN, unsigned seed) {
    std::mt19937 rng(seed);
    int fails = 0;
    for (int t = 0; t < trials && fails == 0; t++) {
        int n = 1 + (int)(rng() % (unsigned)maxN);
        std::vector<int> h(n);
        for (auto& v : h) v = (int)(rng() % 1000000u);
        auto exp = cpuApsep(h);

        int *d_in, *d_out;
        gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
        gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
        gpuAssert(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        launch(d_in, d_out, n);
        gpuAssert(cudaDeviceSynchronize());
        std::vector<int> got(n);
        gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_in); cudaFree(d_out);

        for (int i = 0; i < n; i++) {
            if (got[i] != exp[i]) {
                printf("  [%s] FAIL trial=%d n=%d i=%d val=%d got=%d exp=%d\n",
                       name, t, n, i, h[i], got[i], exp[i]);
                fails++;
                break;
            }
        }
    }
    return fails == 0;
}

// ---------------------------------------------------------------------------
// Correctness: small structured cases for one kernel
// ---------------------------------------------------------------------------

template <typename LaunchFn>
static bool smallCases(LaunchFn launch, const char* name) {
    struct TC { const char* label; std::vector<int> data; };
    constexpr int B = 128 * 4;  // default block size
    TC cases[] = {
        {"single element",      {42}},
        {"two ascending",       {1, 2}},
        {"two descending",      {2, 1}},
        {"all equal",           {5, 5, 5, 5, 5}},
        {"ascending",           {1, 2, 3, 4, 5, 6, 7, 8}},
        {"descending",          {8, 7, 6, 5, 4, 3, 2, 1}},
        {"alternating",         {3, 1, 4, 1, 5, 9, 2, 6}},
        {"exact block (B=512)", std::vector<int>(B)},
        {"two full blocks",     std::vector<int>(2 * B)},
        {"non-power-of-two",    std::vector<int>(B + 13)},
    };
    std::iota(cases[7].data.begin(), cases[7].data.end(), 0);
    for (int i = 0; i < 2 * B; i++) cases[8].data[i] = (i % 7) * 13;
    for (int i = 0; i < B + 13;  i++) cases[9].data[i] = i % 17;

    bool all_ok = true;
    for (auto& tc : cases) {
        int n = (int)tc.data.size();
        auto exp = cpuApsep(tc.data);
        int *d_in, *d_out;
        gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
        gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
        gpuAssert(cudaMemcpy(d_in, tc.data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        launch(d_in, d_out, n);
        gpuAssert(cudaDeviceSynchronize());
        std::vector<int> got(n);
        gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_in); cudaFree(d_out);
        bool ok = (got == exp);
        printf("  [%s] %-28s %s\n", name, tc.label, ok ? "PASS" : "FAIL");
        if (!ok) for (int i = 0; i < n; i++)
            if (got[i] != exp[i])
                printf("    i=%d val=%d got=%d exp=%d\n", i, tc.data[i], got[i], exp[i]);
        all_ok = all_ok && ok;
    }
    return all_ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    auto wstlLaunch = [](int* di, int* dou, int n){ launchWSTL<int,128,4>(di, dou, n); };
    auto sptLaunch  = [](int* di, int* dou, int n){ launchSPT <int,128,4>(di, dou, n); };

    // -------------------------------------------------------------------------
    // 1. Correctness
    // -------------------------------------------------------------------------
    printf("=== Correctness ===\n");

    bool ok_wstl = smallCases(wstlLaunch, "WSTL");
    bool ok_spt  = smallCases(sptLaunch,  "SPT");
    printf("\n");

    printf("  Stress (500 trials, n<=4096) ...\n");
    ok_wstl = ok_wstl && stressKernel(wstlLaunch, "WSTL", 500, 4096,  42);
    ok_spt  = ok_spt  && stressKernel(sptLaunch,  "SPT",  500, 4096,  43);

    printf("  Stress (200 trials, n<=131072) ...\n");
    ok_wstl = ok_wstl && stressKernel(wstlLaunch, "WSTL", 200, 131072, 100);
    ok_spt  = ok_spt  && stressKernel(sptLaunch,  "SPT",  200, 131072, 101);

    printf("  WSTL: %s\n", ok_wstl ? "PASSED" : "FAILED");
    printf("  SPT:  %s\n", ok_spt  ? "PASSED" : "FAILED");

    // -------------------------------------------------------------------------
    // 2. Benchmark
    // -------------------------------------------------------------------------
    const int N = 32 * 1024 * 1024;
    const int B = 128 * 4;  // BLOCK_SIZE * IPT
    const long long bytes_rw = 2LL * N * sizeof(int);

    std::vector<int> h(N);
    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, N * sizeof(int)));

    auto wstlScratch = allocWSTLScratch<int,128,4>(N);
    auto sptScratch  = allocSPTScratch <int,128,4>(N);

    auto runWSTL_ = [&]{ runWSTL<int,128,4>(d_in, d_out, N, wstlScratch); };
    auto runSPT_  = [&]{ runSPT <int,128,4>(d_in, d_out, N, sptScratch);  };

    struct InputCase { const char* label; bool ascending; bool descending; };
    InputCase inputs[] = {
        {"random",     false, false},
        {"descending", false, true},
        {"ascending",  true,  false},
    };

    // -------------------------------------------------------------------------
    // Analytical global memory traffic estimates (worst-case descending input)
    //
    // "Useful I/O" = 2*N*4 bytes (read input + write output).
    // All other accesses are overhead from the algorithm structure.
    //
    // WSTL: P3 does a full 16-level tree ascent per element with no match ->
    //   N*16*4 = 2147 MB of tree reads dominates (10.5x useful I/O total).
    // SPT:  prefix-min scan enables 100% O(1) early exit on descending input ->
    //   no tree reads at all, only 2.6x useful I/O total.
    // -------------------------------------------------------------------------
    {
        const long long num_blocks_ = (N + B - 1) / B;  // 65536
        const long long M_          = 65536;             // next pow2 >= num_blocks
        const long long W_          = B / 32;            // 16 warp-mins per block
        const long long tree_nodes_ = 2 * M_ - 1;
        const long long HS_PASSES   = 16;                // log2(num_blocks)
        const long long TREE_LEVELS = 16;                // log2(M)

        auto mb = [](long long b){ return (double)b / 1e6; };

        long long wstl_total =
            (long long)N * 4              // P1 read input
          + (long long)N * 4              // P1 write output
          + num_blocks_ * B * 4           // P1 write leaves
          + num_blocks_ * W_ * 4          // P1 write warp-mins
          + num_blocks_ * 4               // P1 write block-mins
          + M_ * 4                        // P2 fill tree leaves
          + (tree_nodes_ - M_) * 2 * 4   // P2 tree reduce read (2 children/node)
          + (tree_nodes_ - M_) * 4        // P2 tree reduce write
          + (long long)N * 4              // P3 read d_out
          + (long long)N * 4              // P3 read d_in
          + (long long)N * TREE_LEVELS * 4; // P3 tree ascent (no match, 16 levels)

        long long spt_total =
            (long long)N * 4              // P1 read input
          + (long long)N * 4              // P1 write output
          + num_blocks_ * B * 4           // P1 write leaves
          + num_blocks_ * W_ * 4          // P1 write warp-mins
          + num_blocks_ * 4               // P1 write block-mins
          + M_ * 4                        // P2 fill tree leaves
          + (tree_nodes_ - M_) * 2 * 4   // P2 tree reduce read
          + (tree_nodes_ - M_) * 4        // P2 tree reduce write
          + HS_PASSES * num_blocks_ * 4   // P2 Hillis-Steele prefix-min read
          + HS_PASSES * num_blocks_ * 4   // P2 Hillis-Steele prefix-min write
          + (long long)N * 4              // P3 read d_out
          + (long long)N * 4              // P3 read d_in
          + num_blocks_ * 4;              // P3 read d_prefix_min (once per block)
          // P3 tree reads: 0 (prefix-min early exit fires for 100% of elements)

        printf("\n=== Global memory traffic (analytical, worst-case descending N=%d) ===\n", N);
        printf("  Useful I/O (2*N*4 bytes):   %6.0f MB\n", mb(bytes_rw));
        printf("  WSTL total traffic:         %6.0f MB  (%.1fx useful I/O)\n",
               mb(wstl_total), (double)wstl_total / bytes_rw);
        printf("  SPT  total traffic:         %6.0f MB  (%.1fx useful I/O)\n",
               mb(spt_total),  (double)spt_total  / bytes_rw);
        printf("  Note: WSTL P3 tree ascent alone = %.0f MB (N*16*4); "
               "SPT P3 = 0 MB (prefix-min early exit)\n",
               mb((long long)N * TREE_LEVELS * 4));
    }

    printf("\n=== Benchmark (N=%d, peak=%.0f GB/s) ===\n",
           N, (double)prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8.0 * 2.0 / 1e9);
    printf("  Throughput reported as useful I/O (2*N*4 bytes) / time.\n");
    printf("  %-14s  %10s  %10s\n", "input", "WSTL", "SPT");
    printf("  %-14s  %10s  %10s\n", "-----", "----", "---");

    for (auto& inp : inputs) {
        for (int i = 0; i < N; i++) {
            if      (inp.ascending)  h[i] = i;
            else if (inp.descending) h[i] = N - i;
            else                     h[i] = rand();
        }
        gpuAssert(cudaMemcpy(d_in, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        double gbs_wstl = benchKernel(runWSTL_, bytes_rw, 2, 9);
        double gbs_spt  = benchKernel(runSPT_,  bytes_rw, 2, 9);
        printf("  %-14s  %8.1f GB/s  %8.1f GB/s\n", inp.label, gbs_wstl, gbs_spt);
    }

    freeWSTLScratch<int,128,4>(wstlScratch);
    freeSPTScratch <int,128,4>(sptScratch);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n");
    bool all_ok = ok_wstl && ok_spt;
    printf("Overall: %s\n", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
