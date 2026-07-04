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

// Run kernel fn, return median elapsed time in milliseconds.
template <typename Fn>
static double benchKernelMs(Fn fn, int warmup, int iters) {
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
    return medianMs(ms);
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
    // Analytical logical read/write traffic per input type.
    //
    // "Useful I/O" = 2*N*4 bytes (read input + write output).
    // "Logical traffic" counts all array accesses the algorithm issues.
    // Note: the min-tree (131K nodes, 524 KB) fits in L2 cache, so tree reads
    // are served from L2 and do NOT consume DRAM bandwidth — logical traffic
    // for descending WSTL therefore overstates DRAM pressure significantly.
    // The useful GB/s column (based on 2*N*4) is the standard metric.
    //
    // Assumptions:
    //   random:     ~1.3% of elements need inter-block lookup (measured);
    //               avg tree walk ~16 reads, warp-min+leaf scan ~33 reads.
    //   descending: 100% need inter-block lookup; WSTL ascends 16 levels with
    //               no match; SPT prefix-min early exit fires for all (0 tree reads).
    //   ascending:  0% need inter-block lookup; Phase 3 skips all elements.
    // -------------------------------------------------------------------------
    const long long num_blocks_ = (N + B - 1) / B;  // 65536
    const long long M_          = 65536;
    const long long W_          = B / 32;
    const long long tree_nodes_ = 2 * M_ - 1;
    const long long HS_PASSES   = 16;  // log2(num_blocks) for Hillis-Steele
    const long long TREE_LEVELS = 16;  // log2(M) for tree ascent/descent

    // Traffic shared by both kernels: P1 + P2 tree build.
    const long long common =
          (long long)N * 4             // P1 read input
        + (long long)N * 4             // P1 write output
        + num_blocks_ * B * 4          // P1 write leaves
        + num_blocks_ * W_ * 4         // P1 write warp-mins
        + num_blocks_ * 4              // P1 write block-mins
        + M_ * 4                       // P2 fill tree leaves
        + (tree_nodes_ - M_) * 2 * 4  // P2 tree reduce read
        + (tree_nodes_ - M_) * 4;     // P2 tree reduce write

    // SPT adds Hillis-Steele prefix-min scan on top of common.
    const long long spt_hs = HS_PASSES * num_blocks_ * 4 * 2;  // R+W each pass

    // Per-element inter-block lookup cost (random input):
    // avg tree walk: ascent to find block (~8 levels) + descent (~8 levels) = 16 reads.
    // avg warp-min + leaf scan: W=16 warp-min reads + ~17 leaf reads = 33 reads.
    const long long avg_tree_per_elem    = TREE_LEVELS;  // 16 reads
    const long long avg_wm_leaf_per_elem = W_ + 17;      // 33 reads

    // Per-input analytical traffic (bytes).
    //
    // random:     ~1.3% inter-block; WSTL reads d_in only for those;
    //             SPT reads d_in for ALL (prefix-min check); avg tree walk ~16 reads,
    //             warp-min+leaf ~33 reads per inter-block element.
    // descending: 100% inter-block; WSTL ascends 16 levels with no match, no wm/leaf;
    //             SPT prefix-min fires for all -> 0 tree/wm/leaf reads.
    // ascending:  0% inter-block; P3 trivially skips all elements (d_out != INT_MIN).
    const long long n_rand = (long long)(N * 0.013);  // ~1.3% measured

    struct TrafficRow { const char* label; long long wstl; long long spt; };
    TrafficRow traffic[] = {
        {"random",
            // WSTL random
            common
            + (long long)N * 4                                    // P3 read d_out
            + n_rand * 4                                          // P3 read d_in (inter-block only)
            + n_rand * (avg_tree_per_elem + avg_wm_leaf_per_elem) * 4,
            // SPT random
            common + spt_hs
            + (long long)N * 4                                    // P3 read d_out
            + (long long)N * 4                                    // P3 read d_in (all, prefix-min check)
            + num_blocks_ * 4                                     // P3 read d_prefix_min
            + n_rand * (avg_tree_per_elem + avg_wm_leaf_per_elem) * 4},
        {"descending",
            // WSTL descending: every element ascends 16 levels, no match -> no wm/leaf
            common
            + (long long)N * 4                                    // P3 read d_out
            + (long long)N * 4                                    // P3 read d_in
            + (long long)N * TREE_LEVELS * 4,                    // P3 tree ascent (no match)
            // SPT descending: prefix-min fires for all -> 0 tree/wm/leaf
            common + spt_hs
            + (long long)N * 4                                    // P3 read d_out
            + (long long)N * 4                                    // P3 read d_in (prefix-min check)
            + num_blocks_ * 4},                                   // P3 read d_prefix_min
        {"ascending",
            // WSTL ascending: 0% inter-block, P3 reads d_out only (exits immediately)
            common + (long long)N * 4,
            // SPT ascending: same but also reads d_in + d_prefix_min for the check
            common + spt_hs
            + (long long)N * 4                                    // P3 read d_out
            + (long long)N * 4                                    // P3 read d_in
            + num_blocks_ * 4},                                   // P3 read d_prefix_min
    };

    const double peak_gbps = (double)prop.memoryClockRate * 1e3
                           * prop.memoryBusWidth / 8.0 * 2.0 / 1e9;

    printf("\n=== Benchmark (N=%d, peak=%.0f GB/s) ===\n", N, peak_gbps);
    printf("  useful GB/s = 2*N*4 / time.  logical MB = all array accesses (tree reads\n");
    printf("  may be L2-cached and not consume full DRAM bandwidth).\n");
    printf("  %-11s  %-32s  %-32s\n", "", "------------ WSTL ------------", "------------ SPT  ------------");
    printf("  %-11s  %9s  %10s  %6s  %9s  %10s  %6s\n",
           "input", "useful", "logical MB", "%peak", "useful", "logical MB", "%peak");
    printf("  %-11s  %9s  %10s  %6s  %9s  %10s  %6s\n",
           "-----", "------", "----------", "-----", "------", "----------", "-----");

    int ti = 0;
    for (auto& inp : inputs) {
        for (int i = 0; i < N; i++) {
            if      (inp.ascending)  h[i] = i;
            else if (inp.descending) h[i] = N - i;
            else                     h[i] = rand();
        }
        gpuAssert(cudaMemcpy(d_in, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));

        double ms_wstl = benchKernelMs(runWSTL_, 2, 9);
        double ms_spt  = benchKernelMs(runSPT_,  2, 9);

        long long tw = traffic[ti].wstl;
        long long ts = traffic[ti].spt;
        double useful_w = (double)bytes_rw / (ms_wstl * 1e-3) / 1e9;
        double useful_s = (double)bytes_rw / (ms_spt  * 1e-3) / 1e9;

        printf("  %-11s  %7.1f GB/s  %7lld MB  %5.1f%%  %7.1f GB/s  %7lld MB  %5.1f%%\n",
               inp.label,
               useful_w, tw / 1000000LL, useful_w / peak_gbps * 100.0,
               useful_s, ts / 1000000LL, useful_s / peak_gbps * 100.0);
        ti++;
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
