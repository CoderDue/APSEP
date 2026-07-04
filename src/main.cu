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
// Table printing helpers
//
// A "table" is a sequence of columns. Each column has a header label and a
// fixed display width. Columns are separated by two spaces. A group of
// consecutive columns can be spanned by a banner label, which is printed
// centred (left-padded if shorter, truncated if longer) over exactly the
// combined width of the columns it spans, including the inter-column gaps.
// ---------------------------------------------------------------------------

struct Col {
    const char* header;
    int width;  // display width, not counting the 2-space separator
};

// Returns the total display width of columns [first, last) including gaps.
static int colsWidth(const Col* cols, int first, int last) {
    int w = 0;
    for (int i = first; i < last; i++) {
        if (i > first) w += 2;  // separator
        w += cols[i].width;
    }
    return w;
}

// Print a banner row. Each entry in `spans` is {first_col, last_col, label}.
// Columns not covered by any span are printed as blank space.
struct Span { int first, last; const char* label; };

static void printBanner(const Col* cols, int ncols,
                        const Col* rowkey,           // leading row-key column (may be null)
                        const Span* spans, int nspans) {
    // Leading row-key blank
    if (rowkey) printf("  %-*s", rowkey->width, "");

    for (int c = 0; c < ncols; ) {
        // find if this column starts a span
        const Span* sp = nullptr;
        for (int s = 0; s < nspans; s++)
            if (spans[s].first == c) { sp = &spans[s]; break; }

        printf("  ");  // separator before every column group
        if (sp) {
            int w = colsWidth(cols, sp->first, sp->last);
            // centre label: pad evenly on left, remainder on right
            int llen = (int)strlen(sp->label);
            int pad  = w - llen;
            int lpad = pad / 2, rpad = pad - lpad;
            printf("%*s%s%*s", lpad, "", sp->label, rpad, "");
            c = sp->last;
        } else {
            printf("%-*s", cols[c].width, "");
            c++;
        }
    }
    printf("\n");
}

// Print the header row (column labels).
static void printHeader(const Col* cols, int ncols, const Col* rowkey) {
    if (rowkey) printf("  %-*s", rowkey->width, rowkey->header);
    for (int c = 0; c < ncols; c++)
        printf("  %-*s", cols[c].width, cols[c].header);
    printf("\n");
}

// Print the separator row (dashes under each column).
static void printSep(const Col* cols, int ncols, const Col* rowkey) {
    auto dashes = [](int w) {
        for (int i = 0; i < w; i++) putchar('-');
    };
    printf("  ");
    if (rowkey) { dashes(rowkey->width); printf("  "); }
    for (int c = 0; c < ncols; c++) {
        if (c > 0) printf("  ");
        dashes(cols[c].width);
    }
    printf("\n");
}

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

    // Traffic shared by both kernels: P1 + P2 tree build.
    const long long common =
          (long long)N * 4             // P1 read input
        + (long long)N * 4             // P1 write output
        + num_blocks_ * W_ * 4         // P1 write warp-mins
        + num_blocks_ * 4              // P1 write block-mins
        + M_ * 4                       // P2 fill tree leaves
        + (tree_nodes_ - M_) * 2 * 4  // P2 tree reduce read
        + (tree_nodes_ - M_) * 4;     // P2 tree reduce write

    // WSTL Pass 1 additionally writes the leaves array; SPT skips it (its
    // Phase 3 reads d_in directly, which is byte-identical to the leaves).
    const long long wstl_common = common + num_blocks_ * B * 4;
    // SPT writes d_prefix_min once (tree-ascent reads are L2-cached), the
    // unresolved bitmask (1 bit/element, written in P1 and read in P3), and
    // reads each block's first element in P3 for the whole-block early-out
    // (first element is always unresolved and is the max of the block's
    // unresolved elements, which form a non-increasing sequence).
    const long long bm_bytes    = num_blocks_ * (B / 32) * 4;  // N/8 bytes
    const long long spt_common  = common
        + num_blocks_ * 4        // write d_prefix_min
        + num_blocks_ * 4        // P3 read d_in[first-of-block] early-out test
        + bm_bytes;

    // Per-element inter-block lookup cost (random input); tree-walk reads
    // are L2-cached and excluded.
    // WSTL scalar scan: W=16 warp-min reads + ~17 leaf reads (early exit) = 33 reads.
    // SPT warp-coop scan: full W=16 warp-min load + 32-leaf load = 48 reads.
    const long long avg_wm_leaf_per_elem = W_ + 17;      // 33 reads (WSTL)
    const long long spt_wm_leaf_per_elem = W_ + 32;      // 48 reads (SPT)

    // Per-input analytical traffic (bytes).
    //
    // random:     ~1.3% inter-block; WSTL reads d_in only for those;
    //             SPT reads d_in for ALL (prefix-min check); avg tree walk ~16 reads,
    //             warp-min+leaf ~33 reads per inter-block element.
    // descending: 100% inter-block; WSTL ascends 16 levels with no match, no wm/leaf;
    //             SPT prefix-min fires for all -> 0 tree/wm/leaf reads.
    // ascending:  0% inter-block; P3 trivially skips all elements (zero bitmask words).
    const long long n_rand = (long long)(N * 0.013);  // ~1.3% measured

    // The min-tree (2*M-1 nodes, 524 KB) fits in L2 cache (1.5 MB), so tree reads
    // do not consume DRAM bandwidth and are excluded from traffic estimates.
    // Leaf arrays (num_blocks*B*4 = 134 MB) and warp-min arrays (4 MB) are too
    // large for L2 and do hit DRAM.

    struct TrafficRow { const char* label; long long wstl; long long spt; };
    TrafficRow traffic[] = {
        {"random",
            // WSTL random: ~1.3% of elements do inter-block lookup
            // tree reads excluded (L2-cached); warp-min+leaf reads are DRAM
            wstl_common
            + (long long)N * 4                                    // P3 read d_out
            + n_rand * 4                                          // P3 read d_in (inter-block only)
            + n_rand * avg_wm_leaf_per_elem * 4,                 // P3 warp-min+leaf scan
            // SPT random: d_in read only for unresolved elements (bitmask-driven P3)
            spt_common
            + bm_bytes                                            // P3 read bitmask
            + n_rand * 4                                          // P3 read d_in (unresolved only)
            + num_blocks_ * 4                                     // P3 read d_prefix_min
            + n_rand * spt_wm_leaf_per_elem * 4},                // P3 warp-coop wm+leaf scan
        {"descending",
            // WSTL descending: all N elements do tree ascent (no match, no wm/leaf)
            // tree reads excluded (L2-cached); only d_out and d_in reads remain in P3
            wstl_common
            + (long long)N * 4                                    // P3 read d_out
            + (long long)N * 4,                                   // P3 read d_in
            // SPT descending: whole-block early-out fires for every block ->
            // P3 reads only the bitmask + first-of-block d_in (in spt_common)
            spt_common
            + bm_bytes                                            // P3 read bitmask
            + num_blocks_ * 4},                                   // P3 read d_prefix_min
        {"ascending",
            // WSTL ascending: 0% inter-block; P3 reads d_out and exits immediately
            wstl_common + (long long)N * 4,
            // SPT ascending: 1 unresolved element per block (its first element)
            spt_common
            + bm_bytes                                            // P3 read bitmask
            + num_blocks_ * 4                                     // P3 read d_in (first-of-block)
            + num_blocks_ * 4                                     // P3 read d_prefix_min
            + num_blocks_ * spt_wm_leaf_per_elem * 4},           // P3 warp-coop wm+leaf scan
    };

    const double peak_gbps = (double)prop.memoryClockRate * 1e3
                           * prop.memoryBusWidth / 8.0 * 2.0 / 1e9;

    // Columns per kernel: useful GB/s | total GB/s | %peak
    //   useful:  input+output only (2*N*4 bytes / time)
    //   total:   all reads+writes including intermediates / time
    //   %peak:   total GB/s as % of hardware peak (288 GB/s)
    // Note: tree reads excluded (524 KB tree fits in L2, not DRAM traffic).
    printf("\n=== Benchmark (N=%d, peak=%.0f GB/s) ===\n", N, peak_gbps);
    printf("  useful = input+output only.  total = all DRAM reads+writes (analytical).\n");
    {
        Col rowkey = {"input", 11};
        Col cols[] = {{"useful", 11}, {"total", 11}, {"useful", 11}, {"total", 11}};
        Span spans[] = {{0, 2, "WSTL"}, {2, 4, "SPT"}};
        printBanner(cols, 4, &rowkey, spans, 2);
        printHeader(cols, 4, &rowkey);
        printSep   (cols, 4, &rowkey);
    }

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
        double total_w  = (double)tw / (ms_wstl * 1e-3) / 1e9;
        double total_s  = (double)ts / (ms_spt  * 1e-3) / 1e9;

        printf("  %-11s  %6.1f GB/s  %6.1f GB/s  %6.1f GB/s  %6.1f GB/s\n",
               inp.label, useful_w, total_w, useful_s, total_s);
        ti++;
    }

    freeWSTLScratch<int,128,4>(wstlScratch);
    freeSPTScratch <int,128,4>(sptScratch);
    cudaFree(d_in);
    cudaFree(d_out);

    // -------------------------------------------------------------------------
    // N sweep: useful GB/s (2*N*4 / time) across input sizes
    // columns: random | descending (worst) | ascending (best)
    // -------------------------------------------------------------------------
    printf("\n=== N sweep (useful GB/s = 2*N*4 / time) ===\n");
    {
        Col rowkey = {"N", 14};
        Col cols[] = {
            {"WSTL", 11}, {"SPT", 11},
            {"WSTL", 11}, {"SPT", 11},
            {"WSTL", 11}, {"SPT", 11},
        };
        Span spans[] = {{0, 2, "random"}, {2, 4, "descend"}, {4, 6, "ascend"}};
        printBanner(cols, 6, &rowkey, spans, 3);
        printHeader(cols, 6, &rowkey);
        printSep   (cols, 6, &rowkey);
    }

    int ns[] = {
        1 << 14,          //   16K
        1 << 16,          //   64K
        1 << 18,          //  256K
        1 << 20,          //    1M
        1 << 22,          //    4M
        1 << 24,          //   16M
        32 * 1024 * 1024, //   32M
        64 * 1024 * 1024, //   64M
        128 * 1024 * 1024,// ~128M
    };

    for (int n : ns) {
        long long brw = 2LL * n * sizeof(int);

        int *din, *dout;
        gpuAssert(cudaMalloc(&din,  n * sizeof(int)));
        gpuAssert(cudaMalloc(&dout, n * sizeof(int)));

        auto sw = allocWSTLScratch<int,128,4>(n);
        auto ss = allocSPTScratch <int,128,4>(n);

        // --- random ---
        {
            std::vector<int> hn(n);
            for (int i = 0; i < n; i++) hn[i] = rand();
            gpuAssert(cudaMemcpy(din, hn.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        }
        double ms_wr = benchKernelMs([&]{ runWSTL<int,128,4>(din, dout, n, sw); }, 2, 7);
        double ms_sr = benchKernelMs([&]{ runSPT <int,128,4>(din, dout, n, ss); }, 2, 7);

        // --- descending (worst case: every PSE = -1) ---
        {
            std::vector<int> hn(n);
            for (int i = 0; i < n; i++) hn[i] = n - i;
            gpuAssert(cudaMemcpy(din, hn.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        }
        double ms_wd = benchKernelMs([&]{ runWSTL<int,128,4>(din, dout, n, sw); }, 2, 7);
        double ms_sd = benchKernelMs([&]{ runSPT <int,128,4>(din, dout, n, ss); }, 2, 7);

        // --- ascending (best case: every PSE = i-1) ---
        {
            std::vector<int> hn(n);
            for (int i = 0; i < n; i++) hn[i] = i;
            gpuAssert(cudaMemcpy(din, hn.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        }
        double ms_wa = benchKernelMs([&]{ runWSTL<int,128,4>(din, dout, n, sw); }, 2, 7);
        double ms_sa = benchKernelMs([&]{ runSPT <int,128,4>(din, dout, n, ss); }, 2, 7);

        freeWSTLScratch<int,128,4>(sw);
        freeSPTScratch <int,128,4>(ss);
        cudaFree(din);
        cudaFree(dout);

        double gbs = (double)brw / 1e9;
        double gbs_wr = gbs / (ms_wr * 1e-3);
        double gbs_sr = gbs / (ms_sr * 1e-3);
        double gbs_wd = gbs / (ms_wd * 1e-3);
        double gbs_sd = gbs / (ms_sd * 1e-3);
        double gbs_wa = gbs / (ms_wa * 1e-3);
        double gbs_sa = gbs / (ms_sa * 1e-3);
        printf("  %-14d  %6.1f GB/s  %6.1f GB/s  %6.1f GB/s  %6.1f GB/s  %6.1f GB/s  %6.1f GB/s\n",
               n, gbs_wr, gbs_sr, gbs_wd, gbs_sd, gbs_wa, gbs_sa);
    }

    printf("\n");
    bool all_ok = ok_wstl && ok_spt;
    printf("Overall: %s\n", all_ok ? "PASSED" : "FAILED");
    return all_ok ? 0 : 1;
}
