// bench_sweep.cu — sweep WarpScanLeaves parameters
#include "apsep.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define gpuAssert(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

static std::vector<int> cpuApsep(const std::vector<int>& a) {
    int n = (int)a.size();
    std::vector<int> r(n, -1);
    std::vector<int> stk;
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && a[stk.back()] >= a[i]) stk.pop_back();
        r[i] = stk.empty() ? -1 : stk.back();
        stk.push_back(i);
    }
    return r;
}

static void computeDescriptors(std::vector<float>& ms, long long bytes) {
    std::sort(ms.begin(), ms.end());
    float med = ms[ms.size()/2];
    float lo  = ms[ms.size()/4];
    float hi  = ms[ms.size()*3/4];
    double gbs = (double)bytes / (med * 1e-3) / 1e9;
    printf("%9.0fµs (CI [%.1f,%.1f]); %.1f GB/s\n",
           med * 1000.0f, lo * 1000.0f, hi * 1000.0f, gbs);
}

#define BENCH(label, kernel_fn, bytes, warmup, iters)                        \
do {                                                                         \
    for (int _w = 0; _w < (warmup); _w++) { (kernel_fn)(); gpuAssert(cudaDeviceSynchronize()); } \
    cudaEvent_t _t0, _t1; gpuAssert(cudaEventCreate(&_t0)); gpuAssert(cudaEventCreate(&_t1)); \
    std::vector<float> _ms(iters);                                           \
    for (int _i = 0; _i < (iters); _i++) {                                  \
        gpuAssert(cudaEventRecord(_t0));                                     \
        (kernel_fn)();                                                       \
        gpuAssert(cudaEventRecord(_t1));                                     \
        gpuAssert(cudaEventSynchronize(_t1));                                \
        gpuAssert(cudaEventElapsedTime(&_ms[_i], _t0, _t1));                \
    }                                                                        \
    gpuAssert(cudaEventDestroy(_t0)); gpuAssert(cudaEventDestroy(_t1));      \
    printf("  %-50s  ", label);                                              \
    computeDescriptors(_ms, bytes);                                          \
} while(0)

template <typename RunFn>
static bool quickCheck(RunFn fn, const char* name, int* d_in, int* d_out, int n) {
    std::vector<int> h(n);
    srand(99);
    for (auto& v : h) v = rand() % 1000;
    auto expected = cpuApsep(h);
    gpuAssert(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    fn(d_in, d_out, n);
    gpuAssert(cudaDeviceSynchronize());
    std::vector<int> got(n);
    gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    int nfail = 0;
    for (int i = 0; i < n && nfail < 3; i++)
        if (got[i] != expected[i]) {
            printf("  [%s] FAIL i=%d val=%d got=%d exp=%d\n", name, i, h[i], got[i], expected[i]);
            nfail++;
        }
    bool ok = (nfail == 0);
    printf("  %-40s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

template <int BS, int IPT, int K>
static void benchWSL(const int* d_in, int* d_out, int N, long long bytes_rw) {
    char label[80];
    snprintf(label, sizeof(label), "WarpScanLeaves BS=%d IPT=%d K=%d", BS, IPT, K);
    auto s = allocWarpScanLeavesScratch<int, BS, IPT, K>(N);
    BENCH(label, ([&]{ runWarpScanLeaves<int, BS, IPT, K>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
    freeWarpScanLeavesScratch<int, BS, IPT, K>(s);
}

template <int BS, int IPT, int K>
static void benchWSNT(const int* d_in, int* d_out, int N, long long bytes_rw) {
    char label[80];
    snprintf(label, sizeof(label), "WarpScanNoTree BS=%d IPT=%d K=%d", BS, IPT, K);
    auto s = allocWarpScanNoTreeScratch<int, BS, IPT, K>(N);
    BENCH(label, ([&]{ runWarpScanNoTree<int, BS, IPT, K>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
    freeWarpScanNoTreeScratch<int, BS, IPT, K>(s);
}

template <int BS, int IPT, int K, int G>
static void benchWSNT2L(const int* d_in, int* d_out, int N, long long bytes_rw) {
    char label[80];
    snprintf(label, sizeof(label), "NoTree2L BS=%d IPT=%d K=%d G=%d", BS, IPT, K, G);
    auto s = allocWarpScanNoTree2LScratch<int, BS, IPT, K, G>(N);
    BENCH(label, ([&]{ runWarpScanNoTree2L<int, BS, IPT, K, G>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
    freeWarpScanNoTree2LScratch<int, BS, IPT, K, G>(s);
}

template <int BS, int IPT, int K>
static void benchWMH(const int* d_in, int* d_out, int N, long long bytes_rw) {
    char label[80];
    snprintf(label, sizeof(label), "WarpMinHierarchy BS=%d IPT=%d K=%d", BS, IPT, K);
    auto s = allocWarpMinHierarchyScratch<int, BS, IPT, K>(N);
    BENCH(label, ([&]{ runWarpMinHierarchy<int, BS, IPT, K>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
    freeWarpMinHierarchyScratch<int, BS, IPT, K>(s);
}

template <int BS, int IPT, int K>
static void benchWCL(const int* d_in, int* d_out, int N, long long bytes_rw) {
    char label[80];
    snprintf(label, sizeof(label), "WarpCoopLeaf BS=%d IPT=%d K=%d", BS, IPT, K);
    auto s = allocWarpCoopLeafScratch<int, BS, IPT, K>(N);
    BENCH(label, ([&]{ runWarpCoopLeaf<int, BS, IPT, K>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
    freeWarpCoopLeafScratch<int, BS, IPT, K>(s);
}

int main() {
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const int N = 128 * 1024 * 1024 / sizeof(int);
    long long bytes_rw = (long long)N * sizeof(int) * 3;

    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  (size_t)N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)N * sizeof(int)));

    {
        std::vector<int> h(N);
        srand(42);
        for (int i = 0; i < N; i++) h[i] = rand();
        gpuAssert(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    }

    printf("=== Correctness check ===\n");
    const int NC = 8192;
    int *dc_in, *dc_out;
    gpuAssert(cudaMalloc(&dc_in,  NC * sizeof(int)));
    gpuAssert(cudaMalloc(&dc_out, NC * sizeof(int)));
    quickCheck([](int* di, int* dou, int n){ launchWarpScanLeaves<int,128,2,8>(di,dou,n); },
               "WarpScanLeaves IPT=2 K=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanLeaves<int,128,4,8>(di,dou,n); },
               "WarpScanLeaves IPT=4 K=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,2,8>(di,dou,n); },
               "WarpScanNoTree IPT=2 K=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,4,8>(di,dou,n); },
               "WarpScanNoTree IPT=4 K=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,4,32>(di,dou,n); },
               "WarpScanNoTree IPT=4 K=32", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,4,64>(di,dou,n); },
               "WarpScanNoTree IPT=4 K=64", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,4,128>(di,dou,n); },
               "WarpScanNoTree IPT=4 K=128", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,4,256>(di,dou,n); },
               "WarpScanNoTree IPT=4 K=256", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,8,8>(di,dou,n); },
               "WarpScanNoTree IPT=8 K=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,128,8,64>(di,dou,n); },
               "WarpScanNoTree IPT=8 K=64", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree<int,256,2,64>(di,dou,n); },
               "WarpScanNoTree BS=256 IPT=2 K=64", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree2L<int,128,4,16,16>(di,dou,n); },
               "NoTree2L K=16 G=16", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree2L<int,128,4,32,32>(di,dou,n); },
               "NoTree2L K=32 G=32", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpScanNoTree2L<int,128,4,64,64>(di,dou,n); },
               "NoTree2L K=64 G=64", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpCoopLeaf<int,128,4,64>(di,dou,n); },
               "WarpCoopLeaf IPT=4 K=64", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpMinHierarchy<int,128,4,8>(di,dou,n); },
               "WarpMinHierarchy IPT=4 K=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpMinHierarchy<int,128,4,64>(di,dou,n); },
               "WarpMinHierarchy IPT=4 K=64", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWarpMinHierarchy<int,128,4,128>(di,dou,n); },
               "WarpMinHierarchy IPT=4 K=128", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchWSTL<int,128,4>(di,dou,n); },
               "WSTL IPT=4", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPT<int,128,4>(di,dou,n); },
               "SPT BS=128 IPT=4", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPTAtomic<int,128,4>(di,dou,n); },
               "SPTAtomic BS=128 IPT=4", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPT<int,64,4>(di,dou,n); },
               "SPT BS=64 IPT=4", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPT<int,64,2>(di,dou,n); },
               "SPT BS=64 IPT=2", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPT<int,64,8>(di,dou,n); },
               "SPT BS=64 IPT=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPT<int,128,2>(di,dou,n); },
               "SPT BS=128 IPT=2", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPT<int,128,8>(di,dou,n); },
               "SPT BS=128 IPT=8", dc_in, dc_out, NC);
    quickCheck([](int* di, int* dou, int n){ launchSPT<int,128,16>(di,dou,n); },
               "SPT BS=128 IPT=16", dc_in, dc_out, NC);
    cudaFree(dc_in); cudaFree(dc_out);

    printf("\n=== WarpScanLeaves K sweep at BS=128 IPT=4 ===\n");
    benchWSL<128, 4,  4>(d_in, d_out, N, bytes_rw);
    benchWSL<128, 4,  6>(d_in, d_out, N, bytes_rw);
    benchWSL<128, 4,  8>(d_in, d_out, N, bytes_rw);
    benchWSL<128, 4, 10>(d_in, d_out, N, bytes_rw);
    benchWSL<128, 4, 12>(d_in, d_out, N, bytes_rw);
    benchWSL<128, 4, 16>(d_in, d_out, N, bytes_rw);

    printf("\n=== WarpScanNoTree K sweep at BS=128 IPT=4 ===\n");
    benchWSNT<128, 4,  8>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4, 32>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4, 64>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4,128>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4,256>(d_in, d_out, N, bytes_rw);

    printf("\n=== WarpScanNoTree IPT=8 K sweep (B=1024, fewer serial steps) ===\n");
    benchWSNT<128, 8,  8>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 8, 32>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 8, 64>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 8,128>(d_in, d_out, N, bytes_rw);

    printf("\n=== WarpScanNoTree IPT=2 K sweep (B=256, reference) ===\n");
    benchWSNT<128, 2,  8>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 2, 64>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 2,128>(d_in, d_out, N, bytes_rw);

    printf("\n=== Approach A: Two-level SB hierarchy (NoTree2L) ===\n");
    benchWSNT2L<128, 4, 16, 16>(d_in, d_out, N, bytes_rw);
    benchWSNT2L<128, 4, 32, 32>(d_in, d_out, N, bytes_rw);
    benchWSNT2L<128, 4, 64, 64>(d_in, d_out, N, bytes_rw);

    printf("\n=== Approach B: Static assign + warp-coop leaf scan (WarpCoopLeaf) ===\n");
    benchWCL<128, 4,  8>(d_in, d_out, N, bytes_rw);
    benchWCL<128, 4, 32>(d_in, d_out, N, bytes_rw);
    benchWCL<128, 4, 64>(d_in, d_out, N, bytes_rw);

    printf("\n=== WarpScanNoTree BS=256 IPT=2 (B=512, same tile, more threads) ===\n");
    benchWSNT<256, 2, 32>(d_in, d_out, N, bytes_rw);
    benchWSNT<256, 2, 64>(d_in, d_out, N, bytes_rw);
    benchWSNT<256, 2,128>(d_in, d_out, N, bytes_rw);

    printf("\n=== Fine comparison: WSNT IPT=4 K=64 vs K=128 (10 iters each) ===\n");
    {
        auto s64  = allocWarpScanNoTreeScratch<int,128,4, 64>(N);
        auto s128 = allocWarpScanNoTreeScratch<int,128,4,128>(N);
        BENCH("WarpScanNoTree IPT=4 K=64",
              ([&]{ runWarpScanNoTree<int,128,4, 64>(d_in, d_out, N, s64);  }), bytes_rw, 3, 10);
        BENCH("WarpScanNoTree IPT=4 K=128",
              ([&]{ runWarpScanNoTree<int,128,4,128>(d_in, d_out, N, s128); }), bytes_rw, 3, 10);
        freeWarpScanNoTreeScratch<int,128,4, 64>(s64);
        freeWarpScanNoTreeScratch<int,128,4,128>(s128);
    }
    printf("\n=== Reference: WSL IPT=2 K=16 and WSNT IPT=4 K=128 ===\n");
    benchWSL<128, 2, 16>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4,128>(d_in, d_out, N, bytes_rw);

    printf("\n=== WarpMinHierarchy K sweep at BS=128 IPT=4 ===\n");
    benchWMH<128, 4,  8>(d_in, d_out, N, bytes_rw);
    benchWMH<128, 4, 32>(d_in, d_out, N, bytes_rw);
    benchWMH<128, 4, 64>(d_in, d_out, N, bytes_rw);
    benchWMH<128, 4,128>(d_in, d_out, N, bytes_rw);
    benchWMH<128, 4,256>(d_in, d_out, N, bytes_rw);

    printf("\n=== WSTL (two-pass: intra-block + tree lookup) ===\n");
    {
        auto s = allocWSTLScratch<int,128,4>(N);
        BENCH("WSTL IPT=4 B=512",
              ([&]{ runWSTL<int,128,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeWSTLScratch<int,128,4>(s);
    }

    printf("\n=== SPT BS/IPT sweep (cooperative: grid.sync tree build) ===\n");
    {
        // BS=128, IPT=4 (baseline: 192 physical blocks, B=512)
        {
            auto s = allocSPTScratch<int,128,4>(N);
            printf("  SPT BS=128 IPT=4: %d phys blocks, B=512\n", s.num_phys);
            BENCH("SPT BS=128 IPT=4 B=512",
                  ([&]{ runSPT<int,128,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
            freeSPTScratch<int,128,4>(s);
        }
        // BS=64, IPT=4 (384 physical blocks, B=256) — 16 blocks/SM vs 8
        {
            auto s = allocSPTScratch<int,64,4>(N);
            printf("  SPT BS=64 IPT=4: %d phys blocks, B=256\n", s.num_phys);
            BENCH("SPT BS=64 IPT=4 B=256",
                  ([&]{ runSPT<int,64,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
            freeSPTScratch<int,64,4>(s);
        }
        // BS=64, IPT=2 (384 physical blocks, B=128)
        {
            auto s = allocSPTScratch<int,64,2>(N);
            printf("  SPT BS=64 IPT=2: %d phys blocks, B=128\n", s.num_phys);
            BENCH("SPT BS=64 IPT=2 B=128",
                  ([&]{ runSPT<int,64,2>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
            freeSPTScratch<int,64,2>(s);
        }
        // BS=64, IPT=8 (384 physical blocks, B=512)
        {
            auto s = allocSPTScratch<int,64,8>(N);
            printf("  SPT BS=64 IPT=8: %d phys blocks, B=512\n", s.num_phys);
            BENCH("SPT BS=64 IPT=8 B=512",
                  ([&]{ runSPT<int,64,8>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
            freeSPTScratch<int,64,8>(s);
        }
        // BS=128, IPT=2 (192 physical blocks, B=256)
        {
            auto s = allocSPTScratch<int,128,2>(N);
            printf("  SPT BS=128 IPT=2: %d phys blocks, B=256\n", s.num_phys);
            BENCH("SPT BS=128 IPT=2 B=256",
                  ([&]{ runSPT<int,128,2>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
            freeSPTScratch<int,128,2>(s);
        }
        // BS=128, IPT=8 (192 physical blocks, B=1024) — 2x fewer logical blocks
        {
            auto s = allocSPTScratch<int,128,8>(N);
            printf("  SPT BS=128 IPT=8: %d phys blocks, B=1024\n", s.num_phys);
            BENCH("SPT BS=128 IPT=8 B=1024",
                  ([&]{ runSPT<int,128,8>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
            freeSPTScratch<int,128,8>(s);
        }
        // BS=128, IPT=16 (192 physical blocks, B=2048) — 4x fewer logical blocks
        {
            auto s = allocSPTScratch<int,128,16>(N);
            printf("  SPT BS=128 IPT=16: %d phys blocks, B=2048\n", s.num_phys);
            BENCH("SPT BS=128 IPT=16 B=2048",
                  ([&]{ runSPT<int,128,16>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
            freeSPTScratch<int,128,16>(s);
        }
    }

    printf("\n=== SPTAtomic (atomicMin tree build, no Phase 2 grid.syncs) ===\n");
    {
        auto s = allocSPTScratch<int,128,4>(N);
        int bps = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, apsepKernelSPTAtomic<int,128,4>, 128, 0);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
        printf("  SPTAtomic BS=128 IPT=4: %d phys blocks, B=512\n", bps * prop.multiProcessorCount);
        BENCH("SPTAtomic BS=128 IPT=4 B=512",
              ([&]{ runSPTAtomic<int,128,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,128,4>(s);
    }

    printf("\n=== BSZ two-stage ===\n");
    {
        // Correctness check first
        int *dc_bsz_in, *dc_bsz_out;
        gpuAssert(cudaMalloc(&dc_bsz_in,  8192 * sizeof(int)));
        gpuAssert(cudaMalloc(&dc_bsz_out, 8192 * sizeof(int)));
        quickCheck([](int* di, int* dou, int n){ launchBSZ<int,128,4>(di,dou,n); },
                   "BSZ B=512", dc_bsz_in, dc_bsz_out, 8192);
        cudaFree(dc_bsz_in); cudaFree(dc_bsz_out);
        // Benchmark
        auto s = allocBSZScratch<int,128,4>(N);
        char label[80];
        snprintf(label, sizeof(label), "BSZ two-stage B=512");
        BENCH(label, ([&]{ runBSZ<int,128,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeBSZScratch<int,128,4>(s);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
