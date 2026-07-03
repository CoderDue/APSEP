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
    benchWSNT<128, 4, 16>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4, 32>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4, 64>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4,128>(d_in, d_out, N, bytes_rw);
    benchWSNT<128, 4,256>(d_in, d_out, N, bytes_rw);

    printf("\n=== Reference: WSL IPT=2 K=16 ===\n");
    benchWSL<128, 2, 16>(d_in, d_out, N, bytes_rw);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
