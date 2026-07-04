// bench_spt_worst.cu — SPT BS/IPT sweep on worst-case (descending) input
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

static void computeStats(std::vector<float>& ms, long long bytes) {
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
    printf("  %-45s", label);                                                \
    computeStats(_ms, bytes);                                                \
} while(0)

int main() {
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const int N = 32 * 1024 * 1024;
    const long long bytes_rw = 2LL * N * sizeof(int);

    std::vector<int> h(N);
    // Descending: worst case (all PSE = -1)
    for (int i = 0; i < N; i++) h[i] = N - i;

    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  (size_t)N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)N * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    printf("=== Worst-case (descending N=%d) SPT BS/IPT sweep ===\n\n", N);

    // WSTL reference
    {
        auto s = allocWSTLScratch<int,128,4>(N);
        BENCH("WSTL BS=128 IPT=4 (reference)",
              ([&]{ runWSTL<int,128,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeWSTLScratch<int,128,4>(s);
    }

    // SPT variants
    {
        auto s = allocSPTScratch<int,128,4>(N);
        printf("  [SPT BS=128 IPT=4: %d phys blocks, B=512]\n", s.num_phys);
        BENCH("SPT BS=128 IPT=4 B=512",
              ([&]{ runSPT<int,128,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,128,4>(s);
    }
    {
        auto s = allocSPTScratch<int,64,4>(N);
        printf("  [SPT BS=64 IPT=4: %d phys blocks, B=256]\n", s.num_phys);
        BENCH("SPT BS=64 IPT=4 B=256",
              ([&]{ runSPT<int,64,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,64,4>(s);
    }
    {
        auto s = allocSPTScratch<int,128,8>(N);
        printf("  [SPT BS=128 IPT=8: %d phys blocks, B=1024]\n", s.num_phys);
        BENCH("SPT BS=128 IPT=8 B=1024",
              ([&]{ runSPT<int,128,8>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,128,8>(s);
    }
    {
        auto s = allocSPTScratch<int,64,8>(N);
        printf("  [SPT BS=64 IPT=8: %d phys blocks, B=512]\n", s.num_phys);
        BENCH("SPT BS=64 IPT=8 B=512",
              ([&]{ runSPT<int,64,8>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,64,8>(s);
    }
    {
        auto s = allocSPTScratch<int,128,2>(N);
        printf("  [SPT BS=128 IPT=2: %d phys blocks, B=256]\n", s.num_phys);
        BENCH("SPT BS=128 IPT=2 B=256",
              ([&]{ runSPT<int,128,2>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,128,2>(s);
    }
    {
        auto s = allocSPTScratch<int,64,2>(N);
        printf("  [SPT BS=64 IPT=2: %d phys blocks, B=128]\n", s.num_phys);
        BENCH("SPT BS=64 IPT=2 B=128",
              ([&]{ runSPT<int,64,2>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,64,2>(s);
    }

    // SPTAtomic
    {
        auto s = allocSPTScratch<int,128,4>(N);
        int bps = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, apsepKernelSPTAtomic<int,128,4>, 128, 0);
        cudaDeviceProp prop2; cudaGetDeviceProperties(&prop2, 0);
        printf("  [SPTAtomic BS=128 IPT=4: %d phys blocks]\n", bps * prop2.multiProcessorCount);
        BENCH("SPTAtomic BS=128 IPT=4 B=512",
              ([&]{ runSPTAtomic<int,128,4>(d_in, d_out, N, s); }), bytes_rw, 2, 7);
        freeSPTScratch<int,128,4>(s);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
