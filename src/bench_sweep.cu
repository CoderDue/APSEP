// bench_sweep.cu — sweep K values for WarpScanLeaves kernel
#include "apsep.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>

#define gpuAssert(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

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
    printf("  %-45s  ", label);                                              \
    computeDescriptors(_ms, bytes);                                          \
} while(0)

template <int K>
static void benchWSL(const int* d_in, int* d_out, int N, long long bytes_rw) {
    char label[64];
    snprintf(label, sizeof(label), "WarpScanLeaves K=%d", K);
    auto s = allocWarpScanLeavesScratch<int, 128, 2, K>(N);
    BENCH(label,
          ([&]{ runWarpScanLeaves<int, 128, 2, K>(d_in, d_out, N, s); }),
          bytes_rw, 2, 5);
    freeWarpScanLeavesScratch<int, 128, 2, K>(s);
}

template <int K>
static void benchBaseline(const int* d_in, int* d_out, int N, long long bytes_rw) {
    char label[64];
    snprintf(label, sizeof(label), "Baseline (apsepKernel) K=%d", K);
    auto s = allocApsepScratch<int, 128, 2, K>(N);
    BENCH(label,
          ([&]{ runApsep<int, 128, 2, K>(d_in, d_out, N, s); }),
          bytes_rw, 2, 5);
    freeApsepScratch<int, 128, 2, K>(s);
}

int main() {
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const int N = 128 * 1024 * 1024 / sizeof(int);  // 128 MiB
    long long bytes_rw = (long long)N * sizeof(int) * 3;  // read + 2*write (approx)

    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  (size_t)N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)N * sizeof(int)));

    // Fill with random data
    {
        std::vector<int> h(N);
        srand(42);
        for (int i = 0; i < N; i++) h[i] = rand();
        gpuAssert(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    }

    printf("=== Baseline K sweep (BS=128, IPT=2) ===\n");
    benchBaseline<1> (d_in, d_out, N, bytes_rw);
    benchBaseline<2> (d_in, d_out, N, bytes_rw);
    benchBaseline<4> (d_in, d_out, N, bytes_rw);
    benchBaseline<8> (d_in, d_out, N, bytes_rw);
    benchBaseline<16>(d_in, d_out, N, bytes_rw);
    benchBaseline<32>(d_in, d_out, N, bytes_rw);

    printf("\n=== WarpScanLeaves K sweep (BS=128, IPT=2) ===\n");
    benchWSL<1> (d_in, d_out, N, bytes_rw);
    benchWSL<2> (d_in, d_out, N, bytes_rw);
    benchWSL<4> (d_in, d_out, N, bytes_rw);
    benchWSL<8> (d_in, d_out, N, bytes_rw);
    benchWSL<16>(d_in, d_out, N, bytes_rw);
    benchWSL<32>(d_in, d_out, N, bytes_rw);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
