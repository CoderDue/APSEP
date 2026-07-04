// Minimal profiling target for WSTL and SPT kernels.
// Usage: ./profile_bottleneck <wstl|spt> <random|desc|asc>
#include "apsep.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    const char* kernel = (argc > 1) ? argv[1] : "wstl";
    const char* input  = (argc > 2) ? argv[2] : "random";

    const int N = 32 * 1024 * 1024;

    std::vector<int> h(N);
    if (strcmp(input, "desc") == 0)
        for (int i = 0; i < N; i++) h[i] = N - i;
    else if (strcmp(input, "asc") == 0)
        for (int i = 0; i < N; i++) h[i] = i;
    else
        for (int i = 0; i < N; i++) h[i] = rand();

    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, N * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Warmup
    if (strcmp(kernel, "spt") == 0) {
        auto s = allocSPTScratch<int,128,4>(N);
        runSPT<int,128,4>(d_in, d_out, N, s);
        gpuAssert(cudaDeviceSynchronize());
        // Timed run (profiler captures this)
        runSPT<int,128,4>(d_in, d_out, N, s);
        gpuAssert(cudaDeviceSynchronize());
        freeSPTScratch<int,128,4>(s);
    } else {
        auto s = allocWSTLScratch<int,128,4>(N);
        runWSTL<int,128,4>(d_in, d_out, N, s);
        gpuAssert(cudaDeviceSynchronize());
        runWSTL<int,128,4>(d_in, d_out, N, s);
        gpuAssert(cudaDeviceSynchronize());
        freeWSTLScratch<int,128,4>(s);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    printf("done kernel=%s input=%s\n", kernel, input);
    return 0;
}
