// Minimal binary for NCU profiling of apsepKernelWarpScanNoTree.
// Run: ncu --set full -o wsnt_random ./profile_wsnt random
//      ncu --set full -o wsnt_desc   ./profile_wsnt desc

#include "apsep.cuh"
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "random";

    constexpr int N = 32 * 1024 * 1024;  // 32M ints = 128 MiB

    std::vector<int> h(N);
    if (strcmp(mode, "desc") == 0) {
        for (int i = 0; i < N; i++) h[i] = N - i;
        printf("Input: descending (worst case)\n");
    } else {
        std::mt19937 rng(0xBEEF);
        std::uniform_int_distribution<int> dist(0, N);
        for (auto& v : h) v = dist(rng);
        printf("Input: uniform random\n");
    }

    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  (size_t)N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)N * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    // WarpScanNoTree BS=128, IPT=4, K=64
    auto s = allocWarpScanNoTreeScratch<int, 128, 4, 64>(N);
    runWarpScanNoTree<int, 128, 4, 64>(d_in, d_out, N, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWarpScanNoTreeScratch<int, 128, 4, 64>(s);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
