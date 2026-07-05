// bench_spt_le.cu — correctness + timing for the PSE(<=) variant of SPT
// (apsepKernelSPT<..., INCL=true>), added for the alpacc LLP parser backend
// (bracket matching and parent vectors need previous smaller-OR-EQUAL).
//
// Checks the <= variant against a CPU stack reference, re-checks the strict
// variant against a CPU reference (the comparator refactor must not change
// it), and times both variants (expected identical: the predicate is a
// compile-time constant).
#include "apsep.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define gpuAssert2(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// CPU reference: nearest j < i with (le ? a[j] <= a[i] : a[j] < a[i]), else -1.
static void cpuPSE(const std::vector<int>& a, int n, bool le, std::vector<int>& out) {
    std::vector<int> stk;   // indices with strictly... maintained per semantics
    stk.clear();
    for (int i = 0; i < n; i++) {
        // pop while stack top does NOT qualify as answer
        while (!stk.empty() &&
               !(le ? (a[stk.back()] <= a[i]) : (a[stk.back()] < a[i])))
            stk.pop_back();
        out[i] = stk.empty() ? -1 : stk.back();
        stk.push_back(i);
    }
}

template <typename Fn>
static float timeKernel(Fn fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) { fn(); gpuAssert2(cudaDeviceSynchronize()); }
    cudaEvent_t t0, t1;
    gpuAssert2(cudaEventCreate(&t0)); gpuAssert2(cudaEventCreate(&t1));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; i++) {
        gpuAssert2(cudaEventRecord(t0));
        fn();
        gpuAssert2(cudaEventRecord(t1));
        gpuAssert2(cudaEventSynchronize(t1));
        gpuAssert2(cudaEventElapsedTime(&ms[i], t0, t1));
    }
    gpuAssert2(cudaEventDestroy(t0)); gpuAssert2(cudaEventDestroy(t1));
    std::sort(ms.begin(), ms.end());
    return ms[iters / 2];
}

int main() {
    cudaDeviceProp prop;
    gpuAssert2(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    const int N = 32 * 1024 * 1024;
    const long long bytes_rw = 2LL * N * sizeof(int);

    std::vector<int> h(N), ref(N), got(N);
    int *d_in, *d_out;
    gpuAssert2(cudaMalloc(&d_in,  N * sizeof(int)));
    gpuAssert2(cudaMalloc(&d_out, N * sizeof(int)));

    auto s_lt = allocSPTScratch<int, 128, 4, false>(N);
    auto s_le = allocSPTScratch<int, 128, 4, true>(N);
    printf("num_phys: strict=%d, incl=%d\n\n", s_lt.num_phys, s_le.num_phys);

    bool ok = true;
    auto check = [&](const char* label, int n, bool le) {
        gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
        gpuAssert2(cudaMemset(d_out, 0xAB, (size_t)n * sizeof(int)));
        if (le) runSPT<int, 128, 4, true >(d_in, d_out, n, s_le);
        else    runSPT<int, 128, 4, false>(d_in, d_out, n, s_lt);
        gpuAssert2(cudaDeviceSynchronize());
        gpuAssert2(cudaMemcpy(got.data(), d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
        cpuPSE(h, n, le, ref);
        int bad = -1;
        for (int i = 0; i < n; i++)
            if (ref[i] != got[i]) { bad = i; break; }
        if (bad >= 0)
            printf("  [%-4s %-10s n=%9d] MISMATCH at %d: ref=%d got=%d (val=%d)\n",
                   le ? "<=" : "<", label, n, bad, ref[bad], got[bad], h[bad]);
        else
            printf("  [%-4s %-10s n=%9d] PASS\n", le ? "<=" : "<", label, n);
        ok &= (bad < 0);
    };

    auto fill_and_check = [&](const char* label) {
        for (int n : {N, 1000003, 513, 512, 511, 1}) {
            check(label, n, true);
            check(label, n, false);
        }
    };

    printf("=== Correctness (vs CPU stack reference) ===\n");
    srand(1234);
    for (int i = 0; i < N; i++) h[i] = rand();
    fill_and_check("random");

    srand(99);
    for (int i = 0; i < N; i++) h[i] = rand() % 16;
    fill_and_check("dup16");

    srand(7);
    for (int i = 0; i < N; i++) h[i] = rand() % 2;
    fill_and_check("dup2");

    for (int i = 0; i < N; i++) h[i] = 42;
    fill_and_check("all-equal");

    for (int i = 0; i < N; i++) h[i] = N - i;
    fill_and_check("descending");

    for (int i = 0; i < N; i++) h[i] = i;
    fill_and_check("ascending");

    // INT_MAX stress: INF-valued queries exercise the lane<W ballot guard.
    srand(5);
    for (int i = 0; i < N; i++) h[i] = (rand() % 4 == 0) ? 2147483647 : rand();
    fill_and_check("intmax");

    if (!ok) { printf("\nCORRECTNESS FAILED — not benchmarking.\n"); return 1; }

    printf("\n=== Timing (N=32M, median of 7) ===\n");
    auto bench = [&](const char* label) {
        gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
        float ms_lt = timeKernel([&]{ runSPT<int,128,4,false>(d_in, d_out, N, s_lt); }, 2, 7);
        float ms_le = timeKernel([&]{ runSPT<int,128,4,true >(d_in, d_out, N, s_le); }, 2, 7);
        printf("  %-11s  strict %7.0f µs (%.1f GB/s)   incl %7.0f µs (%.1f GB/s)\n",
               label,
               ms_lt * 1000, bytes_rw / (ms_lt * 1e-3) / 1e9,
               ms_le * 1000, bytes_rw / (ms_le * 1e-3) / 1e9);
    };

    srand(1234);
    for (int i = 0; i < N; i++) h[i] = rand();
    bench("random");

    srand(99);
    for (int i = 0; i < N; i++) h[i] = rand() % 16;
    bench("dup16");

    for (int i = 0; i < N; i++) h[i] = N - i;
    bench("descending");

    for (int i = 0; i < N; i++) h[i] = i;
    bench("ascending");

    cudaFree(d_in); cudaFree(d_out);
    freeSPTScratch<int,128,4>(s_lt);
    freeSPTScratch<int,128,4>(s_le);
    return 0;
}
