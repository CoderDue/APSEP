// bench_spt_warpauto.cu — prototype of SPT with warp-autonomous blocks.
//
// Hypothesis: 21% of random-input warp stalls are on the one __syncthreads
// per Phase 1 block iteration (needed because the cross-warp fallback reads
// other warps' shared data). Making each *warp* own its logical block
// (B = 32*IPT = 128 elements) removes shared memory and every __syncthreads:
// elements stay in registers, the spine-walk exact scan fetches the target
// thread's 4 elements by shuffle, and the cross-warp fallback disappears
// (those elements go to the bitmask and resolve in Phase 3 instead).
//
// Costs to watch: 4x more logical blocks -> the min-tree grows to ~2 MB
// (exceeds the 1.5 MB L2) and random input has ~3x more genuine Phase 3
// lookups (unresolved elements are block prefix minima: ~H(128) per 128
// elements vs ~H(512) per 512).
//
// Bitmask permutation is unchanged (chunks are still 8 threads x 4 slots):
// mybit = 8*(lane&3) + (lane>>2).
//
// This binary checks the prototype against runSPT output and times both.
#include "apsep.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

namespace cg = cooperative_groups;

template <typename T, int BLOCK_SIZE, int IPT>
__global__
void apsepKernelSPTWarpAuto(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,   // ceil(n / (32*IPT))
        int                     M,
        int                     leaf_offset,
        unsigned* __restrict__  d_unres,
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins,
        T* __restrict__         d_tree,
        T* __restrict__         d_prefix_min)
{
    cg::grid_group grid = cg::this_grid();

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int B         = 32 * IPT;   // elements per *warp* block
    constexpr int W         = B / 32;     // chunk-mins per block (= IPT)
    const     T   INF       = ApsepInfinity<T>::value();

    const int phys_bid   = (int)blockIdx.x;
    const int num_phys   = (int)gridDim.x;
    const int lane       = threadIdx.x & 31;
    const int warp_id    = threadIdx.x >> 5;
    const int gw          = phys_bid * NUM_WARPS + warp_id;
    const int warps_total = num_phys * NUM_WARPS;

    static_assert(IPT == 4, "assumes IPT == 4");
    static_assert(sizeof(T) == 4, "assumes 4-byte T (int4 loads)");

    // ---- Phase 1: warp-autonomous, no shared memory, no __syncthreads ----
    for (int block_id = gw; block_id < num_blocks; block_id += warps_total) {
        const int glb_offs = block_id * B;
        const int tbase    = IPT * lane;   // block-relative
        const bool full    = (glb_offs + B <= n);

        T v[IPT];
        if (full) {
            int4 raw = *reinterpret_cast<const int4*>(d_in + glb_offs + tbase);
            v[0] = raw.x; v[1] = raw.y; v[2] = raw.z; v[3] = raw.w;
        } else {
            #pragma unroll
            for (int i = 0; i < IPT; i++) {
                int gid = glb_offs + tbase + i;
                v[i] = (gid < n) ? d_in[gid] : INF;
            }
        }

        // In-thread sequential ANSV (block-relative result, -1 if none)
        int res[IPT];
        res[0] = -1;
        res[1] = (v[0] < v[1]) ? tbase     : -1;
        res[2] = (v[1] < v[2]) ? tbase + 1 : (v[0] < v[2]) ? tbase : -1;
        res[3] = (v[2] < v[3]) ? tbase + 2 : (v[1] < v[3]) ? tbase + 1
                                           : (v[0] < v[3]) ? tbase : -1;

        const T tmin = min(min(v[0], v[1]), min(v[2], v[3]));

        // Warp inclusive prefix-min over thread mins
        T c = tmin;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        const T carry = __shfl_up_sync(0xffffffff, c, 1);  // valid for lane>0

        // Chunk mins (32 consecutive elements = 8 threads) for Phase 3
        {
            T om = tmin;
            om = min(om, __shfl_xor_sync(0xffffffff, om, 1));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 2));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 4));
            if ((lane & 7) == 0)
                d_block_warp_mins[(size_t)block_id * W + (lane >> 3)] = om;
        }

        // Thread-level ANSV chain over tmins (pointer jumping)
        int chain = -1;
        {
            bool has = (lane > 0) && (carry < tmin);
            if (__any_sync(0xffffffff, has)) {
                int g = has ? lane - 1 : -1;
                while (true) {
                    int src = (g >= 0) ? g : 0;
                    T   tg  = __shfl_sync(0xffffffff, tmin, src);
                    int gg  = __shfl_sync(0xffffffff, g,    src);
                    bool jump = (g >= 0) && (tg >= tmin);
                    if (!__any_sync(0xffffffff, jump)) break;
                    if (jump) g = gg;
                }
                chain = g;
            }
        }

        // Per-element resolution; spine-walk cursor is monotone across i.
        // The exact scan fetches the found thread's elements by shuffle
        // (all lanes participate; inactive lanes use src 0).
        unsigned bal[IPT];
        int cur = lane - 1;
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            const bool active = full || (glb_offs + tbase + i < n);
            const T val = v[i];
            bool pending = active && (res[i] < 0) && (lane > 0) && (carry < val);

            if (__any_sync(0xffffffff, pending)) {
                while (true) {
                    int src = (pending && cur >= 0) ? cur : 0;
                    T   tg  = __shfl_sync(0xffffffff, tmin,  src);
                    int cg  = __shfl_sync(0xffffffff, chain, src);
                    bool step = pending && (cur >= 0) && (tg >= val);
                    if (!__any_sync(0xffffffff, step)) break;
                    if (step) cur = cg;
                }
                const int src = pending ? cur : 0;
                const T e0 = __shfl_sync(0xffffffff, v[0], src);
                const T e1 = __shfl_sync(0xffffffff, v[1], src);
                const T e2 = __shfl_sync(0xffffffff, v[2], src);
                const T e3 = __shfl_sync(0xffffffff, v[3], src);
                if (pending) {
                    int k = (e3 < val) ? 3 : (e2 < val) ? 2 : (e1 < val) ? 1 : 0;
                    res[i] = IPT * cur + k;
                }
            }
            // No cross-warp fallback: anything unresolved here goes to Phase 3.

            if (active && res[i] >= 0) d_out[glb_offs + tbase + i] = glb_offs + res[i];
            bal[i] = __ballot_sync(0xffffffff, active && res[i] < 0);
        }

        // Publish unresolved bits (4 words per warp-block, same permutation
        // as production: element 4j+i of a chunk maps to bit 8i+j).
        if (lane < W) {
            unsigned word = ((bal[0] >> (8 * lane)) & 0xffu)
                          | (((bal[1] >> (8 * lane)) & 0xffu) << 8)
                          | (((bal[2] >> (8 * lane)) & 0xffu) << 16)
                          | (((bal[3] >> (8 * lane)) & 0xffu) << 24);
            d_unres[(unsigned)glb_offs / 32 + lane] = word;
        }

        if (lane == 31) d_block_mins[block_id] = c;   // warp min = block min
    }

    grid.sync();

    // ---- Phase 2 (identical to apsepKernelSPT) ----
    for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < M; i += num_phys * BLOCK_SIZE)
        d_tree[leaf_offset + i] = (i < num_blocks) ? d_block_mins[i] : INF;

    {
        int level_size  = M / 2;
        int level_start = M / 2 - 1;
        while (level_size > 0) {
            grid.sync();
            for (int i = phys_bid * BLOCK_SIZE + threadIdx.x;
                 i < level_size;
                 i += num_phys * BLOCK_SIZE) {
                int node = level_start + i;
                d_tree[node] = min(d_tree[2 * node + 1], d_tree[2 * node + 2]);
            }
            level_size  >>= 1;
            level_start  = (level_start - 1) / 2;
        }
    }

    grid.sync();

    for (int b = phys_bid * BLOCK_SIZE + threadIdx.x; b < num_blocks; b += num_phys * BLOCK_SIZE) {
        T pm = INF;
        int node = leaf_offset + b;
        while (node > 0) {
            if (node % 2 == 0)
                pm = min(pm, __ldg(&d_tree[node - 1]));
            node = (node - 1) / 2;
        }
        d_prefix_min[b] = pm;
    }
    grid.sync();

    // ---- Phase 3 (identical to apsepKernelSPT; W is 4 here) ----
    const int num_words = (n + 31) >> 5;
    const int mybit     = 8 * (lane & 3) + (lane >> 2);

    for (int wbase = gw * 32; wbase < num_words; wbase += warps_total * 32) {
        const int wl = wbase + lane;
        const unsigned um_l = (wl < num_words) ? __ldg(&d_unres[wl]) : 0u;

        const int  bid_l = min(wl / W, num_blocks - 1);
        const T    pm_l  = __ldg(&d_prefix_min[bid_l]);
        const bool eo_l  = pm_l >= __ldg(&d_in[(size_t)bid_l * B]);

        if (__all_sync(0xffffffff, eo_l && um_l == 0xffffffffu)) {
            int4* out4 = reinterpret_cast<int4*>(d_out) + (size_t)wbase * 8 + lane;
            const int4 m1 = make_int4(-1, -1, -1, -1);
            #pragma unroll
            for (int i = 0; i < 8; i++)
                out4[i * 32] = m1;
            continue;
        }

        unsigned nz = __ballot_sync(0xffffffff, um_l != 0);

        while (nz) {
            const int j = __ffs(nz) - 1;
            nz &= nz - 1;
            const unsigned um = __shfl_sync(0xffffffff, um_l, j);
            const int w = wbase + j;

            const int  block_id     = __shfl_sync(0xffffffff, bid_l, j);
            const T    prefix_min_b = __shfl_sync(0xffffffff, pm_l, j);
            const bool eo           = __shfl_sync(0xffffffff, (int)eo_l, j);
            const int  base         = w * 32;
            const bool mine         = (um >> mybit) & 1u;

            if (eo) {
                if (mine) d_out[base + lane] = -1;
                continue;
            }

            const int gid = base + lane;
            T val = mine ? __ldg(&d_in[gid]) : INF;
            const bool need = mine && (prefix_min_b < val);
            if (mine && !need) d_out[gid] = -1;

            unsigned pend = __ballot_sync(0xffffffff, need);
            while (pend) {
                const int bit = __ffs(pend) - 1;
                pend &= pend - 1;
                const T qval = __shfl_sync(0xffffffff, val, bit);

                int node = leaf_offset + block_id;
                int found_block = -1;
                while (node > 0) {
                    bool is_right = (node % 2 == 0);
                    if (is_right) {
                        int left_sib = node - 1;
                        if (__ldg(&d_tree[left_sib]) < qval) {
                            node = left_sib;
                            while (node < leaf_offset) {
                                int rc = 2 * node + 2;
                                node = (__ldg(&d_tree[rc]) < qval) ? rc : (2 * node + 1);
                            }
                            found_block = node - leaf_offset;
                            break;
                        }
                    }
                    node = (node - 1) / 2;
                }

                int result = -1;
                if (found_block >= 0) {
                    const T* wm = d_block_warp_mins + (size_t)found_block * W;
                    T wv = (lane < W) ? __ldg(&wm[lane]) : INF;
                    unsigned wmask = __ballot_sync(0xffffffff, wv < qval);
                    int wstar = 31 - __clz(wmask);
                    const T* bl = d_in + (size_t)found_block * B + wstar * 32;
                    unsigned lmask = __ballot_sync(0xffffffff, __ldg(&bl[lane]) < qval);
                    result = found_block * B + wstar * 32 + (31 - __clz(lmask));
                }
                if (lane == 0) d_out[base + bit] = result;
            }
        }
    }
}

#define gpuAssert2(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

struct WAScratch {
    unsigned* d_unres           = nullptr;
    int*      d_block_mins      = nullptr;
    int*      d_block_warp_mins = nullptr;
    int*      d_tree            = nullptr;
    int*      d_prefix_min      = nullptr;
    int       num_blocks = 0, M = 0, leaf_offset = 0;
};

static WAScratch allocWA(int n) {
    constexpr int B = 32 * 4;
    constexpr int W = B / 32;
    WAScratch s;
    s.num_blocks  = (n + B - 1) / B;
    s.M           = nextPow2(s.num_blocks);
    s.leaf_offset = s.M - 1;
    gpuAssert2(cudaMalloc(&s.d_unres,           (size_t)s.num_blocks * W * sizeof(unsigned)));
    gpuAssert2(cudaMalloc(&s.d_block_mins,      (size_t)s.num_blocks     * sizeof(int)));
    gpuAssert2(cudaMalloc(&s.d_block_warp_mins, (size_t)s.num_blocks * W * sizeof(int)));
    gpuAssert2(cudaMalloc(&s.d_tree,            (size_t)(2 * s.M - 1)    * sizeof(int)));
    gpuAssert2(cudaMalloc(&s.d_prefix_min,      (size_t)s.num_blocks     * sizeof(int)));
    return s;
}

static void freeWA(WAScratch& s) {
    cudaFree(s.d_unres); cudaFree(s.d_block_mins); cudaFree(s.d_block_warp_mins);
    cudaFree(s.d_tree);  cudaFree(s.d_prefix_min);
    s = WAScratch{};
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

    std::vector<int> h(N), out_ref(N), out_new(N);
    int *d_in, *d_out;
    gpuAssert2(cudaMalloc(&d_in,  N * sizeof(int)));
    gpuAssert2(cudaMalloc(&d_out, N * sizeof(int)));

    auto s  = allocSPTScratch<int,128,4>(N);
    auto sw = allocWA(N);

    int blk_per_sm = 0;
    gpuAssert2(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blk_per_sm, apsepKernelSPTWarpAuto<int,128,4>, 128, 0));
    const int new_num_phys = blk_per_sm * prop.multiProcessorCount;
    printf("SPT num_phys=%d, warpauto num_phys=%d (%d/SM)\n\n",
           s.num_phys, new_num_phys, blk_per_sm);

    auto runNew = [&](int n, WAScratch& sc) {
        void* args[] = {
            (void*)&d_in, (void*)&d_out, (void*)&n,
            (void*)&sc.num_blocks, (void*)&sc.M, (void*)&sc.leaf_offset,
            (void*)&sc.d_unres, (void*)&sc.d_block_mins,
            (void*)&sc.d_block_warp_mins, (void*)&sc.d_tree,
            (void*)&sc.d_prefix_min
        };
        gpuAssert2(cudaLaunchCooperativeKernel(
            (void*)apsepKernelSPTWarpAuto<int,128,4>, new_num_phys, 128, args, 0, nullptr));
    };

    auto check = [&](const char* label, int n) {
        gpuAssert2(cudaMemset(d_out, 0xAB, (size_t)N * sizeof(int)));
        runSPT<int,128,4>(d_in, d_out, n, s);
        gpuAssert2(cudaDeviceSynchronize());
        gpuAssert2(cudaMemcpy(out_ref.data(), d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

        auto s2 = allocWA(n);
        gpuAssert2(cudaMemset(d_out, 0xAB, (size_t)N * sizeof(int)));
        runNew(n, s2);
        gpuAssert2(cudaDeviceSynchronize());
        gpuAssert2(cudaMemcpy(out_new.data(), d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
        freeWA(s2);

        int bad = -1;
        for (int i = 0; i < n; i++)
            if (out_ref[i] != out_new[i]) { bad = i; break; }
        if (bad >= 0)
            printf("  [%s n=%d] MISMATCH at %d: ref=%d new=%d (val=%d)\n",
                   label, n, bad, out_ref[bad], out_new[bad], h[bad]);
        else
            printf("  [%s n=%d] PASS\n", label, n);
        return bad < 0;
    };

    bool ok = true;
    printf("=== Correctness (warpauto vs SPT reference) ===\n");
    srand(1234);
    for (int i = 0; i < N; i++) h[i] = rand();
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("random", N);
    ok &= check("random", 1000003);
    ok &= check("random", 129);
    ok &= check("random", 127);
    ok &= check("random", 128 * 3 + 17);
    ok &= check("random", 511);

    for (int i = 0; i < N; i++) h[i] = N - i;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("descending", N);
    ok &= check("descending", 1000003);

    for (int i = 0; i < N; i++) h[i] = i;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("ascending", N);

    srand(99);
    for (int i = 0; i < N; i++) h[i] = rand() % 16;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("dup16", N);

    if (!ok) { printf("\nCORRECTNESS FAILED — not benchmarking.\n"); return 1; }

    printf("\n=== Timing (N=32M, median of 7) ===\n");
    auto bench = [&](const char* label) {
        float ms_ref = timeKernel([&]{ runSPT<int,128,4>(d_in, d_out, N, s); }, 2, 7);
        float ms_new = timeKernel([&]{ runNew(N, sw); }, 2, 7);
        printf("  %-11s  SPT %7.0f µs (%.1f GB/s)   warpauto %7.0f µs (%.1f GB/s)\n",
               label,
               ms_ref * 1000, bytes_rw / (ms_ref * 1e-3) / 1e9,
               ms_new * 1000, bytes_rw / (ms_new * 1e-3) / 1e9);
    };

    srand(1234);
    for (int i = 0; i < N; i++) h[i] = rand();
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    bench("random");

    for (int i = 0; i < N; i++) h[i] = N - i;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    bench("descending");

    for (int i = 0; i < N; i++) h[i] = i;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    bench("ascending");

    cudaFree(d_in); cudaFree(d_out);
    freeSPTScratch<int,128,4>(s);
    freeWA(sw);
    return 0;
}
