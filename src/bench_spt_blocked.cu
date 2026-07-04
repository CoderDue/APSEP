// bench_spt_blocked.cu — prototype of a blocked-layout Phase 1 for SPT.
//
// Hypothesis: Phase 1 dominates SPT (67% of random runtime). The striped
// layout does a 5-shuffle warp scan per 32 elements. A blocked layout
// (thread owns IPT consecutive elements, int4 load) resolves ~half the
// elements sequentially in registers and needs only one warp scan per 128
// elements over per-thread mins.
//
// Phase 2 and Phase 3 are copied from apsepKernelSPT; only the d_unres bit
// layout changes (P1 ballots produce a lane-major permutation within each
// 32-element word: element 4j+i of a chunk maps to bit 8i+j).
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
void apsepKernelSPTBlocked(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     M,
        int                     leaf_offset,
        unsigned* __restrict__  d_unres,
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins,
        T* __restrict__         d_tree,
        T* __restrict__         d_prefix_min)
{
    cg::grid_group grid = cg::this_grid();

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W         = B / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    static_assert(IPT == 4, "blocked P1 prototype assumes IPT == 4");
    static_assert(sizeof(T) == 4, "blocked P1 prototype assumes 4-byte T");
    static_assert(W <= 32, "warp-cooperative Phase 3 assumes W <= 32");

    const int phys_bid   = (int)blockIdx.x;
    const int num_phys   = (int)gridDim.x;
    const int lane       = threadIdx.x & 31;
    const int warp_id    = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_tmin[BLOCK_SIZE];
    __shared__ T s_warp_min[NUM_WARPS];

    // -------------------------------------------------------------------------
    // Phase 1 (blocked layout)
    // -------------------------------------------------------------------------
    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        const int tbase    = IPT * threadIdx.x;   // block-relative
        const bool full    = (glb_offs + B <= n);

        T v[IPT];
        if (full) {
            int4 raw = *reinterpret_cast<const int4*>(d_in + glb_offs + tbase);
            v[0] = raw.x; v[1] = raw.y; v[2] = raw.z; v[3] = raw.w;
            *reinterpret_cast<int4*>(s_elems + tbase) = raw;
        } else {
            #pragma unroll
            for (int i = 0; i < IPT; i++) {
                int gid = glb_offs + tbase + i;
                v[i] = (gid < n) ? d_in[gid] : INF;
                s_elems[tbase + i] = v[i];
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
        s_tmin[threadIdx.x] = tmin;

        // Warp inclusive prefix-min over thread mins
        T c = tmin;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        const T carry = __shfl_up_sync(0xffffffff, c, 1);  // valid for lane>0
        if (lane == 31) s_warp_min[warp_id] = c;

        // Chunk mins (32 consecutive elements = 8 threads) for Phase 3
        {
            T om = tmin;
            om = min(om, __shfl_xor_sync(0xffffffff, om, 1));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 2));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 4));
            if ((lane & 7) == 0)
                d_block_warp_mins[(size_t)block_id * W + warp_id * (32/8) + (lane >> 3)] = om;
        }

        __syncthreads();  // covers cross-thread s_elems/s_tmin/s_warp_min reads

        // Thread-level ANSV chain over tmins (pointer jumping; each lane's
        // query is its own tmin, so the gap-bound argument of the original
        // element-level algorithm carries over verbatim).
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

        // Per-element resolution.  Locally-unresolved elements of a thread
        // are its prefix minima (non-increasing), so the spine walk position
        // `cur` is monotone across i and can be reused.
        unsigned bal[IPT];
        int cur = lane - 1;
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            const bool active = full || (glb_offs + tbase + i < n);
            const T val = v[i];
            bool pending = active && (res[i] < 0) && (lane > 0) && (carry < val);

            if (__any_sync(0xffffffff, pending)) {
                // Walk cur along the chain while tmin[cur] >= val.  Threads
                // strictly between cur and lane always have tmin >= val, so
                // the walk cannot skip the answer; pending guarantees the
                // answer exists, so cur lands on it (never -1).
                while (true) {
                    int src = (pending && cur >= 0) ? cur : 0;
                    T   tg  = __shfl_sync(0xffffffff, tmin,  src);
                    int cg  = __shfl_sync(0xffffffff, chain, src);
                    bool step = pending && (cur >= 0) && (tg >= val);
                    if (!__any_sync(0xffffffff, step)) break;
                    if (step) cur = cg;
                }
                if (pending) {
                    int tt = warp_id * 32 + cur;
                    const T* e = s_elems + IPT * tt;
                    int k = (e[3] < val) ? 3 : (e[2] < val) ? 2 : (e[1] < val) ? 1 : 0;
                    res[i] = IPT * tt + k;
                }
            }

            // Cross-warp fallback: warp min, then thread min, then exact.
            if (active && res[i] < 0) {
                for (int w = warp_id - 1; w >= 0 && res[i] < 0; w--) {
                    if (s_warp_min[w] < val) {
                        for (int tt = 32 * w + 31; tt >= 32 * w; tt--) {
                            if (s_tmin[tt] < val) {
                                const T* e = s_elems + IPT * tt;
                                int k = (e[3] < val) ? 3 : (e[2] < val) ? 2
                                      : (e[1] < val) ? 1 : 0;
                                res[i] = IPT * tt + k;
                                break;
                            }
                        }
                    }
                }
            }

            if (active && res[i] >= 0) d_out[glb_offs + tbase + i] = glb_offs + res[i];
            bal[i] = __ballot_sync(0xffffffff, active && res[i] < 0);
        }

        // Publish unresolved bits: word for chunk c of this warp packs the
        // 4 ballots' bytes c — element 4j+i of the chunk maps to bit 8i+j.
        if (lane < 32/8) {
            unsigned word = ((bal[0] >> (8 * lane)) & 0xffu)
                          | (((bal[1] >> (8 * lane)) & 0xffu) << 8)
                          | (((bal[2] >> (8 * lane)) & 0xffu) << 16)
                          | (((bal[3] >> (8 * lane)) & 0xffu) << 24);
            d_unres[(unsigned)glb_offs / 32 + warp_id * (32/8) + lane] = word;
        }

        if (threadIdx.x == 0) {
            T bmin = s_warp_min[0];
            #pragma unroll
            for (int w = 1; w < NUM_WARPS; w++) bmin = min(bmin, s_warp_min[w]);
            d_block_mins[block_id] = bmin;
        }
        __syncthreads();  // shared reused next iteration
    }

    // -------------------------------------------------------------------------
    // Phase 2 (identical to apsepKernelSPT)
    // -------------------------------------------------------------------------
    grid.sync();

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

    // -------------------------------------------------------------------------
    // Phase 3 (identical to apsepKernelSPT except the permuted bit index)
    // -------------------------------------------------------------------------
    const int num_words   = (n + 31) >> 5;
    const int warps_total = num_phys * NUM_WARPS;
    const int mybit       = 8 * (lane & 3) + (lane >> 2);  // permuted bit for my element

    for (int wbase = (phys_bid * NUM_WARPS + warp_id) * 32;
         wbase < num_words;
         wbase += warps_total * 32) {
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

    std::vector<int> h(N), out_ref(N), out_blk(N);
    int *d_in, *d_out;
    gpuAssert2(cudaMalloc(&d_in,  N * sizeof(int)));
    gpuAssert2(cudaMalloc(&d_out, N * sizeof(int)));

    auto s = allocSPTScratch<int,128,4>(N);

    int blk_per_sm = 0;
    gpuAssert2(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blk_per_sm, apsepKernelSPTBlocked<int,128,4>, 128, 0));
    const int blk_num_phys = blk_per_sm * prop.multiProcessorCount;
    printf("SPT num_phys=%d, blocked num_phys=%d (%d/SM)\n\n",
           s.num_phys, blk_num_phys, blk_per_sm);

    auto runBlocked = [&](int n) {
        int nb = (n + 511) / 512;
        int M = s.M, lo = s.leaf_offset;
        // reuse full-size scratch; nb/M/lo recomputed only for full-N runs
        (void)nb;
        void* args[] = {
            (void*)&d_in, (void*)&d_out, (void*)&n,
            (void*)&s.num_blocks, (void*)&M, (void*)&lo,
            (void*)&s.d_unres, (void*)&s.d_block_mins,
            (void*)&s.d_block_warp_mins, (void*)&s.d_tree,
            (void*)&s.d_prefix_min
        };
        gpuAssert2(cudaLaunchCooperativeKernel(
            (void*)apsepKernelSPTBlocked<int,128,4>, blk_num_phys, 128, args, 0, nullptr));
    };

    auto check = [&](const char* label, int n) {
        // reference
        gpuAssert2(cudaMemset(d_out, 0xAB, (size_t)N * sizeof(int)));
        runSPT<int,128,4>(d_in, d_out, n, s);
        gpuAssert2(cudaDeviceSynchronize());
        gpuAssert2(cudaMemcpy(out_ref.data(), d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
        // blocked (needs its own scratch sizing for partial n: reuse via launchSPT-like path)
        auto s2 = allocSPTScratch<int,128,4>(n);
        void* args[] = {
            (void*)&d_in, (void*)&d_out, (void*)&n,
            (void*)&s2.num_blocks, (void*)&s2.M, (void*)&s2.leaf_offset,
            (void*)&s2.d_unres, (void*)&s2.d_block_mins,
            (void*)&s2.d_block_warp_mins, (void*)&s2.d_tree,
            (void*)&s2.d_prefix_min
        };
        gpuAssert2(cudaMemset(d_out, 0xAB, (size_t)N * sizeof(int)));
        gpuAssert2(cudaLaunchCooperativeKernel(
            (void*)apsepKernelSPTBlocked<int,128,4>, blk_num_phys, 128, args, 0, nullptr));
        gpuAssert2(cudaDeviceSynchronize());
        gpuAssert2(cudaMemcpy(out_blk.data(), d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
        freeSPTScratch<int,128,4>(s2);

        int bad = -1;
        for (int i = 0; i < n; i++)
            if (out_ref[i] != out_blk[i]) { bad = i; break; }
        if (bad >= 0)
            printf("  [%s n=%d] MISMATCH at %d: ref=%d blk=%d (val=%d)\n",
                   label, n, bad, out_ref[bad], out_blk[bad], h[bad]);
        else
            printf("  [%s n=%d] PASS\n", label, n);
        return bad < 0;
    };

    bool ok = true;
    printf("=== Correctness (blocked vs SPT reference) ===\n");
    srand(1234);
    for (int i = 0; i < N; i++) h[i] = rand();
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("random", N);
    ok &= check("random", 1000003);
    ok &= check("random", 513);
    ok &= check("random", 511);
    ok &= check("random", 512 * 3 + 17);

    for (int i = 0; i < N; i++) h[i] = N - i;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("descending", N);
    ok &= check("descending", 1000003);

    for (int i = 0; i < N; i++) h[i] = i;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("ascending", N);

    // duplicates-heavy input (equal runs stress the < / <= boundaries)
    srand(99);
    for (int i = 0; i < N; i++) h[i] = rand() % 16;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("dup16", N);

    if (!ok) { printf("\nCORRECTNESS FAILED — not benchmarking.\n"); return 1; }

    printf("\n=== Timing (N=32M, median of 7) ===\n");
    auto bench = [&](const char* label) {
        float ms_ref = timeKernel([&]{ runSPT<int,128,4>(d_in, d_out, N, s); }, 2, 7);
        float ms_blk = timeKernel([&]{ runBlocked(N); }, 2, 7);
        printf("  %-12s SPT %7.0f µs (%.1f GB/s)   blocked %7.0f µs (%.1f GB/s)\n",
               label,
               ms_ref * 1000, (double)bytes_rw / (ms_ref * 1e-3) / 1e9,
               ms_blk * 1000, (double)bytes_rw / (ms_blk * 1e-3) / 1e9);
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

    freeSPTScratch<int,128,4>(s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
