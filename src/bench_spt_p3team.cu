// bench_spt_p3team.cu — prototype of a sub-warp-team Phase 3 for SPT.
//
// Hypothesis: on random input Phase 3 serializes ~446K warp-cooperative
// lookups, one element per warp at a time (~580 dependent lookup chains per
// warp). Splitting each warp into 4 teams of 8 lanes that process nonzero
// bitmask words concurrently gives 4 independent latency chains per warp;
// Volta+ independent thread scheduling interleaves them.
//
// Phase 1 and Phase 2 are identical to apsepKernelSPT. Team lookups need
// W == 16 (each of the 8 lanes covers 2 chunk-mins and 4 leaf elements).
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
void apsepKernelSPTP3Team(
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

    static_assert(IPT == 4, "blocked P1 assumes IPT == 4");
    static_assert(sizeof(T) == 4, "blocked P1 assumes 4-byte T");
    static_assert(W == 16, "team P3 assumes W == 16 (8 lanes x 2 chunk-mins)");

    const int phys_bid   = (int)blockIdx.x;
    const int num_phys   = (int)gridDim.x;
    const int lane       = threadIdx.x & 31;
    const int warp_id    = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_tmin[BLOCK_SIZE];
    __shared__ T s_warp_min[NUM_WARPS];

    // ---- Phase 1 (identical to apsepKernelSPT: blocked layout) ----
    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        const int tbase    = IPT * threadIdx.x;
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

        int res[IPT];
        res[0] = -1;
        res[1] = (v[0] < v[1]) ? tbase     : -1;
        res[2] = (v[1] < v[2]) ? tbase + 1 : (v[0] < v[2]) ? tbase : -1;
        res[3] = (v[2] < v[3]) ? tbase + 2 : (v[1] < v[3]) ? tbase + 1
                                           : (v[0] < v[3]) ? tbase : -1;

        const T tmin = min(min(v[0], v[1]), min(v[2], v[3]));
        s_tmin[threadIdx.x] = tmin;

        T c = tmin;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        const T carry = __shfl_up_sync(0xffffffff, c, 1);
        if (lane == 31) s_warp_min[warp_id] = c;

        {
            T om = tmin;
            om = min(om, __shfl_xor_sync(0xffffffff, om, 1));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 2));
            om = min(om, __shfl_xor_sync(0xffffffff, om, 4));
            if ((lane & 7) == 0)
                d_block_warp_mins[(size_t)block_id * W + warp_id * (32/8) + (lane >> 3)] = om;
        }

        __syncthreads();

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
                if (pending) {
                    int tt = warp_id * 32 + cur;
                    const T* e = s_elems + IPT * tt;
                    int k = (e[3] < val) ? 3 : (e[2] < val) ? 2 : (e[1] < val) ? 1 : 0;
                    res[i] = IPT * tt + k;
                }
            }

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
        __syncthreads();
    }

    // ---- Phase 2 (identical to apsepKernelSPT) ----
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

    // ---- Phase 3 (sub-warp teams) ----
    // A warp still owns an aligned 32-word group (proven balanced), but the
    // nonzero words are handed round-robin to 4 teams of 8 lanes.  Teams are
    // internally converged (their control flow depends only on team-uniform
    // values), so ballots/shuffles use the team mask; metadata is re-fetched
    // via L2 instead of shuffled across team boundaries.
    const int num_words   = (n + 31) >> 5;
    const int warps_total = num_phys * NUM_WARPS;
    const int team_id     = lane >> 3;
    const int tl          = lane & 7;
    const unsigned tmask  = 0xffu << (8 * team_id);

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

        const unsigned nz  = __ballot_sync(0xffffffff, um_l != 0);
        const int      nnz = __popc(nz);

        for (int k = team_id; k < nnz; k += 4) {
            unsigned tmp = nz;
            for (int q = 0; q < k; q++) tmp &= tmp - 1;
            const int j = __ffs(tmp) - 1;
            const int w = wbase + j;

            // Re-derived per word (L2 hits; cross-team shuffles are illegal)
            const unsigned um       = __ldg(&d_unres[w]);
            const int  block_id     = w / W;
            const T    prefix_min_b = __ldg(&d_prefix_min[block_id]);
            const int  base         = w * 32;
            const bool eo           = prefix_min_b >= __ldg(&d_in[(size_t)block_id * B]);

            // lane tl owns the word's elements 4*tl .. 4*tl+3 (raw order);
            // blocked-P1 permutation: element 4j+i <-> bit 8i+j
            bool mine[4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                mine[i] = (um >> (8 * i + tl)) & 1u;

            if (eo) {
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    if (mine[i]) d_out[base + 4 * tl + i] = -1;
                continue;
            }

            T v4[4];
            if (base + 32 <= n) {
                int4 raw = __ldg(reinterpret_cast<const int4*>(d_in + base) + tl);
                v4[0] = raw.x; v4[1] = raw.y; v4[2] = raw.z; v4[3] = raw.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    v4[i] = mine[i] ? __ldg(&d_in[base + 4 * tl + i]) : INF;
            }

            unsigned pend[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const bool need = mine[i] && (prefix_min_b < v4[i]);
                if (mine[i] && !need) d_out[base + 4 * tl + i] = -1;
                pend[i] = __ballot_sync(tmask, need);
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned p = pend[i];
                while (p) {
                    const int owner = __ffs(p) - 1;   // absolute lane index
                    p &= p - 1;
                    const T qval = __shfl_sync(tmask, v4[i], owner);

                    // Tree ascent+descent, redundant on the 8 team lanes
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
                        // chunk-min scan: 8 lanes x 2 of the W=16 chunk mins
                        const T* wm = d_block_warp_mins + (size_t)found_block * W;
                        const T w0 = __ldg(&wm[tl]);
                        const T w1 = __ldg(&wm[tl + 8]);
                        const unsigned m0 = __ballot_sync(tmask, w0 < qval) >> (8 * team_id);
                        const unsigned m1 = __ballot_sync(tmask, w1 < qval) >> (8 * team_id);
                        const unsigned wmask = (m1 << 8) | m0;
                        const int wstar = 31 - __clz(wmask);
                        // leaf scan: 8 lanes x int4 over the 32-element chunk
                        const T* bl = d_in + (size_t)found_block * B + wstar * 32;
                        int4 lraw = __ldg(reinterpret_cast<const int4*>(bl) + tl);
                        const int kl = (lraw.w < qval) ? 3 : (lraw.z < qval) ? 2
                                     : (lraw.y < qval) ? 1 : (lraw.x < qval) ? 0 : -1;
                        const unsigned lm = __ballot_sync(tmask, kl >= 0) >> (8 * team_id);
                        const int lstar = 31 - __clz(lm);
                        const int kk = __shfl_sync(tmask, kl, 8 * team_id + lstar);
                        result = found_block * B + wstar * 32 + 4 * lstar + kk;
                    }
                    if (tl == 0)
                        d_out[base + 4 * (owner - 8 * team_id) + i] = result;
                }
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

    std::vector<int> h(N), out_ref(N), out_new(N);
    int *d_in, *d_out;
    gpuAssert2(cudaMalloc(&d_in,  N * sizeof(int)));
    gpuAssert2(cudaMalloc(&d_out, N * sizeof(int)));

    auto s = allocSPTScratch<int,128,4>(N);

    int blk_per_sm = 0;
    gpuAssert2(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blk_per_sm, apsepKernelSPTP3Team<int,128,4>, 128, 0));
    const int new_num_phys = blk_per_sm * prop.multiProcessorCount;
    printf("SPT num_phys=%d, p3team num_phys=%d (%d/SM)\n\n",
           s.num_phys, new_num_phys, blk_per_sm);

    auto runNew = [&](int n) {
        void* args[] = {
            (void*)&d_in, (void*)&d_out, (void*)&n,
            (void*)&s.num_blocks, (void*)&s.M, (void*)&s.leaf_offset,
            (void*)&s.d_unres, (void*)&s.d_block_mins,
            (void*)&s.d_block_warp_mins, (void*)&s.d_tree,
            (void*)&s.d_prefix_min
        };
        gpuAssert2(cudaLaunchCooperativeKernel(
            (void*)apsepKernelSPTP3Team<int,128,4>, new_num_phys, 128, args, 0, nullptr));
    };

    auto check = [&](const char* label, int n) {
        gpuAssert2(cudaMemset(d_out, 0xAB, (size_t)N * sizeof(int)));
        runSPT<int,128,4>(d_in, d_out, n, s);
        gpuAssert2(cudaDeviceSynchronize());
        gpuAssert2(cudaMemcpy(out_ref.data(), d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));

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
            (void*)apsepKernelSPTP3Team<int,128,4>, new_num_phys, 128, args, 0, nullptr));
        gpuAssert2(cudaDeviceSynchronize());
        gpuAssert2(cudaMemcpy(out_new.data(), d_out, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost));
        freeSPTScratch<int,128,4>(s2);

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
    printf("=== Correctness (p3team vs SPT reference) ===\n");
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

    srand(99);
    for (int i = 0; i < N; i++) h[i] = rand() % 16;
    gpuAssert2(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    ok &= check("dup16", N);

    if (!ok) { printf("\nCORRECTNESS FAILED — not benchmarking.\n"); return 1; }

    printf("\n=== Timing (N=32M, median of 7) ===\n");
    auto bench = [&](const char* label) {
        float ms_ref = timeKernel([&]{ runSPT<int,128,4>(d_in, d_out, N, s); }, 2, 7);
        float ms_new = timeKernel([&]{ runNew(N); }, 2, 7);
        printf("  %-11s  SPT %7.0f µs (%.1f GB/s)   p3team %7.0f µs (%.1f GB/s)\n",
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
    return 0;
}
