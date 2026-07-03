// Profiling tool for the APSEP decoupled look-back kernel.
//
// Instruments the kernel with per-phase clock timers and look-back step
// counters to understand where time is actually spent.

#include "apsep.cuh"

#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// Instrumented kernel: same logic as apsepKernel<T,128,2,K> but with
// per-phase clock measurements and look-back step counting.
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__ void apsepKernelInstrumented(
        const T* __restrict__    d_in,
        int*                     d_out,
        int                      n,
        int                      num_blocks,
        int                      num_superblocks,
        BlockState<T>*           d_states,
        T* __restrict__          d_block_trees,
        SuperBlockState<T>*      d_sb_states,
        T* __restrict__          d_sb_trees,
        volatile uint32_t*       d_dyn_idx,
        // Instrumentation outputs
        unsigned long long*      d_phase_clocks,   // [num_blocks * 4]: intra, publish, sbuild, lookback
        unsigned long long*      d_lookback_steps, // total look-back steps across all elements
        unsigned long long*      d_spin_cycles,    // cycles spent spinning in lookback
        int*                     d_lb_hist)        // histogram of per-element look-back step counts [0..64]
{
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;
    const     T   INF = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B];

    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;

    // ---- Phase 1: intra-block ----
    unsigned long long t0 = clock64();

    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();

    buildMinTree<T, BLOCK_SIZE, IPT>(s_elems, s_tree);

    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n) {
            int local = treePrevSmaller<T>(s_tree, B, lid, s_elems[lid]);
            d_out[gid] = (local >= 0) ? (glb_offs + local) : INT_MIN;
        }
    }

    unsigned long long t1 = clock64();

    // ---- Phase 2: publish block ----
    T* g_block_tree = d_block_trees + (size_t)block_id * (2 * B);
    for (int i = threadIdx.x; i < 2 * B; i += BLOCK_SIZE)
        g_block_tree[i] = s_tree[i];
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        d_states[block_id].block_min = s_tree[1];
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    unsigned long long t2 = clock64();

    // ---- Phase 3: superblock build (last block in SB only) ----
    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        T* g_sb_tree = d_sb_trees + (size_t)sb_id * (2 * KB);
        const T* g_block_trees_sb = d_block_trees + (size_t)sb_first * (2 * B);
        buildSuperBlockTree<T, BLOCK_SIZE, IPT, K>(
            g_block_trees_sb, g_sb_tree, sb_size, n, sb_first, INF);

        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0) {
            d_sb_states[sb_id].sb_min = g_sb_tree[1];
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    unsigned long long t3 = clock64();

    // ---- Phase 4: decoupled look-back ----
    // Count steps and spin cycles per element, accumulate atomically.
    constexpr int KB_val = K * B;

    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;
            int steps = 0;
            unsigned long long spin = 0;

            for (int sb = sb_id - 1; sb >= 0; sb--) {
                steps++;
                unsigned long long spin_start = clock64();
                while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }
                spin += clock64() - spin_start;

                if (d_sb_states[sb].sb_min < val) {
                    const T* st = d_sb_trees + (size_t)sb * (2 * KB_val);
                    int leaf = treeRightmostSmaller<T>(st, KB_val, val);
                    if (leaf >= 0) { result = sb * KB_val + leaf; break; }
                }
            }
            d_out[gid] = result;

            // Accumulate counters (one atomic per element is expensive but fine for profiling)
            atomicAdd(d_lookback_steps, (unsigned long long)steps);
            atomicAdd(d_spin_cycles,    spin);
            int bucket = min(steps, 64);
            atomicAdd(&d_lb_hist[bucket], 1);
        }
    }

    unsigned long long t4 = clock64();

    // Lane 0 of each block records the phase timings
    if (threadIdx.x == 0) {
        d_phase_clocks[block_id * 4 + 0] = t1 - t0;  // intra-block
        d_phase_clocks[block_id * 4 + 1] = t2 - t1;  // publish
        d_phase_clocks[block_id * 4 + 2] = t3 - t2;  // superblock build (0 for non-last)
        d_phase_clocks[block_id * 4 + 3] = t4 - t3;  // look-back
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

template <int K>
static void profileConfig(const int* d_in, int* d_out, int n,
                          const char* label) {
    constexpr int BLOCK_SIZE = 128;
    constexpr int IPT        = 2;
    constexpr int B          = BLOCK_SIZE * IPT;
    constexpr int KB         = K * B;

    int num_blocks      = (n + B - 1) / B;
    int num_superblocks = (num_blocks + K - 1) / K;

    // APSEP scratch
    BlockState<int>*      d_states      = nullptr;
    int*                  d_block_trees = nullptr;
    SuperBlockState<int>* d_sb_states   = nullptr;
    int*                  d_sb_trees    = nullptr;
    uint32_t*             d_dyn_idx     = nullptr;

    gpuAssert(cudaMalloc(&d_states,      (size_t)num_blocks      * sizeof(BlockState<int>)));
    gpuAssert(cudaMalloc(&d_block_trees, (size_t)num_blocks      * 2 * B  * sizeof(int)));
    gpuAssert(cudaMalloc(&d_sb_states,   (size_t)num_superblocks * sizeof(SuperBlockState<int>)));
    gpuAssert(cudaMalloc(&d_sb_trees,    (size_t)num_superblocks * 2 * KB * sizeof(int)));
    gpuAssert(cudaMalloc(&d_dyn_idx,     sizeof(uint32_t)));

    // Instrumentation buffers
    unsigned long long* d_phase_clocks   = nullptr;
    unsigned long long* d_lookback_steps = nullptr;
    unsigned long long* d_spin_cycles    = nullptr;
    int*                d_lb_hist        = nullptr;

    gpuAssert(cudaMalloc(&d_phase_clocks,   (size_t)num_blocks * 4 * sizeof(unsigned long long)));
    gpuAssert(cudaMalloc(&d_lookback_steps, sizeof(unsigned long long)));
    gpuAssert(cudaMalloc(&d_spin_cycles,    sizeof(unsigned long long)));
    gpuAssert(cudaMalloc(&d_lb_hist,        65 * sizeof(int)));

    // Reset
    gpuAssert(cudaMemset(d_states,      0, (size_t)num_blocks      * sizeof(BlockState<int>)));
    gpuAssert(cudaMemset(d_sb_states,   0, (size_t)num_superblocks * sizeof(SuperBlockState<int>)));
    gpuAssert(cudaMemset(d_dyn_idx,     0, sizeof(uint32_t)));
    gpuAssert(cudaMemset(d_phase_clocks,   0, (size_t)num_blocks * 4 * sizeof(unsigned long long)));
    gpuAssert(cudaMemset(d_lookback_steps, 0, sizeof(unsigned long long)));
    gpuAssert(cudaMemset(d_spin_cycles,    0, sizeof(unsigned long long)));
    gpuAssert(cudaMemset(d_lb_hist,        0, 65 * sizeof(int)));

    // Launch
    apsepKernelInstrumented<int, BLOCK_SIZE, IPT, K>
        <<<num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, num_blocks, num_superblocks,
            d_states, d_block_trees, d_sb_states, d_sb_trees, d_dyn_idx,
            d_phase_clocks, d_lookback_steps, d_spin_cycles, d_lb_hist);
    gpuAssert(cudaDeviceSynchronize());
    gpuAssert(cudaGetLastError());

    // Retrieve results
    std::vector<unsigned long long> h_phase(num_blocks * 4);
    unsigned long long h_lb_steps = 0, h_spin = 0;
    std::vector<int> h_hist(65);

    gpuAssert(cudaMemcpy(h_phase.data(), d_phase_clocks,
                         (size_t)num_blocks * 4 * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&h_lb_steps, d_lookback_steps, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(&h_spin,     d_spin_cycles,    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    gpuAssert(cudaMemcpy(h_hist.data(), d_lb_hist,      65 * sizeof(int),           cudaMemcpyDeviceToHost));

    // Count elements that needed look-back
    int lb_elements = 0;
    for (int i = 0; i <= 64; i++) lb_elements += h_hist[i];
    // (bucket 0 = elements that needed look-back but sb_id==0, i.e. 0 steps)

    // Per-phase aggregate: mean, max across blocks
    double sum_intra = 0, sum_pub = 0, sum_sbuild = 0, sum_lb = 0;
    unsigned long long max_intra = 0, max_pub = 0, max_sbuild = 0, max_lb = 0;
    int sbuild_count = 0;
    double sum_sbuild_nonzero = 0;

    for (int b = 0; b < num_blocks; b++) {
        unsigned long long intra  = h_phase[b*4+0];
        unsigned long long pub    = h_phase[b*4+1];
        unsigned long long sbuild = h_phase[b*4+2];
        unsigned long long lb     = h_phase[b*4+3];
        sum_intra  += intra;  max_intra  = max(max_intra,  intra);
        sum_pub    += pub;    max_pub    = max(max_pub,    pub);
        sum_sbuild += sbuild; max_sbuild = max(max_sbuild, sbuild);
        sum_lb     += lb;     max_lb     = max(max_lb,     lb);
        if (sbuild > 0) { sbuild_count++; sum_sbuild_nonzero += sbuild; }
    }

    // GPU clock frequency (approximate from SM clock)
    // Use 1590 MHz typical for GTX 1660 Ti boost; we report in microseconds.
    // We'll just report raw cycles and let the reader convert (1 cycle ≈ 0.63 ns at 1590 MHz).
    double mhz = 1590.0;  // approximate boost clock
    auto cy2us = [&](double cy) { return cy / (mhz * 1e3); };

    printf("\n=== %s (K=%d, %d blocks, %d superblocks) ===\n",
           label, K, num_blocks, num_superblocks);
    printf("  Phase breakdown (mean / max across blocks, µs at %.0f MHz):\n", mhz);
    printf("    Intra-block (load+tree+query): mean=%.1f  max=%.1f\n",
           cy2us(sum_intra  / num_blocks), cy2us(max_intra));
    printf("    Publish block state:           mean=%.1f  max=%.1f\n",
           cy2us(sum_pub    / num_blocks), cy2us(max_pub));
    printf("    Superblock build (last only):  mean=%.1f  max=%.1f  (n=%d blocks)\n",
           sbuild_count > 0 ? cy2us(sum_sbuild_nonzero / sbuild_count) : 0.0,
           cy2us(max_sbuild), sbuild_count);
    printf("    Look-back phase total:         mean=%.1f  max=%.1f\n",
           cy2us(sum_lb     / num_blocks), cy2us(max_lb));

    printf("\n  Look-back stats (%d elements needed inter-superblock look-back):\n",
           lb_elements);
    if (lb_elements > 0) {
        double mean_steps = (double)h_lb_steps / lb_elements;
        double spin_us_total = cy2us((double)h_spin);
        printf("    Mean steps per element:  %.2f\n", mean_steps);
        printf("    Total spin wait:         %.1f µs (summed across all threads)\n",
               spin_us_total);
        printf("    Spin as %% of look-back:  %.1f%%\n",
               100.0 * (double)h_spin / (sum_lb > 0 ? sum_lb : 1));

        printf("    Step count histogram (steps -> #elements):\n");
        printf("      ");
        bool printed = false;
        for (int i = 0; i <= 64; i++) {
            if (h_hist[i] > 0) {
                printf("%s%d:%d", printed ? " " : "", i, h_hist[i]);
                printed = true;
            }
        }
        printf("\n");
    }

    // Compute fraction of elements needing look-back
    int valid_n = 0;
    for (int b = 0; b < num_blocks; b++) {
        int bstart = b * B;
        int bend   = min(bstart + B, n);
        valid_n   += bend - bstart;
    }
    // valid_n == n
    printf("    Fraction needing look-back: %.1f%% of %d elements\n",
           100.0 * lb_elements / n, n);

    // Cleanup
    cudaFree(d_states); cudaFree(d_block_trees);
    cudaFree(d_sb_states); cudaFree(d_sb_trees); cudaFree(d_dyn_idx);
    cudaFree(d_phase_clocks); cudaFree(d_lookback_steps);
    cudaFree(d_spin_cycles); cudaFree(d_lb_hist);
}

int main() {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s  (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // 64 MiB — enough to get representative stats without OOM on per-block arrays
    constexpr int N = 64 * 1024 * 1024 / sizeof(int);

    std::vector<int> h(N);
    {
        std::mt19937 rng(0xBEEF);
        std::uniform_int_distribution<int> dist(0, N);
        for (auto& v : h) v = dist(rng);
    }

    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  (size_t)N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)N * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    printf("Profiling with uniform random data (%d MiB)...\n",
           (int)(N * sizeof(int) / (1024*1024)));

    profileConfig<8> (d_in, d_out, N, "K=8  (best from sweep)");
    profileConfig<32>(d_in, d_out, N, "K=32 (large superblocks)");

    // Also profile a pathological case: mostly-ones (worst-case look-back distance)
    printf("\n\nProfiling with mostly-ones [0,1,1,1,...] (%d MiB)...\n",
           (int)(N * sizeof(int) / (1024*1024)));
    std::vector<int> h2(N, 1);
    h2[0] = 0;
    gpuAssert(cudaMemcpy(d_in, h2.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));
    gpuAssert(cudaMemset(d_out, 0, (size_t)N * sizeof(int)));

    profileConfig<8> (d_in, d_out, N, "mostly-ones K=8");
    profileConfig<32>(d_in, d_out, N, "mostly-ones K=32");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
