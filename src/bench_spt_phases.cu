// bench_spt_phases.cu — time SPT phases individually to find bottleneck
#include "apsep.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace cg = cooperative_groups;

#define gpuAssert(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// Phase 1 only — no tree build, no phase 3
template <typename T, int BLOCK_SIZE, int IPT>
__global__
void sptPhase1Only(
        const T* __restrict__ d_in, int* d_out, int n, int num_blocks,
        T* __restrict__ d_block_leaves, T* __restrict__ d_block_mins,
        T* __restrict__ d_block_warp_mins) {
    cg::grid_group grid = cg::this_grid();
    constexpr int B = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W = B / 32;
    const T INF = ApsepInfinity<T>::value();
    const int phys_bid = (int)blockIdx.x;
    const int num_phys = (int)gridDim.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];

    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            int lid = i * BLOCK_SIZE + threadIdx.x;
            int gid = glb_offs + lid;
            s_elems[lid] = (gid < n) ? d_in[gid] : INF;
        }
        __syncthreads();
        T left_carry[IPT];
        #pragma unroll
        for (int ipt = 0; ipt < IPT; ipt++) {
            T c = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
            #pragma unroll
            for (int step = 1; step <= 16; step <<= 1) {
                T nb = __shfl_up_sync(0xffffffff, c, step);
                if (lane >= step) c = min(c, nb);
            }
            left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
            T wm = __shfl_sync(0xffffffff, c, 31);
            if (lane == 0) s_stripe_min[ipt][warp_id] = wm;
        }
        __syncthreads();
        #pragma unroll
        for (int ipt = 0; ipt < IPT; ipt++) {
            int lid = ipt * BLOCK_SIZE + threadIdx.x;
            int gid = glb_offs + lid;
            if (gid >= n) continue;
            T val = s_elems[lid];
            int result = -1;
            if (lane > 0 && left_carry[ipt] < val) {
                int base = ipt * BLOCK_SIZE + warp_id * 32;
                for (int k = lane - 1; k >= 0; k--)
                    if (s_elems[base + k] < val) { result = base + k; break; }
            }
            if (result < 0) {
                for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[ipt][w] < val) {
                        int wb = ipt * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--)
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
            if (result < 0) {
                for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                    for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                        if (s_stripe_min[i][w] < val) {
                            int wb = i * BLOCK_SIZE + w * 32;
                            for (int k = 31; k >= 0; k--)
                                if (s_elems[wb + k] < val) { result = wb + k; break; }
                        }
                    }
                }
            }
            d_out[gid] = (result >= 0) ? (glb_offs + result) : INT_MIN;
        }
        T* g_leaves = d_block_leaves + (size_t)block_id * B;
        for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
            g_leaves[i] = s_elems[i];
        if (lane == 0) {
            T* g_wm = d_block_warp_mins + (size_t)block_id * W;
            for (int ipt = 0; ipt < IPT; ipt++)
                g_wm[ipt * NUM_WARPS + warp_id] = s_stripe_min[ipt][warp_id];
        }
        if (threadIdx.x == 0) {
            T bmin = s_stripe_min[0][0];
            for (int ipt = 0; ipt < IPT; ipt++)
                for (int w = 0; w < NUM_WARPS; w++)
                    bmin = min(bmin, s_stripe_min[ipt][w]);
            d_block_mins[block_id] = bmin;
        }
        __syncthreads();
    }
    grid.sync();  // keep cooperative launch requirement
}

// Phase 1+2 only — no phase 3
template <typename T, int BLOCK_SIZE, int IPT>
__global__
void sptPhase12Only(
        const T* __restrict__ d_in, int* d_out, int n,
        int num_blocks, int M, int leaf_offset,
        T* __restrict__ d_block_leaves, T* __restrict__ d_block_mins,
        T* __restrict__ d_block_warp_mins, T* __restrict__ d_tree) {
    cg::grid_group grid = cg::this_grid();
    constexpr int B = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W = B / 32;
    const T INF = ApsepInfinity<T>::value();
    const int phys_bid = (int)blockIdx.x;
    const int num_phys = (int)gridDim.x;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];

    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            int lid = i * BLOCK_SIZE + threadIdx.x;
            int gid = glb_offs + lid;
            s_elems[lid] = (gid < n) ? d_in[gid] : INF;
        }
        __syncthreads();
        T left_carry[IPT];
        #pragma unroll
        for (int ipt = 0; ipt < IPT; ipt++) {
            T c = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
            #pragma unroll
            for (int step = 1; step <= 16; step <<= 1) {
                T nb = __shfl_up_sync(0xffffffff, c, step);
                if (lane >= step) c = min(c, nb);
            }
            left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
            T wm = __shfl_sync(0xffffffff, c, 31);
            if (lane == 0) s_stripe_min[ipt][warp_id] = wm;
        }
        __syncthreads();
        #pragma unroll
        for (int ipt = 0; ipt < IPT; ipt++) {
            int lid = ipt * BLOCK_SIZE + threadIdx.x;
            int gid = glb_offs + lid;
            if (gid >= n) continue;
            T val = s_elems[lid];
            int result = -1;
            if (lane > 0 && left_carry[ipt] < val) {
                int base = ipt * BLOCK_SIZE + warp_id * 32;
                for (int k = lane - 1; k >= 0; k--)
                    if (s_elems[base + k] < val) { result = base + k; break; }
            }
            if (result < 0) {
                for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[ipt][w] < val) {
                        int wb = ipt * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--)
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
            if (result < 0) {
                for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                    for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                        if (s_stripe_min[i][w] < val) {
                            int wb = i * BLOCK_SIZE + w * 32;
                            for (int k = 31; k >= 0; k--)
                                if (s_elems[wb + k] < val) { result = wb + k; break; }
                        }
                    }
                }
            }
            d_out[gid] = (result >= 0) ? (glb_offs + result) : INT_MIN;
        }
        T* g_leaves = d_block_leaves + (size_t)block_id * B;
        for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
            g_leaves[i] = s_elems[i];
        if (lane == 0) {
            T* g_wm = d_block_warp_mins + (size_t)block_id * W;
            for (int ipt = 0; ipt < IPT; ipt++)
                g_wm[ipt * NUM_WARPS + warp_id] = s_stripe_min[ipt][warp_id];
        }
        if (threadIdx.x == 0) {
            T bmin = s_stripe_min[0][0];
            for (int ipt = 0; ipt < IPT; ipt++)
                for (int w = 0; w < NUM_WARPS; w++)
                    bmin = min(bmin, s_stripe_min[ipt][w]);
            d_block_mins[block_id] = bmin;
        }
        __syncthreads();
    }
    grid.sync();
    // Phase 2
    for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < M; i += num_phys * BLOCK_SIZE)
        d_tree[leaf_offset + i] = (i < num_blocks) ? d_block_mins[i] : INF;
    {
        int level_size = M / 2, level_start = M / 2 - 1;
        while (level_size > 0) {
            grid.sync();
            for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < level_size; i += num_phys * BLOCK_SIZE) {
                int node = level_start + i;
                d_tree[node] = min(d_tree[2*node+1], d_tree[2*node+2]);
            }
            level_size >>= 1;
            level_start = (level_start - 1) / 2;
        }
    }
    grid.sync();
}

template <typename Fn>
static float timeKernel(Fn fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) { fn(); gpuAssert(cudaDeviceSynchronize()); }
    cudaEvent_t t0, t1;
    gpuAssert(cudaEventCreate(&t0)); gpuAssert(cudaEventCreate(&t1));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; i++) {
        gpuAssert(cudaEventRecord(t0));
        fn();
        gpuAssert(cudaEventRecord(t1));
        gpuAssert(cudaEventSynchronize(t1));
        gpuAssert(cudaEventElapsedTime(&ms[i], t0, t1));
    }
    gpuAssert(cudaEventDestroy(t0)); gpuAssert(cudaEventDestroy(t1));
    std::sort(ms.begin(), ms.end());
    return ms[iters / 2];
}

int main() {
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const int N = 32 * 1024 * 1024;
    const long long bytes_rw = 2LL * N * sizeof(int);

    std::vector<int> h(N);

    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, N * sizeof(int)));

    // Alloc SPT scratch (BS=128, IPT=4)
    auto s = allocSPTScratch<int,128,4>(N);
    printf("SPT: %d physical blocks, B=512, num_blocks=%d\n\n", s.num_phys, s.num_blocks);

    auto runPhases = [&](const char* label) {
        printf("--- %s ---\n", label);

    // Full SPT
    {
        float ms = timeKernel([&]{ runSPT<int,128,4>(d_in, d_out, N, s); }, 2, 7);
        printf("Full SPT:      %6.0fµs  %.1f GB/s\n", ms*1000, (double)bytes_rw/(ms*1e-3)/1e9);
    }

    // Phase 1 only
    {
        int blocks_per_sm = 0;
        gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, sptPhase1Only<int,128,4>, 128, 0));
        int num_phys = blocks_per_sm * prop.multiProcessorCount;
        void* args[] = { (void*)&d_in, (void*)&d_out, (void*)&N, (void*)&s.num_blocks,
                         (void*)&s.d_block_leaves, (void*)&s.d_block_mins, (void*)&s.d_block_warp_mins };
        float ms = timeKernel([&]{
            gpuAssert(cudaLaunchCooperativeKernel((void*)sptPhase1Only<int,128,4>, num_phys, 128, args, 0, nullptr));
        }, 2, 7);
        printf("Phase 1 only:  %6.0fµs  %.1f GB/s (implied)\n", ms*1000, (double)bytes_rw/(ms*1e-3)/1e9);
    }

    // Phase 1+2 only
    {
        int blocks_per_sm = 0;
        gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, sptPhase12Only<int,128,4>, 128, 0));
        int num_phys = blocks_per_sm * prop.multiProcessorCount;
        void* args[] = { (void*)&d_in, (void*)&d_out, (void*)&N,
                         (void*)&s.num_blocks, (void*)&s.M, (void*)&s.leaf_offset,
                         (void*)&s.d_block_leaves, (void*)&s.d_block_mins,
                         (void*)&s.d_block_warp_mins, (void*)&s.d_tree };
        float ms = timeKernel([&]{
            gpuAssert(cudaLaunchCooperativeKernel((void*)sptPhase12Only<int,128,4>, num_phys, 128, args, 0, nullptr));
        }, 2, 7);
        printf("Phase 1+2:     %6.0fµs  %.1f GB/s (implied)\n", ms*1000, (double)bytes_rw/(ms*1e-3)/1e9);
        printf("Phase 2 alone: %6.0fµs\n", (ms - /* phase1 */0)*1000);
    }

    // SPTAtomic full
    {
        int blocks_per_sm2 = 0;
        gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm2, apsepKernelSPTAtomic<int,128,4>, 128, 0));
        int num_phys2 = blocks_per_sm2 * prop.multiProcessorCount;
        void* args2[] = {
            (void*)&d_in, (void*)&d_out, (void*)&N,
            (void*)&s.num_blocks, (void*)&s.M, (void*)&s.leaf_offset,
            (void*)&s.d_block_leaves, (void*)&s.d_block_mins,
            (void*)&s.d_block_warp_mins, (void*)&s.d_tree
        };
        float ms = timeKernel([&]{
            gpuAssert(cudaLaunchCooperativeKernel((void*)apsepKernelSPTAtomic<int,128,4>, num_phys2, 128, args2, 0, nullptr));
        }, 2, 7);
        printf("SPTAtomic:     %6.0fµs  %.1f GB/s\n", ms*1000, (double)bytes_rw/(ms*1e-3)/1e9);
    }

    // WSTL reference
    {
        auto ws = allocWSTLScratch<int,128,4>(N);
        float ms = timeKernel([&]{ runWSTL<int,128,4>(d_in, d_out, N, ws); }, 2, 7);
        printf("WSTL:          %6.0fµs  %.1f GB/s\n", ms*1000, (double)bytes_rw/(ms*1e-3)/1e9);
        freeWSTLScratch<int,128,4>(ws);
    }

        printf("\n");
    };  // end runPhases lambda

    // Descending worst case
    for (int i = 0; i < N; i++) h[i] = N - i;
    gpuAssert(cudaMemcpy(d_in, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    runPhases("Descending (worst-case)");

    // Random
    srand(42); for (int i = 0; i < N; i++) h[i] = rand();
    gpuAssert(cudaMemcpy(d_in, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    runPhases("Random");

    freeSPTScratch<int,128,4>(s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
