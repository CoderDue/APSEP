// Benchmark suite for all experimental intra-block PSE kernel variants.
//
// Champion: blockwiseKernelWarpScanV3 at BS=64, IPT=2 (~127 GB/s on GTX 1660 Ti).
// All other variants were investigated and benchmarked for comparison.
// See FINDINGS.md for analysis of why each approach falls short.

#include "apsep.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

static std::vector<int> cpuApsep(const std::vector<int>& arr) {
    int n = (int)arr.size();
    std::vector<int> result(n, -1);
    std::vector<int> stk;
    stk.reserve(n);
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && arr[stk.back()] >= arr[i])
            stk.pop_back();
        result[i] = stk.empty() ? -1 : stk.back();
        stk.push_back(i);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Benchmark statistics
// ---------------------------------------------------------------------------

static void computeDescriptors(const std::vector<float>& measurements, size_t bytes) {
    size_t size = measurements.size();
    double sample_mean = 0, sample_variance = 0, sample_gbps = 0;
    double factor = (double)bytes / (1000.0 * (double)size);
    for (size_t i = 0; i < size; i++) {
        double diff = std::max(1e3 * (double)measurements[i], 0.5);
        sample_mean     += diff / (double)size;
        sample_variance += (diff * diff) / (double)size;
        sample_gbps     += factor / diff;
    }
    double sample_std = std::sqrt(sample_variance);
    double bound = (0.95 * sample_std) / std::sqrt((double)size);
    printf("%.0fμs (95%% CI: [%.1fμs, %.1fμs]); %.1f GB/s",
           sample_mean, sample_mean - bound, sample_mean + bound, sample_gbps);
}

__global__ void copyKernel(const int* __restrict__ src, int* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

static double baselineBandwidth(int n, int warmup = 5, int iters = 50) {
    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  (size_t)n * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)n * sizeof(int)));
    gpuAssert(cudaMemset(d_in, 0, (size_t)n * sizeof(int)));
    constexpr int BS = 256;
    int grid = (n + BS - 1) / BS;
    for (int i = 0; i < warmup; i++)
        copyKernel<<<grid, BS>>>(d_in, d_out, n);
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart); cudaEventCreate(&tstop);
    std::vector<float> measurements(iters);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(tstart);
        copyKernel<<<grid, BS>>>(d_in, d_out, n);
        cudaEventRecord(tstop);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&measurements[i], tstart, tstop);
    }
    cudaEventDestroy(tstart); cudaEventDestroy(tstop);
    cudaFree(d_in); cudaFree(d_out);
    size_t bytes = (size_t)n * 2 * sizeof(int);
    double mean_ms = 0;
    for (float m : measurements) mean_ms += m / iters;
    return (double)bytes / 1e9 / (mean_ms * 1e-3);
}

// ===========================================================================
// Kernel variants
// ===========================================================================

// ---------------------------------------------------------------------------
// Min-tree V1 (original: separate s_elems + s_tree)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernel(const T* __restrict__ d_in, int* d_out, int n) {
    constexpr int B  = BLOCK_SIZE * IPT;
    const     T   INF = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_tree[2 * B];
    const int glb_offs = blockIdx.x * B;
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
            d_out[gid] = (local >= 0) ? (glb_offs + local) : -1;
        }
    }
}

// ---------------------------------------------------------------------------
// Min-tree V2 (load directly into leaf area)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelV2(const T* __restrict__ d_in, int* d_out, int n) {
    constexpr int B   = BLOCK_SIZE * IPT;
    const     T   INF = ApsepInfinity<T>::value();
    __shared__ T s_tree[2 * B];
    const int glb_offs = blockIdx.x * B;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_tree[B + lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    for (int half = B >> 1; half >= 1; half >>= 1) {
        for (int i = threadIdx.x; i < half; i += BLOCK_SIZE) {
            int node = half + i;
            s_tree[node] = min(s_tree[2 * node], s_tree[2 * node + 1]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n) {
            T val  = s_tree[B + lid];
            int local = treePrevSmaller<T>(s_tree, B, lid, val);
            d_out[gid] = (local >= 0) ? (glb_offs + local) : -1;
        }
    }
}

// ---------------------------------------------------------------------------
// WarpScan (original multi-sync: one sync per stripe)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScan(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const int lid  = ipt * BLOCK_SIZE + threadIdx.x;
        const int base = ipt * BLOCK_SIZE + warp_id * 32;
        const T   val  = s_elems[lid];
        T carry = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, carry, step);
            if (lane >= step) carry = min(carry, nb);
        }
        T left_carry = __shfl_up_sync(0xffffffff, carry, 1);
        int pse_idx = -1;
        if (lane > 0 && left_carry < val) {
            for (int k = lane - 1; k >= 0; k--) {
                if (s_elems[base + k] < val) { pse_idx = base + k; break; }
            }
        }
        r_out[ipt] = (pse_idx >= 0) ? (glb_offs + pse_idx) : INT_MIN;
        T stripe_min = __shfl_sync(0xffffffff, carry, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = stripe_min;
        __syncthreads();
        if ((glb_offs + lid) < n && r_out[ipt] == INT_MIN) {
            const T v = val;
            int result = -1;
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < v) {
                    int wb = ipt * BLOCK_SIZE + w * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (s_elems[wb + k] < v) { result = wb + k; break; }
                    }
                }
            }
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < v) {
                        int wb = i * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--) {
                            if (s_elems[wb + k] < v) { result = wb + k; break; }
                        }
                    }
                }
            }
            r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
        }
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V3: champion (single-sync variant, 2 syncs total)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScanV3(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();  // Sync 1
    T carry[IPT], left_carry[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
        T c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        T warp_min = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = warp_min;
    }
    __syncthreads();  // Sync 2
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const int lid = ipt * BLOCK_SIZE + threadIdx.x;
        const T   val = s_elems[lid];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            const int base = ipt * BLOCK_SIZE + warp_id * 32;
            for (int k = lane - 1; k >= 0; k--) {
                if (s_elems[base + k] < val) { result = base + k; break; }
            }
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * BLOCK_SIZE + w * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--) {
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                        }
                    }
                }
            }
        }
        r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V5: non-divergent branchless linear scan
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScanV5(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    T carry[IPT], left_carry[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
        T c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        T warp_min = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = warp_min;
    }
    __syncthreads();
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const int lid = ipt * BLOCK_SIZE + threadIdx.x;
        const T   val = s_elems[lid];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            const int base = ipt * BLOCK_SIZE + warp_id * 32;
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                if (k < lane && s_elems[base + k] < val) result = base + k;
            }
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * BLOCK_SIZE + w * 32;
                    int best = -1;
                    #pragma unroll
                    for (int k = 0; k < 32; k++) {
                        if (s_elems[wb + k] < val) best = k;
                    }
                    if (best >= 0) { result = wb + best; break; }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * BLOCK_SIZE + w * 32;
                        int best = -1;
                        #pragma unroll
                        for (int k = 0; k < 32; k++) {
                            if (s_elems[wb + k] < val) best = k;
                        }
                        if (best >= 0) { result = wb + best; break; }
                    }
                }
            }
        }
        r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V6: shuffle-based rightmost-smaller (buggy — only checks 5 lanes)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScanV6(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    T carry[IPT], left_carry[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
        T c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        T warp_min = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = warp_min;
    }
    __syncthreads();
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const int lid = ipt * BLOCK_SIZE + threadIdx.x;
        const T   val = s_elems[lid];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            int best_k = -1;
            T   prop_val = val;
            const int base = ipt * BLOCK_SIZE + warp_id * 32;
            #pragma unroll
            for (int step = 1; step <= 16; step <<= 1) {
                T   recv_val = __shfl_up_sync(0xffffffff, prop_val, step);
                int recv_k   = lane - step;
                if (lane >= step && recv_val < val) best_k = recv_k;
            }
            if (best_k >= 0) result = base + best_k;
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * BLOCK_SIZE + w * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--) {
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                        }
                    }
                }
            }
        }
        r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V7: padded shared memory layout (PAD=16 per warp slot)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScanV7(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int NUM_WARPS  = BLOCK_SIZE / 32;
    constexpr int PAD        = 16;
    constexpr int WARP_SLOT  = 32 + PAD;
    constexpr int STRIPE_PAD = NUM_WARPS * WARP_SLOT;
    constexpr int B_PAD      = IPT * STRIPE_PAD;
    const     T   INF        = ApsepInfinity<T>::value();
    __shared__ T s_pad[B_PAD];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    const int glb_offs = blockIdx.x * (BLOCK_SIZE * IPT);
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        int gid = glb_offs + ipt * BLOCK_SIZE + warp_id * 32 + lane;
        s_pad[ipt * STRIPE_PAD + warp_id * WARP_SLOT + lane] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    T carry[IPT], left_carry[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = s_pad[ipt * STRIPE_PAD + warp_id * WARP_SLOT + lane];
        T c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        T warp_min = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = warp_min;
    }
    __syncthreads();
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = s_pad[ipt * STRIPE_PAD + warp_id * WARP_SLOT + lane];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            int base = ipt * STRIPE_PAD + warp_id * WARP_SLOT;
            for (int k = lane - 1; k >= 0; k--) {
                if (s_pad[base + k] < val) { result = base + k; break; }
            }
            if (result >= 0) {
                int pad_off = result;
                int ipt_r   = pad_off / STRIPE_PAD;
                int rem     = pad_off % STRIPE_PAD;
                int w_r     = rem / WARP_SLOT;
                int k_r     = rem % WARP_SLOT;
                result = glb_offs + ipt_r * BLOCK_SIZE + w_r * 32 + k_r;
            }
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * STRIPE_PAD + w * WARP_SLOT;
                    for (int k = 31; k >= 0; k--) {
                        if (s_pad[wb + k] < val) {
                            result = glb_offs + ipt * BLOCK_SIZE + w * 32 + k;
                            break;
                        }
                    }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * STRIPE_PAD + w * WARP_SLOT;
                        for (int k = 31; k >= 0; k--) {
                            if (s_pad[wb + k] < val) {
                                result = glb_offs + i * BLOCK_SIZE + w * 32 + k;
                                break;
                            }
                        }
                    }
                }
            }
        }
        r_out[ipt] = result;
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + warp_id * 32 + lane;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V8: ballot-based intra-warp (incorrect base addressing + slow)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScanV8(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    T carry[IPT], left_carry[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
        T c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        T warp_min = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = warp_min;
    }
    __syncthreads();
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const int lid = ipt * BLOCK_SIZE + threadIdx.x;
        const T   val = s_elems[lid];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            const int base = ipt * BLOCK_SIZE + warp_id * 32;
            int best_k = -1;
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                T src_val = __shfl_sync(0xffffffff, s_elems[base + k], k);
                unsigned mask = __ballot_sync(0xffffffff, src_val < val);
                if ((mask >> lane) & 1u) best_k = k;
            }
            if (best_k >= 0) result = base + best_k;
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * BLOCK_SIZE + w * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--) {
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                        }
                    }
                }
            }
        }
        r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V9: CUB BlockSort-based PSE (O(B log B) sort + serial scan)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelCubSort(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B = BLOCK_SIZE * IPT;
    using BlockSort = cub::BlockRadixSort<T, BLOCK_SIZE, IPT, int>;
    using BlockScan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ union {
        typename BlockSort::TempStorage sort;
        typename BlockScan::TempStorage scan;
    } tmp;
    __shared__ T   s_sorted_val[B];
    __shared__ int s_sorted_idx[B];
    __shared__ int s_result[B];
    const int glb_offs = blockIdx.x * B;
    const T   INF      = ApsepInfinity<T>::value();
    T   thread_keys[IPT];
    int thread_vals[IPT];
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        thread_keys[i] = (gid < n) ? d_in[gid] : INF;
        thread_vals[i] = lid;
    }
    BlockSort(tmp.sort).SortBlockedToStriped(thread_keys, thread_vals);
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int rank = i * BLOCK_SIZE + threadIdx.x;
        s_sorted_val[rank] = thread_keys[i];
        s_sorted_idx[rank] = thread_vals[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int max_lt    = -1;
        int group_max = -1;
        T   group_val = INF;
        for (int r = 0; r < B; r++) {
            T   cur_val = s_sorted_val[r];
            int cur_idx = s_sorted_idx[r];
            if (cur_val != group_val) {
                if (group_max > max_lt) max_lt = group_max;
                group_val = cur_val;
                group_max = -1;
            }
            s_result[cur_idx] = max_lt;
            if (cur_idx > group_max) group_max = cur_idx;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n) {
            int pse_local = s_result[lid];
            d_out[gid] = (pse_local >= 0) ? (glb_offs + pse_local) : -1;
        }
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V10: binary search via shuffles (buggy — misses rightmost element)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScanV10(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    T carry[IPT], left_carry[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
        T c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        T warp_min = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = warp_min;
    }
    __syncthreads();
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const int lid = ipt * BLOCK_SIZE + threadIdx.x;
        const T   val = s_elems[lid];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            int lo = 0, hi = lane - 1;
            int best_k = -1;
            const int base = ipt * BLOCK_SIZE + warp_id * 32;
            #pragma unroll
            for (int step = 16; step >= 1; step >>= 1) {
                int mid = lo + step - 1;
                if (mid <= hi) {
                    T mid_val = __shfl_sync(0xffffffff, s_elems[base + mid], mid);
                    if (mid_val < val) {
                        best_k = mid;
                        lo = mid + 1;
                    }
                }
            }
            if (best_k >= 0) result = base + best_k;
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * BLOCK_SIZE + w * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--) {
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                        }
                    }
                }
            }
        }
        r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpScan-V11: __ldg (read-only cache) for Phase B — correct but slower
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpScanV11(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    T reg_val[IPT];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        reg_val[i] = (gid < n) ? __ldg(&d_in[gid]) : INF;
    }
    T carry[IPT], left_carry[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        T c = reg_val[ipt];
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        T warp_min = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = warp_min;
    }
    __syncthreads();
    int r_out[IPT];
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const T val = reg_val[ipt];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            const int base_gid = glb_offs + ipt * BLOCK_SIZE + warp_id * 32;
            for (int k = lane - 1; k >= 0; k--) {
                T elem = (base_gid + k < n) ? __ldg(&d_in[base_gid + k]) : INF;
                if (elem < val) { result = ipt * BLOCK_SIZE + warp_id * 32 + k; break; }
            }
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb_gid = glb_offs + ipt * BLOCK_SIZE + w * 32;
                    for (int k = 31; k >= 0; k--) {
                        T elem = (wb_gid + k < n) ? __ldg(&d_in[wb_gid + k]) : INF;
                        if (elem < val) { result = ipt * BLOCK_SIZE + w * 32 + k; break; }
                    }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb_gid = glb_offs + i * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--) {
                            T elem = (wb_gid + k < n) ? __ldg(&d_in[wb_gid + k]) : INF;
                            if (elem < val) { result = i * BLOCK_SIZE + w * 32 + k; break; }
                        }
                    }
                }
            }
        }
        r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
    }
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }
}

// ---------------------------------------------------------------------------
// WarpTree: register min-tree + inter-warp smem tree
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelWarpTree(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_inter_tree[2 * NUM_WARPS];
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        const int lid = ipt * BLOCK_SIZE + threadIdx.x;
        const T   val = s_elems[lid];
        T reg_tree[6];
        reg_tree[0] = val;
        #pragma unroll
        for (int h = 1; h <= 5; h++) {
            T peer = __shfl_xor_sync(0xffffffff, reg_tree[h-1], 1 << (h-1));
            reg_tree[h] = min(reg_tree[h-1], peer);
        }
        if (lane == 0) s_inter_tree[NUM_WARPS + warp_id] = reg_tree[5];
        __syncthreads();
        #pragma unroll
        for (int half = NUM_WARPS >> 1; half >= 1; half >>= 1) {
            if (threadIdx.x < half) {
                int node = half + threadIdx.x;
                s_inter_tree[node] = min(s_inter_tree[2*node], s_inter_tree[2*node+1]);
            }
            __syncthreads();
        }
        T pmn = reg_tree[0];
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            T nb = __shfl_up_sync(0xffffffff, pmn, step);
            if (lane >= step) pmn = min(pmn, nb);
        }
        T left_pmn = __shfl_up_sync(0xffffffff, pmn, 1);
        if (lane == 0) left_pmn = INF;
        int pse_idx = -1;
        if (left_pmn < val) {
            int base = ipt * BLOCK_SIZE + warp_id * 32;
            for (int k = lane - 1; k >= 0; k--) {
                if (s_elems[base + k] < val) { pse_idx = base + k; break; }
            }
        }
        if (pse_idx < 0 && warp_id > 0) {
            if (s_inter_tree[1] < val) {
                int curr = 1;
                while (curr < NUM_WARPS) {
                    int right = 2 * curr + 1;
                    int right_warp = right - NUM_WARPS;
                    if (right_warp < warp_id && s_inter_tree[right] < val)
                        curr = right;
                    else
                        curr = 2 * curr;
                }
                int found_warp = curr - NUM_WARPS;
                if (found_warp < warp_id) {
                    int base = ipt * BLOCK_SIZE + found_warp * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (s_elems[base + k] < val) { pse_idx = base + k; break; }
                    }
                }
            }
        }
        __syncthreads();
        int gid = glb_offs + lid;
        if (gid < n)
            d_out[gid] = (pse_idx >= 0) ? (glb_offs + pse_idx) : -1;
    }
}

// ---------------------------------------------------------------------------
// Stack kernel: per-thread sequential monotone stack
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void blockwiseKernelStack(const T* __restrict__ d_in, int* d_out, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a warp multiple");
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    __shared__ T s_elems[B];
    __shared__ T s_warp_min[NUM_WARPS];
    const int glb_offs   = blockIdx.x * B;
    const int chunk_base = threadIdx.x * IPT;
    const int lane       = threadIdx.x & 31;
    const int warp_id    = threadIdx.x >> 5;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = chunk_base + i;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();
    T   stk_val[IPT];
    int stk_idx[IPT];
    int stk_top = 0;
    T   chunk_min = INF;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        T v = s_elems[chunk_base + i];
        while (stk_top > 0 && stk_val[stk_top - 1] >= v)
            stk_top--;
        int lid = chunk_base + i;
        int gid = glb_offs + lid;
        if (gid < n) {
            d_out[gid] = (stk_top > 0) ? (glb_offs + stk_idx[stk_top - 1]) : -2;
        }
        stk_val[stk_top] = v;
        stk_idx[stk_top] = lid;
        stk_top++;
        chunk_min = min(chunk_min, v);
    }
    T carry = chunk_min;
    #pragma unroll
    for (int step = 1; step <= 16; step <<= 1) {
        T nb = __shfl_up_sync(0xffffffff, carry, step);
        if (lane >= step) carry = min(carry, nb);
    }
    T warp_min = __shfl_sync(0xffffffff, carry, 31);
    if (lane == 0) s_warp_min[warp_id] = warp_min;
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = chunk_base + i;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == -2) {
            T v = s_elems[lid];
            int result = -1;
            T left_warp_carry = __shfl_up_sync(0xffffffff, carry, 1);
            if (lane > 0 && left_warp_carry < v) {
                int scan_end = chunk_base;
                for (int k = scan_end - 1; k >= warp_id * 32 * IPT; k--) {
                    if (s_elems[k] < v) { result = k; break; }
                }
            }
            if (result < 0 && warp_id > 0) {
                for (int w = warp_id - 1; w >= 0; w--) {
                    if (s_warp_min[w] < v) {
                        int warp_end = (w * 32 + 32) * IPT;
                        for (int k = warp_end - 1; k >= w * 32 * IPT; k--) {
                            if (s_elems[k] < v) { result = k; break; }
                        }
                        break;
                    }
                }
            }
            d_out[gid] = (result >= 0) ? (glb_offs + result) : -1;
        }
    }
}

// ===========================================================================
// Bandwidth measurement helpers (templated per kernel/BS/IPT)
// ===========================================================================

template <typename KernelFn>
static double measureBandwidth(const char* label, KernelFn launch,
                               const int* d_in, int* d_out, int n,
                               int warmup = 5, int iters = 50) {
    for (int i = 0; i < warmup; i++) launch(d_in, d_out, n);
    std::vector<float> measurements(iters);
    cudaEvent_t tstart, tstop;
    cudaEventCreate(&tstart); cudaEventCreate(&tstop);
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(tstart);
        launch(d_in, d_out, n);
        cudaEventRecord(tstop);
        cudaEventSynchronize(tstop);
        cudaEventElapsedTime(&measurements[i], tstart, tstop);
    }
    cudaEventDestroy(tstart); cudaEventDestroy(tstop);
    size_t bytes = (size_t)n * 2 * sizeof(int);
    printf("  %s  ", label);
    computeDescriptors(measurements, bytes);
    printf("\n");
    double mean_ms = 0;
    for (float m : measurements) mean_ms += m / iters;
    return (double)bytes / 1e9 / (mean_ms * 1e-3);
}

// ---------------------------------------------------------------------------
// Correctness test for block-wise variants (intra-block PSE only)
// ---------------------------------------------------------------------------

template <typename KernelFn>
static bool testBlockwiseVariant(const char* label, KernelFn launch,
                                 const std::vector<int>& h_in, int B) {
    int n = (int)h_in.size();
    auto h_cpu = cpuApsep(h_in);
    int *d_in = nullptr, *d_out = nullptr;
    gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    launch(d_in, d_out, n);
    gpuAssert(cudaDeviceSynchronize());
    std::vector<int> h_out(n);
    gpuAssert(cudaMemcpy(h_out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);
    bool ok = true;
    for (int i = 0; i < n; i++) {
        int block_of_i = i / B;
        int cpu_ans    = h_cpu[i];
        int expected   = (cpu_ans >= 0 && cpu_ans / B == block_of_i) ? cpu_ans : -1;
        if (h_out[i] != expected) {
            if (ok)
                printf("  FAIL %s: i=%d val=%d got=%d expected=%d\n",
                       label, i, h_in[i], h_out[i], expected);
            ok = false;
        }
    }
    return ok;
}

static bool runBlockwiseVariantTests() {
    constexpr int BS64  = 64;
    constexpr int B2_64 = BS64 * 2;
    constexpr int B4_64 = BS64 * 4;

    struct TC { const char* name; std::vector<int> data; };
    TC cases[] = {
        {"single",      {42}},
        {"ascending",   {1,2,3,4,5,6,7,8}},
        {"descending",  {8,7,6,5,4,3,2,1}},
        {"alternating", {3,1,4,1,5,9,2,6}},
        {"two blocks",  {}},
        {"odd length",  {}},
    };
    cases[4].data.resize(2 * B4_64); for (int i=0;i<2*B4_64;i++) cases[4].data[i]=(i%7)*13;
    cases[5].data.resize(B4_64+13);  for (int i=0;i<B4_64+13;i++) cases[5].data[i]=i%17;

    bool all_ok = true;
    for (auto& tc : cases) {
        printf("  testing %s...\n", tc.name); fflush(stdout);

#define TEST_VARIANT(label, kernel, bs, ipt, block_size) \
        { \
            printf("    " label "... "); fflush(stdout); \
            bool ok = testBlockwiseVariant(label, [](const int* in, int* out, int n){ \
                int nb = (n + (block_size)-1)/(block_size); \
                kernel<int,(bs),(ipt)><<<nb,(bs)>>>(in,out,n); \
            }, tc.data, (block_size)); \
            printf("%s\n", ok ? "ok" : "FAIL"); fflush(stdout); \
            all_ok = all_ok && ok; \
        }

        TEST_VARIANT("wscan-v3 BS=64 IPT=2",  blockwiseKernelWarpScanV3, 64, 2, B2_64)
        TEST_VARIANT("wscan-v3 BS=32 IPT=2",  blockwiseKernelWarpScanV3, 32, 2, 64)
        TEST_VARIANT("stack BS=64 IPT=2",      blockwiseKernelStack,       64, 2, B2_64)
        TEST_VARIANT("wscan-v7 BS=64 IPT=2",  blockwiseKernelWarpScanV7, 64, 2, B2_64)
        TEST_VARIANT("wscan-v11 BS=64 IPT=2", blockwiseKernelWarpScanV11, 64, 2, B2_64)

#undef TEST_VARIANT

        printf("  %-22s %s\n", tc.name, all_ok ? "PASS" : "FAIL");
    }
    return all_ok;
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s  (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    double peak_gbps = (double)prop.memoryClockRate * 1e3
                     * (double)prop.memoryBusWidth / 8.0 * 2.0 / 1e9;

    constexpr int N = 500 * 1024 * 1024 / sizeof(int);

    double baseline = baselineBandwidth(N);
    printf("Theoretical peak: %.1f GB/s  copy baseline: %.1f GB/s\n\n",
           peak_gbps, baseline);

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

    // ---- Correctness ----
    printf("=== Block-wise variant correctness tests ===\n");
    runBlockwiseVariantTests();
    printf("\n");

    // Helper lambda that runs measureBandwidth for IPT=1,2,4,8
#define BENCH4(label_fmt, kernel, bs) \
    do { \
        char lbl[64]; \
        snprintf(lbl, sizeof(lbl), label_fmt " IPT=1 B=%-4d", (bs)*1); \
        measureBandwidth(lbl, [&](const int* i, int* o, int n){ \
            int nb=(n+(bs)*1-1)/((bs)*1); kernel<int,(bs),1><<<nb,(bs)>>>(i,o,n); }, d_in, d_out, N); \
        snprintf(lbl, sizeof(lbl), label_fmt " IPT=2 B=%-4d", (bs)*2); \
        measureBandwidth(lbl, [&](const int* i, int* o, int n){ \
            int nb=(n+(bs)*2-1)/((bs)*2); kernel<int,(bs),2><<<nb,(bs)>>>(i,o,n); }, d_in, d_out, N); \
        snprintf(lbl, sizeof(lbl), label_fmt " IPT=4 B=%-4d", (bs)*4); \
        measureBandwidth(lbl, [&](const int* i, int* o, int n){ \
            int nb=(n+(bs)*4-1)/((bs)*4); kernel<int,(bs),4><<<nb,(bs)>>>(i,o,n); }, d_in, d_out, N); \
        snprintf(lbl, sizeof(lbl), label_fmt " IPT=8 B=%-4d", (bs)*8); \
        measureBandwidth(lbl, [&](const int* i, int* o, int n){ \
            int nb=(n+(bs)*8-1)/((bs)*8); kernel<int,(bs),8><<<nb,(bs)>>>(i,o,n); }, d_in, d_out, N); \
    } while(0)

    printf("--- Champion: wscan-v3 BS=64 ---\n");
    BENCH4("wscan-v3 BS=64", blockwiseKernelWarpScanV3, 64);

    printf("\n--- A: stack BS=64 ---\n");
    BENCH4("stack BS=64", blockwiseKernelStack, 64);

    printf("\n--- B: wscan-v3 BS=32 ---\n");
    BENCH4("wscan-v3 BS=32", blockwiseKernelWarpScanV3, 32);

    printf("\n--- D: wscan-v7 BS=64 (padded smem) ---\n");
    BENCH4("wscan-v7 BS=64", blockwiseKernelWarpScanV7, 64);

    printf("\n--- E: wscan-v8 BS=64 (ballot) ---\n");
    BENCH4("wscan-v8 BS=64", blockwiseKernelWarpScanV8, 64);

    printf("\n--- F: cub-sort BS=64 ---\n");
    BENCH4("cub-sort BS=64", blockwiseKernelCubSort, 64);

    printf("\n--- G: wscan-v10 BS=64 (binary search shuffles) ---\n");
    BENCH4("wscan-v10 BS=64", blockwiseKernelWarpScanV10, 64);

    printf("\n--- H: wscan-v11 BS=64 (__ldg) ---\n");
    BENCH4("wscan-v11 BS=64", blockwiseKernelWarpScanV11, 64);

    printf("\n--- V5: wscan-v5 BS=64 (non-divergent) ---\n");
    BENCH4("wscan-v5 BS=64", blockwiseKernelWarpScanV5, 64);

    printf("\n--- mintree V1 BS=128 ---\n");
    BENCH4("mintree-v1 BS=128", blockwiseKernel, 128);

    printf("\n--- mintree V2 BS=128 ---\n");
    BENCH4("mintree-v2 BS=128", blockwiseKernelV2, 128);

#undef BENCH4

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
