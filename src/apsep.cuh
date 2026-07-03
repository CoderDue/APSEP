// Single-Pass All Previous Smaller or Equal Problem (APSEP)
//
// For each element d_in[i], computes d_out[i] = the index j of the nearest
// previous (j < i) element satisfying d_in[j] < d_in[i], or -1 if none.
//
// Algorithm (Bar-on & Vishkin 1985, adapted for GPU single-pass):
//
//  Per block (BLOCK_SIZE threads, IPT items/thread, B = BLOCK_SIZE*IPT elems):
//  1. Load B elements into shared memory.
//  2. Build a complete binary min-tree over the B elements (parallel, O(log B)).
//  3. For each element, query the local tree for its intra-block PSE (parallel,
//     O(log B) per element).
//  4. Publish local tree + block_min. Last block in each super-block (K blocks)
//     waits for all K block-mins, builds a merged super-block tree over K*B
//     elements in global memory, then marks the super-block READY.
//  5. Elements without a local PSE perform a decoupled look-back at the
//     super-block granularity (num_superblocks steps instead of num_blocks),
//     reducing serialization by a factor of K.
//
// Template parameters:
//   T          – element type; must support min() and <
//   BLOCK_SIZE – threads per CUDA block
//   IPT        – items per thread  (B = BLOCK_SIZE * IPT must be a power of two)
//   K          – super-block size in blocks (K=1 → original single-block trees)

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <stdio.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#define gpuAssert(x) _gpuAssert((x), __FILE__, __LINE__)

static inline void _gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line,
                cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------
// Type-specific "infinity" (used as out-of-bounds padding)
// ---------------------------------------------------------------------------

template <typename T> struct ApsepInfinity {
    __host__ __device__ static T value();
};
template <> struct ApsepInfinity<int> {
    __host__ __device__ static int value() { return INT_MAX; }
};
template <> struct ApsepInfinity<unsigned> {
    __host__ __device__ static unsigned value() { return UINT_MAX; }
};
template <> struct ApsepInfinity<long long> {
    __host__ __device__ static long long value() { return LLONG_MAX; }
};
template <> struct ApsepInfinity<float> {
    __host__ __device__ static float value() { return FLT_MAX; }
};
template <> struct ApsepInfinity<double> {
    __host__ __device__ static double value() { return DBL_MAX; }
};

// ---------------------------------------------------------------------------
// Status flags
// ---------------------------------------------------------------------------

enum ApsepStatus : uint8_t {
    APSEP_INVALID = 0,
    APSEP_READY   = 1
};

// ---------------------------------------------------------------------------
// Per-block state: used by intermediate blocks to signal their block_min to
// the last block in the super-block, which builds the merged tree.
// ---------------------------------------------------------------------------

template <typename T>
struct alignas(16) BlockState {
    T                    block_min;
    volatile ApsepStatus status;
};

// ---------------------------------------------------------------------------
// Per-super-block state: published after the last block in the super-block
// builds the merged global tree.
// ---------------------------------------------------------------------------

template <typename T>
struct alignas(16) SuperBlockState {
    T                    sb_min;
    volatile ApsepStatus status;
};

// ---------------------------------------------------------------------------
// Min-tree construction in shared memory (1-indexed BFS layout)
// B must be a power of two.
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__device__ void buildMinTree(const volatile T* __restrict__ elems,
                             volatile T*       __restrict__ tree) {
    constexpr int B = BLOCK_SIZE * IPT;

    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        tree[B + lid] = elems[lid];
    }
    __syncthreads();

    for (int half = B >> 1; half >= 1; half >>= 1) {
        for (int i = threadIdx.x; i < half; i += BLOCK_SIZE) {
            int node = half + i;
            tree[node] = min(tree[2 * node], tree[2 * node + 1]);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Build a merged super-block tree in global memory from K per-block trees.
//
// The K per-block trees are already in g_block_trees[0..K-1] (each of size
// 2*B). We build a new tree of size 2*(K*B) in g_sb_tree, using the leaves
// of each block's tree (at offsets [B..2B-1]) as the leaves of the merged
// tree (at offsets [K*B .. 2*K*B - 1]).
//
// All BLOCK_SIZE threads cooperate. The merged tree is written directly to
// global memory; a __threadfence() after is the caller's responsibility.
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT, int K>
__device__ void buildSuperBlockTree(const T* __restrict__ g_block_trees,
                                    T*       __restrict__ g_sb_tree,
                                    int sb_blocks,  // actual blocks in this SB (<=K)
                                    int n_total,    // total elements in input
                                    int sb_first_block, // global block index of first block in SB
                                    T INF) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;

    // Copy leaves from per-block trees into the super-block tree's leaf area.
    // Leaf area: g_sb_tree[KB .. 2*KB-1].
    // Block b contributes leaves g_block_trees[b*(2*B) + B .. b*(2*B) + 2*B-1]
    // into g_sb_tree[KB + b*B .. KB + (b+1)*B - 1].
    for (int i = threadIdx.x; i < KB; i += BLOCK_SIZE) {
        int b   = i / B;   // which block
        int lid = i % B;   // which leaf within that block
        T val;
        if (b < sb_blocks) {
            // Check if this leaf corresponds to a real global element
            int gid = (sb_first_block + b) * B + lid;
            val = (gid < n_total) ? g_block_trees[b * (2 * B) + B + lid] : INF;
        } else {
            val = INF;
        }
        g_sb_tree[KB + i] = val;
    }
    __syncthreads();

    // Bottom-up reduction over the super-block tree
    for (int half = KB >> 1; half >= 1; half >>= 1) {
        for (int i = threadIdx.x; i < half; i += BLOCK_SIZE) {
            int node = half + i;
            g_sb_tree[node] = min(g_sb_tree[2 * node], g_sb_tree[2 * node + 1]);
        }
        __syncthreads();
    }
    // g_sb_tree[1] is now the super-block minimum
}

// ---------------------------------------------------------------------------
// Intra-block PSE via tree query (shared memory tree, size B)
// ---------------------------------------------------------------------------

template <typename T>
__device__ int treePrevSmaller(const volatile T* __restrict__ tree,
                               int B, int start_leaf, T val) {
    if (start_leaf <= 0) return -1;

    int node = B + start_leaf;

    while (node > 1) {
        if ((node & 1) != 0) {
            if (tree[node - 1] < val) {
                int curr = node - 1;
                while (curr < B) {
                    int right = 2 * curr + 1;
                    curr = (tree[right] < val) ? right : (2 * curr);
                }
                return curr - B;
            }
        }
        node >>= 1;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Inter-super-block PSE query
//
// Searches the merged super-block tree (KB leaves) for the rightmost leaf
// whose value < val. Returns leaf index in [0, KB-1], or -1.
// ---------------------------------------------------------------------------

template <typename T>
__device__ int treeRightmostSmaller(const volatile T* __restrict__ tree,
                                    int KB, T val) {
    if (tree[1] >= val) return -1;

    int curr = 1;
    while (curr < KB) {
        int right = 2 * curr + 1;
        curr = (tree[right] < val) ? right : (2 * curr);
    }
    return curr - KB;
}

// ---------------------------------------------------------------------------
// Decoupled look-back at super-block granularity
//
// Scans previous super-blocks (sb_id-1, sb_id-2, …). For each super-block,
// spins until APSEP_READY, checks sb_min, then searches the merged tree.
// Returns a global element index, or -1.
// ---------------------------------------------------------------------------

template <typename T, int B, int K>
__device__ int decoupledLookback(int                            sb_id,
                                 T                              val,
                                 const SuperBlockState<T>*      d_sb_states,
                                 const T* __restrict__          d_sb_trees) {
    constexpr int KB = K * B;

    for (int sb = sb_id - 1; sb >= 0; sb--) {
        while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }

        if (d_sb_states[sb].sb_min < val) {
            const T* st  = d_sb_trees + (size_t)sb * (2 * KB);
            int leaf = treeRightmostSmaller<T>(st, KB, val);
            // leaf is a position within the super-block's KB elements
            return (leaf >= 0) ? sb * KB + leaf : -1;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__ void apsepKernel(const T* __restrict__         d_in,
                            int*                          d_out,
                            int                           n,
                            int                           num_blocks,
                            int                           num_superblocks,
                            BlockState<T>*                d_states,
                            T* __restrict__               d_block_trees,
                            SuperBlockState<T>*           d_sb_states,
                            T* __restrict__               d_sb_trees,
                            volatile uint32_t*            d_dyn_idx) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;
    const     T   INF = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B];

    // ---- 1. Dynamic block assignment ----
    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;
    const int sb_id    = block_id / K;          // which super-block
    const int sb_local = block_id % K;          // position within super-block
    const int sb_first = sb_id * K;             // first block in this super-block

    // ---- 2. Load elements (pad with INF) ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();

    // ---- 3. Build intra-block min-tree in shared memory ----
    buildMinTree<T, BLOCK_SIZE, IPT>(s_elems, s_tree);

    // ---- 4. Intra-block PSE queries ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n) {
            int local = treePrevSmaller<T>(s_tree, B, lid, s_elems[lid]);
            d_out[gid] = (local >= 0) ? (glb_offs + local) : INT_MIN;
        }
    }

    // ---- 5. Publish per-block tree and mark block READY ----
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

    // ---- 6. Last block in super-block builds merged super-block tree ----
    // Determine how many blocks are in this super-block (last SB may be short)
    int sb_size = min(K, num_blocks - sb_first);  // actual blocks in this SB
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        // Wait for all earlier blocks in this super-block to publish
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        // Build the merged super-block tree in global memory
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

    // ---- 7. Decoupled look-back at super-block granularity ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            d_out[gid] = decoupledLookback<T, B, K>(
                sb_id, s_elems[lid], d_sb_states, d_sb_trees);
        }
    }
}

// ---------------------------------------------------------------------------
// Scratch buffers
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 1>
struct ApsepScratch {
    BlockState<T>*      d_states      = nullptr;
    T*                  d_block_trees = nullptr;
    SuperBlockState<T>* d_sb_states   = nullptr;
    T*                  d_sb_trees    = nullptr;
    uint32_t*           d_dyn_idx     = nullptr;
    int                 num_blocks     = 0;
    int                 num_superblocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 1>
ApsepScratch<T, BLOCK_SIZE, IPT, K> allocApsepScratch(int n) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;
    static_assert((B & (B - 1)) == 0, "BLOCK_SIZE * IPT must be a power of two");

    ApsepScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;

    gpuAssert(cudaMalloc(&s.d_states,      (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_trees, (size_t)s.num_blocks      * 2 * B  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,   (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_sb_trees,    (size_t)s.num_superblocks * 2 * KB * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,     sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 1>
void freeApsepScratch(ApsepScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_trees);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_sb_trees);
    cudaFree(s.d_dyn_idx);
    s = ApsepScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 1>
void runApsep(const T* d_in, int* d_out, int n,
              ApsepScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));

    apsepKernel<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_trees,
            s.d_sb_states, s.d_sb_trees,
            s.d_dyn_idx);

    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 1>
void launchApsep(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocApsepScratch<T, BLOCK_SIZE, IPT, K>(n);
    runApsep<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeApsepScratch<T, BLOCK_SIZE, IPT, K>(s);
}

// ===========================================================================
// Stack look-back APSEP
// ===========================================================================
//
// Like the K=1 tree look-back but uses a compact per-block monotone suffix
// stack instead of a min-tree.  For random input the stack has O(log B)
// entries on average, so look-back reads are tiny and typically served from
// L2.  Intra-block PSE uses the same warp-scan as blockwiseKernelWarpScan.
//
// Published per block:
//   volatile int  status     – 0 = not ready, 1 = ready
//   int           block_min  – minimum value in this block (fast screen)
//   int           depth      – number of stack entries
//   T             vals[B]    – monotone increasing stack values
//   int           idxs[B]    – corresponding global indices
// ===========================================================================

template <typename T, int B>
struct alignas(16) BlockStack {
    volatile int status;   // 0 = not ready, 1 = ready
    T            block_min;
    int          depth;
    T            vals[B];
    int          idxs[B];
};

// Intra-block: build the block's monotone suffix stack into s_vals/s_idxs,
// returning the depth.  Uses a parallel suffix-min + prefix-sum approach,
// identical to the persistent kernel's carry-update logic.
// Requires s_tmp[B] scratch (overwritten).  Called after warp-scan phases.
// s_elems[B] must be populated.  n_tile = number of valid elements in block.
template <typename T, int BLOCK_SIZE, int IPT>
__device__ int buildBlockStack(
        T* __restrict__ s_elems,
        T* __restrict__ s_tmp,
        T* __restrict__ s_vals,
        int* __restrict__ s_idxs,
        int glb_offs, int n,
        T INF)
{
    constexpr int B = BLOCK_SIZE * IPT;

    // Step 1: suffix-min into s_tmp
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        s_tmp[lid] = s_elems[lid];
    }
    __syncthreads();
    #pragma unroll
    for (int half = 1; half < B; half <<= 1) {
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            int lid = i * BLOCK_SIZE + threadIdx.x;
            if (lid + half < B)
                s_tmp[lid] = min(s_tmp[lid], s_tmp[lid + half]);
        }
        __syncthreads();
    }

    // Step 2: survive flag (1 if element is a suffix minimum = left-to-right
    //         minimum from right), then inclusive prefix-sum
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        T suf_next = (lid + 1 < B) ? s_tmp[lid + 1] : INF;
        s_tmp[lid] = (gid < n && s_elems[lid] < suf_next) ? 1 : 0;
    }
    __syncthreads();
    // Hillis-Steele inclusive prefix-sum
    #pragma unroll
    for (int half = 1; half < B; half <<= 1) {
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            int lid = i * BLOCK_SIZE + threadIdx.x;
            int v = s_tmp[lid] + (lid >= half ? s_tmp[lid - half] : 0);
            __syncthreads();
            s_tmp[lid] = v;
        }
        __syncthreads();
    }
    // Now s_tmp[k] = inclusive prefix-sum of survive flags.
    // Total survivors = s_tmp[B-1].

    // Step 3: scatter survivors into s_vals / s_idxs
    const int total = s_tmp[B - 1];
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        int excl = (lid == 0) ? 0 : s_tmp[lid - 1];
        int incl = s_tmp[lid];
        if (incl - excl == 1) {
            s_vals[excl] = s_elems[lid];
            s_idxs[excl] = gid;
        }
    }
    __syncthreads();
    return total;
}

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void apsepStackLookbackKernel(
        const T* __restrict__ d_in,
        int*                  d_out,
        int                   n,
        int                   num_blocks,
        BlockStack<T, BLOCK_SIZE * IPT>* d_stacks)
{
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const T INF = ApsepInfinity<T>::value();

    __shared__ T   s_elems[B];
    __shared__ T   s_stripe_min[IPT][NUM_WARPS];
    __shared__ T   s_tmp[B];      // scratch for suffix-min + prefix-sum
    __shared__ T   s_vals[B];     // block's suffix stack values
    __shared__ int s_idxs[B];    // block's suffix stack indices

    const int block_id = blockIdx.x;
    const int glb_offs = block_id * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    // ---- Load ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }

    // ---- Warp-scan intra-block PSE (identical to blockwiseKernelWarpScan) ----
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
        r_out[ipt] = (pse_idx >= 0) ? (glb_offs + pse_idx) : -1;

        T stripe_min = __shfl_sync(0xffffffff, carry, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = stripe_min;
        __syncthreads();

        if ((glb_offs + lid) < n && r_out[ipt] == -1) {
            const T v = val;
            int result = -1;
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < v) {
                    int wb = ipt * BLOCK_SIZE + w * 32;
                    for (int k = 31; k >= 0; k--)
                        if (s_elems[wb + k] < v) { result = wb + k; break; }
                }
            }
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < v) {
                        int wb = i * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--)
                            if (s_elems[wb + k] < v) { result = wb + k; break; }
                    }
                }
            }
            r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
        }
    }

    // ---- Write intra-block results ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int gid = glb_offs + i * BLOCK_SIZE + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }

    // ---- Build block suffix stack ----
    int depth = buildBlockStack<T, BLOCK_SIZE, IPT>(
            s_elems, s_tmp, s_vals, s_idxs, glb_offs, n, INF);

    // ---- Publish block stack ----
    // Compute block_min from stripe_mins (all s_stripe_min slots valid after
    // the warp-scan phase above).
    T bmin = INF;
    if (threadIdx.x == 0) {
        for (int i = 0; i < IPT; i++)
            for (int w = 0; w < NUM_WARPS; w++)
                bmin = min(bmin, s_stripe_min[i][w]);
    }
    // Write vals/idxs to global
    for (int i = threadIdx.x; i < depth; i += BLOCK_SIZE) {
        d_stacks[block_id].vals[i] = s_vals[i];
        d_stacks[block_id].idxs[i] = s_idxs[i];
    }
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        d_stacks[block_id].depth     = depth;
        d_stacks[block_id].block_min = bmin;
        __threadfence();
        d_stacks[block_id].status    = 1;
    }

    // ---- Inter-block look-back ----
    // For each element that has no intra-block PSE, scan blocks b-1, b-2, ...
    // Binary-search each block's stack for the rightmost entry < val.
    // All threads in the block collaborate: the warp collectively scans back
    // using the block_min as a fast screen.  Each thread then resolves its
    // own element independently.
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid >= n || r_out[i] != -1) continue;

        const T v = s_elems[lid];
        int result = -1;
        for (int b = block_id - 1; b >= 0 && result < 0; b--) {
            // spin until block b is ready
            while (d_stacks[b].status == 0) { /* spin */ }
            if (d_stacks[b].block_min >= v) continue;

            // binary search the stack: find rightmost entry with val < v
            const int dep = d_stacks[b].depth;
            int lo = 0, hi = dep;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (d_stacks[b].vals[mid] < v) lo = mid + 1;
                else                             hi = mid;
            }
            if (lo > 0) result = d_stacks[b].idxs[lo - 1];
        }
        d_out[gid] = result;
    }
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for stack look-back kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
struct ApsepStackScratch {
    BlockStack<T, BLOCK_SIZE * IPT>* d_stacks = nullptr;
    int num_blocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
ApsepStackScratch<T, BLOCK_SIZE, IPT> allocStackScratch(int n) {
    ApsepStackScratch<T, BLOCK_SIZE, IPT> s;
    constexpr int B = BLOCK_SIZE * IPT;
    s.num_blocks = (n + B - 1) / B;
    gpuAssert(cudaMalloc(&s.d_stacks,
        (size_t)s.num_blocks * sizeof(BlockStack<T, B>)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void freeStackScratch(ApsepStackScratch<T, BLOCK_SIZE, IPT>& s) {
    cudaFree(s.d_stacks);
    s = ApsepStackScratch<T, BLOCK_SIZE, IPT>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void runStackApsep(const T* d_in, int* d_out, int n,
                   ApsepStackScratch<T, BLOCK_SIZE, IPT>& s) {
    gpuAssert(cudaMemset(s.d_stacks, 0,
        (size_t)s.num_blocks * sizeof(BlockStack<T, BLOCK_SIZE * IPT>)));
    apsepStackLookbackKernel<T, BLOCK_SIZE, IPT>
        <<<s.num_blocks, BLOCK_SIZE>>>(d_in, d_out, n, s.num_blocks, s.d_stacks);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void launchStackApsep(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocStackScratch<T, BLOCK_SIZE, IPT>(n);
    runStackApsep<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeStackScratch<T, BLOCK_SIZE, IPT>(s);
}

// (Persistent-thread variant removed — see FINDINGS.md for analysis.)
