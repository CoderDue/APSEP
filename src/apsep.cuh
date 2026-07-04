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

    // ---- 7. Decoupled look-back ----
    // First scan earlier blocks within the same superblock (per-block trees),
    // then scan previous superblocks (superblock trees).
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;

            // Scan blocks sb_first .. block_id-1 within this superblock
            for (int b = block_id - 1; b >= sb_first && result < 0; b--) {
                while (d_states[b].status == APSEP_INVALID) { /* spin */ }
                if (d_states[b].block_min < val) {
                    const T* bt = d_block_trees + (size_t)b * (2 * B);
                    int leaf = treeRightmostSmaller<T>(bt, B, val);
                    result = (leaf >= 0) ? b * B + leaf : -1;
                }
            }

            // If still not found, scan previous superblocks
            if (result < 0)
                result = decoupledLookback<T, B, K>(
                    sb_id, val, d_sb_states, d_sb_trees);

            d_out[gid] = result;
        }
    }
}

// ===========================================================================
// Option A: Per-block look-back (no superblocks)
//
// Each block publishes its block_min + full tree immediately after intra-block
// work, with no waiting for siblings.  Look-back scans individual BlockState
// entries and their per-block trees.  Eliminates the last-in-SB wait entirely.
// ===========================================================================

template <typename T, int B>
__device__ int decoupledLookbackPerBlock(
        int                          block_id,
        T                            val,
        const BlockState<T>*         d_states,
        const T* __restrict__        d_block_trees) {
    for (int b = block_id - 1; b >= 0; b--) {
        while (d_states[b].status == APSEP_INVALID) { /* spin */ }
        if (d_states[b].block_min < val) {
            const T* bt = d_block_trees + (size_t)b * (2 * B);
            int leaf = treeRightmostSmaller<T>(bt, B, val);
            return (leaf >= 0) ? b * B + leaf : -1;
        }
    }
    return -1;
}

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void apsepKernelPerBlock(
        const T* __restrict__  d_in,
        int*                   d_out,
        int                    n,
        BlockState<T>*         d_states,
        T* __restrict__        d_block_trees,
        volatile uint32_t*     d_dyn_idx) {
    constexpr int B   = BLOCK_SIZE * IPT;
    const     T   INF = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B];

    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;

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

    // Publish immediately — no waiting for siblings
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

    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            d_out[gid] = decoupledLookbackPerBlock<T, B>(
                block_id, s_elems[lid], d_states, d_block_trees);
        }
    }
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for per-block look-back kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
struct ApsepPerBlockScratch {
    BlockState<T>* d_states      = nullptr;
    T*             d_block_trees = nullptr;
    uint32_t*      d_dyn_idx     = nullptr;
    int            num_blocks    = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
ApsepPerBlockScratch<T, BLOCK_SIZE, IPT> allocPerBlockScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    static_assert((B & (B - 1)) == 0, "BLOCK_SIZE * IPT must be a power of two");
    ApsepPerBlockScratch<T, BLOCK_SIZE, IPT> s;
    s.num_blocks = (n + B - 1) / B;
    gpuAssert(cudaMalloc(&s.d_states,      (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_trees, (size_t)s.num_blocks * 2 * B * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,     sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void freePerBlockScratch(ApsepPerBlockScratch<T, BLOCK_SIZE, IPT>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_trees);
    cudaFree(s.d_dyn_idx);
    s = ApsepPerBlockScratch<T, BLOCK_SIZE, IPT>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void runPerBlockApsep(const T* d_in, int* d_out, int n,
                      ApsepPerBlockScratch<T, BLOCK_SIZE, IPT>& s) {
    gpuAssert(cudaMemset(s.d_states,   0, (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,  0, sizeof(uint32_t)));
    apsepKernelPerBlock<T, BLOCK_SIZE, IPT>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.d_states, s.d_block_trees, s.d_dyn_idx);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void launchPerBlockApsep(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocPerBlockScratch<T, BLOCK_SIZE, IPT>(n);
    runPerBlockApsep<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freePerBlockScratch<T, BLOCK_SIZE, IPT>(s);
}

// ===========================================================================
// Option B: Warp-cooperative look-back (keep superblocks)
//
// Same superblock structure as apsepKernel but look-back is performed
// cooperatively per warp: lane 0 polls the SB status, then all 32 lanes
// together descend the superblock tree in parallel.  One warp handles one
// element that needs look-back.  Reduces the inactive-thread cost and
// divergence in the look-back loop.
//
// Elements needing look-back are collected into shared memory so warps can be
// assigned contiguously.  With BS=128 (4 warps) and IPT=2 (256 elements/block),
// at most 256 elements need look-back — each warp handles up to 64 elements
// sequentially.
// ===========================================================================

// Warp-cooperative rightmost-smaller search over a superblock tree of KB leaves.
// All 32 lanes participate; lane 0 does the descent, result is broadcast.
template <typename T>
__device__ int warpRightmostSmaller(const T* __restrict__ tree, int KB, T val) {
    int result;
    if (threadIdx.x % 32 == 0) {
        result = treeRightmostSmaller<T>(tree, KB, val);
    }
    result = __shfl_sync(0xffffffff, result, 0);
    return result;
}

// Parallel tree descent: all 32 lanes of a warp cooperate to descend one level
// per cycle.  Each lane tracks its own node position starting from a different
// starting point in the tree (spread across the leaf level).  This version
// assigns each lane a contiguous chunk of the leaf range to search in parallel.
template <typename T>
__device__ int warpParallelRightmostSmaller(const volatile T* __restrict__ tree,
                                             int KB, T val) {
    // Divide KB leaves into 32 equal segments; each lane finds the rightmost
    // element < val in its segment.  Then lane 31 (rightmost) is checked first,
    // then lane 30, etc., to find the overall rightmost.
    const int lane   = threadIdx.x & 31;
    const int seg    = (KB + 31) / 32;   // segment size (leaves per lane)
    const int lo_idx = lane * seg;
    const int hi_idx = min(lo_idx + seg, KB);

    // Each lane scans its leaf segment right-to-left
    int found = -1;
    for (int idx = hi_idx - 1; idx >= lo_idx; idx--) {
        if (tree[KB + idx] < val) { found = idx; break; }
    }

    // Find the rightmost across all lanes using warp shuffles
    // Lane with highest lane-index that has found >= 0 wins
    for (int mask = 16; mask >= 1; mask >>= 1) {
        int peer = __shfl_xor_sync(0xffffffff, found, mask);
        if (found < 0 || (peer >= 0 && peer > found)) found = peer;
    }
    return found;  // -1 if none found
}

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__ void apsepKernelWarpCoop(
        const T* __restrict__  d_in,
        int*                   d_out,
        int                    n,
        int                    num_blocks,
        int                    num_superblocks,
        BlockState<T>*         d_states,
        T* __restrict__        d_block_trees,
        SuperBlockState<T>*    d_sb_states,
        T* __restrict__        d_sb_trees,
        volatile uint32_t*     d_dyn_idx) {
    constexpr int B          = BLOCK_SIZE * IPT;
    constexpr int KB         = K * B;
    constexpr int NUM_WARPS  = BLOCK_SIZE / 32;
    const     T   INF        = ApsepInfinity<T>::value();
    const int     lane       = threadIdx.x & 31;
    const int     warp_id    = threadIdx.x >> 5;

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B];
    // Elements needing look-back: their values and output slots
    __shared__ T        s_lb_vals[B];
    __shared__ int      s_lb_gids[B];
    __shared__ int      s_lb_count;

    if (threadIdx.x == 0) {
        s_bid      = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
        s_lb_count = 0;
    }
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;

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
            if (local >= 0) {
                d_out[gid] = glb_offs + local;
            } else {
                d_out[gid] = INT_MIN;
                // Enqueue for look-back
                int slot = atomicAdd(&s_lb_count, 1);
                s_lb_vals[slot] = s_elems[lid];
                s_lb_gids[slot] = gid;
            }
        }
    }

    // Publish per-block tree
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

    // Last block in SB builds merged superblock tree
    int sb_size       = min(K, num_blocks - sb_first);
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

    // Warp-cooperative look-back: each warp handles a disjoint slice of
    // s_lb_vals[0..s_lb_count-1].  Within the warp, lane 0 drives the
    // superblock scan; all 32 lanes cooperate on the tree search.
    const int lb_count = s_lb_count;
    // Round up so each warp gets a contiguous slice
    // Warp warp_id handles elements [warp_id * ceil(lb_count/NUM_WARPS) .. ...)
    const int per_warp = (lb_count + NUM_WARPS - 1) / NUM_WARPS;
    const int my_start = warp_id * per_warp;
    const int my_end   = min(my_start + per_warp, lb_count);

    for (int ei = my_start; ei < my_end; ei++) {
        const T   val = s_lb_vals[ei];
        const int gid = s_lb_gids[ei];
        int result = -1;

        for (int sb = sb_id - 1; sb >= 0; sb--) {
            // Lane 0 spins; status broadcast to whole warp
            int status;
            if (lane == 0) {
                while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }
                status = (d_sb_states[sb].sb_min < val) ? 1 : 0;
            }
            status = __shfl_sync(0xffffffff, status, 0);

            if (status == 0) continue;  // min >= val, skip this SB

            // All 32 lanes cooperate: parallel leaf scan over KB leaves
            const T* st = d_sb_trees + (size_t)sb * (2 * KB);
            int leaf = warpParallelRightmostSmaller<T>(st, KB, val);
            result = (leaf >= 0) ? sb * KB + leaf : -1;
            break;
        }

        if (lane == 0)
            d_out[gid] = result;
    }
    // Threads not in look-back warps still need to write -1 for elements with
    // no look-back needed — already written as INT_MIN in the intra-block loop.
    // Fix up: elements still marked INT_MIN that weren't enqueued (n boundary)
    // are already guarded by `gid < n` above. Elements enqueued but result=-1
    // are written by lane 0 of their warp.  Elements resolved intra-block were
    // written immediately.  The only remaining issue: lb_count == 0 and no
    // lane writes INT_MIN elements.  But those were already written as valid
    // indices in the intra-block phase. We're good.
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for warp-cooperative look-back kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
struct ApsepWarpCoopScratch {
    BlockState<T>*      d_states      = nullptr;
    T*                  d_block_trees = nullptr;
    SuperBlockState<T>* d_sb_states   = nullptr;
    T*                  d_sb_trees    = nullptr;
    uint32_t*           d_dyn_idx     = nullptr;
    int                 num_blocks    = 0;
    int                 num_superblocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
ApsepWarpCoopScratch<T, BLOCK_SIZE, IPT, K> allocWarpCoopScratch(int n) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;
    static_assert((B & (B - 1)) == 0, "BLOCK_SIZE * IPT must be a power of two");
    ApsepWarpCoopScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;
    gpuAssert(cudaMalloc(&s.d_states,      (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_trees, (size_t)s.num_blocks      * 2 * B  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,   (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_sb_trees,    (size_t)s.num_superblocks * 2 * KB * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,     sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void freeWarpCoopScratch(ApsepWarpCoopScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_trees);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_sb_trees);
    cudaFree(s.d_dyn_idx);
    s = ApsepWarpCoopScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void runWarpCoopApsep(const T* d_in, int* d_out, int n,
                      ApsepWarpCoopScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));
    apsepKernelWarpCoop<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_trees,
            s.d_sb_states, s.d_sb_trees,
            s.d_dyn_idx);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void launchWarpCoopApsep(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocWarpCoopScratch<T, BLOCK_SIZE, IPT, K>(n);
    runWarpCoopApsep<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWarpCoopScratch<T, BLOCK_SIZE, IPT, K>(s);
}

// ===========================================================================
// Variant 1 (V1): Warp-per-element, per-block trees
//
// Like Option A (publish immediately, no superblocks) but look-back uses one
// full warp per element.  Lane 0 spins on BlockState.status; all 32 lanes
// then scan the per-block tree in parallel.  Increases active threads during
// look-back from ~8/32 to 32/32 for that phase.
// ===========================================================================

// All 32 lanes of the calling warp scan tree[B..2B-1] (the leaf layer of a
// per-block 1-indexed min-tree of size 2*B) right-to-left in parallel.
// Each lane covers a contiguous segment; the global rightmost index < val
// is found via a max-reduction across lanes.
template <typename T, int B>
__device__ int warpScanPerBlockTree(const T* __restrict__ tree, T val) {
    const int lane = threadIdx.x & 31;
    constexpr int SEG = (B + 31) / 32;
    const int lo = lane * SEG;
    const int hi = min(lo + SEG, B);
    int found = -1;
    for (int idx = hi - 1; idx >= lo; idx--) {
        if (tree[B + idx] < val) { found = idx; break; }
    }
    // Max-reduce across all 32 lanes
    for (int mask = 16; mask >= 1; mask >>= 1) {
        int peer = __shfl_xor_sync(0xffffffff, found, mask);
        if (found < 0 || (peer >= 0 && peer > found)) found = peer;
    }
    return found;
}

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void apsepKernelV1(
        const T* __restrict__  d_in,
        int*                   d_out,
        int                    n,
        BlockState<T>*         d_states,
        T* __restrict__        d_block_trees,
        volatile uint32_t*     d_dyn_idx) {
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    const int     lane      = threadIdx.x & 31;
    const int     warp_id   = threadIdx.x >> 5;

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B];
    __shared__ T        s_lb_vals[B];
    __shared__ int      s_lb_gids[B];
    __shared__ int      s_lb_count;

    if (threadIdx.x == 0) {
        s_bid      = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
        s_lb_count = 0;
    }
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;

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
            if (local >= 0) {
                d_out[gid] = glb_offs + local;
            } else {
                int slot = atomicAdd(&s_lb_count, 1);
                s_lb_vals[slot] = s_elems[lid];
                s_lb_gids[slot] = gid;
                d_out[gid] = -1;
            }
        }
    }

    // Publish immediately
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

    // Warp-per-element look-back: warp warp_id handles elements
    // [warp_id * ceil(lb_count/NUM_WARPS) .. ...)
    const int lb_count = s_lb_count;
    const int per_warp = (lb_count + NUM_WARPS - 1) / NUM_WARPS;
    const int my_start = warp_id * per_warp;
    const int my_end   = min(my_start + per_warp, lb_count);

    for (int ei = my_start; ei < my_end; ei++) {
        const T   val = s_lb_vals[ei];
        const int gid = s_lb_gids[ei];
        int result = -1;

        for (int b = block_id - 1; b >= 0; b--) {
            // Lane 0 spins; broadcasts status
            int ready;
            if (lane == 0) {
                while (d_states[b].status == APSEP_INVALID) {}
                ready = (d_states[b].block_min < val) ? 1 : 0;
            }
            ready = __shfl_sync(0xffffffff, ready, 0);
            if (!ready) continue;

            const T* bt = d_block_trees + (size_t)b * (2 * B);
            int leaf = warpScanPerBlockTree<T, B>(bt, val);
            result = (leaf >= 0) ? b * B + leaf : -1;
            break;
        }
        if (lane == 0)
            d_out[gid] = result;
    }
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for V1
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
struct ApsepV1Scratch {
    BlockState<T>* d_states      = nullptr;
    T*             d_block_trees = nullptr;
    uint32_t*      d_dyn_idx     = nullptr;
    int            num_blocks    = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
ApsepV1Scratch<T, BLOCK_SIZE, IPT> allocV1Scratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    static_assert((B & (B - 1)) == 0, "B must be power of two");
    ApsepV1Scratch<T, BLOCK_SIZE, IPT> s;
    s.num_blocks = (n + B - 1) / B;
    gpuAssert(cudaMalloc(&s.d_states,      (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_trees, (size_t)s.num_blocks * 2 * B * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,     sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void freeV1Scratch(ApsepV1Scratch<T, BLOCK_SIZE, IPT>& s) {
    cudaFree(s.d_states); cudaFree(s.d_block_trees); cudaFree(s.d_dyn_idx);
    s = ApsepV1Scratch<T, BLOCK_SIZE, IPT>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void runV1Apsep(const T* d_in, int* d_out, int n, ApsepV1Scratch<T, BLOCK_SIZE, IPT>& s) {
    gpuAssert(cudaMemset(s.d_states,  0, (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx, 0, sizeof(uint32_t)));
    apsepKernelV1<T, BLOCK_SIZE, IPT>
        <<<s.num_blocks, BLOCK_SIZE>>>(d_in, d_out, n, s.d_states, s.d_block_trees, s.d_dyn_idx);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void launchV1Apsep(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocV1Scratch<T, BLOCK_SIZE, IPT>(n);
    runV1Apsep<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeV1Scratch<T, BLOCK_SIZE, IPT>(s);
}

// ===========================================================================
// Variant 2 (V2): Full-block cooperative, one element at a time
//
// Like V1 but the entire BLOCK_SIZE threads cooperate on each look-back
// element serially.  Thread 0 spins; all BLOCK_SIZE threads scan the tree
// in parallel.  For BLOCK_SIZE=128 and B=256: 128 threads cover 256 leaves
// in 2 reads each — vs the tree-descent O(log B).  Maximises parallel work
// per spin-wait.
// ===========================================================================

template <typename T, int B, int BLOCK_SIZE>
__device__ int blockScanPerBlockTree(const T* __restrict__ tree, T val) {
    // Each thread covers (2B / BLOCK_SIZE) leaf positions
    // We want the rightmost leaf < val.  Each thread finds its local rightmost,
    // then we reduce across the block via shared memory.
    constexpr int SEG = (B + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int lo = threadIdx.x * SEG;
    const int hi = min(lo + SEG, B);
    int found = -1;
    for (int idx = hi - 1; idx >= lo; idx--) {
        if (tree[B + idx] < val) { found = idx; break; }
    }
    // Warp-level max
    for (int mask = 16; mask >= 1; mask >>= 1) {
        int peer = __shfl_xor_sync(0xffffffff, found, mask);
        if (found < 0 || (peer >= 0 && peer > found)) found = peer;
    }
    // Cross-warp reduction via shared memory
    // Caller must provide smem — we use a fixed-size approach with warp leaders
    return found;  // each warp holds its own max; caller reduces across warps
}

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void apsepKernelV2(
        const T* __restrict__  d_in,
        int*                   d_out,
        int                    n,
        BlockState<T>*         d_states,
        T* __restrict__        d_block_trees,
        volatile uint32_t*     d_dyn_idx) {
    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();
    const int     lane      = threadIdx.x & 31;
    const int     warp_id   = threadIdx.x >> 5;

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B];
    __shared__ T        s_lb_vals[B];
    __shared__ int      s_lb_gids[B];
    __shared__ int      s_lb_count;
    __shared__ int      s_warp_found[NUM_WARPS];  // cross-warp reduction scratch

    if (threadIdx.x == 0) {
        s_bid      = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
        s_lb_count = 0;
    }
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;

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
            if (local >= 0) {
                d_out[gid] = glb_offs + local;
            } else {
                int slot = atomicAdd(&s_lb_count, 1);
                s_lb_vals[slot] = s_elems[lid];
                s_lb_gids[slot] = gid;
                d_out[gid] = -1;
            }
        }
    }

    // Publish immediately
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

    // Full-block cooperative look-back, one element at a time
    const int lb_count = s_lb_count;
    for (int ei = 0; ei < lb_count; ei++) {
        const T   val = s_lb_vals[ei];
        const int gid = s_lb_gids[ei];
        int result = -1;

        for (int b = block_id - 1; b >= 0; b--) {
            // Thread 0 spins; broadcasts readiness
            int ready = 0;
            if (threadIdx.x == 0) {
                while (d_states[b].status == APSEP_INVALID) {}
                ready = (d_states[b].block_min < val) ? 1 : 0;
            }
            // Broadcast via shared mem (no __ballot needed, just a smem write)
            if (threadIdx.x == 0) s_warp_found[0] = ready;
            __syncthreads();
            ready = s_warp_found[0];
            if (!ready) continue;

            // All BLOCK_SIZE threads scan the leaf layer in parallel
            const T* bt = d_block_trees + (size_t)b * (2 * B);
            constexpr int SEG = (B + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int lo = threadIdx.x * SEG;
            const int hi = min(lo + SEG, B);
            int found = -1;
            for (int idx = hi - 1; idx >= lo; idx--) {
                if (bt[B + idx] < val) { found = idx; break; }
            }
            // Warp-level max
            for (int mask = 16; mask >= 1; mask >>= 1) {
                int peer = __shfl_xor_sync(0xffffffff, found, mask);
                if (found < 0 || (peer >= 0 && peer > found)) found = peer;
            }
            // Warp leaders write to shared; thread 0 reduces
            if (lane == 0) s_warp_found[warp_id] = found;
            __syncthreads();
            if (threadIdx.x == 0) {
                int best = -1;
                for (int w = 0; w < NUM_WARPS; w++) {
                    int v = s_warp_found[w];
                    if (v > best) best = v;
                }
                result = (best >= 0) ? b * B + best : -1;
                d_out[gid] = result;
                s_warp_found[0] = result;  // signal done to avoid re-scan
            }
            __syncthreads();
            result = s_warp_found[0];
            break;
        }
        // If loop exhausted without break, result == -1, already written
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for V2
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
using ApsepV2Scratch = ApsepV1Scratch<T, BLOCK_SIZE, IPT>;

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
ApsepV2Scratch<T, BLOCK_SIZE, IPT> allocV2Scratch(int n) { return allocV1Scratch<T, BLOCK_SIZE, IPT>(n); }

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void freeV2Scratch(ApsepV2Scratch<T, BLOCK_SIZE, IPT>& s) { freeV1Scratch<T, BLOCK_SIZE, IPT>(s); }

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void runV2Apsep(const T* d_in, int* d_out, int n, ApsepV2Scratch<T, BLOCK_SIZE, IPT>& s) {
    gpuAssert(cudaMemset(s.d_states,  0, (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx, 0, sizeof(uint32_t)));
    apsepKernelV2<T, BLOCK_SIZE, IPT>
        <<<s.num_blocks, BLOCK_SIZE>>>(d_in, d_out, n, s.d_states, s.d_block_trees, s.d_dyn_idx);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void launchV2Apsep(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocV2Scratch<T, BLOCK_SIZE, IPT>(n);
    runV2Apsep<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeV2Scratch<T, BLOCK_SIZE, IPT>(s);
}

// ===========================================================================
// Variant 3 (V3): Full-block batched look-back
//
// Collect all M look-back elements.  Then scan candidate blocks right-to-left:
// for each candidate block, all BLOCK_SIZE threads read its leaf layer and
// compute, for each of the M queued elements, whether that block contains
// its answer.  This amortizes the per-block spin-wait across all M elements
// at once — one spin per candidate block, not M spins.
//
// Implementation: each thread handles one queued element and one leaf position
// per candidate block.  With B leaves and BLOCK_SIZE threads, one pass covers
// a contiguous segment of leaves per thread.  After checking all leaves for
// the candidate block, each queued element (thread) records its answer if
// found and is marked done.  The outer loop terminates when all elements are
// resolved or no more candidate blocks remain.
//
// Layout: thread t handles element t (for t < lb_count) and also participates
// in the leaf scan.  Since lb_count <= B and BLOCK_SIZE <= B, each thread maps
// 1-to-1 to a queued element (if lb_count <= BLOCK_SIZE) or threads cycle
// through elements in chunks.
// ===========================================================================

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void apsepKernelV3(
        const T* __restrict__  d_in,
        int*                   d_out,
        int                    n,
        BlockState<T>*         d_states,
        T* __restrict__        d_block_trees,
        volatile uint32_t*     d_dyn_idx) {
    constexpr int B         = BLOCK_SIZE * IPT;
    const     T   INF       = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B];
    __shared__ T        s_lb_vals[B];
    __shared__ int      s_lb_gids[B];
    __shared__ int      s_lb_ans[B];
    __shared__ int      s_lb_count;
    __shared__ int      s_remaining;
    __shared__ int      s_block_ready;

    if (threadIdx.x == 0) {
        s_bid      = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
        s_lb_count = 0;
        s_remaining = 0;
    }
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;

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
            if (local >= 0) {
                d_out[gid] = glb_offs + local;
            } else {
                int slot = atomicAdd(&s_lb_count, 1);
                s_lb_vals[slot] = s_elems[lid];
                s_lb_gids[slot] = gid;
                s_lb_ans[slot]  = -1;
                d_out[gid] = -1;
            }
        }
    }

    // Publish immediately
    T* g_block_tree = d_block_trees + (size_t)block_id * (2 * B);
    for (int i = threadIdx.x; i < 2 * B; i += BLOCK_SIZE)
        g_block_tree[i] = s_tree[i];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        d_states[block_id].block_min = s_tree[1];
        __threadfence();
        d_states[block_id].status = APSEP_READY;
        s_remaining = s_lb_count;
    }
    __syncthreads();

    const int lb_count = s_lb_count;
    if (lb_count == 0) return;

    // Batched look-back: scan candidate blocks right-to-left.
    // For each candidate block:
    //   1. Thread 0 spins on status.
    //   2. All threads check: is block_min < my element's val?
    //      (Thread t handles element t, for t < lb_count.)
    //   3. Each thread t whose element is not yet answered and block_min < val[t]:
    //      scan the candidate block's B leaves in my assigned segment to find
    //      the rightmost leaf < val[t].
    //   4. Each thread t records its answer in s_lb_ans[t] if found, writes
    //      d_out[gid] and marks itself done.
    //   5. After all threads are done or remaining==0, stop.
    //
    // Thread assignment: thread t handles queued element t (t < lb_count).
    // For the leaf scan: each thread also covers leaf segment
    // [t * SEG .. (t+1)*SEG - 1] for its own element.
    // This works cleanly when lb_count == BLOCK_SIZE; when lb_count < BLOCK_SIZE
    // the extra threads participate only in the leaf scan for elements < lb_count
    // (they help by covering more leaves per element via the s_warp_max path below).
    //
    // For simplicity: each thread t < lb_count does its own full leaf scan
    // (iterating over all B leaves) independently.  This is O(B) per thread per
    // candidate block, but all lb_count threads run in parallel.  The key gain
    // is one spin-wait per candidate block instead of lb_count spin-waits.

    for (int b = block_id - 1; b >= 0 && s_remaining > 0; b--) {
        // Thread 0 waits for candidate block
        if (threadIdx.x == 0) {
            while (d_states[b].status == APSEP_INVALID) {}
            s_block_ready = 1;  // will check per-element below
        }
        __syncthreads();

        // Each thread t (for t < lb_count) checks if this block can answer it
        const T* bt = d_block_trees + (size_t)b * (2 * B);
        const T  bmin = d_states[b].block_min;

        // Each thread t independently scans leaves if not yet answered
        if (threadIdx.x < lb_count && s_lb_ans[threadIdx.x] == -1) {
            const T val = s_lb_vals[threadIdx.x];
            if (bmin < val) {
                // Scan leaves right-to-left for this element
                int found = -1;
                for (int idx = B - 1; idx >= 0; idx--) {
                    if (bt[B + idx] < val) { found = idx; break; }
                }
                if (found >= 0) {
                    s_lb_ans[threadIdx.x] = b * B + found;
                    d_out[s_lb_gids[threadIdx.x]] = b * B + found;
                    atomicSub(&s_remaining, 1);
                }
            }
        }
        __syncthreads();
    }
    // Elements still at -1 have no answer (d_out already -1)
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for V3 (shares layout with V1)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
using ApsepV3Scratch = ApsepV1Scratch<T, BLOCK_SIZE, IPT>;

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
ApsepV3Scratch<T, BLOCK_SIZE, IPT> allocV3Scratch(int n) { return allocV1Scratch<T, BLOCK_SIZE, IPT>(n); }

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void freeV3Scratch(ApsepV3Scratch<T, BLOCK_SIZE, IPT>& s) { freeV1Scratch<T, BLOCK_SIZE, IPT>(s); }

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void runV3Apsep(const T* d_in, int* d_out, int n, ApsepV3Scratch<T, BLOCK_SIZE, IPT>& s) {
    gpuAssert(cudaMemset(s.d_states,  0, (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx, 0, sizeof(uint32_t)));
    apsepKernelV3<T, BLOCK_SIZE, IPT>
        <<<s.num_blocks, BLOCK_SIZE>>>(d_in, d_out, n, s.d_states, s.d_block_trees, s.d_dyn_idx);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2>
void launchV3Apsep(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocV3Scratch<T, BLOCK_SIZE, IPT>(n);
    runV3Apsep<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeV3Scratch<T, BLOCK_SIZE, IPT>(s);
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
// WarpScanLeaves APSEP
//
// Replaces the shared-memory min-tree with a warp prefix-min scan for
// intra-block PSE, and publishes only B leaf values (raw elements) to global
// memory instead of the full 2*B tree.  This cuts shared memory from ~3 KB
// to ~1 KB per block, roughly tripling occupancy on SM 7.5 and eliminating
// N bytes of global writes.
//
// The SB tree is built from the stored leaf values (identical content to the
// min-tree leaves, just read directly).  Intra-SB cross-block look-back does
// a linear scan over the B stored leaves instead of a tree query.
// ===========================================================================

// Build SB tree from per-block leaf buffers (B values per block)
template <typename T, int BLOCK_SIZE, int IPT, int K>
__device__ void buildSuperBlockTreeFromLeaves(
        const T* __restrict__ g_block_leaves,  // num_blocks * B
        T*       __restrict__ g_sb_tree,        // 2*KB output
        int sb_blocks, int n_total, int sb_first_block, T INF) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;

    for (int i = threadIdx.x; i < KB; i += BLOCK_SIZE) {
        int b   = i / B;
        int lid = i % B;
        T val;
        if (b < sb_blocks) {
            int gid = (sb_first_block + b) * B + lid;
            val = (gid < n_total) ? g_block_leaves[(size_t)(sb_first_block + b) * B + lid] : INF;
        } else {
            val = INF;
        }
        g_sb_tree[KB + i] = val;
    }
    __syncthreads();

    for (int half = KB >> 1; half >= 1; half >>= 1) {
        for (int i = threadIdx.x; i < half; i += BLOCK_SIZE) {
            int node = half + i;
            g_sb_tree[node] = min(g_sb_tree[2 * node], g_sb_tree[2 * node + 1]);
        }
        __syncthreads();
    }
}

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__
void apsepKernelWarpScanLeaves(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     num_superblocks,
        BlockState<T>*          d_states,
        T* __restrict__         d_block_leaves,   // B per block (raw elements)
        SuperBlockState<T>*     d_sb_states,
        T* __restrict__         d_sb_trees,
        volatile uint32_t*      d_dyn_idx) {

    constexpr int B        = BLOCK_SIZE * IPT;
    constexpr int KB       = K * B;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF      = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_stripe_min[IPT][NUM_WARPS];

    // ---- 1. Dynamic block assignment ----
    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    // ---- 2. Load elements ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();

    // ---- 3. Warp prefix-min scan — intra-block PSE ----
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

    // ---- 4. Intra-block backward search ----
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

    // ---- 5. Publish B leaf values and mark block READY ----
    T* g_leaves = d_block_leaves + (size_t)block_id * B;
    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
        g_leaves[i] = s_elems[i];
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        // block_min = min of all elements = s_stripe_min[IPT-1][NUM_WARPS-1]
        // but easier: just read s_elems minimum via warp reduction already in s_stripe_min
        T bmin = s_stripe_min[0][0];
        for (int ipt = 0; ipt < IPT; ipt++)
            for (int w = 0; w < NUM_WARPS; w++)
                bmin = min(bmin, s_stripe_min[ipt][w]);
        d_states[block_id].block_min = bmin;
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    // ---- 6. Last block in SB builds merged SB tree ----
    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        T* g_sb_tree = d_sb_trees + (size_t)sb_id * (2 * KB);
        buildSuperBlockTreeFromLeaves<T, BLOCK_SIZE, IPT, K>(
            d_block_leaves, g_sb_tree, sb_size, n, sb_first, INF);

        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0) {
            d_sb_states[sb_id].sb_min = g_sb_tree[1];
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // ---- 7. Decoupled look-back ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;

            // Intra-SB: linear scan over earlier blocks' leaves
            for (int b = block_id - 1; b >= sb_first && result < 0; b--) {
                while (d_states[b].status == APSEP_INVALID) { /* spin */ }
                if (d_states[b].block_min < val) {
                    const T* bl = d_block_leaves + (size_t)b * B;
                    for (int k = B - 1; k >= 0; k--)
                        if (bl[k] < val) { result = b * B + k; break; }
                }
            }

            // Inter-SB: tree look-back over previous superblocks
            if (result < 0)
                result = decoupledLookback<T, B, K>(
                    sb_id, val, d_sb_states, d_sb_trees);

            d_out[gid] = result;
        }
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
struct ApsepWarpScanLeavesScratch {
    BlockState<T>*      d_states        = nullptr;
    T*                  d_block_leaves  = nullptr;
    SuperBlockState<T>* d_sb_states     = nullptr;
    T*                  d_sb_trees      = nullptr;
    uint32_t*           d_dyn_idx       = nullptr;
    int                 num_blocks      = 0;
    int                 num_superblocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
ApsepWarpScanLeavesScratch<T, BLOCK_SIZE, IPT, K> allocWarpScanLeavesScratch(int n) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;
    static_assert((B & (B - 1)) == 0, "BLOCK_SIZE * IPT must be a power of two");

    ApsepWarpScanLeavesScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;

    gpuAssert(cudaMalloc(&s.d_states,       (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_leaves, (size_t)s.num_blocks      * B  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,    (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_sb_trees,     (size_t)s.num_superblocks * 2 * KB * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,      sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void freeWarpScanLeavesScratch(ApsepWarpScanLeavesScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_sb_trees);
    cudaFree(s.d_dyn_idx);
    s = ApsepWarpScanLeavesScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void runWarpScanLeaves(const T* d_in, int* d_out, int n,
                       ApsepWarpScanLeavesScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));

    apsepKernelWarpScanLeaves<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_leaves,
            s.d_sb_states, s.d_sb_trees,
            s.d_dyn_idx);

    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void launchWarpScanLeaves(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocWarpScanLeavesScratch<T, BLOCK_SIZE, IPT, K>(n);
    runWarpScanLeaves<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWarpScanLeavesScratch<T, BLOCK_SIZE, IPT, K>(s);
}

// ===========================================================================
// WarpScanNoTree APSEP
//
// Like WarpScanLeaves but removes the SB min-tree entirely.  The inter-SB
// decoupled look-back replaces tree traversal with:
//   1. Check sb_min (one read, same as before)
//   2. Scan K block_mins in that SB to find the rightmost matching block
//   3. Linear scan of that block's B leaf values
//
// This eliminates the d_sb_trees allocation (saves ~32 KB × num_SBs) and
// replaces O(log KB) pointer-chasing tree reads with sequential block_min +
// leaf reads.  For large N where the SB tree working set doesn't fit in L2,
// sequential access patterns enable hardware prefetching and reduce misses.
// ===========================================================================

template <typename T, int B, int K>
// d_block_mins is a packed T array (4 bytes/entry) vs BlockState (16 bytes/entry),
// giving 4× better cache utilization when scanning K block_mins per SB.
__device__ int decoupledLookbackNoTree(
        int                             sb_id,
        T                               val,
        const SuperBlockState<T>*       d_sb_states,
        const T* __restrict__           d_block_mins,
        const T* __restrict__           d_block_leaves,
        int                             num_blocks) {
    for (int sb = sb_id - 1; sb >= 0; sb--) {
        while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }
        if (d_sb_states[sb].sb_min >= val) continue;

        int sb_first = sb * K;
        int sb_last  = min(sb_first + K, num_blocks) - 1;

        // scan blocks right-to-left using packed block_min array
        for (int b = sb_last; b >= sb_first; b--) {
            if (__ldg(&d_block_mins[b]) < val) {
                const T* bl = d_block_leaves + (size_t)b * B;
                for (int k = B - 1; k >= 0; k--)
                    if (__ldg(&bl[k]) < val) return b * B + k;
            }
        }
    }
    return -1;
}

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__
void apsepKernelWarpScanNoTree(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     num_superblocks,
        BlockState<T>*          d_states,
        T* __restrict__         d_block_leaves,
        T* __restrict__         d_block_mins,   // packed block_min array (4 bytes/entry)
        SuperBlockState<T>*     d_sb_states,
        volatile uint32_t*      d_dyn_idx) {

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_stripe_min[IPT][NUM_WARPS];

    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

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

    // Publish B leaves
    T* g_leaves = d_block_leaves + (size_t)block_id * B;
    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
        g_leaves[i] = s_elems[i];
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        T bmin = s_stripe_min[0][0];
        for (int ipt = 0; ipt < IPT; ipt++)
            for (int w = 0; w < NUM_WARPS; w++)
                bmin = min(bmin, s_stripe_min[ipt][w]);
        d_block_mins[block_id] = bmin;           // packed: 4-byte stride
        d_states[block_id].block_min = bmin;     // also write to states for intra-SB spin
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    // Last block in SB: wait for siblings, publish SB state (no tree build)
    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        if (threadIdx.x == 0) {
            T sb_min = d_block_mins[sb_first];
            for (int b = sb_first + 1; b <= block_id; b++)
                sb_min = min(sb_min, d_block_mins[b]);
            __threadfence();
            d_sb_states[sb_id].sb_min = sb_min;
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // Decoupled look-back (no tree)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;

            // Intra-SB: spin on status (in d_states), then use packed block_min
            for (int b = block_id - 1; b >= sb_first && result < 0; b--) {
                while (d_states[b].status == APSEP_INVALID) { /* spin */ }
                if (__ldg(&d_block_mins[b]) < val) {
                    const T* bl = d_block_leaves + (size_t)b * B;
                    for (int k = B - 1; k >= 0; k--)
                        if (__ldg(&bl[k]) < val) { result = b * B + k; break; }
                }
            }

            // Inter-SB: scan packed block_mins + leaves (no tree)
            if (result < 0)
                result = decoupledLookbackNoTree<T, B, K>(
                    sb_id, val, d_sb_states, d_block_mins, d_block_leaves, num_blocks);

            d_out[gid] = result;
        }
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
struct ApsepWarpScanNoTreeScratch {
    BlockState<T>*      d_states        = nullptr;
    T*                  d_block_leaves  = nullptr;
    T*                  d_block_mins    = nullptr;  // packed 4-byte-stride block_min array
    SuperBlockState<T>* d_sb_states     = nullptr;
    uint32_t*           d_dyn_idx       = nullptr;
    int                 num_blocks      = 0;
    int                 num_superblocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
ApsepWarpScanNoTreeScratch<T, BLOCK_SIZE, IPT, K> allocWarpScanNoTreeScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;

    ApsepWarpScanNoTreeScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;

    gpuAssert(cudaMalloc(&s.d_states,       (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_leaves, (size_t)s.num_blocks      * B  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_mins,   (size_t)s.num_blocks      * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,    (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,      sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void freeWarpScanNoTreeScratch(ApsepWarpScanNoTreeScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_block_mins);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_dyn_idx);
    s = ApsepWarpScanNoTreeScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void runWarpScanNoTree(const T* d_in, int* d_out, int n,
                       ApsepWarpScanNoTreeScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));

    apsepKernelWarpScanNoTree<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_leaves, s.d_block_mins,
            s.d_sb_states, s.d_dyn_idx);

    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void launchWarpScanNoTree(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocWarpScanNoTreeScratch<T, BLOCK_SIZE, IPT, K>(n);
    runWarpScanNoTree<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWarpScanNoTreeScratch<T, BLOCK_SIZE, IPT, K>(s);
}

// ===========================================================================
// WarpMinHierarchy APSEP
//
// Like WarpScanNoTree but replaces the flat B-element leaf array with a
// two-level structure per block:
//   d_block_warp_mins[num_blocks * W]  – W = B/32 warp-min values (packed T)
//   d_block_leaves[num_blocks * B]     – B full leaf values
//
// The backward leaf scan becomes:
//   1. Scan W warp-mins backward to find the rightmost matching warp (W reads,
//      all in 1-2 cache lines since W=16 for B=512).
//   2. Scan that warp's 32 leaves backward (32 reads = 1 cache line).
//
// This vs WarpScanNoTree: instead of scanning up to B=512 leaves with 9.2/32
// sector utilization (early exit leaves most of each cache line unused), we
// scan W=16 warp-mins (100% utilization, 1 cache line) then exactly 32 leaves
// (100% utilization, 1 cache line).  Expected ~4-5x reduction in L1/L2 traffic
// for the leaf-scan phase on random data.
// ===========================================================================

template <typename T, int B, int K>
__device__ int decoupledLookbackWarpMin(
        int                             sb_id,
        T                               val,
        const SuperBlockState<T>*       d_sb_states,
        const T* __restrict__           d_block_mins,
        const T* __restrict__           d_block_warp_mins,  // W = B/32 entries per block
        const T* __restrict__           d_block_leaves,
        int                             num_blocks) {
    constexpr int W = B / 32;  // number of warps per block

    for (int sb = sb_id - 1; sb >= 0; sb--) {
        while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }
        if (d_sb_states[sb].sb_min >= val) continue;

        int sb_first = sb * K;
        int sb_last  = min(sb_first + K, num_blocks) - 1;

        for (int b = sb_last; b >= sb_first; b--) {
            if (__ldg(&d_block_mins[b]) < val) {
                const T* wm = d_block_warp_mins + (size_t)b * W;
                // Find rightmost matching warp
                for (int w = W - 1; w >= 0; w--) {
                    if (__ldg(&wm[w]) < val) {
                        const T* bl = d_block_leaves + (size_t)b * B + w * 32;
                        for (int k = 31; k >= 0; k--)
                            if (__ldg(&bl[k]) < val) return b * B + w * 32 + k;
                    }
                }
            }
        }
    }
    return -1;
}

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__
void apsepKernelWarpMinHierarchy(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     num_superblocks,
        BlockState<T>*          d_states,
        T* __restrict__         d_block_leaves,
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins,  // W = B/32 entries per block
        SuperBlockState<T>*     d_sb_states,
        volatile uint32_t*      d_dyn_idx) {

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W         = B / 32;  // warp-mins per block = NUM_WARPS * IPT
    const     T   INF       = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_stripe_min[IPT][NUM_WARPS];

    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

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

    // Publish leaves
    T* g_leaves = d_block_leaves + (size_t)block_id * B;
    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
        g_leaves[i] = s_elems[i];

    // Publish per-warp mins: each warp stores its warp-min from s_stripe_min.
    // Layout: d_block_warp_mins[block_id * W + ipt * NUM_WARPS + warp_id]
    // All W = IPT * NUM_WARPS entries written by separate warps.
    if (lane == 0) {
        T* g_wm = d_block_warp_mins + (size_t)block_id * W;
        for (int ipt = 0; ipt < IPT; ipt++)
            g_wm[ipt * NUM_WARPS + warp_id] = s_stripe_min[ipt][warp_id];
    }
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        T bmin = s_stripe_min[0][0];
        for (int ipt = 0; ipt < IPT; ipt++)
            for (int w = 0; w < NUM_WARPS; w++)
                bmin = min(bmin, s_stripe_min[ipt][w]);
        d_block_mins[block_id] = bmin;
        d_states[block_id].block_min = bmin;
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        if (threadIdx.x == 0) {
            T sb_min = d_block_mins[sb_first];
            for (int b = sb_first + 1; b <= block_id; b++)
                sb_min = min(sb_min, d_block_mins[b]);
            __threadfence();
            d_sb_states[sb_id].sb_min = sb_min;
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // Decoupled look-back using warp-min hierarchy
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;

            // Intra-SB: spin on per-block status, then warp-min hierarchy
            for (int b = block_id - 1; b >= sb_first && result < 0; b--) {
                while (d_states[b].status == APSEP_INVALID) { /* spin */ }
                if (__ldg(&d_block_mins[b]) < val) {
                    constexpr int W_local = B / 32;
                    const T* wm = d_block_warp_mins + (size_t)b * W_local;
                    for (int w = W_local - 1; w >= 0; w--) {
                        if (__ldg(&wm[w]) < val) {
                            const T* bl = d_block_leaves + (size_t)b * B + w * 32;
                            for (int k = 31; k >= 0; k--)
                                if (__ldg(&bl[k]) < val) { result = b * B + w * 32 + k; break; }
                            break;
                        }
                    }
                }
            }

            if (result < 0)
                result = decoupledLookbackWarpMin<T, B, K>(
                    sb_id, val, d_sb_states, d_block_mins, d_block_warp_mins,
                    d_block_leaves, num_blocks);

            d_out[gid] = result;
        }
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
struct ApsepWarpMinHierarchyScratch {
    BlockState<T>*      d_states          = nullptr;
    T*                  d_block_leaves    = nullptr;
    T*                  d_block_mins      = nullptr;
    T*                  d_block_warp_mins = nullptr;
    SuperBlockState<T>* d_sb_states       = nullptr;
    uint32_t*           d_dyn_idx         = nullptr;
    int                 num_blocks        = 0;
    int                 num_superblocks   = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
ApsepWarpMinHierarchyScratch<T, BLOCK_SIZE, IPT, K> allocWarpMinHierarchyScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    constexpr int W = B / 32;

    ApsepWarpMinHierarchyScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;

    gpuAssert(cudaMalloc(&s.d_states,          (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_leaves,    (size_t)s.num_blocks      * B  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_mins,      (size_t)s.num_blocks          * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_warp_mins, (size_t)s.num_blocks      * W  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,       (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,         sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void freeWarpMinHierarchyScratch(ApsepWarpMinHierarchyScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_block_mins);
    cudaFree(s.d_block_warp_mins);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_dyn_idx);
    s = ApsepWarpMinHierarchyScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void runWarpMinHierarchy(const T* d_in, int* d_out, int n,
                         ApsepWarpMinHierarchyScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));

    apsepKernelWarpMinHierarchy<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_leaves, s.d_block_mins, s.d_block_warp_mins,
            s.d_sb_states, s.d_dyn_idx);

    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void launchWarpMinHierarchy(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocWarpMinHierarchyScratch<T, BLOCK_SIZE, IPT, K>(n);
    runWarpMinHierarchy<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWarpMinHierarchyScratch<T, BLOCK_SIZE, IPT, K>(s);
}

// ===========================================================================
// WarpScanTreeLookup (WSTL) — two-pass hybrid
//
// Problem: worst-case descending input makes all spin-based single-pass
// kernels collapse to ~1 GB/s because blocks spin-wait on a O(N/B) serial
// chain.  NCU confirms: 99%+ of samples are on ISETP.NE.AND status checks.
//
// Solution: decouple the intra-block phase from the inter-block look-back.
//   Pass 1: One kernel handles intra-block PSE (fully parallel, no waits).
//           Thread 0 writes block_min and B leaves; marks INT_MIN in d_out
//           for elements needing inter-block look-back.  No spin-waits at all.
//   Pass 2: Build a 0-indexed segment min-tree over d_block_mins[num_blocks].
//           Uses ceil-to-power-of-2 leaves; tree has 2*M-1 nodes.
//           For N=131M: num_blocks=256K, tree ≈ 2 MB — fits in L2 cache.
//   Pass 3: Each element with d_out==INT_MIN queries the tree to find the
//           rightmost block with a smaller element, then linearly scans that
//           block's W warp-mins + 32 leaves.  Fully parallel, O(log num_blocks)
//           tree traversal, input-independent cost.
//
// vs BSZ: BSZ builds a tree over all N elements (N=131M → 512 MB tree,
// doesn't fit in L2).  WSTL builds a tree over num_blocks=256K block_mins
// (2 MB tree, fits in L2), then does one block-leaf scan.  Should be faster.
// ===========================================================================

// --- Pass 1: intra-block only, no spin-wait ---
template <typename T, int BLOCK_SIZE, int IPT>
__global__
void wstlPass1Kernel(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        T* __restrict__         d_block_leaves,
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins) {

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W         = B / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    const int block_id = (int)blockIdx.x;
    const int glb_offs = block_id * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];

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
        // Mark INT_MIN if needs inter-block look-back, else write global index
        d_out[gid] = (result >= 0) ? (glb_offs + result) : INT_MIN;
    }

    // Publish leaves and warp-mins
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
}

// --- Pass 2: build segment min-tree over d_block_mins[num_blocks] ---
// 0-indexed tree: root=0, left=2i+1, right=2i+2, leaves at [M-1..2M-2].
// M = next power of two >= num_blocks.

template <typename T>
__global__
void wstlFillLeavesKernel(
        const T* __restrict__ d_block_mins,
        T* __restrict__       d_tree,
        int                   num_blocks,
        int                   leaf_offset,  // = M-1
        int                   M) {
    const T INF = ApsepInfinity<T>::value();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    d_tree[leaf_offset + i] = (i < num_blocks) ? d_block_mins[i] : INF;
}

template <typename T>
__global__
void wstlReduceLevelKernel(T* __restrict__ d_tree, int level_start, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int node = level_start + i;
    d_tree[node] = min(d_tree[2 * node + 1], d_tree[2 * node + 2]);
}

// --- Pass 3: tree query + leaf scan for elements needing inter-block lookup ---
// For each element with d_out==INT_MIN, do Futhark-style ascent+descent on
// the block-level segment tree to find the rightmost block with val-in[i],
// then scan that block's warp-mins and leaves.
template <typename T, int B>
__device__ int wstlTreeQuery(
        const T* __restrict__ d_tree,
        int                   leaf_offset,  // M-1
        int                   block_id,     // query must be in [0, block_id-1]
        T                     val) {
    // Start at the leaf for block_id and ascend, looking left
    int node = leaf_offset + block_id;
    while (node > 0) {
        bool is_right = (node % 2 == 0);
        if (is_right) {
            int left_sib = node - 1;
            if (d_tree[left_sib] < val) {
                // Descend into left sibling preferring right child
                node = left_sib;
                while (node < leaf_offset) {
                    int rc = 2 * node + 2;
                    node = (d_tree[rc] < val) ? rc : (2 * node + 1);
                }
                return node - leaf_offset;  // block index
            }
        }
        node = (node - 1) / 2;
    }
    return -1;
}

template <typename T, int BLOCK_SIZE, int IPT>
__global__
void wstlPass3Kernel(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        const T* __restrict__   d_block_leaves,
        const T* __restrict__   d_block_mins,
        const T* __restrict__   d_block_warp_mins,
        const T* __restrict__   d_tree,
        int                     leaf_offset) {

    constexpr int B = BLOCK_SIZE * IPT;
    constexpr int W = B / 32;

    const int block_id = (int)blockIdx.x;
    const int glb_offs = block_id * B;

    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE) {
        int gid = glb_offs + i;
        if (gid >= n) continue;
        if (d_out[gid] != INT_MIN) continue;  // already resolved intra-block

        T val = d_in[gid];

        // Tree query: find rightmost block b < block_id with block_min < val
        int b = wstlTreeQuery<T, B>(d_tree, leaf_offset, block_id, val);
        if (b < 0) { d_out[gid] = -1; continue; }

        // Scan warp-mins then leaves within block b
        const T* wm = d_block_warp_mins + (size_t)b * W;
        int result = -1;
        for (int w = W - 1; w >= 0 && result < 0; w--) {
            if (__ldg(&wm[w]) < val) {
                const T* bl = d_block_leaves + (size_t)b * B + w * 32;
                for (int k = 31; k >= 0; k--)
                    if (__ldg(&bl[k]) < val) { result = b * B + w * 32 + k; break; }
            }
        }
        d_out[gid] = result;
    }
}

// Power-of-two ceiling
static inline int nextPow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
struct WSTLScratch {
    T*    d_block_leaves    = nullptr;
    T*    d_block_mins      = nullptr;
    T*    d_block_warp_mins = nullptr;
    T*    d_tree            = nullptr;
    int   num_blocks        = 0;
    int   M                 = 0;   // next power of two >= num_blocks
    int   leaf_offset       = 0;   // M - 1
    int   tree_size         = 0;   // 2*M - 1
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
WSTLScratch<T, BLOCK_SIZE, IPT> allocWSTLScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    constexpr int W = B / 32;

    WSTLScratch<T, BLOCK_SIZE, IPT> s;
    s.num_blocks  = (n + B - 1) / B;
    s.M           = nextPow2(s.num_blocks);
    s.leaf_offset = s.M - 1;
    s.tree_size   = 2 * s.M - 1;

    gpuAssert(cudaMalloc(&s.d_block_leaves,    (size_t)s.num_blocks * B  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_mins,      (size_t)s.num_blocks      * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_warp_mins, (size_t)s.num_blocks * W  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_tree,            (size_t)s.tree_size        * sizeof(T)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void freeWSTLScratch(WSTLScratch<T, BLOCK_SIZE, IPT>& s) {
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_block_mins);
    cudaFree(s.d_block_warp_mins);
    cudaFree(s.d_tree);
    s = WSTLScratch<T, BLOCK_SIZE, IPT>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void runWSTL(const T* d_in, int* d_out, int n, WSTLScratch<T, BLOCK_SIZE, IPT>& s) {
    if (n <= 0) return;
    constexpr int B = BLOCK_SIZE * IPT;

    // Pass 1: intra-block, no spin-waits
    wstlPass1Kernel<T, BLOCK_SIZE, IPT>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks,
            s.d_block_leaves, s.d_block_mins, s.d_block_warp_mins);
    gpuAssert(cudaGetLastError());

    // Pass 2: build block-level segment min-tree bottom-up
    {
        int tpb = 256;
        int blocks = (s.M + tpb - 1) / tpb;
        wstlFillLeavesKernel<T><<<blocks, tpb>>>(
            s.d_block_mins, s.d_tree, s.num_blocks, s.leaf_offset, s.M);
        gpuAssert(cudaGetLastError());

        // Reduce level by level bottom-up
        int level_size = s.M / 2;
        int level_start = s.leaf_offset / 2;  // parent of first leaf = (M-1-1)/2 = (M-2)/2
        // Actually: parent of leaf[0] = (leaf_offset - 1)/2
        // Bottom level of internal nodes: children are s.leaf_offset..s.leaf_offset+s.M-1
        // Parent of leaf_offset+i is (leaf_offset+i-1)/2
        // Level start of parent level = (leaf_offset - 1) / 2  ... not exactly
        // Simpler: iterate from the bottom of the internal nodes
        // The internal nodes are [0, leaf_offset-1] = [0, M-2]
        // Bottom level: nodes [M/2-1 .. M-2], count = M/2
        level_start = s.M / 2 - 1;
        level_size  = s.M / 2;
        while (level_size > 0) {
            int b2 = (level_size + tpb - 1) / tpb;
            wstlReduceLevelKernel<T><<<b2, tpb>>>(s.d_tree, level_start, level_size);
            gpuAssert(cudaGetLastError());
            level_size  >>= 1;
            level_start  = (level_start - 1) / 2;
        }
    }

    // Pass 3: tree query + leaf scan for elements needing inter-block lookup
    wstlPass3Kernel<T, BLOCK_SIZE, IPT>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks,
            s.d_block_leaves, s.d_block_mins, s.d_block_warp_mins,
            s.d_tree, s.leaf_offset);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void launchWSTL(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocWSTLScratch<T, BLOCK_SIZE, IPT>(n);
    runWSTL<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWSTLScratch<T, BLOCK_SIZE, IPT>(s);
}

// ===========================================================================
// SinglePassTree (SPT) — cooperative single-pass kernel
//
// Uses CUDA cooperative launch (grid.sync()) to implement the WSTL pipeline
// in a single kernel invocation with persistent thread-blocks:
//
//   Phase 1: each physical block processes its assigned logical blocks
//            sequentially (stride = num_physical_blocks).  Computes intra-block
//            PSE, publishes block_min[b], warp-mins, leaves.
//   grid.sync()
//   Phase 2: build segment min-tree over block_mins bottom-up, one level per
//            grid.sync().  Physical blocks share the work at each level.
//   grid.sync() (after final tree level)
//   Phase 3: each physical block processes its logical blocks, querying the
//            tree (ascent+descent O(log num_blocks)) for elements with
//            d_out==INT_MIN, then scans the found block's warp-mins + 32 leaves.
//
// Constraint: grid must fit on-chip for grid.sync() → launch exactly
//   cudaOccupancyMaxActiveBlocksPerMultiprocessor × num_SMs physical blocks.
//   Each physical block loops over ~num_blocks / num_physical_blocks logical
//   blocks.  No spin-waits anywhere after the grid.sync().
// ===========================================================================

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename T, int BLOCK_SIZE, int IPT>
__global__
void apsepKernelSPT(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     M,            // next pow2 >= num_blocks
        int                     leaf_offset,  // M - 1
        unsigned* __restrict__  d_unres,      // 1 bit/element: needs inter-block lookup
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins,
        T* __restrict__         d_tree,       // 2*M-1 nodes
        T* __restrict__         d_prefix_min) // prefix_min[b] = min(block_min[0..b-1]), INF for b=0
{
    cg::grid_group grid = cg::this_grid();

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W         = B / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    const int phys_bid   = (int)blockIdx.x;
    const int num_phys   = (int)gridDim.x;
    const int lane       = threadIdx.x & 31;
    const int warp_id    = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];
    __shared__ int s_queue[B];   // Phase 3: indices needing inter-block lookup
    __shared__ int s_qcount;

    static_assert(W <= 32, "warp-cooperative Phase 3 assumes W <= 32");

    // -------------------------------------------------------------------------
    // Phase 1: each physical block processes its logical blocks
    // -------------------------------------------------------------------------
    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        T* se = s_elems;
        T (*sm)[NUM_WARPS] = s_stripe_min;

        // No barrier between the shared store and the warp scan: the scan
        // only needs the thread's own element, kept in v[]; cross-thread
        // se reads happen after the __syncthreads below.
        T v[IPT];
        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            int lid = i * BLOCK_SIZE + threadIdx.x;
            int gid = glb_offs + lid;
            v[i] = (gid < n) ? d_in[gid] : INF;
            se[lid] = v[i];
        }

        T left_carry[IPT];
        #pragma unroll
        for (int ipt = 0; ipt < IPT; ipt++) {
            T c = v[ipt];
            #pragma unroll
            for (int step = 1; step <= 16; step <<= 1) {
                T nb = __shfl_up_sync(0xffffffff, c, step);
                if (lane >= step) c = min(c, nb);
            }
            left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
            T wm = __shfl_sync(0xffffffff, c, 31);
            if (lane == 0) sm[ipt][warp_id] = wm;
        }
        __syncthreads();

        // Intra-block PSE
        #pragma unroll
        for (int ipt = 0; ipt < IPT; ipt++) {
            int lid = ipt * BLOCK_SIZE + threadIdx.x;
            int gid = glb_offs + lid;
            bool active = (gid < n);
            T val = v[ipt];
            int result = -1;

            // Within-warp ANSV via synchronous pointer jumping: each lane
            // tracks a candidate index g; lanes whose candidate is not yet
            // smaller jump to the candidate's candidate (distance doubles
            // per round).  Uniform warp execution replaces the divergent
            // backward scan (profiled at 10.5/32 active threads on random).
            // All lanes participate in the shuffles (inactive lanes hold
            // val=INF and g=-1, never acting as jump sources for active
            // lanes since lower lanes in a warp are always active).
            bool has_within = active && (lane > 0) && (left_carry[ipt] < val);
            if (__any_sync(0xffffffff, has_within)) {
                int g = has_within ? lane - 1 : -1;
                while (true) {
                    int src = (g >= 0) ? g : 0;
                    T   eg  = __shfl_sync(0xffffffff, val, src);
                    int gg  = __shfl_sync(0xffffffff, g,   src);
                    bool jump = (g >= 0) && (eg >= val);
                    if (!__any_sync(0xffffffff, jump)) break;
                    if (jump) g = gg;
                }
                if (g >= 0)
                    result = ipt * BLOCK_SIZE + warp_id * 32 + g;
            }
            if (active) {
                if (result < 0) {
                    for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                        if (sm[ipt][w] < val) {
                            int wb = ipt * BLOCK_SIZE + w * 32;
                            for (int k = 31; k >= 0; k--)
                                if (se[wb + k] < val) { result = wb + k; break; }
                        }
                    }
                }
                if (result < 0) {
                    for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                        for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                            if (sm[i][w] < val) {
                                int wb = i * BLOCK_SIZE + w * 32;
                                for (int k = 31; k >= 0; k--)
                                    if (se[wb + k] < val) { result = wb + k; break; }
                            }
                        }
                    }
                }
                if (result >= 0) d_out[gid] = glb_offs + result;
            }
            // Publish unresolved bits, one word per warp.  Unresolved
            // elements get no d_out write here — Phase 3 writes them exactly
            // once, so every output byte is written once total (the old
            // INT_MIN sentinel forced a full d_out re-read in Phase 3).
            unsigned um = __ballot_sync(0xffffffff, active && result < 0);
            if (lane == 0)
                d_unres[(unsigned)(glb_offs + ipt * BLOCK_SIZE + warp_id * 32) >> 5] = um;
        }

        // Publish warp-mins and block_min.  No leaf publish: the leaves of a
        // complete block are byte-identical to d_in at the same offsets, so
        // Phase 3 reads d_in directly (saves a 4-byte-per-element write).
        if (lane == 0) {
            T* g_wm = d_block_warp_mins + (size_t)block_id * W;
            for (int ipt = 0; ipt < IPT; ipt++)
                g_wm[ipt * NUM_WARPS + warp_id] = sm[ipt][warp_id];
        }

        // Reduce block_min using first warp across all IPT*NUM_WARPS stripe mins (W values)
        if (warp_id == 0) {
            T bmin = (lane < W) ? sm[lane / NUM_WARPS][lane % NUM_WARPS] : INF;
            #pragma unroll
            for (int step = 16; step >= 1; step >>= 1)
                bmin = min(bmin, __shfl_xor_sync(0xffffffff, bmin, step));
            if (lane == 0) d_block_mins[block_id] = bmin;
        }
        __syncthreads();  // s_elems/s_stripe_min still read above; next iteration overwrites
    }

    // -------------------------------------------------------------------------
    // grid.sync(): all block_mins are now populated
    // -------------------------------------------------------------------------
    grid.sync();

    // -------------------------------------------------------------------------
    // Phase 2: build segment min-tree bottom-up
    // Fill leaves: d_tree[leaf_offset + i] = (i < num_blocks) ? block_mins[i] : INF
    // -------------------------------------------------------------------------
    for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < M; i += num_phys * BLOCK_SIZE)
        d_tree[leaf_offset + i] = (i < num_blocks) ? d_block_mins[i] : INF;

    // Reduce level by level
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

    // -------------------------------------------------------------------------
    // grid.sync(): tree is fully built
    // -------------------------------------------------------------------------
    grid.sync();

    // Exclusive prefix-min: d_prefix_min[b] = min(block_min[0..b-1]).
    // Computed per block by ascending the already-built min-tree and taking
    // the min over left siblings on the leaf-to-root path (their subtrees
    // cover exactly leaves [0, b)).  O(log M) L2-cached reads per block and
    // a single pass, replacing the Hillis-Steele scan's log2(M) grid.sync
    // passes and 8 MB of DRAM traffic.
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
    // Phase 3: tree query + leaf scan for elements flagged in d_unres
    // -------------------------------------------------------------------------

    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        const T prefix_min_b = d_prefix_min[block_id];  // min(block_min[0..block_id-1])

        // Whole-block early-out: unresolved elements within a block form a
        // non-increasing sequence (were an earlier one smaller, the later one
        // would have resolved intra-block), and the block's first element is
        // always unresolved — so it is the max of all unresolved elements.
        // If prefix_min_b >= that first element, every unresolved element in
        // the block resolves to -1 (the descending worst case takes this path
        // for all blocks).
        if (prefix_min_b >= __ldg(&d_in[glb_offs])) {
            for (int i = threadIdx.x; i < B; i += BLOCK_SIZE) {
                int gid = glb_offs + i;
                if (gid >= n) continue;
                unsigned um = __ldg(&d_unres[(unsigned)gid >> 5]);
                if ((um >> (gid & 31)) & 1u) d_out[gid] = -1;
            }
            continue;
        }

        // Collect elements needing inter-block lookup into a shared queue,
        // applying the O(1) prefix-min early exit inline.
        if (threadIdx.x == 0) s_qcount = 0;
        __syncthreads();

        for (int i = threadIdx.x; i < B; i += BLOCK_SIZE) {
            int gid = glb_offs + i;
            if (gid >= n) continue;
            unsigned um = __ldg(&d_unres[(unsigned)gid >> 5]);  // warp-uniform word
            if (!((um >> (gid & 31)) & 1u)) continue;

            T val = d_in[gid];
            if (prefix_min_b >= val) { d_out[gid] = -1; continue; }
            s_queue[atomicAdd(&s_qcount, 1)] = i;
        }
        __syncthreads();

        // Warp-cooperative processing, one queued element per warp.  The
        // tree walk runs redundantly on all lanes (identical L2-cached
        // addresses broadcast within the warp); the warp-min and leaf scans
        // become two 32-wide coalesced loads + __ballot_sync instead of up
        // to 48 dependent scalar loads with early exit.
        const int qn = s_qcount;
        for (int q = warp_id; q < qn; q += NUM_WARPS) {
            int gid = glb_offs + s_queue[q];
            T val = __ldg(&d_in[gid]);

            // Ascent+descent on block-level tree
            int node = leaf_offset + block_id;
            int found_block = -1;
            while (node > 0) {
                bool is_right = (node % 2 == 0);
                if (is_right) {
                    int left_sib = node - 1;
                    if (__ldg(&d_tree[left_sib]) < val) {
                        node = left_sib;
                        while (node < leaf_offset) {
                            int rc = 2 * node + 2;
                            node = (__ldg(&d_tree[rc]) < val) ? rc : (2 * node + 1);
                        }
                        found_block = node - leaf_offset;
                        break;
                    }
                }
                node = (node - 1) / 2;
            }

            int result = -1;
            if (found_block >= 0) {
                // found_block's block_min < val, so both ballots are nonzero.
                const T* wm = d_block_warp_mins + (size_t)found_block * W;
                T wv = (lane < W) ? __ldg(&wm[lane]) : INF;
                unsigned wmask = __ballot_sync(0xffffffff, wv < val);
                int wstar = 31 - __clz(wmask);
                // found_block < block_id, so all B of its elements are < n:
                // reading d_in here is safe and identical to the old leaves.
                const T* bl = d_in + (size_t)found_block * B + wstar * 32;
                unsigned lmask = __ballot_sync(0xffffffff, __ldg(&bl[lane]) < val);
                result = found_block * B + wstar * 32 + (31 - __clz(lmask));
            }
            if (lane == 0) d_out[gid] = result;
        }
        __syncthreads();
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
struct SPTScratch {
    unsigned* d_unres       = nullptr;  // 1 bit/element: needs inter-block lookup
    T*    d_block_mins      = nullptr;
    T*    d_block_warp_mins = nullptr;
    T*    d_tree            = nullptr;
    T*    d_prefix_min      = nullptr;  // prefix_min[b] = min(block_min[0..b-1])
    int   num_blocks        = 0;
    int   M                 = 0;
    int   leaf_offset       = 0;
    int   num_phys          = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
SPTScratch<T, BLOCK_SIZE, IPT> allocSPTScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    constexpr int W = B / 32;

    SPTScratch<T, BLOCK_SIZE, IPT> s;
    s.num_blocks  = (n + B - 1) / B;
    s.M           = nextPow2(s.num_blocks);
    s.leaf_offset = s.M - 1;

    // Determine physical block count: occupancy × SM count
    int blocks_per_sm = 0;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm,
        apsepKernelSPT<T, BLOCK_SIZE, IPT>,
        BLOCK_SIZE, 0));
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    s.num_phys = blocks_per_sm * prop.multiProcessorCount;

    gpuAssert(cudaMalloc(&s.d_unres,           (size_t)s.num_blocks * (B / 32) * sizeof(unsigned)));
    gpuAssert(cudaMalloc(&s.d_block_mins,      (size_t)s.num_blocks      * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_warp_mins, (size_t)s.num_blocks * W  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_tree,            (size_t)(2 * s.M - 1)     * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_prefix_min,      (size_t)s.num_blocks      * sizeof(T)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void freeSPTScratch(SPTScratch<T, BLOCK_SIZE, IPT>& s) {
    cudaFree(s.d_unres);
    cudaFree(s.d_block_mins);
    cudaFree(s.d_block_warp_mins);
    cudaFree(s.d_tree);
    cudaFree(s.d_prefix_min);
    s = SPTScratch<T, BLOCK_SIZE, IPT>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void runSPT(const T* d_in, int* d_out, int n, SPTScratch<T, BLOCK_SIZE, IPT>& s) {
    if (n <= 0) return;

    void* args[] = {
        (void*)&d_in, (void*)&d_out, (void*)&n,
        (void*)&s.num_blocks, (void*)&s.M, (void*)&s.leaf_offset,
        (void*)&s.d_unres, (void*)&s.d_block_mins,
        (void*)&s.d_block_warp_mins, (void*)&s.d_tree,
        (void*)&s.d_prefix_min
    };
    gpuAssert(cudaLaunchCooperativeKernel(
        (void*)apsepKernelSPT<T, BLOCK_SIZE, IPT>,
        s.num_phys, BLOCK_SIZE, args, 0, nullptr));
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void launchSPT(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocSPTScratch<T, BLOCK_SIZE, IPT>(n);
    runSPT<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeSPTScratch<T, BLOCK_SIZE, IPT>(s);
}

// ===========================================================================
// SPT-Atomic: cooperative single-pass kernel using atomicMin tree construction
//
// Same as SPT but Phase 2 (bottom-up grid.sync tree build) is replaced with
// inline atomicMin updates during Phase 1:
//   After computing bmin for a logical block, thread 0 walks from
//   d_tree[leaf_offset + block_id] up to root, doing atomicMin on each ancestor.
//
// This eliminates ~16 grid.sync() calls for tree construction, at the cost of
// ~log2(M) = 16 atomicMin operations per logical block.
//
// Requires T to be an unsigned integer type (atomicMin on unsigned int).
// Tree is initialized to INF (all-bits-set) before Phase 1 via cooperative init.
// ===========================================================================

template <typename T, int BLOCK_SIZE, int IPT>
__global__
void apsepKernelSPTAtomic(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     M,
        int                     leaf_offset,
        T* __restrict__         d_block_leaves,
        T* __restrict__         d_block_mins,
        T* __restrict__         d_block_warp_mins,
        T* __restrict__         d_tree) {

    cg::grid_group grid = cg::this_grid();

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    constexpr int W         = B / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    const int phys_bid = (int)blockIdx.x;
    const int num_phys = (int)gridDim.x;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];

    // Initialize tree to INF cooperatively before Phase 1
    const int tree_size = 2 * M - 1;
    for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < tree_size; i += num_phys * BLOCK_SIZE)
        d_tree[i] = INF;
    grid.sync();

    // -------------------------------------------------------------------------
    // Phase 1: intra-block PSE + atomicMin tree construction
    // -------------------------------------------------------------------------
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

        // Publish leaves and warp-mins
        T* g_leaves = d_block_leaves + (size_t)block_id * B;
        for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
            g_leaves[i] = s_elems[i];
        if (lane == 0) {
            T* g_wm = d_block_warp_mins + (size_t)block_id * W;
            for (int ipt = 0; ipt < IPT; ipt++)
                g_wm[ipt * NUM_WARPS + warp_id] = s_stripe_min[ipt][warp_id];
        }

        // Compute block_min and atomicMin up the tree (warp 0, thread 0)
        if (warp_id == 0) {
            T bmin = (lane < W) ? s_stripe_min[lane / NUM_WARPS][lane % NUM_WARPS] : INF;
            #pragma unroll
            for (int step = 16; step >= 1; step >>= 1)
                bmin = min(bmin, __shfl_xor_sync(0xffffffff, bmin, step));
            if (lane == 0) {
                d_block_mins[block_id] = bmin;
                // Walk up the tree from this block's leaf, atomicMin at each node
                int node = leaf_offset + block_id;
                while (node >= 0) {
                    atomicMin(&d_tree[node], bmin);
                    if (node == 0) break;
                    node = (node - 1) / 2;
                }
            }
        }
        __syncthreads();
    }

    // One grid.sync: all block_mins and tree updates complete
    grid.sync();

    // -------------------------------------------------------------------------
    // Phase 3: tree query + leaf scan (same as SPT, no Phase 2 needed)
    // -------------------------------------------------------------------------
    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;

        for (int i = threadIdx.x; i < B; i += BLOCK_SIZE) {
            int gid = glb_offs + i;
            if (gid >= n) continue;
            if (d_out[gid] != INT_MIN) continue;

            T val = d_in[gid];

            int node = leaf_offset + block_id;
            int found_block = -1;
            while (node > 0) {
                bool is_right = (node % 2 == 0);
                if (is_right) {
                    int left_sib = node - 1;
                    if (__ldg(&d_tree[left_sib]) < val) {
                        node = left_sib;
                        while (node < leaf_offset) {
                            int rc = 2 * node + 2;
                            node = (__ldg(&d_tree[rc]) < val) ? rc : (2 * node + 1);
                        }
                        found_block = node - leaf_offset;
                        break;
                    }
                }
                node = (node - 1) / 2;
            }

            if (found_block < 0) { d_out[gid] = -1; continue; }

            const T* wm = d_block_warp_mins + (size_t)found_block * W;
            int result = -1;
            for (int w = W - 1; w >= 0 && result < 0; w--) {
                if (__ldg(&wm[w]) < val) {
                    const T* bl = d_block_leaves + (size_t)found_block * B + w * 32;
                    for (int k = 31; k >= 0; k--)
                        if (__ldg(&bl[k]) < val) { result = found_block * B + w * 32 + k; break; }
                }
            }
            d_out[gid] = result;
        }
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void runSPTAtomic(const T* d_in, int* d_out, int n, SPTScratch<T, BLOCK_SIZE, IPT>& s) {
    if (n <= 0) return;
    // SPTAtomic uses same scratch as SPT; reuse allocSPTScratch
    // but num_phys must be queried for apsepKernelSPTAtomic
    int blocks_per_sm = 0;
    gpuAssert(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, apsepKernelSPTAtomic<T, BLOCK_SIZE, IPT>, BLOCK_SIZE, 0));
    cudaDeviceProp prop;
    gpuAssert(cudaGetDeviceProperties(&prop, 0));
    int num_phys = blocks_per_sm * prop.multiProcessorCount;

    void* args[] = {
        (void*)&d_in, (void*)&d_out, (void*)&n,
        (void*)&s.num_blocks, (void*)&s.M, (void*)&s.leaf_offset,
        (void*)&s.d_block_leaves, (void*)&s.d_block_mins,
        (void*)&s.d_block_warp_mins, (void*)&s.d_tree
    };
    gpuAssert(cudaLaunchCooperativeKernel(
        (void*)apsepKernelSPTAtomic<T, BLOCK_SIZE, IPT>,
        num_phys, BLOCK_SIZE, args, 0, nullptr));
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void launchSPTAtomic(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocSPTScratch<T, BLOCK_SIZE, IPT>(n);
    runSPTAtomic<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeSPTScratch<T, BLOCK_SIZE, IPT>(s);
}

// ===========================================================================
// Approach A: Two-level superblock hierarchy (WarpScanNoTree2L)
//
// Adds a "super-superblock" (SSB) level grouping G superblocks.
// Serial chain length: N / (G * K * B).  At N=131M, K=64, B=512, G=64:
// ~65 steps instead of ~4K.  Per-step cost: G sb_min reads + K block_min
// reads + B leaf reads, all sequential via __ldg.
//
// New template param G = number of SBs per SSB.
// ===========================================================================

template <typename T>
struct alignas(16) SuperSuperBlockState {
    T                    ssb_min;
    volatile ApsepStatus status;
};

template <typename T, int B, int K, int G>
__device__ int decoupledLookback2L(
        int                                 ssb_id,
        T                                   val,
        const SuperSuperBlockState<T>*      d_ssb_states,
        const SuperBlockState<T>*           d_sb_states,
        const T* __restrict__               d_block_mins,
        const T* __restrict__               d_block_leaves,
        int                                 num_blocks,
        int                                 num_sbs) {
    for (int ssb = ssb_id - 1; ssb >= 0; ssb--) {
        while (d_ssb_states[ssb].status == APSEP_INVALID) { /* spin */ }
        if (d_ssb_states[ssb].ssb_min >= val) continue;

        int ssb_sb_first = ssb * G;
        int ssb_sb_last  = min(ssb_sb_first + G, num_sbs) - 1;

        for (int sb = ssb_sb_last; sb >= ssb_sb_first; sb--) {
            if (__ldg(&d_sb_states[sb].sb_min) >= val) continue;

            int sb_first = sb * K;
            int sb_last  = min(sb_first + K, num_blocks) - 1;

            for (int b = sb_last; b >= sb_first; b--) {
                if (__ldg(&d_block_mins[b]) < val) {
                    const T* bl = d_block_leaves + (size_t)b * B;
                    for (int k = B - 1; k >= 0; k--)
                        if (__ldg(&bl[k]) < val) return b * B + k;
                }
            }
        }
    }
    return -1;
}

template <typename T, int BLOCK_SIZE, int IPT, int K, int G>
__global__
void apsepKernelWarpScanNoTree2L(
        const T* __restrict__       d_in,
        int*                        d_out,
        int                         n,
        int                         num_blocks,
        int                         num_sbs,
        int                         num_ssbs,
        BlockState<T>*              d_states,
        T* __restrict__             d_block_leaves,
        T* __restrict__             d_block_mins,
        SuperBlockState<T>*         d_sb_states,
        SuperSuperBlockState<T>*    d_ssb_states,
        volatile uint32_t*          d_dyn_idx) {

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_stripe_min[IPT][NUM_WARPS];

    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id  = (int)s_bid;
    const int glb_offs  = block_id * B;
    const int sb_id     = block_id / K;
    const int sb_local  = block_id % K;
    const int sb_first  = sb_id * K;
    const int ssb_id    = sb_id / G;
    const int ssb_local = sb_id % G;
    const int ssb_sb_first = ssb_id * G;
    const int lane      = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;

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

    // Publish leaves + block state
    T* g_leaves = d_block_leaves + (size_t)block_id * B;
    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
        g_leaves[i] = s_elems[i];
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        T bmin = s_stripe_min[0][0];
        for (int ipt = 0; ipt < IPT; ipt++)
            for (int w = 0; w < NUM_WARPS; w++)
                bmin = min(bmin, s_stripe_min[ipt][w]);
        d_block_mins[block_id] = bmin;
        d_states[block_id].block_min = bmin;
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    // Last block in SB: publish SB state
    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        if (threadIdx.x == 0) {
            T sb_min = d_block_mins[sb_first];
            for (int b = sb_first + 1; b <= block_id; b++)
                sb_min = min(sb_min, d_block_mins[b]);
            d_sb_states[sb_id].sb_min = sb_min;
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // Last SB in SSB: publish SSB state
    int ssb_sb_size = min(G, num_sbs - ssb_sb_first);
    bool is_last_sb_in_ssb = is_last_in_sb && (ssb_local == ssb_sb_size - 1);

    if (is_last_sb_in_ssb) {
        for (int sb = ssb_sb_first; sb < sb_id; sb++)
            while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }

        if (threadIdx.x == 0) {
            T ssb_min = d_sb_states[ssb_sb_first].sb_min;
            for (int sb = ssb_sb_first + 1; sb <= sb_id; sb++)
                ssb_min = min(ssb_min, d_sb_states[sb].sb_min);
            d_ssb_states[ssb_id].ssb_min = ssb_min;
            __threadfence();
            d_ssb_states[ssb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // Decoupled look-back (two-level)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;

            // Intra-SB: spin + packed block_min + leaves
            for (int b = block_id - 1; b >= sb_first && result < 0; b--) {
                while (d_states[b].status == APSEP_INVALID) { /* spin */ }
                if (__ldg(&d_block_mins[b]) < val) {
                    const T* bl = d_block_leaves + (size_t)b * B;
                    for (int k = B - 1; k >= 0; k--)
                        if (__ldg(&bl[k]) < val) { result = b * B + k; break; }
                }
            }

            // Intra-SSB: scan earlier SBs in same SSB
            if (result < 0) {
                for (int sb = sb_id - 1; sb >= ssb_sb_first && result < 0; sb--) {
                    while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }
                    if (__ldg(&d_sb_states[sb].sb_min) >= val) continue;
                    int sf = sb * K, sl = min(sf + K, num_blocks) - 1;
                    for (int b = sl; b >= sf && result < 0; b--) {
                        if (__ldg(&d_block_mins[b]) < val) {
                            const T* bl = d_block_leaves + (size_t)b * B;
                            for (int k = B - 1; k >= 0; k--)
                                if (__ldg(&bl[k]) < val) { result = b * B + k; break; }
                        }
                    }
                }
            }

            // Inter-SSB: two-level decoupled look-back
            if (result < 0)
                result = decoupledLookback2L<T, B, K, G>(
                    ssb_id, val, d_ssb_states, d_sb_states,
                    d_block_mins, d_block_leaves, num_blocks, num_sbs);

            d_out[gid] = result;
        }
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64, int G = 64>
struct ApsepWarpScanNoTree2LScratch {
    BlockState<T>*           d_states        = nullptr;
    T*                       d_block_leaves  = nullptr;
    T*                       d_block_mins    = nullptr;
    SuperBlockState<T>*      d_sb_states     = nullptr;
    SuperSuperBlockState<T>* d_ssb_states    = nullptr;
    uint32_t*                d_dyn_idx       = nullptr;
    int                      num_blocks      = 0;
    int                      num_sbs         = 0;
    int                      num_ssbs        = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64, int G = 64>
ApsepWarpScanNoTree2LScratch<T, BLOCK_SIZE, IPT, K, G> allocWarpScanNoTree2LScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    ApsepWarpScanNoTree2LScratch<T, BLOCK_SIZE, IPT, K, G> s;
    s.num_blocks = (n + B - 1) / B;
    s.num_sbs    = (s.num_blocks + K - 1) / K;
    s.num_ssbs   = (s.num_sbs   + G - 1) / G;
    gpuAssert(cudaMalloc(&s.d_states,      (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_leaves,(size_t)s.num_blocks * B * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_mins,  (size_t)s.num_blocks * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,   (size_t)s.num_sbs    * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_ssb_states,  (size_t)s.num_ssbs   * sizeof(SuperSuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,     sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64, int G = 64>
void freeWarpScanNoTree2LScratch(ApsepWarpScanNoTree2LScratch<T, BLOCK_SIZE, IPT, K, G>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_block_mins);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_ssb_states);
    cudaFree(s.d_dyn_idx);
    s = ApsepWarpScanNoTree2LScratch<T, BLOCK_SIZE, IPT, K, G>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64, int G = 64>
void runWarpScanNoTree2L(const T* d_in, int* d_out, int n,
                          ApsepWarpScanNoTree2LScratch<T, BLOCK_SIZE, IPT, K, G>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_sbs    * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_ssb_states,0, (size_t)s.num_ssbs   * sizeof(SuperSuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));
    apsepKernelWarpScanNoTree2L<T, BLOCK_SIZE, IPT, K, G>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_sbs, s.num_ssbs,
            s.d_states, s.d_block_leaves, s.d_block_mins,
            s.d_sb_states, s.d_ssb_states, s.d_dyn_idx);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64, int G = 64>
void launchWarpScanNoTree2L(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocWarpScanNoTree2LScratch<T, BLOCK_SIZE, IPT, K, G>(n);
    runWarpScanNoTree2L<T, BLOCK_SIZE, IPT, K, G>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWarpScanNoTree2LScratch<T, BLOCK_SIZE, IPT, K, G>(s);
}

// ===========================================================================
// Approach B: Static block assignment + warp-cooperative leaf scan (WarpCoopLeaf)
//
// Removes the d_dyn_idx atomicAdd: block_id = blockIdx.x.  Adjacent threads
// in a warp process adjacent blocks, so they all look back at the same
// previous block.  The leaf scan broadcasts a single leaf value across all
// 32 lanes, each comparing against their own val — one L2 read per step
// instead of 32.  Early termination when all lanes have found their answer.
// ===========================================================================

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__
void apsepKernelWarpCoopLeaf(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     num_superblocks,
        BlockState<T>*          d_states,
        T* __restrict__         d_block_leaves,
        T* __restrict__         d_block_mins,
        SuperBlockState<T>*     d_sb_states,
        volatile uint32_t*      d_dyn_idx) {

    constexpr int B         = BLOCK_SIZE * IPT;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const     T   INF       = ApsepInfinity<T>::value();

    // Static assignment: block_id = blockIdx.x
    const int block_id = (int)blockIdx.x;
    const int glb_offs = block_id * B;
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];

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

    // Intra-block backward search
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

    // Publish leaves + block state
    T* g_leaves = d_block_leaves + (size_t)block_id * B;
    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
        g_leaves[i] = s_elems[i];
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        T bmin = s_stripe_min[0][0];
        for (int ipt = 0; ipt < IPT; ipt++)
            for (int w = 0; w < NUM_WARPS; w++)
                bmin = min(bmin, s_stripe_min[ipt][w]);
        d_block_mins[block_id] = bmin;
        d_states[block_id].block_min = bmin;
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    // Last block in SB: publish SB state
    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        if (threadIdx.x == 0) {
            T sb_min = d_block_mins[sb_first];
            for (int b = sb_first + 1; b <= block_id; b++)
                sb_min = min(sb_min, d_block_mins[b]);
            __threadfence();
            d_sb_states[sb_id].sb_min = sb_min;
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // Decoupled look-back with warp-cooperative leaf scan.
    // All 32 lanes in a warp advance k together over the same leaf block,
    // broadcasting a single global load to all lanes.  One L2 read per step.
    #pragma unroll
    for (int ipt = 0; ipt < IPT; ipt++) {
        int lid = ipt * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        T val = (gid < n) ? s_elems[lid] : INF;
        int result = (gid < n && d_out[gid] == INT_MIN) ? -2 : -1;
        // result == -2: needs look-back; -1: done (intra-block resolved or out-of-range)

        // Intra-SB warp-cooperative leaf scan
        for (int b = block_id - 1; b >= sb_first; b--) {
            // Check if any lane still needs an answer
            uint32_t need = __ballot_sync(0xffffffff, result == -2);
            if (!need) break;

            while (d_states[b].status == APSEP_INVALID) {
                need = __ballot_sync(0xffffffff, result == -2);
                if (!need) goto intra_sb_done;
            }
            // All lanes check block_min
            T bmin = __ldg(&d_block_mins[b]);
            uint32_t candidate = __ballot_sync(0xffffffff, result == -2 && bmin < val);
            if (!candidate) continue;

            // Warp-cooperative backward leaf scan: one load per step, shared across all lanes
            {
                const T* bl = d_block_leaves + (size_t)b * B;
                for (int k = B - 1; k >= 0; k--) {
                    uint32_t still_need = __ballot_sync(0xffffffff, result == -2);
                    if (!still_need) break;
                    T leaf = __ldg(&bl[k]);  // one load, broadcast to all lanes
                    // Each lane checks against its own val
                    uint32_t found = __ballot_sync(0xffffffff, result == -2 && leaf < val);
                    if (found & (1u << lane)) result = b * B + k;
                }
            }
        }
        intra_sb_done:;

        // Inter-SB look-back (independent per thread — no warp coop across SBs)
        if (result == -2) {
            result = -1;
            // fallback to scalar inter-SB scan
            for (int sb = sb_id - 1; sb >= 0 && result < 0; sb--) {
                while (d_sb_states[sb].status == APSEP_INVALID) { /* spin */ }
                if (d_sb_states[sb].sb_min >= val) continue;
                int sf = sb * K, sl = min(sf + K, num_blocks) - 1;
                for (int b = sl; b >= sf && result < 0; b--) {
                    if (__ldg(&d_block_mins[b]) < val) {
                        const T* bl = d_block_leaves + (size_t)b * B;
                        for (int k = B - 1; k >= 0; k--)
                            if (__ldg(&bl[k]) < val) { result = b * B + k; break; }
                    }
                }
            }
        }

        if (gid < n) d_out[gid] = (result >= 0) ? result : ((d_out[gid] == INT_MIN) ? -1 : d_out[gid]);
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64>
struct ApsepWarpCoopLeafScratch {
    BlockState<T>*      d_states        = nullptr;
    T*                  d_block_leaves  = nullptr;
    T*                  d_block_mins    = nullptr;
    SuperBlockState<T>* d_sb_states     = nullptr;
    uint32_t*           d_dyn_idx       = nullptr;  // unused but kept for alloc symmetry
    int                 num_blocks      = 0;
    int                 num_superblocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64>
ApsepWarpCoopLeafScratch<T, BLOCK_SIZE, IPT, K> allocWarpCoopLeafScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    ApsepWarpCoopLeafScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;
    gpuAssert(cudaMalloc(&s.d_states,       (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_leaves, (size_t)s.num_blocks      * B  * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_block_mins,   (size_t)s.num_blocks      * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,    (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,      sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64>
void freeWarpCoopLeafScratch(ApsepWarpCoopLeafScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_block_mins);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_dyn_idx);
    s = ApsepWarpCoopLeafScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64>
void runWarpCoopLeaf(const T* d_in, int* d_out, int n,
                     ApsepWarpCoopLeafScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));
    apsepKernelWarpCoopLeaf<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_leaves, s.d_block_mins,
            s.d_sb_states, s.d_dyn_idx);
    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4, int K = 64>
void launchWarpCoopLeaf(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocWarpCoopLeafScratch<T, BLOCK_SIZE, IPT, K>(n);
    runWarpCoopLeaf<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeWarpCoopLeafScratch<T, BLOCK_SIZE, IPT, K>(s);
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

// ===========================================================================
// Leaves-Only variant
//
// Identical to apsepKernel (superblock structure, K blocks per SB) except:
//   - Per-block publish writes only the B leaves (s_tree[B..2B-1]) to a
//     d_block_leaves buffer (B elements per block) instead of the full 2*B tree.
//   - buildSuperBlockTreeFromLeaves reads directly from d_block_leaves[b*B..
//     (b+1)*B-1] (no +B offset into a 2*B array).
//   - Intra-SB look-back (blocks within the same superblock) performs a linear
//     scan over the B stored leaves instead of a tree query.
//
// Net saving: N*sizeof(T) bytes less written to global memory per kernel
// invocation (only N instead of 2*N for the per-block publish step).
// ===========================================================================

// g_block_leaves is already offset to the start of this SB (caller does +sb_first*B)
template <typename T, int BLOCK_SIZE, int IPT, int K>
__device__ void buildSBTreeFromSBRelativeLeaves(
        const T* __restrict__ g_block_leaves,  // d_block_leaves + sb_first*B
        T*       __restrict__ g_sb_tree,
        int sb_blocks,
        int n_total,
        int sb_first_block,
        T INF) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;

    for (int i = threadIdx.x; i < KB; i += BLOCK_SIZE) {
        int b   = i / B;
        int lid = i % B;
        T val;
        if (b < sb_blocks) {
            int gid = (sb_first_block + b) * B + lid;
            val = (gid < n_total) ? g_block_leaves[b * B + lid] : INF;
        } else {
            val = INF;
        }
        g_sb_tree[KB + i] = val;
    }
    __syncthreads();

    for (int half = KB >> 1; half >= 1; half >>= 1) {
        for (int i = threadIdx.x; i < half; i += BLOCK_SIZE) {
            int node = half + i;
            g_sb_tree[node] = min(g_sb_tree[2 * node], g_sb_tree[2 * node + 1]);
        }
        __syncthreads();
    }
}

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__ void apsepKernelLeavesOnly(
        const T* __restrict__         d_in,
        int*                          d_out,
        int                           n,
        int                           num_blocks,
        int                           num_superblocks,
        BlockState<T>*                d_states,
        T* __restrict__               d_block_leaves,   // B elements per block
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
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;

    // ---- 2. Load elements ----
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

    // ---- 5. Publish only the B leaves and block_min ----
    T* g_leaves = d_block_leaves + (size_t)block_id * B;
    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
        g_leaves[i] = s_tree[B + i];   // s_tree[B..2B-1] are the leaves
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        d_states[block_id].block_min = s_tree[1];
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    // ---- 6. Last block in SB builds merged SB tree from leaves ----
    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        T* g_sb_tree = d_sb_trees + (size_t)sb_id * (2 * KB);
        const T* g_block_leaves_sb = d_block_leaves + (size_t)sb_first * B;
        buildSBTreeFromSBRelativeLeaves<T, BLOCK_SIZE, IPT, K>(
            g_block_leaves_sb, g_sb_tree, sb_size, n, sb_first, INF);

        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0) {
            d_sb_states[sb_id].sb_min = g_sb_tree[1];
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // ---- 7. Decoupled look-back ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;

            // Intra-SB look-back: linear scan over stored leaves
            for (int b = block_id - 1; b >= sb_first && result < 0; b--) {
                while (d_states[b].status == APSEP_INVALID) { /* spin */ }
                if (d_states[b].block_min < val) {
                    const T* bl = d_block_leaves + (size_t)b * B;
                    for (int k = B - 1; k >= 0; k--) {
                        if (bl[k] < val) { result = b * B + k; break; }
                    }
                }
            }

            // Inter-SB look-back via SB trees
            if (result < 0)
                result = decoupledLookback<T, B, K>(
                    sb_id, val, d_sb_states, d_sb_trees);

            d_out[gid] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for leaves-only kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
struct ApsepLeavesOnlyScratch {
    BlockState<T>*      d_states       = nullptr;
    T*                  d_block_leaves = nullptr;   // B elements per block
    SuperBlockState<T>* d_sb_states    = nullptr;
    T*                  d_sb_trees     = nullptr;
    uint32_t*           d_dyn_idx      = nullptr;
    int                 num_blocks     = 0;
    int                 num_superblocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
ApsepLeavesOnlyScratch<T, BLOCK_SIZE, IPT, K> allocLeavesOnlyScratch(int n) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;
    static_assert((B & (B - 1)) == 0, "BLOCK_SIZE * IPT must be a power of two");

    ApsepLeavesOnlyScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;

    gpuAssert(cudaMalloc(&s.d_states,       (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_leaves, (size_t)s.num_blocks      * B * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,    (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_sb_trees,     (size_t)s.num_superblocks * 2 * KB * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,      sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void freeLeavesOnlyScratch(ApsepLeavesOnlyScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_sb_trees);
    cudaFree(s.d_dyn_idx);
    s = ApsepLeavesOnlyScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void runLeavesOnly(const T* d_in, int* d_out, int n,
                   ApsepLeavesOnlyScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));

    apsepKernelLeavesOnly<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_leaves,
            s.d_sb_states, s.d_sb_trees,
            s.d_dyn_idx);

    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void launchLeavesOnly(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocLeavesOnlyScratch<T, BLOCK_SIZE, IPT, K>(n);
    runLeavesOnly<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeLeavesOnlyScratch<T, BLOCK_SIZE, IPT, K>(s);
}
// ===========================================================================
// NoBlockTree variant
//
// Eliminates the per-block full-tree write to global memory.  Only the B leaf
// values (s_tree[B..2B-1]) are published per block.  The last-in-SB block
// reads B leaves per earlier block to build the merged SB tree (instead of
// 2*B tree nodes).  Intra-SB look-back does a linear scan over the stored
// leaves (O(B)) instead of a tree descent (O(log B)).
//
// Net savings vs apsepKernel baseline:
//   - Per-block global write: B elements instead of 2*B  (saves N bytes)
//   - Last-in-SB read: K*B leaves instead of K*2*B      (saves K*B reads/SB)
//   - Intra-SB look-back: O(B) linear scan vs O(log B) tree query
//
// Reuses buildSuperBlockTreeFromLeaves (defined in the LeavesOnly section).
// ===========================================================================

template <typename T, int BLOCK_SIZE, int IPT, int K>
__global__ void apsepKernelNoBlockTree(
        const T* __restrict__         d_in,
        int*                          d_out,
        int                           n,
        int                           num_blocks,
        int                           num_superblocks,
        BlockState<T>*                d_states,
        T* __restrict__               d_block_leaves,   // B elements per block
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
    const int sb_id    = block_id / K;
    const int sb_local = block_id % K;
    const int sb_first = sb_id * K;

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

    // ---- 5. Publish B leaf values only (not the full 2*B tree) ----
    // s_tree[B..2B-1] are the leaves (identical to s_elems[0..B-1]).
    T* g_leaves = d_block_leaves + (size_t)block_id * B;
    for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
        g_leaves[i] = s_tree[B + i];
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0) {
        d_states[block_id].block_min = s_tree[1];
        __threadfence();
        d_states[block_id].status = APSEP_READY;
    }
    __syncthreads();

    // ---- 6. Last block in SB builds merged SB tree from stored leaves ----
    int sb_size = min(K, num_blocks - sb_first);
    bool is_last_in_sb = (sb_local == sb_size - 1);

    if (is_last_in_sb) {
        // Wait for all earlier blocks in this SB to publish their leaves
        for (int b = sb_first; b < block_id; b++)
            while (d_states[b].status == APSEP_INVALID) { /* spin */ }

        // Build merged SB tree from B leaves per block (not 2*B trees)
        T* g_sb_tree = d_sb_trees + (size_t)sb_id * (2 * KB);
        const T* g_block_leaves_sb = d_block_leaves + (size_t)sb_first * B;
        buildSBTreeFromSBRelativeLeaves<T, BLOCK_SIZE, IPT, K>(
            g_block_leaves_sb, g_sb_tree, sb_size, n, sb_first, INF);

        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0) {
            d_sb_states[sb_id].sb_min = g_sb_tree[1];
            __threadfence();
            d_sb_states[sb_id].status = APSEP_READY;
        }
    }
    __syncthreads();

    // ---- 7. Decoupled look-back ----
    // Intra-SB: linear scan over stored B leaves per earlier block.
    // Inter-SB: unchanged — uses the merged SB trees.
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            T val = s_elems[lid];
            int result = -1;

            // Intra-SB look-back: linear scan over stored leaves
            for (int b = block_id - 1; b >= sb_first && result < 0; b--) {
                while (d_states[b].status == APSEP_INVALID) { /* spin */ }
                if (d_states[b].block_min < val) {
                    const T* bl = d_block_leaves + (size_t)b * B;
                    for (int k = B - 1; k >= 0; k--) {
                        if (bl[k] < val) { result = b * B + k; break; }
                    }
                }
            }

            // Inter-SB look-back via SB trees
            if (result < 0)
                result = decoupledLookback<T, B, K>(
                    sb_id, val, d_sb_states, d_sb_trees);

            d_out[gid] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// Scratch + wrappers for NoBlockTree kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
struct ApsepNoBlockTreeScratch {
    BlockState<T>*      d_states        = nullptr;
    T*                  d_block_leaves  = nullptr;   // B elements per block
    SuperBlockState<T>* d_sb_states     = nullptr;
    T*                  d_sb_trees      = nullptr;
    uint32_t*           d_dyn_idx       = nullptr;
    int                 num_blocks      = 0;
    int                 num_superblocks = 0;
};

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
ApsepNoBlockTreeScratch<T, BLOCK_SIZE, IPT, K> allocNoBlockTreeScratch(int n) {
    constexpr int B  = BLOCK_SIZE * IPT;
    constexpr int KB = K * B;
    static_assert((B & (B - 1)) == 0, "BLOCK_SIZE * IPT must be a power of two");

    ApsepNoBlockTreeScratch<T, BLOCK_SIZE, IPT, K> s;
    s.num_blocks      = (n + B - 1) / B;
    s.num_superblocks = (s.num_blocks + K - 1) / K;

    gpuAssert(cudaMalloc(&s.d_states,       (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_block_leaves, (size_t)s.num_blocks      * B * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_sb_states,    (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMalloc(&s.d_sb_trees,     (size_t)s.num_superblocks * 2 * KB * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_dyn_idx,      sizeof(uint32_t)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void freeNoBlockTreeScratch(ApsepNoBlockTreeScratch<T, BLOCK_SIZE, IPT, K>& s) {
    cudaFree(s.d_states);
    cudaFree(s.d_block_leaves);
    cudaFree(s.d_sb_states);
    cudaFree(s.d_sb_trees);
    cudaFree(s.d_dyn_idx);
    s = ApsepNoBlockTreeScratch<T, BLOCK_SIZE, IPT, K>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void runNoBlockTree(const T* d_in, int* d_out, int n,
                    ApsepNoBlockTreeScratch<T, BLOCK_SIZE, IPT, K>& s) {
    gpuAssert(cudaMemset(s.d_states,    0, (size_t)s.num_blocks      * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(s.d_sb_states, 0, (size_t)s.num_superblocks * sizeof(SuperBlockState<T>)));
    gpuAssert(cudaMemset(s.d_dyn_idx,   0, sizeof(uint32_t)));

    apsepKernelNoBlockTree<T, BLOCK_SIZE, IPT, K>
        <<<s.num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, s.num_blocks, s.num_superblocks,
            s.d_states, s.d_block_leaves,
            s.d_sb_states, s.d_sb_trees,
            s.d_dyn_idx);

    gpuAssert(cudaGetLastError());
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 2, int K = 8>
void launchNoBlockTree(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocNoBlockTreeScratch<T, BLOCK_SIZE, IPT, K>(n);
    runNoBlockTree<T, BLOCK_SIZE, IPT, K>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeNoBlockTreeScratch<T, BLOCK_SIZE, IPT, K>(s);
}

// ===========================================================================
// BSZ Two-Stage APSEP
//
// Implements the BSZ algorithm (Sitchinava & Svenning, SPAA '24) adapted for
// GPU:
//
//   Stage 1 (bszStage1Kernel): Each CTA processes one tile of B=BS*IPT
//   elements.  A sequential monotone-stack scan finds *local* matches (j in
//   the same tile).  Non-local elements are marked with sentinel INT_MIN in
//   d_out.  The tile minimum value and its global index are stored in
//   d_tile_min_vals / d_tile_min_idxs.
//
//   Stage 2 (bszBuildTreeKernel): A global segment-min tree of depth
//   ceil(log2(n)) is built over d_in.  Each internal node stores the minimum
//   of its subtree, enabling O(log n) "rightmost element < threshold in prefix
//   [0, i-1]" queries.  The tree has M = next_pow2(n) leaves stored at
//   positions [M .. 2M-1] in a 1-indexed array.
//
//   Stage 3 (bszNonlocalKernel): Each non-local element (d_out[i] == INT_MIN)
//   independently queries the global segment-min tree to find the rightmost
//   j < i with d_in[j] < d_in[i] in O(log n) steps.  This is fully parallel
//   with no inter-thread dependency.
//
// Key insight (Observation 1 / Lemma 3 of the paper): the non-local unmatched
// values partition remaining elements into *disjoint independent* subproblems,
// so all queries can run concurrently.  This breaks the serial dependency chain
// of the decoupled look-back approach.
//
// Template parameters:
//   T         – element type
//   BLOCK_SIZE – threads per CTA for stage-1
//   IPT       – items per thread  (B = BLOCK_SIZE * IPT)
// ===========================================================================

// ---------------------------------------------------------------------------
// Stage 1: local matches via sequential monotone-stack scan
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void bszStage1Kernel(
        const T* __restrict__ d_in,
        int*                  d_out,
        int                   n,
        T*  __restrict__      d_tile_min_vals,
        int*                  d_tile_min_idxs)
{
    constexpr int B = BLOCK_SIZE * IPT;
    const int tile  = blockIdx.x;
    const int base  = tile * B;

    // Shared memory: elements + a monotone stack (indices only)
    __shared__ T   s_elems[B];
    __shared__ int s_stack[B];   // monotone-stack index buffer

    // Load tile into shared memory
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = base + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : ApsepInfinity<T>::value();
    }
    __syncthreads();

    // Thread 0 runs the sequential SEQ algorithm (monotone stack) to find
    // local matches and the tile minimum.
    if (threadIdx.x == 0) {
        int depth  = 0;
        T   tmin   = ApsepInfinity<T>::value();
        int tmin_i = base;

        for (int lid = 0; lid < B; lid++) {
            int gid = base + lid;
            if (gid >= n) break;
            T val = s_elems[lid];

            // pop stack while top >= val
            while (depth > 0 && s_elems[s_stack[depth - 1]] >= val)
                depth--;

            // local match = stack top (must be in same tile)
            int match = (depth > 0) ? (base + s_stack[depth - 1]) : INT_MIN;
            // INT_MIN is the "needs non-local look-up" sentinel (never a valid index)
            d_out[gid] = match;

            s_stack[depth++] = lid;

            if (val < tmin) { tmin = val; tmin_i = gid; }
        }

        d_tile_min_vals[tile] = tmin;
        d_tile_min_idxs[tile] = tmin_i;
    }
}

// ---------------------------------------------------------------------------
// Stage 2: build global min-tree in Futhark transparent_reduction_tree layout.
//
// 0-indexed, root at 0, left child of i = 2i+1, right child = 2i+2.
// Leaf offset = M-1 where M = next_pow2(n).  Total tree size = 2M-1.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void bszFillLeavesKernel(
        const T* __restrict__ d_in,
        T*       __restrict__ d_tree,
        int n,
        int leaf_offset,
        int M)
{
    const T INF = ApsepInfinity<T>::value();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += gridDim.x * blockDim.x)
        d_tree[leaf_offset + i] = (i < n) ? d_in[i] : INF;
}

// Reduce one level: nodes [level_start .. level_start+count-1]
template <typename T>
__global__ void bszReduceLevelKernel(
        T* __restrict__ d_tree,
        int level_start,
        int count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        int node = level_start + i;
        d_tree[node] = min(d_tree[2 * node + 1], d_tree[2 * node + 2]);
    }
}

// ---------------------------------------------------------------------------
// Stage 3: ascent + descent query (Futhark transparent_reduction_tree.previous)
//
// For each i where d_out[i] == INT_MIN:
//   find rightmost j < i with d_in[j] < d_in[i]
//
// Ascent from leaf(i): climb while we are either a right child OR the left
//   sibling subtree has no value < val.
// Descent into left sibling: prefer right child whenever it satisfies < val.
//
// Visits exactly O(log n) tree nodes — two root-to-leaf paths.
// ---------------------------------------------------------------------------

template <typename T>
__device__ int bszTreeQuery(
        const T* __restrict__ d_tree,
        int leaf_offset,
        int i,
        T   val)
{
    int node = leaf_offset + i;

    while (node != 0) {
        bool is_right_child = (node % 2 == 0);
        if (is_right_child) {
            int left_sib = node - 1;
            if (d_tree[left_sib] < val) {
                // descend left sibling, always preferring right child
                node = left_sib;
                while (node < leaf_offset) {
                    int rc = 2 * node + 2;
                    int lc = 2 * node + 1;
                    node = (d_tree[rc] < val) ? rc : lc;
                }
                return node - leaf_offset;
            }
        }
        node = (node - 1) / 2;
    }
    return -1;
}

template <typename T>
__global__ void bszNonlocalKernel(
        const T* __restrict__ d_in,
        int*                  d_out,
        int                   n,
        const T* __restrict__ d_tree,
        int                   leaf_offset)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    if (d_out[gid] != INT_MIN) return;

    T val = d_in[gid];
    d_out[gid] = bszTreeQuery<T>(d_tree, leaf_offset, gid, val);
}

// ---------------------------------------------------------------------------
// Scratch + wrappers
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
struct ApsepBSZScratch {
    T*   d_tree          = nullptr;
    T*   d_tile_min_vals = nullptr;
    int* d_tile_min_idxs = nullptr;
    int  num_tiles       = 0;
    int  M               = 0;   // next_pow2(n)
    int  leaf_offset     = 0;   // M - 1  (index of first leaf in 0-indexed tree)
};

static inline int bszNextPow2(int n) {
    int m = 1;
    while (m < n) m <<= 1;
    return m;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
ApsepBSZScratch<T, BLOCK_SIZE, IPT> allocBSZScratch(int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    ApsepBSZScratch<T, BLOCK_SIZE, IPT> s;
    s.num_tiles   = (n + B - 1) / B;
    s.M           = bszNextPow2(n);
    s.leaf_offset = s.M - 1;

    // Tree has 2M-1 nodes (0-indexed, root=0, leaves=[M-1..2M-2])
    gpuAssert(cudaMalloc(&s.d_tree,          (size_t)(2 * s.M - 1) * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_tile_min_vals, (size_t)s.num_tiles * sizeof(T)));
    gpuAssert(cudaMalloc(&s.d_tile_min_idxs, (size_t)s.num_tiles * sizeof(int)));
    return s;
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void freeBSZScratch(ApsepBSZScratch<T, BLOCK_SIZE, IPT>& s) {
    cudaFree(s.d_tree);
    cudaFree(s.d_tile_min_vals);
    cudaFree(s.d_tile_min_idxs);
    s = ApsepBSZScratch<T, BLOCK_SIZE, IPT>{};
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void runBSZ(const T* d_in, int* d_out, int n,
            ApsepBSZScratch<T, BLOCK_SIZE, IPT>& s) {
    constexpr int B = BLOCK_SIZE * IPT;
    const int M     = s.M;

    // --- Stage 1: local matches (sequential SEQ per tile) ---
    bszStage1Kernel<T, BLOCK_SIZE, IPT>
        <<<s.num_tiles, BLOCK_SIZE>>>(
            d_in, d_out, n, s.d_tile_min_vals, s.d_tile_min_idxs);
    gpuAssert(cudaGetLastError());
    gpuAssert(cudaDeviceSynchronize());

    // --- Stage 2: build global min-tree (Futhark layout, 0-indexed) ---
    {
        const int leaf_offset = s.leaf_offset;
        int threads = 256;
        // Fill leaves
        int lblocks = (M + threads - 1) / threads;
        bszFillLeavesKernel<T><<<lblocks, threads>>>(d_in, s.d_tree, n, leaf_offset, M);
        gpuAssert(cudaGetLastError());
        gpuAssert(cudaDeviceSynchronize());
        // Reduce bottom-up: level l has 2^l nodes starting at (2^l - 1)
        // Go from level h-2 down to 0 (root)
        for (int count = M >> 1, start = (M >> 1) - 1; count >= 1; count >>= 1, start = (start - 1) / 2) {
            int rblocks = (count + threads - 1) / threads;
            bszReduceLevelKernel<T><<<rblocks, threads>>>(s.d_tree, start, count);
            gpuAssert(cudaGetLastError());
            gpuAssert(cudaDeviceSynchronize());
        }
    }

    // --- Stage 3: non-local matches via ascent+descent tree query ---
    {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        bszNonlocalKernel<T><<<blocks, threads>>>(d_in, d_out, n, s.d_tree, s.leaf_offset);
        gpuAssert(cudaGetLastError());
    }
}

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void launchBSZ(const T* d_in, int* d_out, int n) {
    if (n <= 0) return;
    auto s = allocBSZScratch<T, BLOCK_SIZE, IPT>(n);
    runBSZ<T, BLOCK_SIZE, IPT>(d_in, d_out, n, s);
    gpuAssert(cudaDeviceSynchronize());
    freeBSZScratch<T, BLOCK_SIZE, IPT>(s);
}
