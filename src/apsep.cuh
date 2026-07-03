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
