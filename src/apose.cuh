// Single-Pass All Previous Or Smaller Element (APOSE)
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
//  4. Write the tree and block-minimum to global memory; mark block READY.
//  5. Elements whose PSE was not found locally perform a decoupled look-back:
//     they scan previous blocks (waiting for each to become READY), check the
//     published block-minimum, and – once a qualifying block is found – search
//     its global tree for the rightmost element that is strictly less than the
//     query value.
//
// Requirements:
//   - B = BLOCK_SIZE * IPT must be a power of two.
//   - T must support min() and <.
//   - The kernel must be launched with at most as many CTAs as the GPU can
//     run concurrently (standard decoupled-look-back requirement).

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

template <typename T> struct AposeInfinity {
    __host__ __device__ static T value();
};
template <> struct AposeInfinity<int> {
    __host__ __device__ static int value() { return INT_MAX; }
};
template <> struct AposeInfinity<unsigned> {
    __host__ __device__ static unsigned value() { return UINT_MAX; }
};
template <> struct AposeInfinity<long long> {
    __host__ __device__ static long long value() { return LLONG_MAX; }
};
template <> struct AposeInfinity<float> {
    __host__ __device__ static float value() { return FLT_MAX; }
};
template <> struct AposeInfinity<double> {
    __host__ __device__ static double value() { return DBL_MAX; }
};

// ---------------------------------------------------------------------------
// Decoupled look-back status flags
// ---------------------------------------------------------------------------

enum AposeStatus : uint8_t {
    APOSE_INVALID = 0, // block has not yet published its tree
    APOSE_READY   = 1  // block has published tree + block_min
};

// ---------------------------------------------------------------------------
// Per-block state (published to global memory for the look-back)
//
// The `status` field is declared volatile so that spinning readers in other
// blocks always see the latest write and the compiler cannot cache it.
// ---------------------------------------------------------------------------

template <typename T>
struct alignas(16) BlockState {
    T                    block_min; // minimum value among all elements
    volatile AposeStatus status;    // APOSE_INVALID → APOSE_READY
};

// ---------------------------------------------------------------------------
// Min-tree construction (shared memory, 1-indexed BFS layout)
//
// Layout: tree[0] is UNUSED.  Root at tree[1].
//         Leaves at tree[B .. 2B-1]; leaf i maps to tree[B + i].
//         B must be a power of two.
//
// All BLOCK_SIZE threads cooperate; caller must issue __syncthreads()
// before using the finished tree (the function ends with a sync).
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__device__ void buildMinTree(const volatile T* __restrict__ elems,
                             volatile T*       __restrict__ tree) {
    constexpr int B = BLOCK_SIZE * IPT;

    // Copy leaves (each thread copies IPT consecutive leaves in strided order)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        tree[B + lid] = elems[lid];
    }
    __syncthreads();

    // Reduce levels bottom-up: half nodes per level, halving each time
    for (int half = B >> 1; half >= 1; half >>= 1) {
        for (int i = threadIdx.x; i < half; i += BLOCK_SIZE) {
            int node = half + i;
            tree[node] = min(tree[2 * node], tree[2 * node + 1]);
        }
        __syncthreads();
    }
    // tree[1] is now the block minimum
}

// ---------------------------------------------------------------------------
// Intra-block PSE via tree query
//
// Finds the rightmost leaf with index strictly less than `start_leaf` whose
// stored value is strictly less than `val`.
//
// Returns the leaf index in [0, B-1], or -1 if no such element exists.
//
// Implementation follows the `previous` function from Bar-on & Vishkin (1985)
// as described in the transparent_reduction_tree Futhark library.
//
// 1-indexed BFS tree conventions used throughout:
//   node `n` is a RIGHT child  ⟺  n is odd   (n = 2*parent + 1)
//   node `n` is a LEFT  child  ⟺  n is even  (n = 2*parent)
//   left sibling of right child n → n - 1
// ---------------------------------------------------------------------------

template <typename T>
__device__ int treePrevSmaller(const volatile T* __restrict__ tree,
                               int B, int start_leaf, T val) {
    if (start_leaf <= 0) return -1;

    int node = B + start_leaf; // start at the leaf for `start_leaf`

    // Ascend until we find a right child whose left sibling subtree
    // contains an element < val (i.e., its min < val).
    while (node > 1) {
        if ((node & 1) != 0) {
            // `node` is a right child; its left sibling is `node - 1`
            if (tree[node - 1] < val) {
                // Descend into the left sibling, always going right when
                // possible (to find the rightmost qualifying element).
                int curr = node - 1;
                while (curr < B) {
                    int right = 2 * curr + 1;
                    curr = (tree[right] < val) ? right : (2 * curr);
                }
                return curr - B; // convert tree node back to leaf index
            }
        }
        node >>= 1; // parent = node / 2
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Inter-block PSE query
//
// Finds the rightmost leaf in [0, B-1] of the given tree whose value < val.
// Used during the look-back when searching an entire previous block's tree.
//
// Returns the leaf index in [0, B-1], or -1 if tree[1] (block min) >= val.
// ---------------------------------------------------------------------------

template <typename T>
__device__ int treeRightmostSmaller(const volatile T* __restrict__ tree,
                                    int B, T val) {
    if (tree[1] >= val) return -1; // block min >= val → nothing qualifies

    // Descend from the root, always preferring the right child
    // (which covers higher indices) when its subtree min < val.
    // Invariant: tree[curr] < val at all times.
    int curr = 1;
    while (curr < B) {
        int right = 2 * curr + 1;
        curr = (tree[right] < val) ? right : (2 * curr);
    }
    return curr - B;
}

// ---------------------------------------------------------------------------
// Decoupled look-back for elements with no intra-block PSE
//
// Scans previous blocks in reverse order (block_id-1, block_id-2, …).
// For each block b, waits (spins) until its status is APOSE_READY, then
// checks whether its block_min < val.  The first (rightmost) such block
// contains the PSE; we descend its global tree and return the global index.
// ---------------------------------------------------------------------------

template <typename T, int B>
__device__ int decoupledLookback(int                         block_id,
                                 T                           val,
                                 const BlockState<T>*        d_states,
                                 const T* __restrict__       d_trees) {
    for (int b = block_id - 1; b >= 0; b--) {
        // Spin until block b has published its state (volatile field)
        while (d_states[b].status == APOSE_INVALID) { /* spin */ }

        if (d_states[b].block_min < val) {
            // PSE is somewhere in block b; search its published tree
            const T* bt = d_trees + (size_t)b * (2 * B);
            int leaf = treeRightmostSmaller<T>(bt, B, val);
            return (leaf >= 0) ? b * B + leaf : -1;
        }
    }
    return -1; // no previous block has any element < val
}

// ---------------------------------------------------------------------------
// Main kernel: single-pass APOSE
//
// Grid/block dimensions:
//   gridDim.x  = number of logical blocks (num_blocks = ceil(n / B))
//   blockDim.x = BLOCK_SIZE
//
// Template parameters:
//   T          – element type
//   BLOCK_SIZE – threads per block
//   IPT        – items per thread  (B = BLOCK_SIZE * IPT must be power of two)
//
// Global memory arrays:
//   d_states  [num_blocks]           – per-block status + min
//   d_trees   [num_blocks * 2 * B]   – per-block min-tree (1-indexed, index 0
//                                       unused, root at [1], leaves at [B..2B-1])
//   d_dyn_idx [1]                    – atomic counter for dynamic block IDs
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE, int IPT>
__global__ void aposeKernel(const T* __restrict__      d_in,
                            int*                       d_out,
                            int                        n,
                            BlockState<T>*             d_states,
                            T* __restrict__            d_trees,
                            volatile uint32_t*         d_dyn_idx) {
    constexpr int B   = BLOCK_SIZE * IPT;
    const     T   INF = AposeInfinity<T>::value();

    // ---- Shared memory ----
    __shared__ uint32_t s_bid;
    __shared__ T        s_elems[B];
    __shared__ T        s_tree[2 * B]; // [0] unused; root=[1]; leaves=[B..2B-1]

    // ---- 1. Dynamic block assignment ----
    // All physical blocks atomically claim a logical block ID to ensure
    // correct decoupled look-back ordering.
    if (threadIdx.x == 0)
        s_bid = atomicAdd(const_cast<uint32_t*>(d_dyn_idx), 1u);
    __syncthreads();

    const int block_id = (int)s_bid;
    const int glb_offs = block_id * B;

    // ---- 2. Load elements (pad out-of-bounds with INF) ----
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INF;
    }
    __syncthreads();

    // ---- 3. Build min-tree in shared memory ----
    // After this call, s_tree[1] holds the block minimum.
    // The function ends with __syncthreads(), so the tree is fully visible.
    buildMinTree<T, BLOCK_SIZE, IPT>(s_elems, s_tree);

    // ---- 4. Intra-block PSE queries (parallel, O(log B) per element) ----
    // Each thread queries the shared tree for its IPT elements independently.
    // Result: d_out[gid] = global PSE index, or INT_MIN as "unresolved" sentinel.
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n) {
            int local = treePrevSmaller<T>(s_tree, B, lid, s_elems[lid]);
            d_out[gid] = (local >= 0) ? (glb_offs + local) : INT_MIN;
        }
    }

    // ---- 5. Publish tree to global memory (all threads cooperate) ----
    T* g_tree = d_trees + (size_t)block_id * (2 * B);
    for (int i = threadIdx.x; i < 2 * B; i += BLOCK_SIZE) {
        g_tree[i] = s_tree[i];
    }
    // Each thread flushes its own writes to the L2 so they are globally
    // visible.  The subsequent __syncthreads() ensures every thread has
    // completed its fence before thread 0 writes the flag.
    __threadfence();
    __syncthreads();

    // ---- 6. Publish block_min and mark block READY ----
    // Thread 0 writes block_min, issues another __threadfence() to order
    // that write before the status flag, then sets status = READY.
    // Readers spin on status (volatile) and are guaranteed to observe the
    // block_min and tree data once they see APOSE_READY.
    if (threadIdx.x == 0) {
        d_states[block_id].block_min = s_tree[1];
        __threadfence();
        d_states[block_id].status = APOSE_READY;
    }
    __syncthreads();

    // ---- 7. Decoupled look-back for unresolved elements ----
    // Blocks at index 0 have no previous block, so all their elements are
    // already resolved above (-1 from intra-block query = truly no PSE).
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int lid = i * BLOCK_SIZE + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid < n && d_out[gid] == INT_MIN) {
            // No intra-block PSE found; search previous blocks.
            d_out[gid] = decoupledLookback<T, B>(
                block_id, s_elems[lid], d_states, d_trees);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
//
// Allocates temporary device buffers, launches the kernel, and frees them.
// The output d_out must already be allocated (n ints).
//
// BLOCK_SIZE * IPT must be a power of two (checked at compile time).
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE = 128, int IPT = 4>
void launchApose(const T* d_in, int* d_out, int n) {
    constexpr int B = BLOCK_SIZE * IPT;
    static_assert((B & (B - 1)) == 0,
                  "BLOCK_SIZE * IPT must be a power of two");

    if (n <= 0) return;

    const int num_blocks = (n + B - 1) / B;

    BlockState<T>* d_states  = nullptr;
    T*             d_trees   = nullptr;
    uint32_t*      d_dyn_idx = nullptr;

    gpuAssert(cudaMalloc(&d_states,  (size_t)num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMalloc(&d_trees,   (size_t)num_blocks * 2 * B * sizeof(T)));
    gpuAssert(cudaMalloc(&d_dyn_idx, sizeof(uint32_t)));

    gpuAssert(cudaMemset(d_states,  0, (size_t)num_blocks * sizeof(BlockState<T>)));
    gpuAssert(cudaMemset(d_dyn_idx, 0, sizeof(uint32_t)));

    aposeKernel<T, BLOCK_SIZE, IPT>
        <<<num_blocks, BLOCK_SIZE>>>(
            d_in, d_out, n, d_states, d_trees, d_dyn_idx);

    gpuAssert(cudaGetLastError());
    gpuAssert(cudaDeviceSynchronize());

    cudaFree(d_states);
    cudaFree(d_trees);
    cudaFree(d_dyn_idx);
}
