# SPT Optimization Prompt

## Context

This is a CUDA implementation of the **All Previous Smaller Element Problem (APSEP)**:
for each element `a[i]`, find the index `j < i` of the nearest preceding element where `a[j] < a[i]`, or -1 if none.

The goal is to make the **SPT (SinglePassTree)** kernel as fast as possible.
Hardware: Nvidia GTX 1660 Ti, SM 7.5, 24 SMs, 288 GB/s peak memory bandwidth, 1.5 MB L2 cache.

## Current SPT architecture

SPT is a single cooperative kernel launched with `cudaLaunchCooperativeKernel`.
192 persistent physical blocks (8 blocks/SM × 24 SMs) loop over all 65 536 logical blocks.
`grid.sync()` separates the three phases.

**Parameters:** `BLOCK_SIZE=128`, `IPT=4`, `B = BLOCK_SIZE * IPT = 512` elements per logical block,
`W = B / 32 = 16` warp-mins per block, `N = 32M` elements → 65 536 logical blocks.

**Phase 1** (each physical block loops over its logical blocks):
1. Load B elements into shared memory `s_elems[B]`.
2. Warp prefix-min scan to compute `left_carry[ipt]` and `s_stripe_min[IPT][NUM_WARPS]`.
3. **Intra-block backward scan** (current bottleneck — see below): each thread independently
   scans shared memory backward to find its PSE. Writes `d_out[gid] = local_idx` or `INT_MIN`.
4. Publish `d_block_leaves[block_id * B]`, `d_block_warp_mins[block_id * W]`, `d_block_mins[block_id]`.

**Phase 2** (grid-wide, after `grid.sync()`):
1. Fill leaf layer of segment min-tree over `d_block_mins`.
2. Bottom-up tree reduce (log2(65536) = 16 `grid.sync()` iterations).
3. **Hillis-Steele parallel prefix-min scan** over `d_block_mins` into `d_prefix_min` (16 more passes,
   each read+write of 65536 × 4 = 256 KB → 8 MB total extra traffic).

**Phase 3** (each physical block loops over its logical blocks, after final `grid.sync()`):
1. For each element in the logical block: if `d_out != INT_MIN`, skip.
2. Read `d_prefix_min[block_id]`. If `prefix_min >= val`, write -1 and skip (O(1) early exit).
3. Otherwise, ascend+descend the segment min-tree (O(log N) pointer-chasing on L2-cached tree).
4. Scan warp-mins then 32 leaves in the found block to get the exact index.

## Profiling results (ncu --set full, N=32M)

| Scenario   | Duration | DRAM%  | Mem GB/s | Occupancy | Threads/warp | Branch eff |
|------------|----------|--------|----------|-----------|--------------|------------|
| Random     | 6.47 ms  | 34.4%  | 95 GB/s  | 88%       | **10.5/32**  | **84%**    |
| Descending | 4.08 ms  | 71.9%  | 199 GB/s | 99%       | 31.3/32      | 99%        |

Phase breakdown (from a separate phase-timing binary):

| Scenario   | Phase 1 | Phase 2 | Phase 3 | Total  |
|------------|---------|---------|---------|--------|
| Random     | 2.60 ms | ~0.01ms | 3.06 ms | 5.66ms |
| Descending | 2.20 ms | ~0.01ms | 1.80 ms | 4.00ms |

Current useful GB/s (2×N×4 / time): **~49 GB/s random, ~70 GB/s descending**.
WSTL (3-pass multi-kernel baseline) achieves: **~65 GB/s random, ~64 GB/s descending**.

## Identified bottlenecks

### Bottleneck 1: Warp divergence in Phase 1 intra-block backward scan

The current code (simplified):
```cuda
// Each thread independently scans backward in shared memory
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
// ... and similarly for earlier stripes
d_out[gid] = (result >= 0) ? (glb_offs + result) : INT_MIN;
```

**Problem:** Threads in the same warp exit the inner loop at different iterations (early-exit on first
match). The warp must wait for the slowest lane. On random input, ncu measures only **10.5 active
threads/warp** (should be 32) and **84% branch efficiency**. Phase 1 takes 2.60 ms on random.
On descending input (every PSE = -1, scan always runs to completion uniformly), divergence disappears:
31.3 threads/warp, 99% branch efficiency, 2.20 ms.

**Suggested fix — `__ballot_sync` backward scan:**
Replace the sequential per-lane loop with a parallel warp vote. All 32 lanes simultaneously check
whether their position contains an element smaller than `val`; a bitmask gives the rightmost match
in O(1):
```cuda
// Within-warp scan: all lanes vote in parallel
uint32_t smaller = __ballot_sync(0xffffffff, lane > 0 && s_elems[base + lane] < val);
// 'smaller' has a 1 bit for each lane whose element is < val and is to our left
uint32_t left_mask = (1u << lane) - 1;  // lanes strictly to the left of current lane
uint32_t candidates = smaller & left_mask;
if (candidates) {
    result = base + (31 - __clz(candidates));  // rightmost set bit = nearest smaller
}
```
This collapses the within-warp scan from O(lane) to a single `__ballot_sync` + `__clz`.
The cross-warp and cross-stripe scans (checking `s_stripe_min` then scanning that warp's 32 lanes)
can use the same pattern.

**Expected impact:** Phase 1 divergence eliminated on all input types. Phase 1 currently takes
2.60 ms (random) and is memory-throughput-limited at 125 GB/s; reducing warp stalls should
improve it closer to the descending case (2.20 ms, 190 GB/s). Potential gain: ~0.4 ms on random.

### Bottleneck 2: SPT Phase 3 is slower than WSTL Pass3 on random input (3.06 ms vs 1.81 ms)

Phase 3 in SPT has 192 physical blocks each looping over 341 logical blocks sequentially.
For each logical block with any `INT_MIN` element, each thread does:
1. Read `d_out[gid]` (global, may miss L2: 128 MB total across all blocks).
2. If `INT_MIN`: read `d_in[gid]`, check `prefix_min`, then do a 16-step pointer-chasing tree walk
   on `d_tree` (L2-cached at 512 KB), then warp-min scan (W=16 reads) + leaf scan (~17 reads).

On random input, ~1.3% of elements need the tree walk. But with B=512 elements per logical block,
each logical block has on average ~7 elements needing lookup. These are processed by single threads
(no warp cooperation), so each causes ~640 cycles of L2 latency stall while its warp idles.

**Key difference from WSTL Pass3:** WSTL launches all 65 536 logical blocks as independent GPU
thread blocks; the GPU scheduler hides latency across blocks. SPT's 192 physical blocks each
process 341 logical blocks sequentially — latency from one tree walk directly delays the next
logical block.

**Suggested fix — warp-cooperative Phase 3:**
Collect all elements needing inter-block lookup into a shared-memory queue during the first pass
over the logical block, then process them with warp cooperation (similar to the WarpCoopLookback
variant in the codebase). One warp handles one queued element at a time: lane 0 spins/reads tree
nodes and broadcasts via `__shfl_sync`; all 32 lanes cooperate on the final leaf scan using
`__ballot_sync`. This amortizes L2 latency across the warp instead of stalling one thread.

Alternatively: **two-pass Phase 3** — first pass reads `d_out` for all elements and writes
the (val, gid) pairs needing lookup into a small per-physical-block queue in shared memory,
then a second pass processes the queue with full warp cooperation. The queue is bounded by B = 512
entries (worst case: all elements need lookup on descending input, but prefix_min catches those).

**Expected impact on random input:** Reduces Phase 3 from 3.06 ms toward WSTL's 1.81 ms.
The SPT total could drop from 5.66 ms to ~3.6–4.0 ms, potentially exceeding WSTL on random.

### Bottleneck 3: Hillis-Steele prefix-min scan (8 MB extra traffic, 16 grid.sync() calls)

Phase 2 includes 16 passes of Hillis-Steele prefix-min scan over `d_block_mins` (65 536 entries × 4
bytes = 256 KB). Each pass reads and writes 256 KB → **8 MB total** extra DRAM traffic.
This is on top of the segment min-tree build (which is only 512 KB and L2-cached).

The prefix-min's sole purpose is to allow the O(1) early exit in Phase 3:
```cuda
if (d_prefix_min[block_id] >= val) { d_out[gid] = -1; continue; }
```
On random input, this check fires for very few elements (only elements smaller than all preceding
block-mins), so the 8 MB cost buys almost nothing. On descending input, every element fires the
early exit (prefix_min is always INF since there are no smaller elements), saving all tree walks.

**Suggested fix:** Replace Hillis-Steele with a **work-efficient prefix scan** (e.g. Blelloch
up-sweep + down-sweep) that does the same 16 grid.sync() passes but reads/writes each element
exactly once per pass instead of the Hillis-Steele pattern (which reads an element up to 16 times).
A work-efficient scan over 65 536 elements costs exactly 2 × 65 536 × 4 = 512 KB vs 8 MB.

Or: remove prefix-min entirely and replace the early exit with a quick tree-root check:
```cuda
if (d_tree[0] >= val) { d_out[gid] = -1; continue; }
```
`d_tree[0]` is the global minimum of all blocks; reading it once per element is cheap (always L2-cached).
This is a weaker early exit than prefix_min (only fires when val is smaller than ALL elements, not
just all preceding ones) but costs zero extra passes. On descending input, the per-element fallback
then needs to walk the tree (log N steps) instead of exiting in O(1) — which would hurt descending
performance significantly.

**Expected impact on random:** Saves ~0.4–0.8 ms (8 MB at 95–125 GB/s). No impact on descending
if prefix_min is kept; hurts descending significantly if removed.

## The current SPT kernel (relevant excerpt)

```cuda
template <typename T, int BLOCK_SIZE, int IPT>
__global__
void apsepKernelSPT(
        const T* __restrict__   d_in,
        int*                    d_out,
        int                     n,
        int                     num_blocks,
        int                     M,            // next pow2 >= num_blocks
        int                     leaf_offset,  // M - 1
        T* __restrict__         d_block_leaves,
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

    const int phys_bid = (int)blockIdx.x;
    const int num_phys = (int)gridDim.x;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    __shared__ T s_elems[B];
    __shared__ T s_stripe_min[IPT][NUM_WARPS];

    // Phase 1
    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;

        for (int i = 0; i < IPT; i++) {
            int lid = i * BLOCK_SIZE + threadIdx.x;
            s_elems[lid] = (glb_offs + lid < n) ? d_in[glb_offs + lid] : INF;
        }
        __syncthreads();

        T left_carry[IPT];
        for (int ipt = 0; ipt < IPT; ipt++) {
            T c = s_elems[ipt * BLOCK_SIZE + threadIdx.x];
            for (int step = 1; step <= 16; step <<= 1) {
                T nb = __shfl_up_sync(0xffffffff, c, step);
                if (lane >= step) c = min(c, nb);
            }
            left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
            T wm = __shfl_sync(0xffffffff, c, 31);
            if (lane == 0) s_stripe_min[ipt][warp_id] = wm;
        }
        __syncthreads();

        for (int ipt = 0; ipt < IPT; ipt++) {
            int lid = ipt * BLOCK_SIZE + threadIdx.x;
            int gid = glb_offs + lid;
            if (gid >= n) continue;
            T val = s_elems[lid];
            int result = -1;

            // Within-warp scan (BOTTLENECK: divergent)
            if (lane > 0 && left_carry[ipt] < val) {
                int base = ipt * BLOCK_SIZE + warp_id * 32;
                for (int k = lane - 1; k >= 0; k--)
                    if (s_elems[base + k] < val) { result = base + k; break; }
            }
            // Cross-warp scan
            if (result < 0) {
                for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[ipt][w] < val) {
                        int wb = ipt * BLOCK_SIZE + w * 32;
                        for (int k = 31; k >= 0; k--)
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
            // Cross-stripe scan
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

        // Publish
        T* g_leaves = d_block_leaves + (size_t)block_id * B;
        for (int i = threadIdx.x; i < B; i += BLOCK_SIZE)
            g_leaves[i] = s_elems[i];
        if (lane == 0) {
            T* g_wm = d_block_warp_mins + (size_t)block_id * W;
            for (int ipt = 0; ipt < IPT; ipt++)
                g_wm[ipt * NUM_WARPS + warp_id] = s_stripe_min[ipt][warp_id];
        }
        if (warp_id == 0) {
            T bmin = (lane < W) ? s_stripe_min[lane / NUM_WARPS][lane % NUM_WARPS] : INF;
            for (int step = 16; step >= 1; step >>= 1)
                bmin = min(bmin, __shfl_xor_sync(0xffffffff, bmin, step));
            if (lane == 0) d_block_mins[block_id] = bmin;
        }
        __syncthreads();
    }

    grid.sync();

    // Phase 2: build segment min-tree
    for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < M; i += num_phys * BLOCK_SIZE)
        d_tree[leaf_offset + i] = (i < num_blocks) ? d_block_mins[i] : INF;
    {
        int level_size = M / 2, level_start = M / 2 - 1;
        while (level_size > 0) {
            grid.sync();
            for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < level_size; i += num_phys * BLOCK_SIZE) {
                int node = level_start + i;
                d_tree[node] = min(d_tree[2 * node + 1], d_tree[2 * node + 2]);
            }
            level_size >>= 1;
            level_start = (level_start - 1) / 2;
        }
    }
    grid.sync();

    // Hillis-Steele prefix-min (OVERHEAD: 8 MB extra traffic)
    for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < num_blocks; i += num_phys * BLOCK_SIZE)
        d_prefix_min[i] = (i > 0) ? d_block_mins[i - 1] : INF;
    for (int stride = 1; stride < num_blocks; stride <<= 1) {
        grid.sync();
        for (int i = phys_bid * BLOCK_SIZE + threadIdx.x; i < num_blocks; i += num_phys * BLOCK_SIZE)
            if (i >= stride)
                d_prefix_min[i] = min(d_prefix_min[i], d_prefix_min[i - stride]);
    }
    grid.sync();

    // Phase 3
    for (int block_id = phys_bid; block_id < num_blocks; block_id += num_phys) {
        const int glb_offs = block_id * B;
        const T prefix_min_b = d_prefix_min[block_id];

        for (int i = threadIdx.x; i < B; i += BLOCK_SIZE) {
            int gid = glb_offs + i;
            if (gid >= n || d_out[gid] != INT_MIN) continue;
            T val = d_in[gid];
            if (prefix_min_b >= val) { d_out[gid] = -1; continue; }

            // Tree walk (pointer-chasing, L2-cached)
            int node = leaf_offset + block_id, found_block = -1;
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

            // Warp-min + leaf scan
            const T* wm = d_block_warp_mins + (size_t)found_block * W;
            int result = -1;
            for (int w = W - 1; w >= 0 && result < 0; w--)
                if (__ldg(&wm[w]) < val) {
                    const T* bl = d_block_leaves + (size_t)found_block * B + w * 32;
                    for (int k = 31; k >= 0; k--)
                        if (__ldg(&bl[k]) < val) { result = found_block * B + w * 32 + k; break; }
                }
            d_out[gid] = result;
        }
    }
}
```

## Your task

Optimize `apsepKernelSPT` to be as fast as possible, focusing on the bottlenecks above.
Suggested priority order:

1. **Apply ballot-based backward scan in Phase 1** — highest confidence, cleanest change.
2. **Make Phase 3 warp-cooperative** — addresses the largest remaining gap vs WSTL on random.
3. **Replace Hillis-Steele with work-efficient prefix scan** — reduces Phase 2 traffic 16×.

The kernel must produce correct results. Correctness is verified by the CPU reference:
```cpp
static std::vector<int> cpuApsep(const std::vector<int>& a) {
    int n = (int)a.size();
    std::vector<int> r(n, -1);
    std::vector<int> stk;
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && a[stk.back()] >= a[i]) stk.pop_back();
        r[i] = stk.empty() ? -1 : stk.back();
        stk.push_back(i);
    }
    return r;
}
```

The kernel is a cooperative launch — `grid.sync()` is available; `__syncthreads()` synchronizes
within a physical block. `BLOCK_SIZE=128`, `IPT=4`, `T=int` is the primary configuration.
