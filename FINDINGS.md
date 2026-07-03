# APSEP GPU Performance Investigation

## Problem

For each element `arr[i]`, find the index `j < i` of the nearest previous element
satisfying `arr[j] < arr[i]`, or -1.  This is the All Previous Smaller Element
Problem (APSEP).

Hardware: NVIDIA GeForce GTX 1660 Ti (SM 7.5, 24 SMs).  
Theoretical peak bandwidth: 288 GB/s.  Copy baseline: ~245 GB/s.

---

## Algorithms explored

### 1. Min-tree block kernel (baseline)

Each block builds a binary min-tree over `B = BLOCK_SIZE × IPT` elements in
shared memory, queries the tree for intra-block PSE, then publishes the tree to
global memory.  Elements without an intra-block answer perform a **decoupled
look-back** at super-block granularity: they spin-wait on status flags of
previous super-blocks of `K` blocks each, then query the merged tree.

**Best result: IPT=2, K=8 → 17.4 GB/s** (6.0% of peak, ~14× serialization; after bug fix — see below).

The K sweep showed a clear optimum around K=8.  Smaller K means more serial
look-back steps; larger K means more work per super-block before publishing,
increasing latency for downstream blocks.

### 2. Warp-scan block kernel (intra-block only)

Replaced the min-tree with a 5-round `__shfl_up_sync` prefix-min scan.  Each
block processes `B` elements in `IPT` stripes; cross-stripe PSE is resolved
via a `s_stripe_min[IPT][NUM_WARPS]` array with one `__syncthreads()` per
stripe.  Results go directly into registers, written once at the end.

**Intra-block throughput: ~107 GB/s** (IPT=2, B=256).  This is 44% of peak —
a solid result for the intra-block phase.

End-to-end (with the original tree look-back at K=8): **17.4 GB/s** (after bug
fix).  The intra-block phase is fast; the bottleneck is the inter-block serial
chain.

### 3. Stack look-back kernel

Same warp-scan intra-block phase, but each block publishes a compact **monotone
suffix stack** instead of a full min-tree.  For random input the suffix stack
has ~O(log B) ≈ 5-6 entries, vs 2B entries for the tree.  Inter-block look-back
binary-searches each previous block's stack.

**Result: 8.0 GB/s** — worse than the tree look-back at K=1 (8.7 GB/s), and
much worse than K=8 (15.6 GB/s).

Why: the per-step savings from reading fewer bytes don't compensate.  The GPU
memory subsystem is efficient at reading cache lines regardless; the binary
search adds CPU-side compute overhead that the tree query avoids via the
hardware-friendly reduction pattern.  More importantly, the stack look-back
runs at K=1 (512K serial steps) while the tree look-back at K=8 has only 64K
steps — 8× fewer serial dependencies wins decisively.

### 4. Persistent-thread kernel

Launch exactly `P = num_SMs × active_blocks_per_SM = 192` CTAs.  Each CTA
owns a contiguous chunk of `ceil(num_tiles / P)` tiles and maintains the
monotone suffix stack in shared memory across tile boundaries.  CTA `c` spins
on a flag set by CTA `c-1` before starting.

**Result: 0.0 GB/s** (~2.5 seconds per 500 MB pass — ~250× slower than K=8).

**Root cause**: the design is fundamentally serial.  CTA `c` cannot start
until CTA `c-1` has finished *all* of its tiles and published its final carry.
So the 192 CTAs execute one at a time.  Total time ≈ 192 × (time per chunk) =
serial over the entire input.

Pipelining would require CTA `c-1` to publish a carry after every individual
tile, which reduces to the original decoupled look-back scheme.

---

## Why the serialization is irreducible

The PSE answer for element `arr[i]` depends on the values of *all* `arr[0..i-1]`.
No block processing elements near index `i` can produce a final answer without
first knowing what happened before it in the array.

The serial dependency chain has length proportional to `N / (tile_size × K)`.
Any algorithm must traverse this chain; the only lever is making each step
faster (larger tile = more work before publishing, but also more latency) or
hiding the latency (K-factor, caching).

The K=8 design sits at the empirical sweet spot: `N / (256 × 8) = 64K` serial
steps, each requiring one L2 round-trip for the status flag plus a tree query
that is often cached.

---

## Bug fix: intra-superblock look-back

Elements needing an answer from an earlier block *within the same superblock*
(`sb_local > 0`, answer not in own block) previously had no path to find it:
the intra-block PSE only covers the element's own block, and `decoupledLookback`
only scanned *previous* superblocks.  These elements incorrectly returned -1.

**Fix:** before calling `decoupledLookback`, scan per-block trees for blocks
`sb_first .. block_id-1` in the same superblock (spin-waiting on each block's
status flag as needed).  This is the same wait that the last-in-SB block already
performs, just scoped to the querying block's own earlier siblings.

**Effect:** throughput improved from ~15.7 GB/s to ~17.4 GB/s because affected
elements no longer spin fruitlessly waiting on a superblock tree that will never
contain their answer.

---

## Alternative full-array approaches (benchmarked 2026-07-03)

Three alternative algorithms were benchmarked against the corrected single-pass
kernel on 500 MiB of uniform random `int` data (GTX 1660 Ti):

### Two-pass: intra-block + inter-block suffix-stack lookback

Pass 1 uses the warp-scan intra-block kernel (B=128) and builds a monotone
suffix stack per block.  Pass 2 launches one thread per element still at -1,
scanning backward over published stacks.

**Result: ~5.8 GB/s.**  Pass 2 has O(N) work per element in the worst case,
and even on random data the backward scan over all prior stacks is slow.

### Segmented scan: CUB DeviceScan with monotone-stack MergeOp

Pass A computes intra-block PSE and builds a per-block carry (monotone suffix
stack).  CUB `DeviceScan::ExclusiveScan` with a custom `MergeOp` accumulates
the prefix carry for each block.  Pass B uses the prefix carry to answer any
element still at -1 via binary search.

**Result: ~9.2 GB/s.**  Faster than two-pass but still ~2× slower than the
single-pass kernel.  CUB's device scan over 516-byte `BlockCarry3` structs has
high per-call overhead, and the `MergeOp` itself (copying + filtering the left
carry) is expensive.

**Bug found and fixed in MergeOp:** the original implementation iterated left
carry entries from `depth-1` to `0` (smallest→largest val) when appending,
breaking the monotone-decreasing invariant.  The binary search then returned
the wrong (leftmost) index.  Fix: find the contiguous suffix of left entries
with `val < right_min` and append in forward (largest→smallest val) order.

---

## Summary table

| Kernel | GB/s | Serial steps | Notes |
|---|---|---|---|
| Warp-scan (intra-block only) | 107 | — | No inter-block answer |
| Min-tree K=1 | 8.7 | 512K | Baseline end-to-end |
| Stack look-back K=1 | 8.0 | 512K | Compact stack; worse than tree |
| Two-pass suffix-stack | 5.8 | — | O(N) pass-2 scan per element |
| Segmented scan (CUB) | 9.2 | — | Two passes; MergeOp overhead |
| Min-tree K=8 (**best**) | **17.4** | 64K | After intra-SB bug fix |
| Persistent-thread | ~0 | 192 × chunk | Effectively serial |

The 17.4 GB/s result is the practical ceiling for single-pass APSEP on this
GPU.  Closing the remaining ~14× gap to peak bandwidth would require a
fundamentally different algorithm (e.g., a multi-pass approach with a
commutative/associative aggregate — which APSEP does not have).
