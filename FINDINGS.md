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

**Best result: IPT=2, K=8 → 15.6 GB/s** (5.4% of peak, ~16× serialization).

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

End-to-end (with the original tree look-back at K=8): **15.6 GB/s**, same as
before.  The intra-block phase is fast; the bottleneck is the inter-block
serial chain.

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

## Summary table

| Kernel | GB/s | Serial steps | Notes |
|---|---|---|---|
| Warp-scan (intra-block only) | 107 | — | No inter-block answer |
| Min-tree K=1 | 8.7 | 512K | Baseline end-to-end |
| Stack look-back K=1 | 8.0 | 512K | Compact stack; worse than tree |
| Min-tree K=8 (**best**) | **15.6** | 64K | Optimal K empirically |
| Persistent-thread | ~0 | 192 × chunk | Effectively serial |

The 15.6 GB/s result is approximately the practical ceiling for single-pass
APSEP on this GPU.  Closing the remaining ~6× gap to memory bandwidth would
require a fundamentally different algorithm (e.g., multi-pass with a
work-efficient prefix-scan over a commutative/associative aggregate — which
APSEP does not have).
