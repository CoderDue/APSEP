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

### 5. WarpScanLeaves kernel (winner)

Combines two improvements over the baseline min-tree kernel:

1. **Warp-scan intra-block PSE**: replaces the `buildMinTree` + `treePrevSmaller`
   shared-memory tree with 5 `__shfl_up_sync` prefix-min steps.  Eliminates the
   `s_tree[2*B]` shared-memory region entirely.

2. **Leaves-only global publish**: writes only the `B` raw input elements to
   `d_block_leaves` instead of the full `2*B` min-tree.  The last-in-SB block
   reads these leaves to build the merged SB tree.  Intra-SB look-back does a
   linear scan over stored leaves.

**Effect on shared memory:** `3076 → 1060 bytes/block` (IPT=2) or `3076 → 2116 bytes/block` (IPT=4).
With 46 registers/block, occupancy is register-limited at 11 blocks/SM regardless.
The shared-memory reduction mainly matters for larger IPT by reducing evictions.

**IPT sweep (BS=128, K=8, N=32M ints):**

| IPT | B | WarpScanLeaves GB/s |
|---|---|---|
| 1 | 128 | 17.0 |
| 2 | 256 | 40.7 |
| **4** | **512** | **46.3** |
| 8 | 1024 | 28.5 |

**K sweep results (BS=128, IPT=4, N=32M ints):**

| K | GB/s |
|---|---|
| 4 | 33.5 |
| 8 | **46.3** |
| 10 | 46.4 |
| 12 | 45.3 |
| 16 | 42.2 |

**Best configuration: WarpScanLeaves IPT=4, K=8 → 46+ GB/s** on N=32M ints.

On production-size 500 MiB (N=131M ints, bench_approaches):

| Config | GB/s |
|---|---|
| Baseline IPT=2 K=8 | 17.7 |
| WarpScanLeaves IPT=2 K=8 | 26.7 |
| **WarpScanLeaves IPT=4 K=8** | **30.5** |

Note the throughput drop from 32M to 131M input: the SB tree working set no
longer fits in L2 cache, making each inter-SB look-back step slower.

Why IPT=4 wins: doubling B from 256→512 halves the number of serial look-back
steps (each block handles 2× more elements, so the chain from first→last block
has half as many links).  The per-step cost is the same, so half the steps = 2×
less serial overhead.

**Optimizations tried and rejected:**

- `__ballot_sync` backward search: fundamentally cannot work — each thread
  searches for its own `val`, so the ballot condition differs per lane.  No
  single ballot call can answer all threads' queries simultaneously.  Attempted
  a uniform-loop variant that always executes ballot across all lanes; correctness
  failure confirmed the approach is wrong.

- `__launch_bounds__(128, 13)`: reduced IPT=4 from 46→44 registers but the
  compiler spills registers, costing ~0.6 GB/s.  Not beneficial.

---

### 6. WarpScanNoTree kernel (new best)

Eliminates the per-superblock min-tree (`d_sb_trees`) entirely.  The inter-SB
decoupled look-back is replaced by:

1. Check `sb_min` from `d_sb_states` (same as before — one read).
2. If `sb_min < val`: iterate `K` block_mins in that SB to find the rightmost
   matching block (K reads from `d_states`, sequential).
3. Linear scan of that block's B leaf values to find the exact position.

**Why the tree was a bottleneck:** for N=131M with K=8, B=512, the SB tree is
32 KB per SB (2×KB×4 bytes), and there are ~32K SBs → 1 GB total tree data.
The L2 cache on GTX 1660 Ti is only 1.5 MB, so every inter-SB tree traversal
is an L2 miss.  The traversal itself does O(log KB) = 12 pointer-chasing hops
through this cold data.

**The no-tree approach** reads K block_mins (sequential, warm) + B leaves
(burst of 512 ints, hardware-prefetcher-friendly).  The access pattern is
entirely sequential rather than pointer-chasing, allowing the GPU memory
subsystem to hide latency.

**Key insight**: with the tree gone, larger K is no longer penalized by larger
tree build/read costs.  K can now be increased to reduce the serial SB chain
length much more aggressively.

**K sweep (BS=128, IPT=4, N=32M ints):**

| K | WarpScanNoTree GB/s | WarpScanLeaves GB/s |
|---|---|---|
| 8 | 44.0 | 46.3 |
| 16 | 50.3 | 42.0 |
| 32 | 53.8 | — |
| 64 | 55.4 | — |
| **128** | **55.6** | — |
| 256 | 55.4 | — |

**Best configuration: WarpScanNoTree IPT=4, K=128 → 55.6 GB/s** (N=32M).

On production-size 500 MiB (N=131M ints, bench_approaches):

| Config | GB/s |
|---|---|
| Baseline IPT=2 K=8 | 16.8 |
| WarpScanLeaves IPT=2 K=8 | 26.8 |
| WarpScanLeaves IPT=4 K=8 | 30.4 |
| **WarpScanNoTree IPT=4 K=128** | **37.2** |

With K=128 and B=512, there are N/(K×B) = 131M/65536 ≈ 2K serial SB chain
steps, each requiring only K+B = 640 sequential int reads instead of 12 L2-cold
tree hops.

---

## Summary table

| Kernel | GB/s (N=32M) | GB/s (N=131M) | Serial steps | Notes |
|---|---|---|---|---|
| Warp-scan (intra-block only) | 107 | — | — | No inter-block answer |
| Min-tree K=1 | 8.7 | — | 512K | Baseline end-to-end |
| Stack look-back K=1 | 8.0 | — | 512K | Compact stack; worse than tree |
| Two-pass suffix-stack | — | 5.8 | — | O(N) pass-2 scan per element |
| Segmented scan (CUB) | — | 9.2 | — | Two passes; MergeOp overhead |
| Min-tree K=8 (baseline) | 17.4 | 16.8 | 64K | After intra-SB bug fix |
| LeavesOnly K=8 | 16.7 | 16.7 | 64K | Saves N write bytes; no SM gain |
| NoBlockTree K=8 | 16.7 | 16.7 | 64K | Same as LeavesOnly |
| WarpScanLeaves K=8 | 46.3 | 30.4 | 64K | Warp-scan + leaves-only |
| WarpScanLeaves K=16 | 43.6 | — | 32K | IPT=2; prior "best" |
| **WarpScanNoTree K=128** | **55.6** | **37.2** | **2K** | **No SB tree; sequential reads** |
| Persistent-thread | ~0 | ~0 | 192×chunk | Effectively serial |

**WarpScanNoTree K=128** at **55.6 GB/s** (N=32M) / **37.2 GB/s** (N=131M) is
the best single-pass result (19.3% of peak at N=32M).  The remaining gap to peak
is inherent to the serial dependency chain in APSEP — no single-pass algorithm
can close it without a fundamentally different aggregate structure.
