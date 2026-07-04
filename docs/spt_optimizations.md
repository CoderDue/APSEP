# SPT optimization log

Chronology of optimizations applied to the single-pass SPT kernel
(`apsepKernelSPT` in `src/apsep.cuh`) during the July 2026 optimization
session, including experiments that were tried and rejected. All numbers are
useful GB/s (= 2·N·4 bytes / time) at N=32M on a GTX 1660 Ti (288 GB/s peak).

| step | random | descending | ascending |
|---|---|---|---|
| baseline | 49.5 | 69.7 | 76.2 |
| 1. pointer-jumping within-warp ANSV | 52.9 | 69.8 | 76.6 |
| 2. warp-cooperative Phase 3 | 58.8 | 75.6 | 77.6 |
| 3. tree-ascent prefix-min | 59.1 | 76.0 | 78.2 |
| 4. leaves-write elimination | 60.1 | 82.9 | 95.6 |
| 5. unresolved bitmask | 65.4 | 95.6 | 109.8 |
| 6. whole-block early-out (first-element test) | 64.4 | 114.6 | 108.6 |
| 7. Phase 1 barrier removal | **64.2** | **116.9** | **106.5** |

WSTL reference: 64.7 / 63.7 / 106.5. SPT now matches or beats WSTL on every
input class; on the descending worst case it is ~1.8× faster.

## Kept optimizations

### 1. Pointer-jumping within-warp ANSV

The per-thread sequential backward search over the warp's elements diverged
badly (10.5/32 active threads on random). Replaced by a warp-uniform
pointer-jumping loop: each lane tracks a candidate index `g`, and lanes whose
candidate value is not yet smaller jump to the candidate's candidate via
`__shfl_sync`, converging in O(log 32) rounds under an `__any_sync` test.
All lanes participate in every shuffle (inactive lanes hold `val=INF`,
`g=-1`, and are never jump sources because lower lanes in a warp are always
active).

*Trade-off:* more instructions on inputs where the sequential search would
exit immediately (ascending), but uniform execution wins everywhere tested.

### 2. Warp-cooperative Phase 3

Phase 3 previously let each thread walk the tree and scan leaves alone —
divergent and latency-serial. Now unresolved elements are collected into a
shared queue (`s_queue`, shared `atomicAdd` counter) and processed one per
warp: the tree walk runs redundantly on all 32 lanes (converged,
L2-broadcast), and the warp-min + leaf scans become two 32-wide coalesced
loads resolved with `__ballot_sync` + `31 - __clz(mask)`.

*Trade-off:* redundant tree-walk arithmetic ×32, and elements with very
cheap scans pay full warp overhead. Wins because the scans dominate and
become coalesced.

### 3. Tree-ascent prefix-min

`d_prefix_min[b] = min(block_min[0..b-1])` was a Hillis-Steele scan:
16 passes × 512 KB = 8 MB of traffic and 16 `grid.sync()`s. Replaced by a
single pass that ascends the already-built min-tree, taking the min over
left siblings on the leaf-to-root path (subtrees covering exactly leaves
`[0, b)`). O(log M) L2-cached reads per block, zero extra grid syncs.

*Trade-off:* none measurable; the tree (524 KB) fits in the 1.5 MB L2.

### 4. Leaves-write elimination

Phase 1 wrote a 128 MB `d_block_leaves` array that was byte-identical to
`d_in` at the same offsets. Phase 3 now reads `d_in` directly. Safe because
the found block is always a strictly earlier complete block, so all its B
elements are in range.

*Trade-off:* Phase 3 leaf reads no longer benefit from any hypothetical
re-layout of leaves; irrelevant since the layout was identical anyway.

### 5. Unresolved bitmask

Phase 1 used to write an `INT_MIN` sentinel to `d_out` for unresolved
elements, forcing Phase 3 to re-read all 128 MB of `d_out` to find them.
Now Phase 1 publishes one bit per element via
`__ballot_sync(active && result < 0)` (one 4-byte word per warp) into
`d_unres`, and unresolved elements get **no** Phase 1 `d_out` write — every
output byte is written exactly once across the whole kernel. Phase 3 tests
bits (N/8 bytes) instead of reading 128 MB.

*Trade-off:* an extra N/8-byte array and slightly more Phase 1 ballot work;
saves ~2×N×4 bytes of traffic on random.

### 6. Whole-block early-out (first-element test)

See `spt_early_out_regression.md` for the full story: the initial block-max
implementation regressed random/ascending and was replaced by the zero-cost
observation that a block's first element is the maximum of its unresolved
elements, so `prefix_min_b >= d_in[glb_offs]` proves every unresolved
element in the block resolves to -1. Descending: 95.6 → 116.9 GB/s.

### 7. Phase 1 barrier removal

The barrier between the shared `s_elems` store and the warp prefix-min scan
was unnecessary: the scan only consumes the thread's *own* element, which is
now kept in registers (`v[IPT]`); the first cross-thread `s_elems` read
happens after the remaining mid-iteration `__syncthreads`. Removes one of
three barriers per logical-block iteration (barrier stalls were 19% of warp
cycles). Descending +2.5 GB/s, others within noise.

## Rejected experiments (measured, then reverted)

Final-round stall profile (random): long_scoreboard 22%, wait 22%,
barrier 19%, short_scoreboard 13% — latency-bound with no dominant stall.

- **IPT=8** (B=1024): ascending 108.6 → 137.0, but random 64.4 → 57.3 and
  descending 114.6 → 74.5. The longer intra-block scan and reduced occupancy
  hurt the worst case far more than the ILP helped. Rejected: worst-case
  latency is SPT's purpose.
- **Software prefetch of the next logical block** into registers: no change
  on random/descending, ascending −3% (register pressure). The compiler
  already overlaps these loads across the loop.
- **Double-buffered shared memory** (drop the end-of-iteration barrier):
  random −2, descending −7 GB/s. The extra shared usage and indexing cost
  more than the barrier saved.
- **int4 vectorized loads** (via shared-memory redistribution, since the
  striped layout can't be vector-loaded directly): descending −7 GB/s. The
  reintroduced barrier + shared round-trip outweighed saving 3 load
  instructions; the scalar loads were already fully coalesced.

## Why the kernel does not reach 80% of peak bandwidth

`ncu` on the final kernel: DRAM busy 24% (random) / 37% (descending); SM
48–55%. Neither pipe is saturated — the kernel is bound by dependent-
operation latency (global loads feeding scans, shuffle chains, barriers),
not bandwidth. The optimizations above *removed* DRAM traffic, so the
analytical-total GB/s metric (all reads+writes / time) went down even as the
kernel got faster; bus saturation is the wrong target for this algorithm at
this point. Reaching 230 GB/s of DRAM traffic would require ~3× more
concurrent memory work than the algorithm has to issue.
