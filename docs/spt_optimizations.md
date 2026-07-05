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
| 7. Phase 1 barrier removal | 64.2 | 116.9 | 106.5 |
| 8. bitmask-driven Phase 3 | 82.5 | 114.5 | 170.4 |
| 9. dense-word fast path + per-group metadata | 84.1 | 117.8 | 171.3 |
| 10. blocked-layout Phase 1 | **98.2** | **181.7** | **174.4** |

WSTL reference: 65.1 / 63.9 / 106.5. SPT now beats WSTL on every input
class: 1.5× on random, 2.8× on the descending worst case, 1.6× on
ascending.

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
(The shared queue was later removed by step 8; the one-element-per-warp
lookup structure survives.)

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

### 8. Bitmask-driven Phase 3

Phase measurement (`bench_spt_phases`) showed the old Phase 3 cost ~1.2 ms
on random/ascending even though the actual tree lookups were worth ~40 µs
(ascending resolves only ~64K elements): the per-logical-block sweep —
loop setup, bitmask scan, shared-queue build, `__syncthreads` — dominated.
Replaced by warps grid-striding over `d_unres` words directly: each warp
loads an *aligned group of 32 consecutive words* (one coalesced load),
ballots the nonzero ones, and processes them one word at a time. No shared
memory, no barriers, no per-block anything in Phase 3.

The first attempt used a bare word stride and regressed random 64 → 26
GB/s despite passing all correctness tests — see
`p3_word_striding_regression.md` for the load-balance analysis and fix.

*Trade-off:* a zero word still costs its share of the group load, and the
per-word metadata (block id, prefix-min, early-out flag) is fetched per
lane then shuffled — slightly more work per dense word than the per-block
sweep paid. Ascending +64 GB/s, random +18 GB/s; descending −2 GB/s
(recovered by step 9).

### 9. Dense-word fast path + per-group metadata

Two refinements to step 8. (a) Each lane precomputes its word's block id,
prefix-min, and whole-block early-out flag at group-load time; the per-word
loop reads them via `__shfl_sync` instead of re-issuing L2 loads. (b) If
all 32 words of the group are fully-unresolved *and* early-out — true for
every group in the descending worst case — the warp writes the whole 4 KB
span of -1s with coalesced int4 stores and skips the per-word loop
entirely. Descending recovered +3 GB/s past its step-7 level; random +1.6.

*Trade-off:* one extra branch per group; unmeasurable on inputs that never
take the path.

### 10. Blocked-layout Phase 1

The striped layout ran the full warp machinery (5-step prefix-min scan +
pointer-jumping ANSV) once per 32 elements. The blocked layout gives each
thread IPT=4 *consecutive* elements (one int4 load):

- In-thread sequential ANSV resolves within-thread matches in registers —
  ~half of all elements on random, *all* non-first elements of a
  descending run — with zero shuffles.
- The warp prefix-min scan and the pointer-jumping chain now run over
  *per-thread mins*: once per 128 elements instead of once per 32 (4×
  fewer shuffle instructions).
- Elements unresolved in-thread walk the finished thread-level chain
  (valid because a chain query equal to the querying thread's own min
  preserves the gap-bound argument; per-element spine walks reuse a
  monotone cursor since a thread's locally-unresolved elements are its
  prefix minima, non-increasing).
- The `__ballot_sync` bit order changes: element `4j+i` of a 32-element
  chunk lands in bit `8i+j`. Phase 3 compensates with the inverse
  permutation (`mybit = 8*(lane&3) + (lane>>2)`) when assigning lanes to
  elements; `d_unres` words and `d_block_warp_mins` chunk order are
  otherwise unchanged.

Kernel instruction count on random dropped 30% (314.7 M → 220.5 M issued
warp-instructions), on descending 3× (202.5 M → 66.4 M). Random +14,
descending +64, ascending +3 GB/s.

*Trade-off:* +5 registers (35 → 40; still 8 blocks/SM), and the in-thread
ANSV adds comparison work that pays off only when consecutive elements
resolve each other — which measurement shows they overwhelmingly do. The
implementation is specialized to IPT=4 and 4-byte T (`static_assert`ed).

## Rejected experiments (measured, then reverted)

Stall profile at step 7 (random): long_scoreboard 22%, wait 22%,
barrier 19%, short_scoreboard 13% — latency-bound with no dominant stall.
(After step 10: wait 23%, barrier 21%, long_scoreboard 18%,
short_scoreboard 14% — same picture, fewer instructions.)

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

Three follow-up prototypes after step 10 (all pass correctness; all kept
8 blocks/SM with zero spills, so the losses are algorithmic; GB/s vs the
same-run SPT baseline at N=32M, prototypes in `src/bench_spt_*.cu`):

- **Sub-warp-team Phase 3** (`bench_spt_p3team`): split each Phase 3 warp
  into 4 teams of 8 lanes so 4 tree lookups run concurrently (4 independent
  latency chains per warp instead of 1). Random 81.1 → 77.6, ascending
  171.4 → 167.9, descending flat. The 4× latency overlap did not pay for
  the narrower probes: each lookup's chunk-min ballot covers W=16 with 8
  lanes (2 mins/lane) and the leaf scan drops from 32-wide to 8-wide, so
  per-lookup work goes up while the tree-walk itself (the serial part) is
  unchanged. Serialized warp-cooperative lookups remain the better trade.
- **IPT=8 on the blocked layout** (`bench_spt_ipt8`, B=1024, 47 regs,
  4.6 KB smem): random 77.2 → 64.8, ascending 174.8 → 110.7, descending
  flat. The striped-layout IPT=8 rejection (above) was re-tested because
  the blocked layout changed the calculus — halving warp machinery per
  element instead of doubling shuffle scans — but it still loses: the
  O(IPT²) in-thread scan, longer exact-scan reads through shared, and
  halved logical-block parallelism (32K blocks for 192 physical) cost more
  than the halved warp machinery saves. IPT=8 is now rejected on *both*
  layouts.
- **Warp-autonomous Phase 1** (`bench_spt_warpauto`, one 128-element
  logical block per warp, zero shared memory, zero `__syncthreads`, no
  cross-warp fallback): random 88.3 → 65.8, ascending 174.3 → 114.9,
  descending +2 (175.9, best descending measured). Removing the barrier
  (21% of stalls) did help the barrier-free-friendly descending case, but
  4× more logical blocks means a ~2 MB min-tree that no longer fits the
  1.5 MB L2, ~4× more genuine Phase 3 lookups on random (unresolved
  prefix-minima scale per-block), and 4× more prefix-min tree ascents.
  The barrier was cheaper than the block-size reduction needed to remove
  it.

## Why the kernel does not reach 80% of peak bandwidth

`ncu` on the final kernel (step 10): random — DRAM busy 35.7%, SM 45%,
issue 49%, occupancy 94.6%; descending — DRAM busy 62.2%, SM 28%. Random
is still bound by dependent-operation latency (global loads feeding scans,
shuffle chains, barriers) with no dominant stall, not bandwidth.
Descending has crossed over: at 62% DRAM busy with SM nearly idle it is
approaching bandwidth-bound, and its analytical total (~192 GB/s of
traffic at benchmark clocks) is 67% of peak — the remaining headroom there
is mostly real bus limits, not scheduling. The optimizations above
*removed* DRAM traffic, so the analytical-total GB/s metric went down even
as the kernel got faster on some steps; bus saturation was the wrong
target until step 10 made descending fast enough to feel the bus.
