# SPT: how it works, and why it is close to its ceiling

This documents the final SPT (SinglePassTree) kernel — `apsepKernelSPT` in
`src/apsep.cuh` — as of the July 2026 optimization session, and the evidence
that significant further speedup is unlikely without an algorithm change.

Problem: for each `a[i]`, find the largest `j < i` with `a[j] < a[i]`, else
-1 (APSEP). Target hardware: GTX 1660 Ti — 24 SMs, 288 GB/s peak DRAM,
1.5 MB L2.

## Architecture overview

SPT is a *single* cooperative kernel (`cudaLaunchCooperativeKernel`) with
three phases separated by `grid.sync()`:

1. **Phase 1** — solve every element *within* its block of B=512 elements;
   flag the rest in a bitmask; publish per-block min summaries.
2. **Phase 2** — build a global min-tree over the 65,536 block-mins and an
   exclusive prefix-min per block.
3. **Phase 3** — warps grid-stride over the bitmask; flagged elements are
   resolved with a tree query + two coalesced scans, behind two early-out
   levels for the common cases.

The grid is *persistent*: 192 physical blocks (8/SM × 24 SMs, from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) loop over all logical
blocks in stride order. Configuration: BLOCK_SIZE=128 threads, IPT=4
elements/thread → B=512 elements and W=16 chunk-mins (one per 32
consecutive elements) per logical block.

Why single-pass: a multi-kernel pipeline (WSTL) re-reads intermediate state
from DRAM between launches. The cooperative kernel keeps everything in one
launch and lets Phase 3 skip nearly all work when the input allows it.

## Data structures

| array | size (N=32M) | purpose |
|---|---|---|
| `d_unres` | N/8 = 4 MB | 1 bit/element: needs inter-block lookup |
| `d_block_mins` | 256 KB | min of each block |
| `d_block_warp_mins` | 4 MB | min of each 32-element warp chunk |
| `d_tree` | 512 KB | binary min-tree over block-mins (fits in L2) |
| `d_prefix_min` | 256 KB | `prefix_min[b]` = min of blocks `0..b-1` |

There is no leaves array: block leaves would be byte-identical to `d_in`,
so Phase 3 reads the input directly.

## Phase 1 — intra-block PSE (blocked layout)

Each thread owns IPT=4 *consecutive* elements, loaded with a single int4
(the load is still fully coalesced across the warp: lane `l` reads bytes
`16l..16l+15`), stored to shared `s_elems` and kept in registers. The warp
machinery then operates on *per-thread mins* — once per 128 elements
instead of once per 32 as in the earlier striped layout.

1. **In-thread sequential ANSV.** Element `i` of a thread checks its own
   elements `i-1, i-2, ...` in registers — pure ALU, no shuffles. On
   random input this resolves ~half of all elements; in a descending run
   every non-first element of a thread resolves here.
2. **Warp inclusive prefix-min scan** (5 `__shfl_up_sync` steps) over
   per-thread mins `tmin` gives each lane `carry` = min of all elements
   owned by lower lanes; lane 31's value is the warp min (`s_warp_min`).
   Three `__shfl_xor_sync`s also produce the 32-element *chunk* mins for
   `d_block_warp_mins` (8 threads = 1 chunk).
3. **Thread-level ANSV chain by pointer jumping.** Built once per warp
   over the `tmin` values (each lane's query is its own `tmin`, which
   preserves the gap-bound convergence argument of the element-level
   version). Elements not resolved in-thread then *walk* this finished
   chain (spine walk) to the nearest thread whose min beats their value,
   and finish with a ≤4-element register scan of that thread's elements in
   shared memory. A thread's locally-unresolved elements are its prefix
   minima (non-increasing), so the walk cursor is monotone and reused
   across the thread's 4 elements.
4. **Cross-warp fallback.** Still-unresolved lanes scan earlier warps'
   mins, then those warps' thread-mins (`s_tmin`), then the exact ≤4
   elements — all in shared memory.
5. **Publish.** Resolved elements write `d_out` directly. Unresolved
   elements write *nothing* — one `__ballot_sync` per warp per element
   slot packs their flags into `d_unres`. Every `d_out` byte is therefore
   written exactly once across the whole kernel. Because ballots are
   lane-major, element `4j+i` of a 32-element chunk lands in bit `8i+j`;
   Phase 3 applies the inverse permutation
   (`mybit = 8*(lane&3) + (lane>>2)`).

Finally thread 0 reduces the warp mins to `d_block_mins[block_id]`.

The blocked layout cut the kernel's issued warp-instructions by 30% on
random (314.7 M → 220.5 M) and 3× on descending, worth +14/+64/+3 GB/s
(random/descending/ascending) over the striped version.

## Phase 2 — min-tree and prefix-min

A standard bottom-up binary min-tree over the block-mins (one `grid.sync`
per level, all 524 KB of it L2-resident). Then, in a single pass, each
block's exclusive prefix-min is computed by *ascending* the finished tree:
walking leaf→root and taking the min over left siblings, whose subtrees
cover exactly leaves `[0, b)`. This replaced a Hillis-Steele scan (16
grid-syncs, 8 MB traffic) with O(log M) L2-cached reads and zero extra
syncs.

## Phase 3 — inter-block resolution

Phase 3 is driven directly by the bitmask: warps grid-stride over `d_unres`
in **aligned groups of 32 consecutive words** (one coalesced load per
group; the alignment matters — see `p3_word_striding_regression.md` for
the load-balance disaster a bare word stride causes). Each lane also
prefetches its word's block id, prefix-min, and early-out flag, shuffled
into the per-word loop. There is no per-block sweep, no shared memory, and
no `__syncthreads` anywhere in Phase 3; a zero word costs one L2-cached
read. Cheapest test first:

1. **Dense-group fast path.** If all 32 words of the group are fully
   unresolved and early-out (below) — true for every group on descending
   input — the warp writes the 4 KB span of -1s with coalesced int4
   stores and moves on.
2. **Whole-block early-out.** Unresolved elements in a block are
   *non-increasing* (if an earlier one were smaller, the later one would
   have resolved against it in Phase 1), and a block's first element is
   always unresolved — so it is the max of them all. If
   `prefix_min[b] >= d_in[block_start]`, every unresolved element in the
   word answers -1: write them and skip the word.
3. **Per-element early exit.** Otherwise each set bit either resolves to
   -1 via `prefix_min[b] >= val` (one coalesced masked load of the
   values) or joins a ballot of genuinely inter-block elements.
4. **Warp-cooperative lookup**, one balloted element at a time:
   - *Tree walk* (redundant on all 32 lanes — identical addresses broadcast
     from L2): ascend from leaf `b` until a left sibling's min beats `val`,
     then descend preferring right children. This lands on the **nearest
     earlier block** whose min < `val`.
   - *Warp-min scan*: one 32-wide coalesced load of that block's 16
     warp-mins + `__ballot_sync`; `31 - __clz(mask)` picks the rightmost
     qualifying warp chunk.
   - *Leaf scan*: one 32-wide coalesced load of that chunk from `d_in` +
     ballot picks the rightmost element `< val`. Safe unguarded: the found
     block precedes `b`, so it is complete.

Total cost per genuinely inter-block element: ~16 L2-cached tree reads and
exactly two 128-byte coalesced DRAM loads. On random input only ~1.3% of
elements get this far.

## Correctness invariants worth preserving

- Phase 1 warp primitives use full masks; boundary handling is via
  `active` flags and INF padding, never early `continue` (which would
  break `__shfl_sync`/`__ballot_sync` in the last partial block).
- The pointer-jumping loop only jumps while `a[g] >= val`, preserving
  "everything strictly between `g` and `i` is `>= val`"; `g` strictly
  decreases, so it terminates at the exact nearest smaller element.
- The first-element early-out depends on Phase 1 resolving against *any*
  strictly-smaller earlier element (`<`, not `<=`). Changing that
  comparison breaks the non-increasing-unresolved argument silently.
- Every `d_out` byte is written exactly once; nothing initializes `d_out`.

## Phase split (bench_spt_phases, N=32M, median of 9)

| | descending | random | ascending |
|---|---|---|---|
| Phase 1 | 917 µs | 2263 µs | 1243 µs |
| Phase 2 + all 19 grid.syncs | ~5 µs | ~0 µs | ~0 µs |
| Phase 3 | 749 µs | 1157 µs | 289 µs |

Phase 1 is ~55–80% of every case; the grid.sync structure itself is free
(a sync-only cooperative kernel with 19 syncs costs 22 µs). Descending's
Phase 3 is now a pure 128 MB int4 -1-fill — bandwidth, not logic.

## Why it won't get much better

Measurements on the final kernel (ncu, N=32M; ncu-run clocks are lower
than benchmark clocks, so compare ratios, not durations, against the
benchmark tables):

| | random | descending |
|---|---|---|
| duration | 3.44 ms | 1.67 ms |
| DRAM busy | 35.7% | 62.2% |
| compute (SM) busy | 45.1% | 28.1% |
| issue-slot utilization | 48.9% | 28.3% |
| issued warp-instructions | 220.5 M | 66.4 M |
| achieved occupancy | 94.6% | 87.6% |
| stalls (random): wait 23%, barrier 21%, long-scoreboard 18%, short-scoreboard 14% |

The two cases now hit *different* walls:

1. **Descending is approaching the bus.** 62% DRAM busy with the SM
   three-quarters idle; the analytical traffic (~192 GB/s at benchmark
   clocks, 182 GB/s useful) is two-thirds of the 288 GB/s peak. What's
   left is bus efficiency (write-heavy traffic, read/write turnaround),
   not algorithm.

2. **Random is still instruction/latency-bound, but 30% closer to its
   floor.** 220.5 M issued warp-instructions against a peak issue rate of
   96 schedulers × 1.51 GHz ≈ 145 G/s puts an absolute lower bound of
   ~1.5 ms even with a perfect scheduler; measured is ~2.2× that, at ~95%
   occupancy (the SM 7.5 hardware limit is fully subscribed — no more
   thread-level parallelism to buy), with no stall reason above 23%. The
   `wait` stalls come from the dependent shuffle chains of the prefix-min
   scan and pointer jumping (inherent); the barrier stalls from the one
   remaining `__syncthreads` per block iteration (required — the
   cross-warp fallback reads other warps' shared data); the scoreboard
   stalls from loads that are already fully coalesced.

3. **The obvious levers were tried and measured** (see
   `spt_optimizations.md`): IPT=8 (worst case −35%), register prefetch of
   the next block (neutral), double-buffered shared memory (−6%), int4
   loads on the striped layout (−6%). The one redesign that *did* pay —
   the blocked layout, which attacks the instruction count itself rather
   than the schedule — is now in (step 10, −30% instructions on random).
   Repeating that trick would need an intra-block PSE with meaningfully
   fewer than ~6.5 warp-instructions per element; nothing measured or
   sketched so far gets there without losing the coalescing or the
   worst-case guarantees.
