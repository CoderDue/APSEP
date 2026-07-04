# SPT whole-block early-out: block-max regression and first-element fix

## The problem

Optimization 6 added a whole-block early-out to SPT Phase 3: if no element
before block `b` is smaller than every unresolved element in `b`, all of
`b`'s unresolved elements resolve to `-1` without reading `d_in[b]` at all.
The descending worst case (100% unresolved) takes this path for every block.

The first implementation tested `prefix_min[b] >= block_max[b]`, which
required computing and storing a per-block maximum:

- Phase 1: extra `s_stripe_max[IPT][NUM_WARPS]` shared array, a second
  `max` shuffle reduction interleaved with the min reduction, and a
  `d_block_maxs` write per block.
- Phase 3: one `d_block_maxs` read per block.

## How the regression manifested

Benchmark at N=32M (SPT useful GB/s = 2·N·4 / time):

| input      | opt5 (before) | opt6 block-max | delta |
|------------|---------------|----------------|-------|
| random     | 65.4          | 60.5           | −7.5% |
| descending | 95.6          | 109.6          | +14.6% |
| ascending  | 109.8         | 95.2           | −13.3% |

Descending improved as intended, but random and ascending regressed.

## How it was identified

Incremental benchmarking after the single change (per the one-change-at-a-time
rule) made the regression obvious. Compile stats showed registers/thread rose
34 → 37 and static shared memory 4164 → 4228 bytes, but `ncu` occupancy limits
were unchanged (still warp-limited at 8 blocks/SM). The cost was therefore the
added Phase 1 *work* — the extra shuffle reduction and global write execute on
every one of the 65,536 logical blocks regardless of input — while the early-out
only pays off on inputs with many unresolved elements.

## The fix

Observation: **unresolved elements within a block form a non-increasing
sequence.** If an earlier unresolved element were smaller than a later one,
the later one would have resolved against it intra-block. And the first
element of every block is always unresolved (nothing precedes it
intra-block). Therefore the block's first element *is* the maximum of its
unresolved elements, and the early-out test is simply

```cuda
if (prefix_min_b >= __ldg(&d_in[glb_offs])) { /* all unresolved -> -1 */ }
```

Zero Phase 1 cost: no shared array, no max reduction, no `d_block_maxs`
array at all. Result:

| input      | opt5 | opt6 block-max | opt6b first-element |
|------------|------|----------------|---------------------|
| random     | 65.4 | 60.5           | 64.4                |
| descending | 95.6 | 109.6          | **114.6**           |
| ascending  | 109.8| 95.2           | 108.6               |

Descending improved further (Phase 1 is lighter than the block-max variant)
while random/ascending returned to opt5 levels.

## Trade-offs of the chosen solution

- Phase 3 issues one extra `d_in` read per block (~256 KB at N=32M) even
  when the early-out fails; this is negligible (cache-line granularity —
  the queue pass would fetch that line anyway on most inputs).
- The correctness argument (non-increasing unresolved sequence) is subtle
  and lives in a comment; a naive future edit to Phase 1's resolution rule
  (e.g. changing `<` to `<=` semantics) could silently break it.
- Random/ascending are ~1 GB/s below opt5 in single runs — within noise,
  but the per-block branch itself is not entirely free.

## Downsides of alternative approaches

- **Block-max array (as implemented first):** taxes Phase 1 on all inputs
  for a benefit only realized on descending-like inputs; +3 registers,
  +64 B shared, +2 global transfers per block. Rejected due to the measured
  regression above.
- **Computing the max only over unresolved elements** (tighter bound than
  block max): even more Phase 1 work (predicated max reduction), and the
  non-increasing property makes it exactly equal to the first element —
  i.e. it degenerates to the chosen solution but at a higher price.
- **No early-out:** descending stays at ~95 GB/s; every block pays the
  full queue pass reading all 512 elements of `d_in` plus bitmask tests.
