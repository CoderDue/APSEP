# Phase 3 word-striding load-balance regression

## What happened

The first implementation of the bitmask-driven Phase 3 (step 8 in
`spt_optimizations.md`) had warps grid-stride over `d_unres` words with a
stride of one word per warp:

```cuda
for (int w = phys_bid * NUM_WARPS + warp_id; w < num_words; w += warps_total)
```

All correctness tests passed, but the kernel *regressed catastrophically*:
random 64 → 26 useful GB/s, ascending 107 → 59 (N=32M). Descending was
unaffected.

## How it manifested / how it was identified

`ncu` on the random case showed the paradox precisely:

- issued warp-instructions unchanged (~313.6 M) — the same work was being
  done,
- duration 11.67 ms (vs ~5.2 ms before),
- achieved occupancy collapsed to 66.8% (from ~97%),
- long-scoreboard stalls up to 28.5%.

Same instructions, twice the time, occupancy collapse ⇒ a *load-balance*
problem, not an efficiency problem.

Root cause: `warps_total` = 192 physical blocks × 4 warps = 768, which is a
multiple of W = 16 (bitmask words per logical block, B=512). So each warp
always landed on the **same word position within every logical block**.
Unresolved bits are not uniformly distributed across word positions: the
elements that fail to resolve intra-block are the block's prefix minima,
which cluster at the *front* of the block — overwhelmingly in word 0. The
48 warps (of 768) whose fixed position was word 0 therefore performed
essentially **all** of the Phase 3 lookups serially, while the other 720
warps finished instantly and exited, deflating occupancy for the remainder
of the kernel.

## How it was solved

Each warp takes an **aligned group of 32 consecutive words** instead of a
single word:

```cuda
for (int wbase = (phys_bid * NUM_WARPS + warp_id) * 32;
     wbase < num_words;
     wbase += warps_total * 32)
```

The 32 lanes load the group's words in one coalesced transaction, ballot
the nonzero ones, and process those one word at a time,
warp-cooperatively. A 32-word group spans exactly two logical blocks
(32/W = 2), so every group contains the same mix of word positions and the
dense front-of-block words distribute evenly over warps. This fixed the
regression and beat the pre-step-8 baseline (random 64 → 82.5, ascending
107 → 170).

## Trade-offs of the fix

- Group-granularity assignment: a warp whose group is entirely zero words
  still pays one coalesced load; acceptable because that load *is* the
  minimum possible bitmask-read cost.
- The tail group is partially guarded (`wl < num_words`), adding one
  comparison per lane.
- Work distribution is still granular at 32 words = 1024 elements; a
  pathological input that concentrates all unresolved elements in one
  group region would still imbalance, but such inputs also concentrate the
  *work itself*, which no static assignment fixes.

## Alternatives considered

- **Skip-scan / compaction of nonzero words first** (a pre-pass writing a
  compact list of nonzero word indices): perfect balance, but costs an
  extra kernel-wide pass plus global atomics or a scan, and the measured
  imbalance was already gone with aligned groups — not worth the traffic.
- **Stride co-prime with W** (e.g. `warps_total | 1`): breaks the
  fixed-position pathology but destroys coalescing of the bitmask reads
  (adjacent warps no longer read adjacent words) and still admits other
  resonances.
- **Dynamic work stealing via a global atomic counter**: robust to any
  distribution, but a contended global atomic per word group costs more
  than the imbalance it prevents at this scale (measured in an earlier
  session for Phase 3 queues: atomics were the reason the shared-memory
  queue was per-block, not global).

## Lesson

A grid-stride whose stride shares a factor with a per-block structure size
samples that structure at a fixed phase. When the work distribution within
the structure is skewed (here: prefix minima at block fronts), the stride
turns a balanced loop into a serial one. Check `stride mod W` whenever
striding over per-block subdivisions.
