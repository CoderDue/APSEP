# APSEP Bottleneck Analysis

Profiled with `ncu --set full` on GTX 1660 Ti (SM 7.5, 24 SMs, 288 GB/s peak).
Kernel config: BLOCK_SIZE=128, IPT=4, B=512 elements/block, N=32M elements, 65 536 logical blocks.

## Summary table

| Scenario          | Kernel     | Duration | DRAM%  | Mem GB/s | Occupancy | Threads/warp | Branch eff |
|-------------------|------------|----------|--------|----------|-----------|--------------|------------|
| Random            | WSTL Pass1 | 3.25 ms  | 45.4%  | 125 GB/s | 89%       | 11.3 / 32    | 80%        |
| Random            | WSTL Pass2 | ~0.1 ms  | tiny   | —        | —         | —            | —          |
| Random            | WSTL Pass3 | 1.81 ms  | 36.3%  | 99 GB/s  | 77%       | 9.2 / 32     | 97%        |
| Descending (worst)| WSTL Pass1 | 2.14 ms  | 69.0%  | 190 GB/s | 91%       | 31.1 / 32    | 99%        |
| Descending (worst)| WSTL Pass3 | 3.05 ms  | 48.0%  | 132 GB/s | 97%       | 32.0 / 32    | 100%       |
| Random            | SPT        | 6.47 ms  | 34.4%  | 95 GB/s  | 88%       | 10.5 / 32    | 84%        |
| Descending (worst)| SPT        | 4.08 ms  | 71.9%  | 199 GB/s | 99%       | 31.3 / 32    | 99%        |

Peak DRAM throughput = 288 GB/s. Useful throughput (2×N×4 bytes) benchmarked at 57–65 GB/s (WSTL) and 44–70 GB/s (SPT).

---

## Bottleneck 1: Warp divergence in the intra-block backward scan (Pass1 / SPT Phase 1)

**What it is.** The intra-block PSE search (Phase 1 / Pass1) is a backward linear scan through shared memory.
Each thread scans right-to-left starting from its own position looking for the first element smaller than itself.
Threads within the same warp do completely different-length scans — the rightmost thread in the warp may scan 0
elements (its PSE is the immediately preceding element) while the leftmost may scan 31 elements.
The warp only retires when every lane finishes, so it executes as many iterations as the slowest lane.

**What ncu says.**
- Random input: **Avg. Active Threads Per Warp = 11.3 / 32** in Pass1, and 9.2 / 32 in Pass3. Branch efficiency 80%.
- Descending input: Avg. Active Threads / Warp = 31.1 (almost all lanes busy) and branch efficiency 99% —
  because on descending data every element's PSE is -1 so the loop runs to its natural limit uniformly.
- The divergence costs are directly visible in the random-vs-descending duration: Pass1 takes 3.25 ms on random
  but only 2.14 ms on descending, even though descending does more DRAM work (reads d_in AND d_out for all N).

**Why it persists.** The backward search is inherently data-dependent. Threads in the same warp start at positions
`[warp_base, warp_base+31]` and scan backward — they diverge at every step because the loop exit condition
(`s_elems[k] < val`) fires at a different `k` for each lane.

---

## Bottleneck 2: Low sector utilization in Phase 3 / Pass3 leaf scan

**What it is.** In Pass3 (WSTL) and Phase 3 (SPT), threads that need an inter-block answer do:
1. Ascend/descend the block-level min-tree (tree fits in L2, no DRAM cost).
2. Scan warp-mins (W=16 entries per block) then the 32 matching leaves in `d_block_leaves`.

The leaf scan is sequential right-to-left within the chosen warp's 32-element range — a single thread scanning
up to 32 addresses, stopping on first match. On random input, the match hits early on average (~17 elements),
so most of the fetched cache lines go unused.

**What ncu says.**
- WSTL Pass3 random: **L2 Hit Rate = 13.4%** (all 65 536 blocks doing random-index leaf reads go mostly to DRAM).
- SPT random: **L2 Hit Rate = 53.4%** but DRAM throughput only 34.4% (95 GB/s) — L2 is absorbing more but
  SPT's extra Hillis-Steele scan also uses L2.
- ncu warns: *"On average, only 19.6 of the 32 bytes transmitted per sector are utilized by each thread"*
  (SPT) and *"9.4 bytes"* (older WSNT variant) — early-exit leaves most sectors partially used.

**Trade-off.** The warp-min hierarchy (W=16 warp-min values per block, then 32 leaves) was added precisely
to improve sector utilization vs the previous flat 512-leaf scan. It helped, but the fundamental issue remains:
a single-threaded backward scan with early exit always wastes the tail of each fetched cache line.

---

## Bottleneck 3: SPT Hillis-Steele prefix-min scan adds significant overhead vs WSTL

**What it is.** SPT uses a Hillis-Steele parallel prefix-min over `num_blocks = 65 536` entries.
log₂(65 536) = 16 passes, each reading and writing all 65 536 × 4 bytes = 256 KB.
Total = 16 × 2 × 256 KB = **8 MB** of extra reads + writes just for prefix-min.

Additionally, SPT uses `grid.sync()` which forces the 192 physical blocks to idle between phases —
Phase 2 (tree build + prefix-min) is serialized at the grid level.

**What ncu says.**
- SPT random takes 6.47 ms total vs WSTL's 3.25 + 1.81 = 5.06 ms total.
  The extra ~1.4 ms matches the expected cost of 8 MB at 95–125 GB/s effective bandwidth.
- SPT DRAM throughput is only 34.4% on random (vs 45.4% for WSTL Pass1 alone), because Phase 2 is a
  small-grid, low-parallelism phase that underutilizes the memory bus.
- SPT descending: 4.08 ms at 71.9% DRAM — Phase 3 dominates and is bandwidth-bound (every element needs
  inter-block lookup with zero early-exit via prefix-min), so SPT's Phase 3 efficiency shines here.

**Why WSTL wins on random, SPT wins on descending.**
WSTL Pass3 on random is short (1.81 ms, only ~1.3% of elements do inter-block lookups).
SPT pays 8 MB of prefix-min overhead regardless of input.
On descending input, every element needs inter-block lookup; SPT's prefix-min enables O(1) early exit
("min of all prior blocks ≥ val → answer is -1") for the deepest elements, cutting tree traversal.
WSTL Pass3 descending must traverse the tree and scan leaves for all N elements (3.05 ms).

---

## Bottleneck 4: Write amplification — block_leaves doubles DRAM write traffic

**What it is.** Both kernels write `d_block_leaves` (B=512 elements × 65 536 blocks × 4 bytes = **128 MB**),
which is identical to the input data re-arranged by block. This is on top of the useful 128 MB output write.
So Phase 1 alone writes 256 MB: 128 MB d_out + 128 MB d_block_leaves, for just 128 MB of input read.

This was a deliberate trade-off (vs the original 2×B min-tree write of 256 MB), but it means DRAM write
traffic is 2× the input size before any inter-block work begins.

**What ncu says.**
- WSTL Pass1 random: Memory Throughput = 125 GB/s, DRAM% = 45.4%. The 128 MB input read + 128 MB leaves
  write + 128 MB d_out write = 384 MB at ~3.25 ms ≈ 118 GB/s — consistent.
- WSTL Pass1 descending: Memory Throughput = 190 GB/s at 69% DRAM — the same 3 writes but faster because
  descending data is more coherent (no early-exit divergence in the scan means all threads write d_out
  simultaneously, improving write coalescing).

---

## Bottleneck 5: Phase 2 tree reduce — many tiny underutilized kernel launches (WSTL only)

**What it is.** WSTL Pass2 builds the block-min tree bottom-up with one kernel launch per level.
With 65 536 blocks → 16 levels → 16 `wstlReduceLevelKernel` launches. The bottom levels have 32 768,
16 384, … nodes, but the top levels have 1–128 nodes and launch one 256-thread block processing
a handful of elements — wildly underutilized.

**What ncu says.**
- `wstlReduceLevelKernel` (grid=1, 1 block): Duration ~1.8–1.9 µs, DRAM% < 1%, Occupancy 16–25%.
- `wstlReduceLevelKernel` (grid=128): Duration ~3 µs, DRAM% 40%, Occupancy 83%.
- Total time across all 16 reduce levels ≈ 16 × 2 µs = ~32 µs — negligible vs Pass1/3 at 3+ ms each.

**Verdict.** Not a significant bottleneck in absolute time. The 16 kernel-launch overheads (~16 × 5–10 µs
launch latency) add ~80–160 µs, which is measurable but not dominant. SPT avoids this by folding Pass2
into the cooperative kernel, but pays the grid.sync() overhead instead.

---

## What is NOT a bottleneck

- **Occupancy**: Both kernels achieve 88–99% occupancy. Register count (16–21 regs/thread) is not the
  limiting factor. Shared memory (B×4 = 2 KB + stripe_min = 256 bytes per block) leaves plenty of room.
- **The min-tree (d_tree) reads**: The tree is 2×65 536 – 1 = 131 071 nodes × 4 bytes ≈ 512 KB, which
  fits in the 1.5 MB L2 cache. L2 hit rates for tree reads are high; tree traversal costs are absorbed
  in L2 and do not stress DRAM.
- **Phase 2 reduce kernels (WSTL)**: Only ~32 µs total, negligible vs the 5–6 ms total runtime.

---

## Potential optimizations (not yet implemented)

1. **Vectorized backward scan**: Instead of per-thread sequential scan, use warp-wide `__ballot_sync`
   to find the rightmost set bit — all lanes query a predicate in parallel, then `__clz(ballot)` gives
   the rightmost match in one instruction. Would raise active threads/warp toward 32 on random input.

2. **Coarser block granularity** (larger B): Increasing IPT from 4 to 8 (B=1024) reduces num_blocks by 2×,
   halving block_leaves traffic and the prefix-min cost. Trade-off: more shared memory pressure per block,
   deeper backward scan divergence.

3. **Skip Phase 3 for blocks with no INT_MIN elements**: A per-block flag (set in Phase 1 if any element
   wrote INT_MIN) could let Pass3 / Phase 3 skip entire blocks early. Attempted previously — found that
   100% of blocks always have at least one element needing inter-block lookup (the first element of each
   block has no intra-block predecessor). So skip rate is 0% and the flag check is pure overhead.

4. **Vectorized d_block_leaves write**: The leaf write is 128-thread strided over B=512 elements (4 passes
   of 128 threads). Using `float4` / 128-bit stores would halve the number of store instructions.
