# CLAUDE.md — Project-specific instructions

## Project goal

Implement a single-pass APSEP (All Previous Smaller Element) CUDA kernel that is as fast as possible on an Nvidia GTX 1660 Ti (SM 7.5, 24 SMs, 288 GB/s peak memory bandwidth). The two active kernels are WSTL (three-pass) and SPT (single cooperative pass); SPT is the primary optimization target.

## Profiling tools

### Nsight Compute (`ncu`)

Available at `/run/current-system/sw/bin/ncu`. Use it to profile individual kernel launches.

**Collect a full profile:**
```sh
ncu --set full -o .claude-artifacts/profile/mykernel ./my_binary [args]
```

**Print a text report from a saved `.ncu-rep`:**
```sh
ncu --import .claude-artifacts/profile/mykernel.ncu-rep \
    --page details --print-summary per-kernel \
    > .claude-artifacts/profile/mykernel_report.log
```

**Collect specific metrics only (faster, fewer replay passes):**
```sh
ncu --metrics metric1,metric2,... -o .claude-artifacts/profile/mykernel ./my_binary
```

**Useful metric names:**
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy
- `lts__t_sector_hit_rate.pct` — L2 hit rate
- `l1tex__t_sector_hit_rate.pct` — L1 hit rate
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` — sector utilization (load)
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` — stalls waiting on L2/DRAM
- `smsp__warp_issue_stalled_barrier_per_warp_active.pct` — stalls on `__syncthreads`
- `smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct` — divergence stalls

**Key sections to read in a full report:**
- `GPU Speed Of Light Throughput` — DRAM%, Memory Throughput, Duration
- `Memory Workload Analysis` — L1 hit rate, L2 hit rate, GB/s
- `Occupancy` — Achieved Occupancy, Block Limit (registers, shared mem, warps)
- `Warp State Statistics` — Warp Cycles Per Issued Instruction, Active Threads Per Warp
- `Scheduler Statistics` — Eligible Warps Per Scheduler (how often the scheduler has work)
- `Source Counters` — Branch Efficiency, Avg Divergent Branches

**What the numbers mean:**
- `DRAM Throughput %` — fraction of peak 288 GB/s being used; below ~40% on a memory-bound kernel suggests latency rather than bandwidth is the limit
- `Avg. Active Threads Per Warp` — should be 32 for fully converged warps; low values (e.g. 11/32) indicate warp divergence
- `Branch Efficiency %` — 100% means no divergent branches; below 90% is worth investigating
- `L2 Hit Rate %` — the block-min tree (512 KB) fits in the 1.5 MB L2 and should show high hit rates; leaf arrays (128 MB) do not

### Nsight Systems (`nsys`)

Available at `/run/current-system/sw/bin/nsys`. Use for timeline profiling across multiple kernels (useful for WSTL's multi-kernel pipeline).

```sh
nsys profile --output .claude-artifacts/profile/timeline ./my_binary
nsys stats .claude-artifacts/profile/timeline.nsys-rep
```

### `cudaOccupancyMaxActiveBlocksPerMultiprocessor`

Used in code to determine how many physical blocks to launch for the SPT cooperative kernel. Current result: 8 blocks/SM × 24 SMs = 192 physical blocks. Check this if shared memory or register count changes.

## Hardware reference (GTX 1660 Ti, SM 7.5)

| Property | Value |
|---|---|
| SMs | 24 |
| Peak memory bandwidth | 288 GB/s |
| L2 cache | 1.5 MB |
| Shared memory per SM | 64 KB |
| Max blocks per SM | 16 |
| Max warps per SM | 32 |
| Warp size | 32 |
| L2 read latency | ~40 cycles |
| Shared memory latency | ~20 cycles |

At BLOCK_SIZE=128, IPT=4: B=512 elements/block, W=16 warp-mins/block, shared memory per block ≈ 2.25 KB → up to 28 blocks per SM (register-limited to ~10 in practice with current SPT).

## Current performance baseline (N=32M, after July 2026 optimization session)

| Input     | WSTL     | SPT      |
|-----------|----------|----------|
| Random    | ~65 GB/s | ~98 GB/s |
| Descending| ~64 GB/s | ~182 GB/s|
| Ascending | ~107 GB/s| ~174 GB/s|

"Useful GB/s" = 2×N×4 bytes / elapsed time (input read + output write only).
Quote the N-sweep table from `apsep_test` as the reference (the fixed-N
section runs at a different clock/thermal state). SPT beats WSTL on all
inputs. See `docs/spt_optimizations.md` for the full optimization log (kept
and rejected changes), `docs/spt_design.md` for how SPT works,
`docs/spt_early_out_regression.md` for the block-max regression, and
`docs/p3_word_striding_regression.md` for the Phase 3 word-stride
load-balance regression.

## Bottleneck status

Resolved in SPT (see `docs/spt_optimizations.md`):
1. ~~Warp divergence in intra-block backward scan~~ — fixed by warp-uniform pointer jumping.
2. ~~Low L2 sector utilization in Phase 3 leaf scan~~ — fixed by warp-cooperative Phase 3 (coalesced 32-wide loads + ballots).
3. ~~Hillis-Steele prefix-min overhead~~ — replaced by single-pass tree-ascent prefix-min.
4. ~~Write amplification (`d_block_leaves`)~~ — eliminated; Phase 3 reads `d_in` directly, and the unresolved bitmask means every `d_out` byte is written exactly once.
5. ~~Phase 3 per-block sweep overhead~~ — replaced by bitmask-driven Phase 3 (aligned 32-word groups per warp) with a dense-word -1-fill fast path.
6. ~~Striped Phase 1 shuffle cost~~ — replaced by blocked layout (in-register sequential ANSV + warp machinery over per-thread mins; −30% instructions on random).

Remaining: on random, SPT is still latency/instruction-bound (ncu: DRAM 36%
busy, SM 45%, occupancy 95%, no dominant stall — wait/barrier/long_scoreboard
each ~20%); on descending it is approaching bandwidth-bound (DRAM 62% busy,
SM 28%). Measured dead ends: IPT=8 (both striped and blocked layouts),
software prefetch, double-buffered shared memory, int4 loads on the striped
layout, sub-warp-team Phase 3 lookups, warp-autonomous barrier-free Phase 1
(all regressed or were neutral on random; details in
`docs/spt_optimizations.md`). `docs/bottlenecks.md` describes the
pre-optimization state; WSTL still has bottlenecks 1, 2, and 4. SPT's Phase 1
is IPT=4 / 4-byte-T specific (static_asserted).
