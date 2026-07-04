# APSEP — All Previous Smaller Element Problem on GPU

For each element `a[i]` in an array, find the index `j < i` of the nearest preceding element satisfying `a[j] < a[i]`, or -1 if none exists. This is the foundation of monotone-stack algorithms and appears in histogram problems, stock-span, parsing, and tree construction.

This project implements and benchmarks parallel APSEP algorithms in CUDA, targeting SM 7.5 (GTX 1660 Ti, 288 GB/s peak memory bandwidth).

## Algorithms

Two production kernels are maintained in `src/apsep.cuh`:

**WSTL — WarpScanTreeLookup** (`runWSTL` / `launchWSTL`)
Three-pass approach:
1. *Pass 1*: each block computes intra-block PSE using a warp prefix-min scan, then publishes its leaf values, warp-mins, and block-min to global memory.
2. *Pass 2*: build a segment min-tree over all block-mins (CPU-launched kernels, one per level).
3. *Pass 3*: elements that did not resolve intra-block query the tree (ascent+descent, O(log N)) then do a warp-min + 32-leaf scan to find the exact answer.

Random input: **~65 GB/s** (N=32M).

**SPT — SinglePassTree** (`runSPT` / `launchSPT`)
Single cooperative kernel launched with `cudaLaunchCooperativeKernel`. 192 persistent physical blocks loop over all logical blocks in stride order, using `grid.sync()` to separate phases. Beyond the WSTL structure it adds: warp-uniform pointer-jumping for the within-warp ANSV, an unresolved-element bitmask (each output byte is written exactly once), an exclusive prefix-min computed by tree ascent, a whole-block early-out (`prefix_min >= first element` proves all unresolved answers are -1), and warp-cooperative Phase 3 queue processing. See `docs/spt_optimizations.md`.

Matches or beats WSTL on all inputs; descending (worst-case): **~117 GB/s** (N=32M), 1.8× WSTL.

## Performance (N=32M, GTX 1660 Ti, useful GB/s = 2·N·4 bytes / time)

```
                random            descend            ascend
              WSTL    SPT       WSTL    SPT        WSTL    SPT
33554432       65     64         64     117         107    107
```

## Building

```sh
make              # build test + correctness binary (apsep_test)
make profile_bottleneck  # build ncu profiling target
```

Override SM architecture with `CUDA_ARCH=<sm>`, e.g. `make CUDA_ARCH=86` for Ampere.

## Running

```sh
./apsep_test      # correctness checks + benchmark across input types and sizes
```

Output includes:
- Structured small-case correctness tests for both kernels
- Random stress tests (500 trials up to n=4096, 200 trials up to n=131072)
- Fixed-N benchmark table (N=32M): useful GB/s and analytical total DRAM traffic
- N sweep table: useful GB/s across 16K–128M for random, descending, and ascending input

## Repository layout

```
src/
  apsep.cuh              — all kernel implementations (WSTL, SPT, and earlier variants)
  main.cu                — correctness tests + benchmark (the primary binary)
  profile_bottleneck.cu  — minimal target for ncu profiling
  bench_*.cu             — historical benchmark drivers (not actively maintained)
  profile_*.cu           — historical profiling drivers
docs/
  bottlenecks.md               — ncu profiling findings and bottleneck analysis
  spt_optimizations.md         — SPT optimization log: what was tried, kept, rejected
  spt_early_out_regression.md  — block-max early-out regression and first-element fix
.claude-artifacts/       — temporary logs, ncu reports, build artifacts (not committed)
```
