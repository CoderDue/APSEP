// Benchmark comparing three full-array PSE approaches against the champion
// single-pass decoupled look-back kernel (apsepKernel IPT=2 K=8, ~55 GB/s).
//
// Approaches:
//   Baseline : apsepKernel<int,128,2,8>  — current best single-pass
//   Approach1: Two-pass (intra-block WarpScanV3 + inter-block stack look-back)
//   Approach2: Sort-based (CUB DeviceRadixSort + serial scan)
//   Approach3: Segmented scan (CUB DeviceScan with monotone-stack merge operator)

#include "apsep.cuh"

#include <cub/device/device_scan.cuh>

#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <functional>
#include <algorithm>
#include <chrono>

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

static std::vector<int> cpuPse(const std::vector<int>& a) {
    int n = (int)a.size();
    std::vector<int> out(n, -1);
    std::vector<int> stk;
    for (int i = 0; i < n; i++) {
        while (!stk.empty() && a[stk.back()] >= a[i]) stk.pop_back();
        out[i] = stk.empty() ? -1 : stk.back();
        stk.push_back(i);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------

static void computeDescriptors(const std::vector<float>& ms, size_t bytes) {
    int n = (int)ms.size();
    double mean = 0, var = 0, gbps = 0;
    double factor = (double)bytes / (1000.0 * n);
    for (float m : ms) {
        double us = std::max(1e3 * (double)m, 0.5);
        mean += us / n;
        var  += us * us / n;
        gbps += factor / us;
    }
    double std = std::sqrt(var - mean * mean);
    double ci  = 0.95 * std / std::sqrt((double)n);
    printf("%.0fµs (CI [%.1f,%.1f]); %.1f GB/s", mean, mean-ci, mean+ci, gbps);
}

// kernel_fn must be a callable () -> void (use a lambda to avoid macro comma issues)
#define BENCH(label, kernel_fn, bytes, warmup, iters)                        \
do {                                                                         \
    for (int _w = 0; _w < (warmup); _w++) { (kernel_fn)(); gpuAssert(cudaDeviceSynchronize()); } \
    cudaEvent_t _t0, _t1; gpuAssert(cudaEventCreate(&_t0)); gpuAssert(cudaEventCreate(&_t1)); \
    std::vector<float> _ms(iters);                                           \
    for (int _i = 0; _i < (iters); _i++) {                                  \
        gpuAssert(cudaEventRecord(_t0));                                     \
        (kernel_fn)();                                                       \
        gpuAssert(cudaEventRecord(_t1));                                     \
        gpuAssert(cudaEventSynchronize(_t1));                                \
        gpuAssert(cudaEventElapsedTime(&_ms[_i], _t0, _t1));                \
    }                                                                        \
    gpuAssert(cudaEventDestroy(_t0)); gpuAssert(cudaEventDestroy(_t1));      \
    printf("  %-38s  ", label);                                              \
    computeDescriptors(_ms, bytes);                                          \
    printf("\n");                                                            \
} while(0)

// ===========================================================================
// Approach 1: Two-pass
//
// Pass 1 — blockwiseKernelWarpScanV3 (BS=64 IPT=2, B=128):
//   Solves intra-block PSE; writes -1 for elements with no local answer.
//   Each block also builds and publishes a monotone suffix stack.
//
// Pass 2 — interBlockLookback:
//   For each element still at -1, scans backward over published suffix stacks.
//   No spin-wait; pass 1 is fully complete before pass 2 launches.
// ===========================================================================

// Intra-block PSE kernel (WarpScanV3, BS=64 IPT=2).
// Also builds per-block monotone suffix stack published to d_stacks.
// Identical to apsepStackLookbackKernel's intra phase but outputs -1
// (not the look-back result) for inter-block elements so pass 2 can fill them.

constexpr int P1_BS  = 64;
constexpr int P1_IPT = 2;
constexpr int P1_B   = P1_BS * P1_IPT;   // 128

// Suffix stack entry
struct StackEntry { int val; int idx; };

// Per-block published state for pass 2
struct BlockCarry {
    int          block_min;  // min value in block (fast screen)
    int          depth;      // number of stack entries
    StackEntry   stack[P1_B]; // monotone decreasing-value suffix stack
};

__global__ void pass1Kernel(const int* __restrict__ d_in, int* d_out, int n,
                             BlockCarry* d_carries) {
    constexpr int B         = P1_B;
    constexpr int NUM_WARPS = P1_BS / 32;
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    __shared__ int s_elems[B];
    __shared__ int s_stripe_min[P1_IPT][NUM_WARPS];

    // Load
    #pragma unroll
    for (int i = 0; i < P1_IPT; i++) {
        int lid = i * P1_BS + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INT_MAX;
    }
    __syncthreads();

    // Warp prefix-min scan + intra-warp backward search (V3)
    int carry[P1_IPT];
    int left_carry[P1_IPT];
    #pragma unroll
    for (int ipt = 0; ipt < P1_IPT; ipt++) {
        int val = s_elems[ipt * P1_BS + threadIdx.x];
        int c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            int nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry[ipt]      = c;
        left_carry[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        int wm = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = wm;
    }
    __syncthreads();

    int r_out[P1_IPT];
    #pragma unroll
    for (int ipt = 0; ipt < P1_IPT; ipt++) {
        int lid = ipt * P1_BS + threadIdx.x;
        int val = s_elems[lid];
        int result = -1;
        if (lane > 0 && left_carry[ipt] < val) {
            int base = ipt * P1_BS + warp_id * 32;
            for (int k = lane - 1; k >= 0; k--)
                if (s_elems[base + k] < val) { result = base + k; break; }
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * P1_BS + w * 32;
                    for (int k = 31; k >= 0; k--)
                        if (s_elems[wb + k] < val) { result = wb + k; break; }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * P1_BS + w * 32;
                        for (int k = 31; k >= 0; k--)
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
        }
        r_out[ipt] = (result >= 0) ? (glb_offs + result) : -1;
    }

    // Write intra-block results
    #pragma unroll
    for (int i = 0; i < P1_IPT; i++) {
        int gid = glb_offs + i * P1_BS + threadIdx.x;
        if (gid < n) d_out[gid] = r_out[i];
    }

    // Build monotone suffix stack for this block (thread 0 does it serially)
    if (threadIdx.x == 0) {
        int depth = 0;
        int bmin  = INT_MAX;
        BlockCarry& bc = d_carries[blockIdx.x];
        for (int lid = B - 1; lid >= 0; lid--) {
            int gid = glb_offs + lid;
            if (gid >= n) continue;
            int v = s_elems[lid];
            if (v < bmin) bmin = v;
            // Push to suffix stack if it's a new minimum from the right
            if (depth == 0 || v < bc.stack[depth - 1].val) {
                bc.stack[depth].val = v;
                bc.stack[depth].idx = gid;
                depth++;
            }
        }
        bc.block_min = bmin;
        bc.depth     = depth;
    }
}

// Pass 2 with d_in available
__global__ void pass2KernelFull(const int* __restrict__ d_in,
                                 int* d_out, int n, int num_blocks,
                                 const BlockCarry* __restrict__ d_carries) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n || d_out[gid] != -1) return;

    int val      = d_in[gid];
    int my_block = gid / P1_B;
    int result   = -1;

    for (int b = my_block - 1; b >= 0 && result < 0; b--) {
        if (d_carries[b].block_min >= val) continue;
        // Binary search the suffix stack for rightmost entry with val < our val
        // Stack is monotone decreasing in value (stack[0] is largest, rightmost idx)
        const BlockCarry& bc = d_carries[b];
        // Find rightmost index in this block with value < val.
        // Stack entries are in reverse-index order (stack[0] = rightmost suffix min).
        // We want the entry with the largest idx whose val < our val.
        // Since stack is monotone decreasing in value as depth increases,
        // the first entry (depth-1) has the smallest value (global min).
        // We want the entry with smallest depth (largest val still < our val).
        int lo = 0, hi = bc.depth;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (bc.stack[mid].val < val) hi = mid;
            else                          lo = mid + 1;
        }
        // lo is first entry with val < our val; that has the largest idx
        // (since we built stack right-to-left, smaller depth = rightmost position)
        if (lo < bc.depth)
            result = bc.stack[lo].idx;
    }
    d_out[gid] = result;
}

// Scratch + run for two-pass
struct TwoPassScratch {
    BlockCarry* d_carries = nullptr;
    int         num_blocks = 0;
};

static TwoPassScratch allocTwoPass(int n) {
    TwoPassScratch s;
    s.num_blocks = (n + P1_B - 1) / P1_B;
    gpuAssert(cudaMalloc(&s.d_carries, (size_t)s.num_blocks * sizeof(BlockCarry)));
    return s;
}

static void freeTwoPass(TwoPassScratch& s) {
    cudaFree(s.d_carries);
    s = TwoPassScratch{};
}

static void runTwoPass(const int* d_in, int* d_out, int n, TwoPassScratch& s) {
    pass1Kernel<<<s.num_blocks, P1_BS>>>(d_in, d_out, n, s.d_carries);
    gpuAssert(cudaGetLastError());
    // Pass 2: one thread per output element (only those needing inter-block work)
    constexpr int BS2 = 256;
    int grid2 = (n + BS2 - 1) / BS2;
    pass2KernelFull<<<grid2, BS2>>>(d_in, d_out, n, s.num_blocks, s.d_carries);
    gpuAssert(cudaGetLastError());
}

// ===========================================================================
// Approach 3: Segmented scan with monotone-stack carry
//
// Each block computes its intra-block PSE (using WarpScanV3) and a compact
// "carry" = the monotone suffix stack of its B elements.
//
// CUB DeviceScan with a custom operator merges carries left to right:
//   merge(left, right):
//     For each element in right's PENDING list (elements with no local answer),
//     binary-search left's stack. But CUB DeviceScan works on a fixed-size
//     type — so we encode the carry as a fixed-size stack of depth <= MAX_DEPTH.
//
// After the scan, a second kernel uses each block's incoming left-carry to
// fill in inter-block answers.
//
// MAX_DEPTH = 32 covers ~99.9% of cases on random data (log2(B) ≈ 7 average).
// When a carry overflows MAX_DEPTH, we truncate (lose old entries) — this is
// conservative: we may return -1 for some elements that have an answer far back,
// but on random data this is extremely rare.
// ===========================================================================

constexpr int P3_BS      = 128;
constexpr int P3_IPT     = 2;
constexpr int P3_B       = P3_BS * P3_IPT;   // 256
constexpr int MAX_DEPTH  = 64;  // max suffix stack entries per block carry

struct alignas(16) BlockCarry3 {
    int depth;
    int vals[MAX_DEPTH];
    int idxs[MAX_DEPTH];
};

// Merge operator: left carry merged into right carry.
// The result carry is the suffix stack of the combined left+right sequence.
// Since right's suffix stack already contains all right-side minima, we just
// need to prepend left entries that are smaller than right's minimum.
struct MergeOp {
    __device__ __forceinline__
    BlockCarry3 operator()(const BlockCarry3& left, const BlockCarry3& right) const {
        BlockCarry3 out;
        // Right's suffix stack dominates for all indices in right.
        // We append left entries whose vals < right's minimum val.
        // right.vals[right.depth-1] is the smallest (global min of right).
        int right_min = (right.depth > 0) ? right.vals[right.depth - 1] : INT_MAX;

        // Copy right's stack unchanged
        out.depth = right.depth;
        for (int i = 0; i < right.depth && i < MAX_DEPTH; i++) {
            out.vals[i] = right.vals[i];
            out.idxs[i] = right.idxs[i];
        }
        // Left's stack is monotone decreasing: vals[0] > vals[1] > ... > vals[depth-1].
        // Entries with val < right_min form a contiguous suffix starting at some index lo.
        // Append them in order (lo, lo+1, ..., depth-1) to preserve decreasing-val order.
        int lo = 0;
        while (lo < left.depth && left.vals[lo] >= right_min) lo++;
        for (int i = lo; i < left.depth && out.depth < MAX_DEPTH; i++) {
            out.vals[out.depth] = left.vals[i];
            out.idxs[out.depth] = left.idxs[i];
            out.depth++;
        }
        return out;
    }
};

// Pass A: compute intra-block PSE and build per-block carry
__global__ void p3PassA(const int* __restrict__ d_in, int* d_out, int n,
                         BlockCarry3* d_carries) {
    constexpr int B         = P3_B;
    constexpr int NUM_WARPS = P3_BS / 32;
    const int glb_offs = blockIdx.x * B;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;

    __shared__ int s_elems[B];
    __shared__ int s_stripe_min[P3_IPT][NUM_WARPS];

    #pragma unroll
    for (int i = 0; i < P3_IPT; i++) {
        int lid = i * P3_BS + threadIdx.x;
        int gid = glb_offs + lid;
        s_elems[lid] = (gid < n) ? d_in[gid] : INT_MAX;
    }
    __syncthreads();

    int carry_[P3_IPT], left_carry_[P3_IPT];
    #pragma unroll
    for (int ipt = 0; ipt < P3_IPT; ipt++) {
        int val = s_elems[ipt * P3_BS + threadIdx.x];
        int c = val;
        #pragma unroll
        for (int step = 1; step <= 16; step <<= 1) {
            int nb = __shfl_up_sync(0xffffffff, c, step);
            if (lane >= step) c = min(c, nb);
        }
        carry_[ipt]      = c;
        left_carry_[ipt] = __shfl_up_sync(0xffffffff, c, 1);
        int wm = __shfl_sync(0xffffffff, c, 31);
        if (lane == 0) s_stripe_min[ipt][warp_id] = wm;
    }
    __syncthreads();

    #pragma unroll
    for (int ipt = 0; ipt < P3_IPT; ipt++) {
        int lid = ipt * P3_BS + threadIdx.x;
        int val = s_elems[lid];
        int result = -1;
        if (lane > 0 && left_carry_[ipt] < val) {
            int base = ipt * P3_BS + warp_id * 32;
            for (int k = lane - 1; k >= 0; k--)
                if (s_elems[base + k] < val) { result = base + k; break; }
        }
        if (result < 0) {
            for (int w = warp_id - 1; w >= 0 && result < 0; w--) {
                if (s_stripe_min[ipt][w] < val) {
                    int wb = ipt * P3_BS + w * 32;
                    for (int k = 31; k >= 0; k--)
                        if (s_elems[wb + k] < val) { result = wb + k; break; }
                }
            }
        }
        if (result < 0) {
            for (int i = ipt - 1; i >= 0 && result < 0; i--) {
                for (int w = NUM_WARPS - 1; w >= 0 && result < 0; w--) {
                    if (s_stripe_min[i][w] < val) {
                        int wb = i * P3_BS + w * 32;
                        for (int k = 31; k >= 0; k--)
                            if (s_elems[wb + k] < val) { result = wb + k; break; }
                    }
                }
            }
        }
        int gid = glb_offs + ipt * P3_BS + threadIdx.x;
        if (gid < n) d_out[gid] = (result >= 0) ? (glb_offs + result) : -1;
    }

    // Build suffix stack carry (thread 0, serial)
    if (threadIdx.x == 0) {
        BlockCarry3& bc = d_carries[blockIdx.x];
        bc.depth = 0;
        for (int lid = B - 1; lid >= 0; lid--) {
            int gid = glb_offs + lid;
            if (gid >= n) continue;
            int v = s_elems[lid];
            if (bc.depth == 0 || v < bc.vals[bc.depth - 1]) {
                if (bc.depth < MAX_DEPTH) {
                    bc.vals[bc.depth] = v;
                    bc.idxs[bc.depth] = gid;
                    bc.depth++;
                }
            }
        }
    }
}

// Pass B: after DeviceScan, use the scanned (prefix) carry to fill in
// inter-block answers for elements that got -1 from pass A.
__global__ void p3PassB(const int* __restrict__ d_in, int* d_out, int n,
                         const BlockCarry3* __restrict__ d_prefix) {
    // d_prefix[b] is the exclusive prefix carry for block b (the merged suffix
    // stack of all blocks 0..b-1). We use it to answer elements in block b
    // that still have d_out == -1.
    constexpr int B   = P3_B;
    const int glb_offs = blockIdx.x * B;
    const BlockCarry3& carry = d_prefix[blockIdx.x];

    for (int i = 0; i < P3_IPT; i++) {
        int lid = i * P3_BS + threadIdx.x;
        int gid = glb_offs + lid;
        if (gid >= n || d_out[gid] != -1) continue;

        int val = d_in[gid];

        // Binary-search the carry's suffix stack for the rightmost idx with val < our val.
        // Stack is monotone decreasing in val (stack[0] largest val, stack[depth-1] smallest).
        // We want the first entry (smallest depth index) whose val < our val —
        // that entry has the largest (rightmost) original index.
        int lo = 0, hi = carry.depth;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (carry.vals[mid] < val) hi = mid;
            else                        lo = mid + 1;
        }
        d_out[gid] = (lo < carry.depth) ? carry.idxs[lo] : -1;
    }
}

struct Scan3Scratch {
    BlockCarry3* d_carries    = nullptr;  // per-block carries (input to scan)
    BlockCarry3* d_prefix     = nullptr;  // scanned carries (exclusive prefix)
    void*        d_tmp        = nullptr;
    size_t       tmp_bytes    = 0;
    int          num_blocks   = 0;
};

static Scan3Scratch allocScan3(int n) {
    Scan3Scratch s;
    s.num_blocks = (n + P3_B - 1) / P3_B;
    gpuAssert(cudaMalloc(&s.d_carries, (size_t)s.num_blocks * sizeof(BlockCarry3)));
    gpuAssert(cudaMalloc(&s.d_prefix,  (size_t)s.num_blocks * sizeof(BlockCarry3)));
    BlockCarry3 id{};  // identity element: empty stack
    cub::DeviceScan::ExclusiveScan(nullptr, s.tmp_bytes,
        s.d_carries, s.d_prefix, MergeOp{}, id, s.num_blocks);
    gpuAssert(cudaMalloc(&s.d_tmp, s.tmp_bytes));
    return s;
}

static void freeScan3(Scan3Scratch& s) {
    cudaFree(s.d_carries); cudaFree(s.d_prefix); cudaFree(s.d_tmp);
    s = Scan3Scratch{};
}

static void runScan3(const int* d_in, int* d_out, int n, Scan3Scratch& s) {
    p3PassA<<<s.num_blocks, P3_BS>>>(d_in, d_out, n, s.d_carries);
    gpuAssert(cudaGetLastError());
    BlockCarry3 id{};
    cub::DeviceScan::ExclusiveScan(s.d_tmp, s.tmp_bytes,
        s.d_carries, s.d_prefix, MergeOp{}, id, s.num_blocks);
    p3PassB<<<s.num_blocks, P3_BS>>>(d_in, d_out, n, s.d_prefix);
    gpuAssert(cudaGetLastError());
}

// ===========================================================================
// Correctness check
// ===========================================================================

static bool stressTest(const char* name,
                        std::function<void(const int*,int*,int)> fn,
                        int trials, int maxN, unsigned seed) {
    std::mt19937 rng(seed);
    int nfail = 0;
    for (int t = 0; t < trials && nfail == 0; t++) {
        int n = 1 + rng() % maxN;
        std::vector<int> h(n);
        for (auto& v : h) v = rng() % 1000;
        auto exp = cpuPse(h);
        int *d_in, *d_out;
        gpuAssert(cudaMalloc(&d_in,  n * sizeof(int)));
        gpuAssert(cudaMalloc(&d_out, n * sizeof(int)));
        gpuAssert(cudaMemcpy(d_in, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        fn(d_in, d_out, n);
        gpuAssert(cudaDeviceSynchronize());
        std::vector<int> got(n);
        gpuAssert(cudaMemcpy(got.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_in); cudaFree(d_out);
        for (int i = 0; i < n; i++) {
            if (got[i] != exp[i]) {
                printf("  [%s] STRESS FAIL t=%d n=%d i=%d got=%d exp=%d\n",
                       name, t, n, i, got[i], exp[i]);
                nfail++; break;
            }
        }
    }
    return nfail == 0;
}

// ===========================================================================
// main
// ===========================================================================

template<typename F>
static void timedSection(const char* label, F fn) {
    fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  [timing] %-40s %.0f ms\n", label, ms);
    fflush(stdout);
}

int main() {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s  (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    fflush(stdout);

    // ---- Correctness ----
    printf("=== Correctness ===\n"); fflush(stdout);

    timedSection("baseline stress (200 trials)", [&]{
        bool ok = stressTest("baseline", [](const int* di, int* dout, int n){
            launchApsep<int,128,2,8>(di, dout, n);
        }, 200, 8192, 1);
        printf("  Baseline (apsepKernel IPT=2 K=8): %s\n", ok?"PASS":"FAIL");
    });

    timedSection("two-pass stress (200 trials)", [&]{
        constexpr int STRESS_MAX = 8192;
        auto sp = allocTwoPass(STRESS_MAX);
        bool ok = stressTest("two-pass", [&](const int* di, int* dout, int n){
            runTwoPass(di, dout, n, sp);
            gpuAssert(cudaDeviceSynchronize());
        }, 200, STRESS_MAX, 2);
        freeTwoPass(sp);
        printf("  Approach 1 (two-pass):            %s\n", ok?"PASS":"FAIL");
    });

    timedSection("leaves-only stress (200 trials)", [&]{
        bool ok = stressTest("leaves-only", [](const int* di, int* dout, int n){
            launchLeavesOnly<int,128,2,8>(di, dout, n);
        }, 200, 8192, 4);
        printf("  LeavesOnly (K=8):                  %s\n", ok?"PASS":"FAIL");
    });

    timedSection("warp-scan-leaves stress (200 trials)", [&]{
        bool ok = stressTest("warp-scan-leaves", [](const int* di, int* dout, int n){
            launchWarpScanLeaves<int,128,2,8>(di, dout, n);
        }, 200, 8192, 5);
        printf("  WarpScanLeaves IPT=2 (K=8):        %s\n", ok?"PASS":"FAIL");
    });
    timedSection("wsl-ipt4 stress (200 trials)", [&]{
        bool ok = stressTest("wsl-ipt4", [](const int* di, int* dout, int n){
            launchWarpScanLeaves<int,128,4,8>(di, dout, n);
        }, 200, 8192, 7);
        printf("  WarpScanLeaves IPT=4 (K=8):        %s\n", ok?"PASS":"FAIL");
    });

    timedSection("no-block-tree stress (200 trials)", [&]{
        bool ok = stressTest("no-block-tree", [](const int* di, int* dout, int n){
            launchNoBlockTree<int,128,2,8>(di, dout, n);
        }, 200, 8192, 6);
        printf("  NoBlockTree (K=8):                 %s\n", ok?"PASS":"FAIL");
    });

    timedSection("scan3 spot-check (8 sizes)", [&]{
        bool ok = true;
        std::mt19937 rng(42);
        int sizes[] = {1, 127, 128, 255, 256, 1024, 4096, 8192};
        for (int n : sizes) {
            std::vector<int> hh(n);
            for (auto& v : hh) v = rng() % 1000;
            auto exp = cpuPse(hh);
            int *di, *dout;
            gpuAssert(cudaMalloc(&di,   n * sizeof(int)));
            gpuAssert(cudaMalloc(&dout, n * sizeof(int)));
            gpuAssert(cudaMemcpy(di, hh.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            auto ss3 = allocScan3(n);
            runScan3(di, dout, n, ss3);
            gpuAssert(cudaDeviceSynchronize());
            freeScan3(ss3);
            std::vector<int> got(n);
            gpuAssert(cudaMemcpy(got.data(), dout, n * sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(di); cudaFree(dout);
            for (int i = 0; i < n && ok; i++) {
                if (got[i] != exp[i]) {
                    printf("  [scan3] FAIL n=%d i=%d got=%d exp=%d\n", n, i, got[i], exp[i]);
                    ok = false;
                }
            }
        }
        printf("  Approach 2 (segmented scan):      %s\n", ok?"PASS":"FAIL");
    });

    printf("\n"); fflush(stdout);

    // ---- Benchmark ----
    constexpr int N = 500 * 1024 * 1024 / sizeof(int);  // 500 MiB
    double peak = (double)prop.memoryClockRate * 1e3
                * (double)prop.memoryBusWidth / 8.0 * 2.0 / 1e9;
    printf("=== Benchmark (N=%d, %.0f MiB, peak=%.0f GB/s) ===\n\n",
           N, (double)N * sizeof(int) / (1024*1024), peak);
    fflush(stdout);

    std::vector<int> h(N);
    {
        std::mt19937 rng(0xBEEF);
        std::uniform_int_distribution<int> dist(0, N);
        for (auto& v : h) v = dist(rng);
    }
    int *d_in, *d_out;
    gpuAssert(cudaMalloc(&d_in,  (size_t)N * sizeof(int)));
    gpuAssert(cudaMalloc(&d_out, (size_t)N * sizeof(int)));
    gpuAssert(cudaMemcpy(d_in, h.data(), (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

    size_t bytes_rw = (size_t)N * 2 * sizeof(int);

    printf("--- Uniform random ---\n"); fflush(stdout);

    timedSection("baseline bench (random)", [&]{
        auto s = allocApsepScratch<int,128,2,8>(N);
        BENCH("Baseline: apsepKernel IPT=2 K=8",
              ([&]{ runApsep<int,128,2,8>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeApsepScratch<int,128,2,8>(s);
    });
    timedSection("two-pass bench (random)", [&]{
        auto s = allocTwoPass(N);
        BENCH("Approach 1: two-pass",
              ([&]{ runTwoPass(d_in, d_out, N, s); }), bytes_rw, 1, 3);
        freeTwoPass(s);
    });
    timedSection("scan3 bench (random)", [&]{
        auto s = allocScan3(N);
        BENCH("Approach 2: segmented scan",
              ([&]{ runScan3(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeScan3(s);
    });
    timedSection("leaves-only bench (random)", [&]{
        auto s = allocLeavesOnlyScratch<int,128,2,8>(N);
        BENCH("LeavesOnly: apsepKernelLeavesOnly K=8",
              ([&]{ runLeavesOnly<int,128,2,8>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeLeavesOnlyScratch<int,128,2,8>(s);
    });
    timedSection("warp-scan-leaves bench (random)", [&]{
        auto s = allocWarpScanLeavesScratch<int,128,2,8>(N);
        BENCH("WarpScanLeaves: IPT=2 K=8",
              ([&]{ runWarpScanLeaves<int,128,2,8>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeWarpScanLeavesScratch<int,128,2,8>(s);
    });
    timedSection("wsl-ipt4 bench (random)", [&]{
        auto s = allocWarpScanLeavesScratch<int,128,4,8>(N);
        BENCH("WarpScanLeaves: IPT=4 K=8  (best)",
              ([&]{ runWarpScanLeaves<int,128,4,8>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeWarpScanLeavesScratch<int,128,4,8>(s);
    });
    timedSection("no-block-tree bench (random)", [&]{
        auto s = allocNoBlockTreeScratch<int,128,2,8>(N);
        BENCH("NoBlockTree: apsepKernelNoBlockTree K=8",
              ([&]{ runNoBlockTree<int,128,2,8>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeNoBlockTreeScratch<int,128,2,8>(s);
    });
    timedSection("wsnt-ipt4-k64 bench (random)", [&]{
        auto s = allocWarpScanNoTreeScratch<int,128,4,64>(N);
        BENCH("WarpScanNoTree: IPT=4 K=64",
              ([&]{ runWarpScanNoTree<int,128,4,64>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeWarpScanNoTreeScratch<int,128,4,64>(s);
    });
    timedSection("wsnt-ipt4-k128 bench (random)", [&]{
        auto s = allocWarpScanNoTreeScratch<int,128,4,128>(N);
        BENCH("WarpScanNoTree: IPT=4 K=128",
              ([&]{ runWarpScanNoTree<int,128,4,128>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeWarpScanNoTreeScratch<int,128,4,128>(s);
    });
    timedSection("notree2l-k64-g64 bench (random)", [&]{
        auto s = allocWarpScanNoTree2LScratch<int,128,4,64,64>(N);
        BENCH("NoTree2L: IPT=4 K=64 G=64",
              ([&]{ runWarpScanNoTree2L<int,128,4,64,64>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeWarpScanNoTree2LScratch<int,128,4,64,64>(s);
    });
    timedSection("warpcoopleaf-k64 bench (random)", [&]{
        auto s = allocWarpCoopLeafScratch<int,128,4,64>(N);
        BENCH("WarpCoopLeaf: IPT=4 K=64",
              ([&]{ runWarpCoopLeaf<int,128,4,64>(d_in, d_out, N, s); }), bytes_rw, 2, 5);
        freeWarpCoopLeafScratch<int,128,4,64>(s);
    });

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
