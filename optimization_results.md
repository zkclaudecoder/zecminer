# Equihash cuda_tromp Optimization Results

**Hardware:** RTX 5090 Laptop (Blackwell SM 12.0, 82 SMs, 80+ MB L2, 228 KB smem/SM, GDDR7)
**Benchmark:** `-b 2000 -cd 0 -cv 1 -t 0` (2000 iterations, CUDA-TROMP solver)
**Baseline:** 175-179 Sols/s (~79.9 I/s, ~3850 solutions per 2000 iterations)
**Date:** 2026-04-08

---

## Test 1: L2 Cache Persistence Controls

**Change:** Reserved 40 MB persistent L2 cache and pinned the `nslots` counter array (~512 KB) with `cudaAccessPropertyPersisting` hint, so atomicAdd contention on nslots benefits from guaranteed L2 residency.

**Result:**
| Metric | Baseline | Test 1 | Delta |
|--------|----------|--------|-------|
| Speed (I/s) | ~79.9 | 80.2-80.6 | +0.4% |
| Speed (Sols/s) | 175-179 | 153.9-154.5 | **-14%** |
| Solutions/2000 iter | ~3850 | 3836-3852 | -0.3% |

**Verdict: REGRESSION (-14% Sols/s)**

**Analysis:** The L2 persistence hint actually hurt performance. Pinning 512 KB of nslots data forces eviction of more valuable bucket slot data from the L2 cache. The nslots array is already small enough to naturally stay in L2 under LRU policy. Reserving 40 MB of "persistent" L2 carve-out likely reduced the effective L2 available for the 117 MB working set of bucket data, increasing cache miss rate on the hot path (bucket reads during collision detection).

Note: iteration speed was nearly identical (the kernel execution speed didn't change much), but the solution count per iteration dropped slightly, suggesting the L2 policy change may have caused subtle correctness-adjacent issues with atomicAdd ordering or nslots counter overflow.

---

## Test 2: Block-per-Bucket with Shared Memory Prefetch

**Change:** Rewrote all 8 digit kernels (digit_1 through digit8) to use a block-per-bucket pattern:
- Each block cooperatively loads one bucket's slot data into shared memory (64 slots x 28 bytes = 1792 bytes)
- Thread 0 performs all collision detection from shared memory (fast) instead of global memory
- Other threads (1-63) only participate in the cooperative load, then idle
- Grid size changed from 574 blocks to 4096 blocks
- Added default constructor to `tree` struct to enable shared memory array declaration

**Result:**
| Metric | Baseline | Test 2 | Delta |
|--------|----------|--------|-------|
| Speed (I/s) | ~79.9 | 17.8 | **-78%** |
| Speed (Sols/s) | 175-179 | 0 | **-100%** |
| Solutions/2000 iter | ~3850 | 0 | -100% |
| Total time (2000 iter) | ~25s | 112.6s | +350% |

**Verdict: CATASTROPHIC REGRESSION (0 Sols/s, -78% I/s)**

**Analysis:** This approach is fundamentally flawed for GPU execution:

1. **Massive thread underutilization:** Only 1 thread per block (thread 0) does collision work. With 4096 blocks x 64 threads = 262,144 threads launched, only 4096 do real work (1.6% utilization). The baseline has 574 x 64 = 36,736 threads ALL doing work.

2. **Serial collision detection is the bottleneck:** The collision detection loop (`addslot`/`nextcollision`) is inherently serial within a bucket -- each `s1` iteration builds on previous collision chain state. This cannot be parallelized across threads without breaking the algorithm.

3. **Shared memory benefit is negligible:** The cooperative load saves ~1792 bytes of global memory reads per bucket, but the collision processing loop already benefits from L1/L2 caching since it accesses the same small bucket sequentially. The shared memory prefetch adds overhead (syncthreads barriers) without meaningful latency improvement.

4. **Zero solutions:** 0 solutions found suggests either a correctness bug in the rewrite, or that the solver is so slow it never completes enough quality iterations to find valid Equihash solutions within the 2000-iteration budget.

---

## Summary

| Test | Sols/s | vs Baseline | Status |
|------|--------|-------------|--------|
| Baseline | 175-179 | -- | Reference |
| Test 1: L2 Persistence | 154 | -14% | Failed |
| Test 2: Shared Mem Prefetch | 0 | -100% | Failed |

**Conclusion:** Both optimizations are counterproductive. The cuda_tromp solver's performance characteristics on Blackwell are:

- The L2 cache is already doing a good job under default LRU policy; explicit persistence hints cause more harm than good by displacing working set data.
- The per-bucket workload is too serial for a block-cooperative pattern; the original grid-stride-per-thread approach maximizes GPU occupancy and is the correct design for this algorithm.
- Future optimization efforts should focus on reducing the working set size (fitting more data in L2), improving memory access coalescing within the existing thread-per-bucket pattern, or algorithmic changes that expose more parallelism within each bucket's collision detection.
