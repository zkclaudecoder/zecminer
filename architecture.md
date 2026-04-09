# nheqminer Architecture & Bottleneck Diagram

## High-Level Data Flow

```
  +------------------+     Stratum TCP      +------------------+
  |   Mining Pool    | <------------------> |  StratumClient   |
  | pool.tazminer.com|   jobs/shares/auth   | (boost::asio)    |
  +------------------+                      +--------+---------+
                                                     |
                                            new job  | submit solution
                                            (header, | (nonce, indices)
                                             target) |
                                                     v
                                            +--------+---------+
                                            |   ZcashMiner     |
                                            | (job dispatcher) |
                                            +--------+---------+
                                                     |
                                         distribute  | nonce ranges
                                         per solver  |
                                                     v
                                            +--------+---------+
                                            |  MinerFactory    |
                                            | (solver manager) |
                                            +--------+---------+
                                                     |
                                    +----------------+----------------+
                                    |                                 |
                              [cuda_tromp]                     [cuda_djezo]
                              WORKING on SM12.0              BROKEN on SM12.0
                              ~179 Sols/s                    (invalid collisions)
                                    |
                                    v
                    +-------------------------------+
                    |     eq_cuda_context            |
                    | cudaMalloc: heap0 (235 MB)     |
                    |             heap1 (184 MB)     |
                    |             nslots (512 KB)    |
                    |             sols (800 B)       |
                    |        Total: ~420 MB VRAM     |
                    +---------------+---------------+
                                    |
                                    v
```

## GPU Kernel Pipeline (per solve iteration)

```
 CPU: prepare BLAKE2b state from (header + nonce)
  |
  | cudaMemcpy (64 bytes blake state -> GPU)
  v
 ============== GPU KERNEL PIPELINE ==============

 +--------------------------------------------------+
 |  digitH (BLAKE2b initial hashing)                 |
 |  574 blocks x 64 threads = 36,736 threads         |
 |                                                    |
 |  Each thread:                                      |
 |    - Compute BLAKE2b(block_index)                  |
 |    - Extract 2 hashes per blake output             |
 |    - bucketid = hash[0:12] (12-bit bucket)         |
 |    - atomicAdd(&nslots[bucketid], 1)  <--- ATOMIC  |
 |    - Write slot to heap0[bucketid][slot]            |
 |                                                    |
 |  2,097,152 hashes -> 65,536 buckets x ~32 slots   |
 |  Memory: WRITE 235 MB to heap0                     |
 +-------------------------+------------------------+
                           |
                           v
 +--------------------------------------------------+
 |  digit_1 (Round 1: find colliding pairs)          |
 |  Reads: heap0 (slot0, 28 bytes/slot)              |
 |  Writes: heap1 (slot1, 28 bytes/slot)             |
 |                                                    |
 |  Each thread processes ~2 buckets:                 |
 |    for each bucket:                                |
 |      for s1 = 0..bsize:           <--- SERIAL     |
 |        xhash = slot.hash bits                      |
 |        cd.addslot(s1, xhash)      collision chain  |
 |        for each collision s0:     <--- SERIAL      |
 |          if equal(s0, s1): skip                    |
 |          XOR hashes                                |
 |          xorbucketid = XOR[0:16]                   |
 |          atomicAdd(&nslots[xorbid]) <--- ATOMIC    |
 |          write XOR result to heap1                 |
 |                                                    |
 |  BOTTLENECK: scattered reads from heap0            |
 |  BOTTLENECK: serial collision chain traversal      |
 |  BOTTLENECK: atomicAdd contention on nslots        |
 +-------------------------+------------------------+
                           |
                           v
 +--------------------------------------------------+
 |  digit2 through digit8 (Rounds 2-8)              |
 |  Same pattern as digit_1, alternating:            |
 |    odd rounds:  read heap0 -> write heap1         |
 |    even rounds: read heap1 -> write heap0         |
 |                                                    |
 |  Each round:                                       |
 |    - Fewer slots per bucket (collisions reduce)    |
 |    - Fewer hash words to XOR (20 bits consumed)    |
 |    - Same serial collision detection               |
 |    - Same atomic contention                        |
 |    - Same scattered memory access                  |
 |                                                    |
 |  Round progression (avg slots/bucket):             |
 |    R1: ~32 -> R2: ~16 -> R3: ~8 -> ... -> R8: ~2  |
 +-------------------------+------------------------+
                           |
                           v
 +--------------------------------------------------+
 |  digitK (Final round: extract solutions)          |
 |  Walk the tree of indices back through all rounds  |
 |  Collect 512 indices per valid solution             |
 |  Write to eq->sols[]                               |
 |  Avg: ~1.9 solutions per iteration                 |
 +-------------------------+------------------------+
                           |
                           | cudaMemcpy (solutions -> CPU)
                           v
 CPU: verify solutions, submit to pool via stratum

 ============== END GPU PIPELINE ==================
```

## Memory Access Pattern (THE CORE BOTTLENECK)

```
                     65,536 buckets
          +----+----+----+----+- ... -+----+
 heap0:   | B0 | B1 | B2 | B3 |      |B65k|   235 MB
          +----+----+----+----+- ... -+----+
            |              |
            |  Thread A    |  Thread B reads
            |  reads B0    |  random bucket B3
            v              v
     +----------+    +----------+
     |slot0     |    |slot0     |    Each bucket: 64 slots x 28 bytes
     |slot1     |    |slot1     |                = 1,792 bytes
     |slot2     |    |slot2     |
     |...       |    |...       |    Slots accessed in ORDER within bucket
     |slot63    |    |slot63    |    but RANDOM across buckets
     +----------+    +----------+
            |              |
            | XOR pairs    | XOR pairs
            v              v
     +-----------+   +-----------+
     | Write to  |   | Write to  |   Output bucket = RANDOM location
     | heap1[X]  |   | heap1[Y]  |   in the other heap
     +-----------+   +-----------+

  L2 Cache (80 MB) can hold ~68% of one heap (235 MB)
  -> ~40-60% cache miss rate on bucket reads
  -> Each miss = ~300ns DRAM roundtrip
  -> THIS is the bottleneck: memory LATENCY, not bandwidth
```

## Bottleneck Summary

```
 PERFORMANCE: 179 Sols/s | 90 I/s | 96% GPU util | 149W

 +---------------------------------------------------------------+
 |                    BOTTLENECK RANKING                          |
 +---------------------------------------------------------------+
 |                                                               |
 |  #1  MEMORY LATENCY (cache misses)           [FUNDAMENTAL]   |
 |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                  |
 |  Working set: 235 MB (heap0) + 184 MB (heap1) = 419 MB       |
 |  L2 cache: 80 MB (holds ~19% of working set)                 |
 |  Miss rate: ~40-60% -> 300ns DRAM penalty per miss            |
 |  Actual BW used: ~35 GB/s of 1,800 GB/s available (1.9%)     |
 |  WHY: bucket access is pseudo-random (hash-determined)        |
 |  FIX: would require algorithmic redesign (different solver)   |
 |                                                               |
 |  #2  SERIAL COLLISION DETECTION              [ALGORITHMIC]    |
 |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                  |
 |  Each bucket's collision chain is traversed by ONE thread     |
 |  Cannot parallelize: chain builds incrementally per s1        |
 |  IPC efficiency: ~37% (pointer chasing limits ILP)            |
 |  FIX: would require different collision algorithm             |
 |                                                               |
 |  #3  ATOMIC CONTENTION on nslots[]           [MODERATE]       |
 |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                  |
 |  2.1M atomicAdd per iteration on 65K counters                 |
 |  32x collision factor per counter                             |
 |  Uses ~73% of GPU atomic throughput                           |
 |  FIX: local histograms, but adds register pressure            |
 |                                                               |
 |  NOT a bottleneck:                                            |
 |  - Compute (only 37% IPC, plenty of ALU headroom)             |
 |  - VRAM capacity (420 MB of 24 GB = 1.7%)                    |
 |  - Memory bandwidth (35 GB/s of 1,800 GB/s = 1.9%)           |
 |  - Power (149W of 150W, but not compute-limited)              |
 |  - BLAKE2b hashing (digitH is fast, not the bottleneck)       |
 +---------------------------------------------------------------+
```

## What would it take to go faster?

```
 Current: ~179 Sols/s (cuda_tromp, 2016-era algorithm)

 To reach ~300+ Sols/s on RTX 5090 would require:
   1. Different memory layout: Structure-of-Arrays instead of
      Array-of-Structures for coalesced access
   2. Different collision algorithm: warp-cooperative collision
      detection instead of per-thread serial chains
   3. Tiled processing: process buckets in tiles that fit in
      L2 cache, reducing miss rate from ~50% to ~10%
   4. This is essentially writing a new solver from scratch

 The djezo solver ATTEMPTED some of these optimizations
 (shared memory atomics, different bucket layout) but its
 approach has Blackwell-specific bugs we couldn't fully fix.
```
