// digit_kernels.cu — Phase 0 prototype kernels for cuda_blackwell.
//
// Phase 0 deliverables (this file):
//   - digit_first_kernel: identical BLAKE2b initial-hash + bucket placement
//     to djezo's `digit_first`, so the round-0 hash table matches byte-for-byte.
//   - digit_1_kernel: producer/consumer warp-specialized first collision round
//     using cp.async + mbarriers. Output round-1 hash table must be
//     bit-identical to djezo's `digit_1`.
//
// Status:
//   [x] digit_first_kernel — implemented, mirrors djezo line-for-line
//   [ ] digit_1_kernel — STUB (TODO: implement producer/consumer pipeline)
//
// PHASE 1+ scope (NOT in this file yet):
//   - digit_2..digit_8 kernels
//   - digit_last_wdc (solution reconstruction)
//   - persistent kernel that fuses all rounds (Phase 2)

#include <cuda_runtime.h>
#include "bucket_layout.hpp"
#include "eq_context.hpp"
#include "blake2b.cuh"

namespace bw {

// Note: cooperative_groups + libcudacxx (cuda::pipeline / cuda::barrier) will be
// pulled in via pipeline.hpp when the digit_1 producer/consumer code lands.
// Avoiding them now keeps Phase 0 free of the CCCL/MSVC preprocessor issue.

// ============================================================================
// digit_first_kernel — mirrors djezo's digit_first in equi_miner.cu
// ============================================================================
//
// Each thread handles one "block" of BLAKE2b output (which produces 2 hashes
// of 200 bits each per BLAKE2b call). For each hash, we extract the leading
// bucket id and append the slot to round0trees[bucketid].
//
// Launch config: <<<NBLOCKS / FD_THREADS, FD_THREADS>>>
//   = <<<1048576 / 128, 128>>>
//   = <<<8192, 128>>>  (matches djezo exactly)
//
// Bit-identical-with-djezo requirement: yes. The bucket placement, slot layout,
// and atomicAdd ordering all match djezo's per-thread approach.
__global__ __launch_bounds__(FD_THREADS, 0)
void digit_first_kernel(equi_state* eq, u32 nonce) {
    const u32 block = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ u64 hash_h[8];
    u32* hash_h32 = (u32*)hash_h;

    // Load BLAKE2b mid-state into shared (one-time per block)
    if (threadIdx.x < 16)
        hash_h32[threadIdx.x] = eq->blake_h32[threadIdx.x];
    __syncthreads();

    // Compute BLAKE2b for this block index — produces 64 bytes of output
    u64 my_h[8] = { hash_h[0], hash_h[1], hash_h[2], hash_h[3],
                    hash_h[4], hash_h[5], hash_h[6], hash_h[7] };
    blake2b_gpu_hash3_bw(my_h, block, nonce);

    // Output is 2 hashes of 200 bits = 50 bytes total in my_h[0..6.something]
    u32* v32 = (u32*)my_h;

    // ---- First hash: bytes 0..24 (200 bits = 25 bytes, but stored aligned) ----
    // round0trees always has 4096 buckets regardless of RB — use bits [12..24]
    // of bexor (12-bit bucket id). Matches djezo's `bfe.u32 %0, bexor, 12, 12`.
    {
        u32 bexor = __byte_perm(v32[0], 0, 0x4012); // first 20 bits
        u32 bucketid = (bexor >> 12) & 0xFFFu;      // 12-bit bucket id
        u32 slotp = atomicAdd(&eq->edata.nslots0[bucketid], 1);
        if (slotp < RB8_NSLOTS) {
            slot* s = &eq->round0trees[bucketid][slotp];
            uint4 tt;
            tt.x = __byte_perm(v32[0], v32[1], 0x1234);
            tt.y = __byte_perm(v32[1], v32[2], 0x1234);
            tt.z = __byte_perm(v32[2], v32[3], 0x1234);
            tt.w = __byte_perm(v32[3], v32[4], 0x1234);
            *(uint4*)(&s->hash[0]) = tt;
            tt.x = __byte_perm(v32[4], v32[5], 0x1234);
            tt.y = __byte_perm(v32[5], v32[6], 0x1234);
            tt.z = 0;
            tt.w = block << 1;
            *(uint4*)(&s->hash[4]) = tt;
        }
    }

    // ---- Second hash: bytes 25..49 ----
    {
        u32 bexor = __byte_perm(v32[6], 0, 0x0123);
        u32 bucketid = (bexor >> 12) & 0xFFFu;      // same bfe 12, 12 as above
        u32 slotp = atomicAdd(&eq->edata.nslots0[bucketid], 1);
        if (slotp < RB8_NSLOTS) {
            slot* s = &eq->round0trees[bucketid][slotp];
            uint4 tt;
            tt.x = __byte_perm(v32[6],  v32[7],  0x2345);
            tt.y = __byte_perm(v32[7],  v32[8],  0x2345);
            tt.z = __byte_perm(v32[8],  v32[9],  0x2345);
            tt.w = __byte_perm(v32[9],  v32[10], 0x2345);
            *(uint4*)(&s->hash[0]) = tt;
            tt.x = __byte_perm(v32[10], v32[11], 0x2345);
            tt.y = __byte_perm(v32[11], v32[12], 0x2345);
            tt.z = 0;
            tt.w = (block << 1) + 1;
            *(uint4*)(&s->hash[4]) = tt;
        }
    }
}

// ============================================================================
// digit_1_kernel — Phase 0 v1: algorithmic port of djezo's digit_1
// ============================================================================
//
// STRATEGY: port djezo's algorithm line-for-line first, validate correctness,
// THEN add producer/consumer warp specialization as a separate optimization.
//
// Launch config: <<<NBUCKETS=4096, THREADS=512>>>
//   - one block per bucket
//   - each thread processes 2 slots (total 1024 slot indices per block, cut
//     off at actual bsize which maxes at RB8_NSLOTS=640)
//
// Shared memory per block (CONFIG_MODE_2 sized):
//   ht[256][SSM-1]        = 256 * 11 * 2 = 5,632 bytes  (xhash-indexed HT)
//   lastword1[640]        = 640 * 8       = 5,120 bytes  (first 64 bits of each slot)
//   lastword2[640]        = 640 * 16      = 10,240 bytes (next 128 bits)
//   ht_len[MAXPAIRS=1024] = 1024 * 4      = 4,096 bytes  (collision counts / pair list)
//   misc scalars                          ~     16 bytes
//   TOTAL: ~25 KB per block → fits ~3 blocks/SM on Blackwell's 100 KB default
//
// Algorithm:
//   1. Load my block's bucket slots from global -> shared lastword1/lastword2
//   2. Compute xhash for each slot, insert into ht[xhash] collision chain
//   3. For each slot, find collisions (same xhash). Process first collision
//      inline; push remaining collision pairs to the pairs list.
//   4. Drain pairs list cooperatively (all 512 threads), producing XOR'd
//      output slots in eq->trees[0][xorbucketid][xorslot].

// packer_default::set_bucketid_and_slots, inlined with RB=8 compile-time constant.
// SLOTBITS for RB=8 is 10, so fields layout: [bucketid:12][s0:10][s1:10]
__device__ __forceinline__
u32 pack_bid_s0_s1(u32 bucketid, u32 s0, u32 s1) {
    // matches packer_default::set_bucketid_and_slots(bid, s0, s1, 8, 640)
    // which is (((bid << SLOTBITS) | s0) << SLOTBITS) | s1  with SLOTBITS=10
    return (((bucketid << 10) | s0) << 10) | s1;
}

// Compile-time MAXPAIRS for digit_1: djezo uses 4*NRESTS = 4*256 = 1024
static constexpr u32 D1_MAXPAIRS = 4u * NRESTS;

__global__ __launch_bounds__(512, 2)
void digit_1_kernel(equi_state* eq) {
    constexpr int  SSM_LOCAL  = 12;
    constexpr u32  THR        = 512;
    constexpr u32  MAXPAIRS   = D1_MAXPAIRS;

    __shared__ u16   ht[256][SSM_LOCAL - 1];
    __shared__ uint2 lastword1[RB8_NSLOTS];
    __shared__ uint4 lastword2[RB8_NSLOTS];
    __shared__ int   ht_len[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    // Reset shared: one thread per ht[] entry (only use first 256), plus scalars.
    if (threadid < 256)
        ht_len[threadid] = 0;
    else if (threadid == (THR - 1))
        pairs_len = 0;
    else if (threadid == (THR - 33))
        next_pair = 0;

    const u32 bsize = min(eq->edata.nslots0[bucketid], (u32)RB8_NSLOTS);

    u32   hr[2];
    int   pos[2];
    pos[0] = pos[1] = SSM_LOCAL;
    uint2 ta[2];
    uint4 tb[2];
    u32   si[2];

    __syncthreads();  // shared-memory init visible

    // --- Phase 1: load my slots, compute xhash, insert into HT ----------------
    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        si[i] = i * THR + threadid;
        if (si[i] >= bsize) break;

        const slot* pslot1 = eq->round0trees[bucketid] + si[i];
        uint4 a1 = *(uint4*)(&pslot1->hash[0]);
        uint2 a2 = *(uint2*)(&pslot1->hash[4]);
        ta[i].x = a1.x; ta[i].y = a1.y;
        lastword1[si[i]] = ta[i];
        tb[i].x = a1.z; tb[i].y = a1.w;
        tb[i].z = a2.x; tb[i].w = a2.y;
        lastword2[si[i]] = tb[i];

        // xhash = bits [20..28) of ta[i].x (BFE width 8, start 20)
        hr[i] = (ta[i].x >> 20) & 0xFFu;
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }

    __syncthreads();
    int* pairs = ht_len;  // reuse ht_len as the pairs list (matches djezo)

    u32 xors[6];
    u32 xorbucketid, xorslot;

    // --- Phase 2: for each of my slots, resolve first collision and push rest --
    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;

        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            *(uint2*)(&xors[0]) = make_uint2(ta[i].x ^ lastword1[p].x,
                                              ta[i].y ^ lastword1[p].y);
            // xorbucketid = bits[RB..RB+BUCKBITS) of xors[0] = bits[8..20)
            xorbucketid = (xors[0] >> 8) & BUCKMASK;
            xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

            if (xorslot < NSLOTS) {
                uint4 l2p = lastword2[p];
                uint4 l2s = tb[i];
                xors[2] = l2s.x ^ l2p.x;
                xors[3] = l2s.y ^ l2p.y;
                xors[4] = l2s.z ^ l2p.z;
                xors[5] = l2s.w ^ l2p.w;

                slot& xs = eq->trees[0][xorbucketid][xorslot];
                *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], xors[3], xors[4]);
                uint4 ttx;
                ttx.x = xors[5];
                ttx.y = xors[0];
                ttx.z = pack_bid_s0_s1(bucketid, si[i], p);
                ttx.w = 0;
                *(uint4*)(&xs.hash[4]) = ttx;
            }

            // Push remaining collisions to the pairs list for cooperative drain
            for (int k = 1; k != pos[i]; ++k) {
                u32 pindex = atomicAdd(&pairs_len, 1);
                if (pindex >= MAXPAIRS) break;
                u16 prev = ht[hr[i]][k];
                pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
            }
        }
    }

    __syncthreads();

    // --- Phase 3: cooperative drain of the pairs list -------------------------
    u32 plen = min(pairs_len, MAXPAIRS);
    u32 ii, kk;
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        int pair = pairs[s];
        ii = __byte_perm(pair, 0, 0x4510);
        kk = __byte_perm(pair, 0, 0x4532);

        uint2 lw1i = lastword1[ii];
        uint2 lw1k = lastword1[kk];
        xors[0] = lw1i.x ^ lw1k.x;
        xors[1] = lw1i.y ^ lw1k.y;

        xorbucketid = (xors[0] >> 8) & BUCKMASK;
        xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

        if (xorslot < NSLOTS) {
            uint4 l2i = lastword2[ii];
            uint4 l2k = lastword2[kk];
            xors[2] = l2i.x ^ l2k.x;
            xors[3] = l2i.y ^ l2k.y;
            xors[4] = l2i.z ^ l2k.z;
            xors[5] = l2i.w ^ l2k.w;

            slot& xs = eq->trees[0][xorbucketid][xorslot];
            *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], xors[3], xors[4]);
            uint4 ttx;
            ttx.x = xors[5];
            ttx.y = xors[0];
            ttx.z = pack_bid_s0_s1(bucketid, ii, kk);
            ttx.w = 0;
            *(uint4*)(&xs.hash[4]) = ttx;
        }
    }
}

// ============================================================================
// Host launchers
// ============================================================================

void launch_digit_first(equi_state* device_eq, uint32_t nonce, cudaStream_t stream) {
    constexpr u32 grid_blocks = NBLOCKS / FD_THREADS;
    digit_first_kernel<<<grid_blocks, FD_THREADS, 0, stream>>>(device_eq, nonce);
}

void launch_digit_1(equi_state* device_eq, cudaStream_t stream) {
    // One block per bucket, 512 threads each. Each thread handles up to 2 slots
    // from round0trees, for up to 1024 slot-indices per block (cut off at bsize
    // which maxes at RB8_NSLOTS=640).
    constexpr u32 grid_blocks = NBUCKETS;   // 4096 for RB=8 / CONFIG_MODE_2
    constexpr u32 threads_per_block = 512;
    digit_1_kernel<<<grid_blocks, threads_per_block, 0, stream>>>(device_eq);
}

} // namespace bw
