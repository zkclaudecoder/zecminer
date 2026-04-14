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

// packer_default accessors for tree walk in digit_last_wdc (RB=8 mode).
// SLOTBITS=10 so layout is [bucketid:12][s0:10][s1:10].
__device__ __forceinline__ u32 pack_get_bucketid(u32 bid) { return (bid >> 20) & 0xFFFu; }
__device__ __forceinline__ u32 pack_get_slot0(u32 bid, u32 /*s1*/) { return (bid >> 10) & 0x3FFu; }
__device__ __forceinline__ u32 pack_get_slot1(u32 bid) { return bid & 0x3FFu; }

// Same helpers but for the RB8_NSLOTS=640-slot arrays indexed by round0 trees;
// for CONFIG_MODE_2 the encoding is identical since djezo calls pack with RB=8
// in both cases (see equi_miner.cu lines 620, 799, etc.).
__device__ __forceinline__ u32 pack_get_bucketid_r0(u32 bid) { return (bid >> 20) & 0xFFFu; }
__device__ __forceinline__ u32 pack_get_slot0_r0(u32 bid, u32 /*s1*/) { return (bid >> 10) & 0x3FFu; }
__device__ __forceinline__ u32 pack_get_slot1_r0(u32 bid) { return bid & 0x3FFu; }

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
// digit_2_kernel — Phase 1: algorithmic port of djezo's digit_2
// ============================================================================
// Reads: eq->trees[0][bucketid]             (slot — 8 u32s)
// Writes: eq->round2trees[xorbid].treessmall[xorslot] (slotsmall — 4 u32s)
//         eq->round2trees[xorbid].treestiny[xorslot]  (slottiny — 2 u32s)
// xhash: bits of the 6th hash word (tty.y & RESTMASK)
// Launch: <<<NBUCKETS, THREADS=512>>>

static constexpr u32 D2_MAXPAIRS = 4u * NRESTS;

__device__ __forceinline__
u32 packer_xor_bucketid_d2(u32 xor0) {
    // xorbucketid = xors[0] >> (12 + RB)   for RB=8 -> >> 20
    return xor0 >> (12 + RB);
}

__global__ __launch_bounds__(512, 2)
void digit_2_kernel(equi_state* eq) {
    constexpr int SSM_LOCAL = 12;
    constexpr u32 THR       = 512;
    constexpr u32 MAXPAIRS  = D2_MAXPAIRS;

    __shared__ u16   ht[NRESTS][SSM_LOCAL - 1];
    __shared__ u32   lastword1[NSLOTS];
    __shared__ uint4 lastword2[NSLOTS];
    __shared__ int   ht_len[NRESTS];
    __shared__ int   pairs[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    if (threadid < NRESTS)
        ht_len[threadid] = 0;
    else if (threadid == (THR - 1))
        pairs_len = 0;
    else if (threadid == (THR - 33))
        next_pair = 0;

    slot* buck = eq->trees[0][bucketid];
    const u32 bsize = min(eq->edata.nslots[1][bucketid], (u32)NSLOTS);

    u32   hr[2];
    int   pos[2]; pos[0] = pos[1] = SSM_LOCAL;
    u32   ta[2];
    uint4 tt[2];
    u32   si[2];

    __syncthreads();

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        si[i] = i * THR + threadid;
        if (si[i] >= bsize) break;
        const slot* pslot1 = buck + si[i];
        uint4 ttx = *(uint4*)(&pslot1->hash[0]);
        lastword1[si[i]] = ta[i] = ttx.x;
        uint2 tty = *(uint2*)(&pslot1->hash[4]);
        tt[i].x = ttx.y; tt[i].y = ttx.z; tt[i].z = ttx.w; tt[i].w = tty.x;
        lastword2[si[i]] = tt[i];
        hr[i] = tty.y & RESTMASK;
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }
    __syncthreads();

    u32 xors[5];
    u32 xorbucketid, xorslot;

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;
        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            xors[0] = ta[i] ^ lastword1[p];
            xorbucketid = packer_xor_bucketid_d2(xors[0]);
            xorslot = atomicAdd(&eq->edata.nslots[2][xorbucketid], 1);
            if (xorslot < NSLOTS) {
                uint4 l2p = lastword2[p];
                xors[1] = tt[i].x ^ l2p.x;
                xors[2] = tt[i].y ^ l2p.y;
                xors[3] = tt[i].z ^ l2p.z;
                xors[4] = tt[i].w ^ l2p.w;
                slotsmall& xs = eq->round2trees[xorbucketid].treessmall[xorslot];
                *(uint4*)(&xs.hash[0]) = make_uint4(xors[0], xors[1], xors[2], xors[3]);
                slottiny& xst = eq->round2trees[xorbucketid].treestiny[xorslot];
                *(uint2*)(&xst.hash[0]) = make_uint2(xors[4], pack_bid_s0_s1(bucketid, si[i], p));
            }
            for (int k = 1; k != pos[i]; ++k) {
                u32 pindex = atomicAdd(&pairs_len, 1);
                if (pindex >= MAXPAIRS) break;
                u16 prev = ht[hr[i]][k];
                pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
            }
        }
    }
    __syncthreads();

    u32 plen = min(pairs_len, MAXPAIRS);
    u32 ii, kk;
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        int pair = pairs[s];
        ii = __byte_perm(pair, 0, 0x4510);
        kk = __byte_perm(pair, 0, 0x4532);
        xors[0] = lastword1[ii] ^ lastword1[kk];
        xorbucketid = packer_xor_bucketid_d2(xors[0]);
        xorslot = atomicAdd(&eq->edata.nslots[2][xorbucketid], 1);
        if (xorslot < NSLOTS) {
            uint4 l2i = lastword2[ii];
            uint4 l2k = lastword2[kk];
            xors[1] = l2i.x ^ l2k.x;
            xors[2] = l2i.y ^ l2k.y;
            xors[3] = l2i.z ^ l2k.z;
            xors[4] = l2i.w ^ l2k.w;
            slotsmall& xs = eq->round2trees[xorbucketid].treessmall[xorslot];
            *(uint4*)(&xs.hash[0]) = make_uint4(xors[0], xors[1], xors[2], xors[3]);
            slottiny& xst = eq->round2trees[xorbucketid].treestiny[xorslot];
            *(uint2*)(&xst.hash[0]) = make_uint2(xors[4], pack_bid_s0_s1(bucketid, ii, kk));
        }
    }
}

// ============================================================================
// digit_3_kernel — Phase 1: port of djezo's digit_3
// ============================================================================
// Reads: eq->round2trees[bucketid].treessmall (uint4) + .treestiny (u32)
// Writes: eq->round3trees[xorbid].treessmall + .treestiny
// xhash: SAFE_BFE(hr, tt.x, 12, RB) = bits[12..12+RB] of the first hash word

static constexpr u32 D3_MAXPAIRS = 4u * NRESTS;

__global__ __launch_bounds__(512, 2)
void digit_3_kernel(equi_state* eq) {
    constexpr int SSM_LOCAL = 12;
    constexpr u32 THR       = 512;
    constexpr u32 MAXPAIRS  = D3_MAXPAIRS;

    __shared__ u16   ht[NRESTS][SSM_LOCAL - 1];
    __shared__ uint4 lastword1[NSLOTS];
    __shared__ u32   lastword2[NSLOTS];
    __shared__ int   ht_len[NRESTS];
    __shared__ int   pairs[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    if (threadid < NRESTS)
        ht_len[threadid] = 0;
    else if (threadid == (THR - 1))
        pairs_len = 0;
    else if (threadid == (THR - 33))
        next_pair = 0;

    const u32 bsize = min(eq->edata.nslots[2][bucketid], (u32)NSLOTS);

    u32   hr[2];
    int   pos[2]; pos[0] = pos[1] = SSM_LOCAL;
    u32   si[2];
    uint4 tt[2];
    u32   ta[2];

    __syncthreads();

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        si[i] = i * THR + threadid;
        if (si[i] >= bsize) break;
        slotsmall& xs  = eq->round2trees[bucketid].treessmall[si[i]];
        slottiny&  xst = eq->round2trees[bucketid].treestiny[si[i]];
        tt[i] = *(uint4*)(&xs.hash[0]);
        lastword1[si[i]] = tt[i];
        ta[i] = xst.hash[0];
        lastword2[si[i]] = ta[i];
        hr[i] = (tt[i].x >> 12) & RESTMASK;   // SAFE_BFE(_, tt.x, 12, RB) w/ RB=8
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }
    __syncthreads();

    u32 xors[5];
    u32 bexor, xorbucketid, xorslot;

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;
        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            xors[4] = ta[i] ^ lastword2[p];
            if (xors[4] != 0) {
                uint4 l1p = lastword1[p];
                xors[0] = tt[i].x ^ l1p.x;
                xors[1] = tt[i].y ^ l1p.y;
                xors[2] = tt[i].z ^ l1p.z;
                xors[3] = tt[i].w ^ l1p.w;
                bexor = __byte_perm(xors[0], xors[1], 0x2107);
                xorbucketid = (bexor >> RB) & BUCKMASK;
                xorslot = atomicAdd(&eq->edata.nslots[3][xorbucketid], 1);
                if (xorslot < NSLOTS) {
                    slotsmall& xs  = eq->round3trees[xorbucketid].treessmall[xorslot];
                    *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], xors[3], xors[4]);
                    slottiny&  xst = eq->round3trees[xorbucketid].treestiny[xorslot];
                    *(uint2*)(&xst.hash[0]) = make_uint2(bexor, pack_bid_s0_s1(bucketid, si[i], p));
                }
            }
            for (int k = 1; k != pos[i]; ++k) {
                u32 pindex = atomicAdd(&pairs_len, 1);
                if (pindex >= MAXPAIRS) break;
                u16 prev = ht[hr[i]][k];
                pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
            }
        }
    }
    __syncthreads();

    u32 plen = min(pairs_len, MAXPAIRS);
    u32 ii, kk;
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        int pair = pairs[s];
        ii = __byte_perm(pair, 0, 0x4510);
        kk = __byte_perm(pair, 0, 0x4532);
        xors[4] = lastword2[ii] ^ lastword2[kk];
        if (xors[4] != 0) {
            uint4 l1i = lastword1[ii];
            uint4 l1k = lastword1[kk];
            xors[0] = l1i.x ^ l1k.x;
            xors[1] = l1i.y ^ l1k.y;
            xors[2] = l1i.z ^ l1k.z;
            xors[3] = l1i.w ^ l1k.w;
            bexor = __byte_perm(xors[0], xors[1], 0x2107);
            xorbucketid = (bexor >> RB) & BUCKMASK;
            xorslot = atomicAdd(&eq->edata.nslots[3][xorbucketid], 1);
            if (xorslot < NSLOTS) {
                slotsmall& xs  = eq->round3trees[xorbucketid].treessmall[xorslot];
                *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], xors[3], xors[4]);
                slottiny&  xst = eq->round3trees[xorbucketid].treestiny[xorslot];
                *(uint2*)(&xst.hash[0]) = make_uint2(bexor, pack_bid_s0_s1(bucketid, ii, kk));
            }
        }
    }
}

// ============================================================================
// digit_4_kernel — port of djezo's digit_4
// ============================================================================
// Reads: eq->round3trees[bucketid].treessmall (uint4) + .treestiny (u32)
// Writes: eq->treessmall[3][xorbucketid][xorslot] (uint4)
//         eq->round4bidandsids[xorbucketid][xorslot] (u32)
// xhash: xst.hash[0] & RESTMASK

static constexpr u32 D4_MAXPAIRS = 4u * NRESTS;

__global__ __launch_bounds__(512, 2)
void digit_4_kernel(equi_state* eq) {
    constexpr int SSM_LOCAL = 12;
    constexpr u32 THR       = 512;
    constexpr u32 MAXPAIRS  = D4_MAXPAIRS;

    __shared__ u16   ht[NRESTS][SSM_LOCAL - 1];
    __shared__ uint4 lastword[NSLOTS];
    __shared__ int   ht_len[NRESTS];
    __shared__ int   pairs[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    if (threadid < NRESTS)
        ht_len[threadid] = 0;
    else if (threadid == (THR - 1))
        pairs_len = 0;
    else if (threadid == (THR - 33))
        next_pair = 0;

    const u32 bsize = min(eq->edata.nslots[3][bucketid], (u32)NSLOTS);

    u32   hr[2];
    int   pos[2]; pos[0] = pos[1] = SSM_LOCAL;
    u32   si[2];
    uint4 tt[2];

    __syncthreads();

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        si[i] = i * THR + threadid;
        if (si[i] >= bsize) break;
        slotsmall& xs  = eq->round3trees[bucketid].treessmall[si[i]];
        slottiny&  xst = eq->round3trees[bucketid].treestiny[si[i]];
        tt[i] = *(uint4*)(&xs.hash[0]);
        lastword[si[i]] = tt[i];
        hr[i] = xst.hash[0] & RESTMASK;
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }
    __syncthreads();

    u32 xors[4];
    u32 xorbucketid, xorslot;

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;
        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            uint4 lp = lastword[p];
            xors[0] = tt[i].x ^ lp.x;
            xors[1] = tt[i].y ^ lp.y;
            xors[2] = tt[i].z ^ lp.z;
            xors[3] = tt[i].w ^ lp.w;
            if (xors[3] != 0) {
                // xorbucketid = SAFE_BFE(xors[0], 4+RB, BUCKBITS) = bits[12..24] for RB=8
                xorbucketid = (xors[0] >> (4 + RB)) & BUCKMASK;
                xorslot = atomicAdd(&eq->edata.nslots[4][xorbucketid], 1);
                if (xorslot < NSLOTS) {
                    slotsmall& xs = eq->treessmall[3][xorbucketid][xorslot];
                    *(uint4*)(&xs.hash[0]) = make_uint4(xors[0], xors[1], xors[2], xors[3]);
                    eq->round4bidandsids[xorbucketid][xorslot] = pack_bid_s0_s1(bucketid, si[i], p);
                }
            }
            for (int k = 1; k != pos[i]; ++k) {
                u32 pindex = atomicAdd(&pairs_len, 1);
                if (pindex >= MAXPAIRS) break;
                u16 prev = ht[hr[i]][k];
                pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
            }
        }
    }
    __syncthreads();

    u32 plen = min(pairs_len, MAXPAIRS);
    u32 ii, kk;
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        int pair = pairs[s];
        ii = __byte_perm(pair, 0, 0x4510);
        kk = __byte_perm(pair, 0, 0x4532);
        uint4 li = lastword[ii];
        uint4 lk = lastword[kk];
        xors[0] = li.x ^ lk.x;
        xors[1] = li.y ^ lk.y;
        xors[2] = li.z ^ lk.z;
        xors[3] = li.w ^ lk.w;
        if (xors[3] != 0) {
            xorbucketid = (xors[0] >> (4 + RB)) & BUCKMASK;
            xorslot = atomicAdd(&eq->edata.nslots[4][xorbucketid], 1);
            if (xorslot < NSLOTS) {
                slotsmall& xs = eq->treessmall[3][xorbucketid][xorslot];
                *(uint4*)(&xs.hash[0]) = make_uint4(xors[0], xors[1], xors[2], xors[3]);
                eq->round4bidandsids[xorbucketid][xorslot] = pack_bid_s0_s1(bucketid, ii, kk);
            }
        }
    }
}

// ============================================================================
// digit_5_kernel — port of djezo's digit_5
// ============================================================================
// Reads: eq->treessmall[3][bucketid] (slotsmall)
// Writes: eq->treessmall[2][xorbucketid][xorslot] (slotsmall)
// xhash: SAFE_BFE(hr, tt.x, 4, RB) = bits[4..12] for RB=8
// bexor = __byte_perm(xors[0], xors[1], 0x1076)
// xorbucketid = SAFE_BFE(_, bexor, RB, BUCKBITS) = bits[8..20] for RB=8

static constexpr u32 D5_MAXPAIRS = 4u * NRESTS;

__global__ __launch_bounds__(512, 2)
void digit_5_kernel(equi_state* eq) {
    constexpr int SSM_LOCAL = 12;
    constexpr u32 THR       = 512;
    constexpr u32 MAXPAIRS  = D5_MAXPAIRS;

    __shared__ u16   ht[NRESTS][SSM_LOCAL - 1];
    __shared__ uint4 lastword[NSLOTS];
    __shared__ int   ht_len[NRESTS];
    __shared__ int   pairs[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    if (threadid < NRESTS)
        ht_len[threadid] = 0;
    else if (threadid == (THR - 1))
        pairs_len = 0;
    else if (threadid == (THR - 33))
        next_pair = 0;

    slotsmall* buck = eq->treessmall[3][bucketid];
    const u32 bsize = min(eq->edata.nslots[4][bucketid], (u32)NSLOTS);

    u32   hr[2];
    int   pos[2]; pos[0] = pos[1] = SSM_LOCAL;
    u32   si[2];
    uint4 tt[2];

    __syncthreads();

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        si[i] = i * THR + threadid;
        if (si[i] >= bsize) break;
        const slotsmall* pslot1 = buck + si[i];
        tt[i] = *(uint4*)(&pslot1->hash[0]);
        lastword[si[i]] = tt[i];
        hr[i] = (tt[i].x >> 4) & RESTMASK;  // SAFE_BFE(_, tt.x, 4, RB) for RB=8
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }
    __syncthreads();

    u32 xors[4];
    u32 bexor, xorbucketid, xorslot;

    #pragma unroll
    for (u32 i = 0; i != 2; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;
        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            uint4 lp = lastword[p];
            xors[0] = tt[i].x ^ lp.x;
            xors[1] = tt[i].y ^ lp.y;
            xors[2] = tt[i].z ^ lp.z;
            xors[3] = tt[i].w ^ lp.w;
            if (xors[3] != 0) {
                bexor = __byte_perm(xors[0], xors[1], 0x1076);
                xorbucketid = (bexor >> RB) & BUCKMASK;
                xorslot = atomicAdd(&eq->edata.nslots[5][xorbucketid], 1);
                if (xorslot < NSLOTS) {
                    slotsmall& xs = eq->treessmall[2][xorbucketid][xorslot];
                    *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], xors[3],
                                                         pack_bid_s0_s1(bucketid, si[i], p));
                }
            }
            for (int k = 1; k != pos[i]; ++k) {
                u32 pindex = atomicAdd(&pairs_len, 1);
                if (pindex >= MAXPAIRS) break;
                u16 prev = ht[hr[i]][k];
                pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
            }
        }
    }
    __syncthreads();

    u32 plen = min(pairs_len, MAXPAIRS);
    u32 ii, kk;
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        int pair = pairs[s];
        ii = __byte_perm(pair, 0, 0x4510);
        kk = __byte_perm(pair, 0, 0x4532);
        uint4 li = lastword[ii];
        uint4 lk = lastword[kk];
        xors[0] = li.x ^ lk.x;
        xors[1] = li.y ^ lk.y;
        xors[2] = li.z ^ lk.z;
        xors[3] = li.w ^ lk.w;
        if (xors[3] != 0) {
            bexor = __byte_perm(xors[0], xors[1], 0x1076);
            xorbucketid = (bexor >> RB) & BUCKMASK;
            xorslot = atomicAdd(&eq->edata.nslots[5][xorbucketid], 1);
            if (xorslot < NSLOTS) {
                slotsmall& xs = eq->treessmall[2][xorbucketid][xorslot];
                *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], xors[3],
                                                     pack_bid_s0_s1(bucketid, ii, kk));
            }
        }
    }
}

// ============================================================================
// digit_6_kernel — port of djezo's digit_6
// ============================================================================
// Different shape from 1-5: NRESTS threads/block, 3 slots per thread,
// SSM-1 (=11) max collisions, first 2 collisions handled inline.
// Reads: eq->treessmall[2][bucketid]
// Writes: eq->treessmall[0][xorbid][xorslot]

static constexpr u32 D6_MAXPAIRS = 3u * NRESTS;

__global__ __launch_bounds__(256, 2)
void digit_6_kernel(equi_state* eq) {
    constexpr int SSM_LOCAL = 11;   // was SSM-1 in djezo (SSM=12)
    constexpr u32 MAXPAIRS  = D6_MAXPAIRS;

    __shared__ u16   ht[NRESTS][SSM_LOCAL - 1];
    __shared__ uint2 lastword1[NSLOTS];
    __shared__ u32   lastword2[NSLOTS];
    __shared__ int   ht_len[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   bsize_sh;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    ht_len[threadid] = 0;
    if (threadid == (NRESTS - 1)) {
        pairs_len = 0;
        next_pair = 0;
    } else if (threadid == (NRESTS - 33)) {
        bsize_sh = min(eq->edata.nslots[5][bucketid], (u32)NSLOTS);
    }

    slotsmall* buck = eq->treessmall[2][bucketid];

    u32   hr[3];
    int   pos[3]; pos[0] = pos[1] = pos[2] = SSM_LOCAL;
    u32   si[3];
    uint4 tt[3];

    __syncthreads();
    u32 bsize = bsize_sh;

    #pragma unroll
    for (u32 i = 0; i != 3; ++i) {
        si[i] = i * NRESTS + threadid;
        if (si[i] >= bsize) break;
        const slotsmall* pslot1 = buck + si[i];
        tt[i] = *(uint4*)(&pslot1->hash[0]);
        lastword1[si[i]] = make_uint2(tt[i].x, tt[i].y);
        lastword2[si[i]] = tt[i].z;
        hr[i] = (tt[i].x >> 16) & RESTMASK;     // SAFE_BFE(_, tt.x, 16, RB=8)
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }

    int* pairs = ht_len;    // reuse ht_len as pairs array (djezo does same)
    __syncthreads();

    u32 xors[3];
    u32 bexor, xorbucketid, xorslot;

    #pragma unroll
    for (u32 i = 0; i != 3; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;
        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            xors[2] = tt[i].z ^ lastword2[p];
            if (xors[2] != 0) {
                uint2 lw = lastword1[p];
                xors[0] = tt[i].x ^ lw.x;
                xors[1] = tt[i].y ^ lw.y;
                bexor = __byte_perm(xors[0], xors[1], 0x1076);
                xorbucketid = bexor >> (12 + RB);
                xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
                if (xorslot < NSLOTS) {
                    slotsmall& xs = eq->treessmall[0][xorbucketid][xorslot];
                    *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], bexor,
                                                         pack_bid_s0_s1(bucketid, si[i], p));
                }
            }
            if (pos[i] > 1) {
                p = ht[hr[i]][1];
                xors[2] = tt[i].z ^ lastword2[p];
                if (xors[2] != 0) {
                    uint2 lw = lastword1[p];
                    xors[0] = tt[i].x ^ lw.x;
                    xors[1] = tt[i].y ^ lw.y;
                    bexor = __byte_perm(xors[0], xors[1], 0x1076);
                    xorbucketid = bexor >> (12 + RB);
                    xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
                    if (xorslot < NSLOTS) {
                        slotsmall& xs = eq->treessmall[0][xorbucketid][xorslot];
                        *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], bexor,
                                                             pack_bid_s0_s1(bucketid, si[i], p));
                    }
                }
                for (int k = 2; k != pos[i]; ++k) {
                    u32 pindex = atomicAdd(&pairs_len, 1);
                    if (pindex >= MAXPAIRS) break;
                    u16 prev = ht[hr[i]][k];
                    pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
                }
            }
        }
    }
    __syncthreads();

    u32 plen = min(pairs_len, MAXPAIRS);
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        u32 pair = pairs[s];
        u32 i = __byte_perm(pair, 0, 0x4510);
        u32 k = __byte_perm(pair, 0, 0x4532);
        xors[2] = lastword2[i] ^ lastword2[k];
        if (xors[2] == 0) continue;
        uint2 lwi = lastword1[i];
        uint2 lwk = lastword1[k];
        xors[0] = lwi.x ^ lwk.x;
        xors[1] = lwi.y ^ lwk.y;
        bexor = __byte_perm(xors[0], xors[1], 0x1076);
        xorbucketid = bexor >> (12 + RB);
        xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
        if (xorslot >= NSLOTS) continue;
        slotsmall& xs = eq->treessmall[0][xorbucketid][xorslot];
        *(uint4*)(&xs.hash[0]) = make_uint4(xors[1], xors[2], bexor,
                                             pack_bid_s0_s1(bucketid, i, k));
    }
}

// ============================================================================
// digit_7_kernel — port of djezo's digit_7
// ============================================================================
// Reads: eq->treessmall[0][bucketid]
// Writes: eq->treessmall[1][xorbid][xorslot]
// xhash: SAFE_BFE(hr, tt.z, 12, RB)

static constexpr u32 D7_MAXPAIRS = 3u * NRESTS;

__global__ __launch_bounds__(256, 2)
void digit_7_kernel(equi_state* eq) {
    constexpr int SSM_LOCAL = 11;
    constexpr u32 MAXPAIRS  = D7_MAXPAIRS;

    __shared__ u16   ht[NRESTS][SSM_LOCAL - 1];
    __shared__ u32   lastword[NSLOTS][2];
    __shared__ int   ht_len[NRESTS];
    __shared__ int   pairs[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   bsize_sh;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    ht_len[threadid] = 0;
    if (threadid == (NRESTS - 1)) {
        pairs_len = 0;
        next_pair = 0;
    } else if (threadid == (NRESTS - 33)) {
        bsize_sh = min(eq->edata.nslots[6][bucketid], (u32)NSLOTS);
    }

    slotsmall* buck = eq->treessmall[0][bucketid];

    u32   hr[3];
    int   pos[3]; pos[0] = pos[1] = pos[2] = SSM_LOCAL;
    u32   si[3];
    uint4 tt[3];

    __syncthreads();
    u32 bsize = bsize_sh;

    #pragma unroll
    for (u32 i = 0; i != 3; ++i) {
        si[i] = i * NRESTS + threadid;
        if (si[i] >= bsize) break;
        const slotsmall* pslot1 = buck + si[i];
        tt[i] = *(uint4*)(&pslot1->hash[0]);
        lastword[si[i]][0] = tt[i].x;
        lastword[si[i]][1] = tt[i].y;
        hr[i] = (tt[i].z >> 12) & RESTMASK;   // SAFE_BFE(_, tt.z, 12, RB=8)
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }
    __syncthreads();

    u32 xors[2];
    u32 xorbucketid, xorslot;

    #pragma unroll
    for (u32 i = 0; i != 3; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;
        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            xors[0] = tt[i].x ^ lastword[p][0];
            xors[1] = tt[i].y ^ lastword[p][1];
            if (xors[1] != 0) {
                xorbucketid = (xors[0] >> (8 + RB)) & BUCKMASK;
                xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
                if (xorslot < NSLOTS) {
                    slotsmall& xs = eq->treessmall[1][xorbucketid][xorslot];
                    *(uint4*)(&xs.hash[0]) = make_uint4(xors[0], xors[1],
                                                         pack_bid_s0_s1(bucketid, si[i], p), 0);
                }
            }
            if (pos[i] > 1) {
                p = ht[hr[i]][1];
                xors[0] = tt[i].x ^ lastword[p][0];
                xors[1] = tt[i].y ^ lastword[p][1];
                if (xors[1] != 0) {
                    xorbucketid = (xors[0] >> (8 + RB)) & BUCKMASK;
                    xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
                    if (xorslot < NSLOTS) {
                        slotsmall& xs = eq->treessmall[1][xorbucketid][xorslot];
                        *(uint4*)(&xs.hash[0]) = make_uint4(xors[0], xors[1],
                                                             pack_bid_s0_s1(bucketid, si[i], p), 0);
                    }
                }
                for (int k = 2; k != pos[i]; ++k) {
                    u32 pindex = atomicAdd(&pairs_len, 1);
                    if (pindex >= MAXPAIRS) break;
                    u16 prev = ht[hr[i]][k];
                    pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
                }
            }
        }
    }
    __syncthreads();

    u32 plen = min(pairs_len, MAXPAIRS);
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        int pair = pairs[s];
        u32 i = __byte_perm(pair, 0, 0x4510);
        u32 k = __byte_perm(pair, 0, 0x4532);
        xors[0] = lastword[i][0] ^ lastword[k][0];
        xors[1] = lastword[i][1] ^ lastword[k][1];
        if (xors[1] == 0) continue;
        xorbucketid = (xors[0] >> (8 + RB)) & BUCKMASK;
        xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
        if (xorslot >= NSLOTS) continue;
        slotsmall& xs = eq->treessmall[1][xorbucketid][xorslot];
        *(uint4*)(&xs.hash[0]) = make_uint4(xors[0], xors[1],
                                             pack_bid_s0_s1(bucketid, i, k), 0);
    }
}

// ============================================================================
// digit_8_kernel — port of djezo's digit_8
// ============================================================================
// Reads: eq->treessmall[1][bucketid]
// Writes: eq->treestiny[0][xorbid][xorslot] (slottiny, 2 u32s)
// Output buckets: eq->edata.nslots8[xorbid] (4096-entry array for round 8+)

static constexpr u32 D8_MAXPAIRS = 3u * NRESTS;

__global__ __launch_bounds__(256, 2)
void digit_8_kernel(equi_state* eq) {
    constexpr int SSM_LOCAL = 11;
    constexpr u32 MAXPAIRS  = D8_MAXPAIRS;

    __shared__ u16   ht[NRESTS][SSM_LOCAL - 1];
    __shared__ u32   lastword[NSLOTS][2];
    __shared__ int   ht_len[NRESTS];
    __shared__ int   pairs[MAXPAIRS];
    __shared__ u32   pairs_len;
    __shared__ u32   bsize_sh;
    __shared__ u32   next_pair;

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    ht_len[threadid] = 0;
    if (threadid == (NRESTS - 1)) {
        next_pair = 0;
        pairs_len = 0;
    } else if (threadid == (NRESTS - 33)) {
        bsize_sh = min(eq->edata.nslots[7][bucketid], (u32)NSLOTS);
    }

    slotsmall* buck = eq->treessmall[1][bucketid];

    u32   hr[3];
    int   pos[3]; pos[0] = pos[1] = pos[2] = SSM_LOCAL;
    u32   si[3];
    uint2 tt[3];

    __syncthreads();
    u32 bsize = bsize_sh;

    #pragma unroll
    for (u32 i = 0; i != 3; ++i) {
        si[i] = i * NRESTS + threadid;
        if (si[i] >= bsize) break;
        const slotsmall* pslot1 = buck + si[i];
        tt[i] = *(uint2*)(&pslot1->hash[0]);
        lastword[si[i]][0] = tt[i].x;
        lastword[si[i]][1] = tt[i].y;
        hr[i] = (tt[i].x >> 8) & RESTMASK;    // SAFE_BFE(_, tt.x, 8, RB=8)
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (SSM_LOCAL - 1))
            ht[hr[i]][pos[i]] = (u16)si[i];
    }
    __syncthreads();

    u32 xors[2];
    u32 bexor, xorbucketid, xorslot;

    #pragma unroll
    for (u32 i = 0; i != 3; ++i) {
        if (pos[i] >= SSM_LOCAL) continue;
        if (pos[i] > 0) {
            u16 p = ht[hr[i]][0];
            xors[0] = tt[i].x ^ lastword[p][0];
            xors[1] = tt[i].y ^ lastword[p][1];
            if (xors[1] != 0) {
                bexor = __byte_perm(xors[0], xors[1], 0x0765);
                xorbucketid = bexor >> (12 + 8);
                xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
                if (xorslot < RB8_NSLOTS_LD) {
                    slottiny& xs = eq->treestiny[0][xorbucketid][xorslot];
                    *(uint2*)(&xs.hash[0]) = make_uint2(xors[1],
                                                        pack_bid_s0_s1(bucketid, si[i], p));
                }
            }
            if (pos[i] > 1) {
                p = ht[hr[i]][1];
                xors[0] = tt[i].x ^ lastword[p][0];
                xors[1] = tt[i].y ^ lastword[p][1];
                if (xors[1] != 0) {
                    bexor = __byte_perm(xors[0], xors[1], 0x0765);
                    xorbucketid = bexor >> (12 + 8);
                    xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
                    if (xorslot < RB8_NSLOTS_LD) {
                        slottiny& xs = eq->treestiny[0][xorbucketid][xorslot];
                        *(uint2*)(&xs.hash[0]) = make_uint2(xors[1],
                                                            pack_bid_s0_s1(bucketid, si[i], p));
                    }
                }
                for (int k = 2; k != pos[i]; ++k) {
                    u32 pindex = atomicAdd(&pairs_len, 1);
                    if (pindex >= MAXPAIRS) break;
                    u16 prev = ht[hr[i]][k];
                    pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
                }
            }
        }
    }
    __syncthreads();

    u32 plen = min(pairs_len, MAXPAIRS);
    for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1)) {
        int pair = pairs[s];
        u32 i = __byte_perm(pair, 0, 0x4510);
        u32 k = __byte_perm(pair, 0, 0x4532);
        xors[0] = lastword[i][0] ^ lastword[k][0];
        xors[1] = lastword[i][1] ^ lastword[k][1];
        if (xors[1] == 0) continue;
        bexor = __byte_perm(xors[0], xors[1], 0x0765);
        xorbucketid = bexor >> (12 + 8);
        xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
        if (xorslot >= RB8_NSLOTS_LD) continue;
        slottiny& xs = eq->treestiny[0][xorbucketid][xorslot];
        *(uint2*)(&xs.hash[0]) = make_uint2(xors[1],
                                            pack_bid_s0_s1(bucketid, i, k));
    }
}

// ============================================================================
// digit_last_wdc_kernel — port of djezo's digit_last_wdc
// ============================================================================
// Solution reconstruction: walks back through the tree structures to recover
// the 512 original indices for each solution candidate.
//
// Launch: <<<4096, 128>>> (djezo uses <<<4096, 256/2>>>)
//   FCT=2, DUPBITS=8, W=4, MAXPAIRS=64
//
// Warp layout: 4 warps per block, each warp processes one candidate at a time
//              via the `par` index.
//
// CRITICAL: The __syncwarp() calls between levels ARE REQUIRED on Blackwell.
// Without them, lane-to-lane shared memory communication races and the kernel
// produces invalid solutions. This was the bug we discovered earlier in djezo.

static constexpr u32 DLWDC_FCT      = 2;
static constexpr u32 DLWDC_DUPBITS  = 8;
static constexpr u32 DLWDC_W        = 4;
static constexpr u32 DLWDC_MAXPAIRS = 64;
static constexpr int DLWDC_SSM      = 9;  // was SSM-3 in djezo

__global__ __launch_bounds__(128, 2)
void digit_last_wdc_kernel(equi_state* eq) {
    __shared__ u8 shared_data[8192];
    int* ht_len    = (int*)(&shared_data[0]);
    int* pairs     = ht_len;
    u32* lastword  = (u32*)(&shared_data[256 * 4]);
    u16* ht        = (u16*)(&shared_data[256 * 4 + RB8_NSLOTS_LD * 4]);
    u32* pairs_len = (u32*)(&shared_data[8188]);

    const u32 threadid = threadIdx.x;
    const u32 bucketid = blockIdx.x;

    // Reset hashtable: 128 threads / FCT=2 -> each writes 2 ht_len entries (256 total).
    #pragma unroll
    for (u32 i = 0; i != DLWDC_FCT; ++i)
        ht_len[i * (256 / DLWDC_FCT) + threadid] = 0;

    if (threadid == ((256 / DLWDC_FCT) - 1))
        *pairs_len = 0;

    slottiny* buck = eq->treestiny[0][bucketid];
    u32 bsize = min(eq->edata.nslots8[bucketid], (u32)RB8_NSLOTS_LD);

    u32 si[3 * DLWDC_FCT];
    u32 hr[3 * DLWDC_FCT];
    int pos[3 * DLWDC_FCT];
    u32 lw[3 * DLWDC_FCT];
    #pragma unroll
    for (u32 i = 0; i != (3 * DLWDC_FCT); ++i)
        pos[i] = DLWDC_SSM;

    __syncthreads();

    #pragma unroll
    for (u32 i = 0; i != (3 * DLWDC_FCT); ++i) {
        si[i] = i * (256 / DLWDC_FCT) + threadid;
        if (si[i] >= bsize) break;
        const slottiny* pslot1 = buck + si[i];
        uint2 tt = *(uint2*)(&pslot1->hash[0]);
        lw[i] = tt.x;
        lastword[si[i]] = lw[i];
        hr[i] = (lw[i] >> 20) & 0xFFu;   // SAFE_BFE(_, lw[i], 20, 8)
        pos[i] = atomicAdd(&ht_len[hr[i]], 1);
        if (pos[i] < (DLWDC_SSM - 1))
            ht[hr[i] * (DLWDC_SSM - 1) + pos[i]] = (u16)si[i];
    }

    __syncthreads();

    #pragma unroll
    for (u32 i = 0; i != (3 * DLWDC_FCT); ++i) {
        if (pos[i] >= DLWDC_SSM) continue;
        for (int k = 0; k != pos[i]; ++k) {
            u16 prev = ht[hr[i] * (DLWDC_SSM - 1) + k];
            if (lw[i] != lastword[prev]) continue;
            u32 pindex = atomicAdd(pairs_len, 1);
            if (pindex >= DLWDC_MAXPAIRS) break;
            pairs[pindex] = (int)__byte_perm(si[i], prev, 0x1054);
        }
    }

    __syncthreads();
    u32 plen = min(*pairs_len, (u32)64);

    // -- Tree walk-back to reconstruct 512 indices per candidate --
    #define CALC_LEVEL(a, b, c, d) do {                                               \
        u32 plvl = levels[b];                                                         \
        u32* bucks = eq->round4bidandsids[pack_get_bucketid(plvl)];                   \
        u32 slot1 = pack_get_slot1(plvl);                                             \
        u32 slot0 = pack_get_slot0(plvl, slot1);                                      \
        levels[b] = bucks[slot1];                                                     \
        levels[c] = bucks[slot0];                                                     \
    } while (0)

    #define CALC_LEVEL_SMALL(a, b, c, d) do {                                         \
        u32 plvl = levels[b];                                                         \
        slotsmall* bucks = eq->treessmall[a][pack_get_bucketid(plvl)];                \
        u32 slot1 = pack_get_slot1(plvl);                                             \
        u32 slot0 = pack_get_slot0(plvl, slot1);                                      \
        levels[b] = bucks[slot1].hash[d];                                             \
        levels[c] = bucks[slot0].hash[d];                                             \
    } while (0)

    u32 lane = threadIdx.x & 0x1f;
    u32 par  = threadIdx.x >> 5;

    u32* levels = (u32*)&pairs[DLWDC_MAXPAIRS + (par << DLWDC_DUPBITS)];
    u32* susp   = levels;

    while (par < plen) {
        int pair = pairs[par];
        par += DLWDC_W;

        if (lane % 16 == 0) {
            u32 plvl;
            if (lane == 0) plvl = buck[__byte_perm(pair, 0, 0x4510)].hash[1];
            else           plvl = buck[__byte_perm(pair, 0, 0x4532)].hash[1];
            slotsmall* bucks = eq->treessmall[1][pack_get_bucketid(plvl)];
            u32 slot1 = pack_get_slot1(plvl);
            u32 slot0 = pack_get_slot0(plvl, slot1);
            levels[lane]     = bucks[slot1].hash[2];
            levels[lane + 8] = bucks[slot0].hash[2];
        }
        __syncwarp(0xFFFFFFFF);

        if (lane % 8 == 0) CALC_LEVEL_SMALL(0, lane, lane + 4, 3);
        __syncwarp(0xFFFFFFFF);

        if (lane % 4 == 0) CALC_LEVEL_SMALL(2, lane, lane + 2, 3);
        __syncwarp(0xFFFFFFFF);

        if (lane % 2 == 0) CALC_LEVEL(0, lane, lane + 1, 4);
        __syncwarp(0xFFFFFFFF);

        u32 ind[16];
        u32 f1 = levels[lane];
        const slottiny* buck_v4 = &eq->round3trees[pack_get_bucketid(f1)].treestiny[0];
        const u32 slot1_v4 = pack_get_slot1(f1);
        const u32 slot0_v4 = pack_get_slot0(f1, slot1_v4);

        susp[lane]      = 0xffffffff;
        susp[32 + lane] = 0xffffffff;
        __syncwarp(0xFFFFFFFF);

        #define CHECK_DUP(a) \
            __any_sync(0xFFFFFFFF, atomicExch(&susp[(ind[a] & ((1u << DLWDC_DUPBITS) - 1u))], (ind[a] >> DLWDC_DUPBITS)) == (ind[a] >> DLWDC_DUPBITS))

        u32 f2 = buck_v4[slot1_v4].hash[1];
        const slottiny* buck_v3_1 = &eq->round2trees[pack_get_bucketid(f2)].treestiny[0];
        const u32 slot1_v3_1 = pack_get_slot1(f2);
        const u32 slot0_v3_1 = pack_get_slot0(f2, slot1_v3_1);

        susp[64 + lane] = 0xffffffff;
        susp[96 + lane] = 0xffffffff;
        __syncwarp(0xFFFFFFFF);

        u32 f0 = buck_v3_1[slot1_v3_1].hash[1];
        const slot* buck_v2_1 = eq->trees[0][pack_get_bucketid(f0)];
        const u32 slot1_v2_1 = pack_get_slot1(f0);
        const u32 slot0_v2_1 = pack_get_slot0(f0, slot1_v2_1);

        susp[128 + lane] = 0xffffffff;
        susp[160 + lane] = 0xffffffff;
        __syncwarp(0xFFFFFFFF);

        u32 f3 = buck_v2_1[slot1_v2_1].hash[6];
        const slot* buck_fin_1 = eq->round0trees[pack_get_bucketid_r0(f3)];
        const u32 slot1_fin_1 = pack_get_slot1_r0(f3);
        const u32 slot0_fin_1 = pack_get_slot0_r0(f3, slot1_fin_1);

        susp[192 + lane] = 0xffffffff;
        susp[224 + lane] = 0xffffffff;
        __syncwarp(0xFFFFFFFF);

        ind[0] = buck_fin_1[slot1_fin_1].hash[7];
        if (CHECK_DUP(0)) continue;
        ind[1] = buck_fin_1[slot0_fin_1].hash[7];
        if (CHECK_DUP(1)) continue;

        u32 f4 = buck_v2_1[slot0_v2_1].hash[6];
        const slot* buck_fin_2 = eq->round0trees[pack_get_bucketid_r0(f4)];
        const u32 slot1_fin_2 = pack_get_slot1_r0(f4);
        const u32 slot0_fin_2 = pack_get_slot0_r0(f4, slot1_fin_2);
        ind[2] = buck_fin_2[slot1_fin_2].hash[7]; if (CHECK_DUP(2)) continue;
        ind[3] = buck_fin_2[slot0_fin_2].hash[7]; if (CHECK_DUP(3)) continue;

        u32 f5 = buck_v3_1[slot0_v3_1].hash[1];
        const slot* buck_v2_2 = eq->trees[0][pack_get_bucketid(f5)];
        const u32 slot1_v2_2 = pack_get_slot1(f5);
        const u32 slot0_v2_2 = pack_get_slot0(f5, slot1_v2_2);
        u32 f6 = buck_v2_2[slot1_v2_2].hash[6];
        const slot* buck_fin_3 = eq->round0trees[pack_get_bucketid_r0(f6)];
        const u32 slot1_fin_3 = pack_get_slot1_r0(f6);
        const u32 slot0_fin_3 = pack_get_slot0_r0(f6, slot1_fin_3);
        ind[4] = buck_fin_3[slot1_fin_3].hash[7]; if (CHECK_DUP(4)) continue;
        ind[5] = buck_fin_3[slot0_fin_3].hash[7]; if (CHECK_DUP(5)) continue;

        u32 f7 = buck_v2_2[slot0_v2_2].hash[6];
        const slot* buck_fin_4 = eq->round0trees[pack_get_bucketid_r0(f7)];
        const u32 slot1_fin_4 = pack_get_slot1_r0(f7);
        const u32 slot0_fin_4 = pack_get_slot0_r0(f7, slot1_fin_4);
        ind[6] = buck_fin_4[slot1_fin_4].hash[7]; if (CHECK_DUP(6)) continue;
        ind[7] = buck_fin_4[slot0_fin_4].hash[7]; if (CHECK_DUP(7)) continue;

        u32 f8 = buck_v4[slot0_v4].hash[1];
        const slottiny* buck_v3_2 = &eq->round2trees[pack_get_bucketid(f8)].treestiny[0];
        const u32 slot1_v3_2 = pack_get_slot1(f8);
        const u32 slot0_v3_2 = pack_get_slot0(f8, slot1_v3_2);
        u32 f9 = buck_v3_2[slot1_v3_2].hash[1];
        const slot* buck_v2_3 = eq->trees[0][pack_get_bucketid(f9)];
        const u32 slot1_v2_3 = pack_get_slot1(f9);
        const u32 slot0_v2_3 = pack_get_slot0(f9, slot1_v2_3);
        u32 f10 = buck_v2_3[slot1_v2_3].hash[6];
        const slot* buck_fin_5 = eq->round0trees[pack_get_bucketid_r0(f10)];
        const u32 slot1_fin_5 = pack_get_slot1_r0(f10);
        const u32 slot0_fin_5 = pack_get_slot0_r0(f10, slot1_fin_5);
        ind[8] = buck_fin_5[slot1_fin_5].hash[7]; if (CHECK_DUP(8)) continue;
        ind[9] = buck_fin_5[slot0_fin_5].hash[7]; if (CHECK_DUP(9)) continue;

        u32 f11 = buck_v2_3[slot0_v2_3].hash[6];
        const slot* buck_fin_6 = eq->round0trees[pack_get_bucketid_r0(f11)];
        const u32 slot1_fin_6 = pack_get_slot1_r0(f11);
        const u32 slot0_fin_6 = pack_get_slot0_r0(f11, slot1_fin_6);
        ind[10] = buck_fin_6[slot1_fin_6].hash[7]; if (CHECK_DUP(10)) continue;
        ind[11] = buck_fin_6[slot0_fin_6].hash[7]; if (CHECK_DUP(11)) continue;

        u32 f12 = buck_v3_2[slot0_v3_2].hash[1];
        const slot* buck_v2_4 = eq->trees[0][pack_get_bucketid(f12)];
        const u32 slot1_v2_4 = pack_get_slot1(f12);
        const u32 slot0_v2_4 = pack_get_slot0(f12, slot1_v2_4);
        u32 f13 = buck_v2_4[slot1_v2_4].hash[6];
        const slot* buck_fin_7 = eq->round0trees[pack_get_bucketid_r0(f13)];
        const u32 slot1_fin_7 = pack_get_slot1_r0(f13);
        const u32 slot0_fin_7 = pack_get_slot0_r0(f13, slot1_fin_7);
        ind[12] = buck_fin_7[slot1_fin_7].hash[7]; if (CHECK_DUP(12)) continue;
        ind[13] = buck_fin_7[slot0_fin_7].hash[7]; if (CHECK_DUP(13)) continue;

        u32 f14 = buck_v2_4[slot0_v2_4].hash[6];
        const slot* buck_fin_8 = eq->round0trees[pack_get_bucketid_r0(f14)];
        const u32 slot1_fin_8 = pack_get_slot1_r0(f14);
        const u32 slot0_fin_8 = pack_get_slot0_r0(f14, slot1_fin_8);
        ind[14] = buck_fin_8[slot1_fin_8].hash[7]; if (CHECK_DUP(14)) continue;
        ind[15] = buck_fin_8[slot0_fin_8].hash[7]; if (CHECK_DUP(15)) continue;

        u32 soli;
        if (lane == 0)
            soli = atomicAdd(&eq->edata.srealcont.nsols, 1);
        soli = __shfl_sync(0xFFFFFFFF, soli, 0);

        if (soli < MAXREALSOLS) {
            u32 pos = lane << 4;
            *(uint4*)(&eq->edata.srealcont.sols[soli][pos])      = *(uint4*)(&ind[0]);
            *(uint4*)(&eq->edata.srealcont.sols[soli][pos + 4])  = *(uint4*)(&ind[4]);
            *(uint4*)(&eq->edata.srealcont.sols[soli][pos + 8])  = *(uint4*)(&ind[8]);
            *(uint4*)(&eq->edata.srealcont.sols[soli][pos + 12]) = *(uint4*)(&ind[12]);
        }
    }
    #undef CHECK_DUP
    #undef CALC_LEVEL
    #undef CALC_LEVEL_SMALL
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

void launch_digit_2(equi_state* device_eq, cudaStream_t stream) {
    digit_2_kernel<<<NBUCKETS, THREADS, 0, stream>>>(device_eq);
}

void launch_digit_3(equi_state* device_eq, cudaStream_t stream) {
    digit_3_kernel<<<NBUCKETS, THREADS, 0, stream>>>(device_eq);
}

void launch_digit_4(equi_state* device_eq, cudaStream_t stream) {
    digit_4_kernel<<<NBUCKETS, THREADS, 0, stream>>>(device_eq);
}

void launch_digit_5(equi_state* device_eq, cudaStream_t stream) {
    digit_5_kernel<<<NBUCKETS, THREADS, 0, stream>>>(device_eq);
}

// digit_6..8 use NRESTS (=256 for RB=8) threads/block
void launch_digit_6(equi_state* device_eq, cudaStream_t stream) {
    digit_6_kernel<<<NBUCKETS, NRESTS, 0, stream>>>(device_eq);
}

void launch_digit_7(equi_state* device_eq, cudaStream_t stream) {
    digit_7_kernel<<<NBUCKETS, NRESTS, 0, stream>>>(device_eq);
}

void launch_digit_8(equi_state* device_eq, cudaStream_t stream) {
    digit_8_kernel<<<NBUCKETS, NRESTS, 0, stream>>>(device_eq);
}

void launch_digit_last_wdc(equi_state* device_eq, cudaStream_t stream) {
    // djezo's layout: <<<4096, 256/2>>> = 4096 blocks, 128 threads/block
    digit_last_wdc_kernel<<<4096, 128, 0, stream>>>(device_eq);
}

} // namespace bw
