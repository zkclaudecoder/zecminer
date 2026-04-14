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
    // Bucket id = top 12 bits of first 32-bit word (per CONFIG_MODE_2: BUCKBITS=12)
    // Matches djezo's logic for RB=8.
    {
        u32 bexor = __byte_perm(v32[0], 0, 0x4012); // first 20 bits
        u32 bucketid = (bexor >> 8) & BUCKMASK;     // 12-bit bucket id
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
        u32 bucketid = (bexor >> 8) & BUCKMASK;
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
// digit_1_kernel — STUB (Phase 0 still TODO)
// ============================================================================
//
// Architecture target: producer/consumer warp specialization.
//
//   __shared__ slot ring[kPipelineDepth][NSLOTS];        // bucket-sized scratch
//   __shared__ cuda::barrier<cuda::thread_scope_block> bar[kPipelineDepth];
//   __shared__ u32 bsize[kPipelineDepth];
//
// At init (one warp does this):
//   for (int i = 0; i < kPipelineDepth; ++i)
//       init(&bar[i], blockDim.x);
//
// PRODUCER warp (warp 0):
//   while (more buckets) {
//       int slot = bucket_idx % kPipelineDepth;
//       wait_for_drain(bar[slot]);
//       size = eq->edata.nslots0[bucket_idx];
//       cuda::memcpy_async(producer_warp, ring[slot], &eq->round0trees[bucket_idx],
//                          size * sizeof(slot), bar[slot]);
//       bsize[slot] = size;
//   }
//
// CONSUMER warps (warps 1..N):
//   while (more buckets) {
//       int slot = bucket_idx % kPipelineDepth;
//       bar[slot].arrive_and_wait();
//       // collision-find on ring[slot][0..bsize[slot]] — exact same logic as
//       // djezo's digit_1 inner loop
//       // emit XOR'd output to eq->trees[0] / eq->edata.nslots[1]
//   }
//
// Validation: must produce IDENTICAL output to djezo's digit_1 (same bucket
// layout, same atomic ordering for nslots[1]). Some atomicAdd order non-determinism
// is acceptable — what we compare is the SET of slots in each bucket, not order.
//
// Estimated time: 3-5 days for first cut, 1-2 weeks to debug to bit-match.

__global__ void digit_1_kernel(equi_state* eq) {
    // TODO(phase0): implement producer/consumer pipeline
}

// ============================================================================
// Host launchers
// ============================================================================

void launch_digit_first(equi_state* device_eq, uint32_t nonce, cudaStream_t stream) {
    constexpr u32 grid_blocks = NBLOCKS / FD_THREADS;
    digit_first_kernel<<<grid_blocks, FD_THREADS, 0, stream>>>(device_eq, nonce);
}

void launch_digit_1(equi_state* device_eq, cudaStream_t stream) {
    // Phase 0 stub — launch config TBD when kernel body is implemented.
    constexpr u32 grid_blocks = 4096;   // matches djezo's <<<4096, 512>>>
    constexpr u32 threads_per_block = 512;
    digit_1_kernel<<<grid_blocks, threads_per_block, 0, stream>>>(device_eq);
}

} // namespace bw
