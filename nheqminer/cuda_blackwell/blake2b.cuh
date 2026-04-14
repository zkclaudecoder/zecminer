#pragma once
// BLAKE2b kernels for cuda_blackwell solver.
//
// PHASE 0: Pure direct port of djezo's `blake2b.cu` device functions, used by
// our own `digit_first_kernel` in digit_kernels.cu. Identical math means we can
// validate digit_first output bit-for-bit against djezo.
//
// LATER PHASES: We may use cp.async to stage the BLAKE2b initial state
// (`blake_h[8]`) into shared memory once per block instead of repeatedly
// reading from device global, but the BLAKE2b math itself stays the same.

#include <cuda_runtime.h>
#include <stdint.h>

namespace bw {

using u64 = uint64_t;
using u32 = uint32_t;

// ---- BLAKE2b helper math (verbatim from djezo's blake2b.cu) ----

static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b) {
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

__device__ __forceinline__ uint2 ROR2(const uint2 a, const int offset) {
    uint2 result;
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
    asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
    return result;
}

__device__ __forceinline__ uint2 SWAPUINT2(uint2 value) {
    return make_uint2(value.y, value.x);
}

__device__ __forceinline__ uint2 ROR24(const uint2 a) {
    uint2 result;
    result.x = __byte_perm(a.y, a.x, 0x2107);
    result.y = __byte_perm(a.y, a.x, 0x6543);
    return result;
}

__device__ __forceinline__ uint2 ROR16(const uint2 a) {
    uint2 result;
    result.x = __byte_perm(a.y, a.x, 0x1076);
    result.y = __byte_perm(a.y, a.x, 0x5432);
    return result;
}

__device__ __constant__
const u64 blake_iv_bw[] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

__device__ __forceinline__
void G2(u64& a, u64& b, u64& c, u64& d, u64 x, u64 y) {
    a = a + b + x;
    ((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
    c = c + d;
    ((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
    a = a + b + y;
    ((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
    c = c + d;
    ((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

// Bytewise BLAKE2b for one block index. `h[8]` is the prepared mid-state from
// the host's `setheader` (after consuming the block header); `block_idx` is the
// per-thread block ordinal that gets folded in as the final m word.
//
// Output: `h[]` is consumed (unchanged on input — the caller passes a copy).
// The compressed state in v[0..6] is XOR'd back into `h[]` per the djezo
// formula at the end.
//
// This is djezo's `blake2b_gpu_hash3` with a Blackwell-friendly inline-PTX
// rotate and a renamed constant array; output is bit-identical.
__device__ __forceinline__
void blake2b_gpu_hash3_bw(u64* h, u32 idx, u32 nonce) {
    u64 m = (u64)idx << 32 | (u64)nonce;
    u64 v[16];

    v[0] = h[0]; v[1] = h[1]; v[2] = h[2]; v[3] = h[3];
    v[4] = h[4]; v[5] = h[5]; v[6] = h[6]; v[7] = h[7];
    v[8]  = blake_iv_bw[0];
    v[9]  = blake_iv_bw[1];
    v[10] = blake_iv_bw[2];
    v[11] = blake_iv_bw[3];
    v[12] = blake_iv_bw[4] ^ (128 + 16);
    v[13] = blake_iv_bw[5];
    v[14] = blake_iv_bw[6] ^ 0xffffffffffffffffULL;
    v[15] = blake_iv_bw[7];

    // 12 rounds, hand-unrolled with the BLAKE2b sigma table folded in (m goes
    // to specific G2 calls). Identical to djezo's blake2b_gpu_hash3.

    // round 0
    G2(v[0], v[4], v[8],  v[12], 0, m);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 1
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], m, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 2
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, m);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 3
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, m);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 4
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, m);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 5
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], m, 0);
    // round 6
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], m, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 7
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, m);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 8
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], m, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 9
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], m, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 10
    G2(v[0], v[4], v[8],  v[12], 0, m);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], 0, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);
    // round 11
    G2(v[0], v[4], v[8],  v[12], 0, 0);  G2(v[1], v[5], v[9],  v[13], 0, 0);
    G2(v[2], v[6], v[10], v[14], 0, 0);  G2(v[3], v[7], v[11], v[15], 0, 0);
    G2(v[0], v[5], v[10], v[15], m, 0);  G2(v[1], v[6], v[11], v[12], 0, 0);
    G2(v[2], v[7], v[8],  v[13], 0, 0);  G2(v[3], v[4], v[9],  v[14], 0, 0);

    h[0] ^= v[0] ^ v[8];
    h[1] ^= v[1] ^ v[9];
    h[2] ^= v[2] ^ v[10];
    h[3] ^= v[3] ^ v[11];
    h[4] ^= v[4] ^ v[12];
    h[5] ^= v[5] ^ v[13];
    h[6] ^= v[6] ^ v[14];
}

} // namespace bw
