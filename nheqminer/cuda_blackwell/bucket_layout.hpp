#pragma once
//
// Equihash 200,9 bucket / slot data layout.
//
// PHASE 0 STRATEGY: We deliberately keep the SAME on-disk layout as djezo's
// `equi<RB,SM>` struct so that Phase 0's prototype kernels can produce
// bit-identical output for byte-for-byte comparison against djezo. This
// validates that our new control-flow architecture (cp.async + producer/consumer
// + cluster sync) produces the same hash table bytes as djezo's classic
// per-thread kernels.
//
// LATER PHASES will switch to a SoA layout for coalesced 16B-aligned access
// once we've confirmed correctness. The layout decision is isolated to this
// header so the change is local.

#include <cstdint>

namespace bw {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

// ---- Equihash 200,9 algorithmic constants ----
static constexpr u32 WN = 200;
static constexpr u32 WK = 9;
static constexpr u32 NDIGITS = WK + 1;                  // 10
static constexpr u32 DIGITBITS = WN / NDIGITS;          // 20
static constexpr u32 PROOFSIZE = 1u << WK;              // 512
static constexpr u32 BASE = 1u << DIGITBITS;            // 2^20
static constexpr u32 NHASHES = 2u * BASE;               // 2^21 = 2,097,152
static constexpr u32 HASHESPERBLAKE = 512u / WN;        // 2 (BLAKE2b output is 64 bytes = 512 bits)
static constexpr u32 HASHOUT = HASHESPERBLAKE * WN / 8; // 50 bytes
static constexpr u32 NBLOCKS = (NHASHES + HASHESPERBLAKE - 1) / HASHESPERBLAKE; // 1,048,576
static constexpr u32 FD_THREADS = 128;

// ---- Phase 0 config: matches djezo CONFIG_MODE_2 ----
// We start with mode 2 (smaller buckets) since that's our current production
// mode on Blackwell. Phase 1 may switch to mode 1 or a custom config.
static constexpr u32 RB = 8;
static constexpr u32 SM_PARAM = 640;     // NSLOTS in djezo terminology
static constexpr int SSM = 12;           // soft-stack max collisions per xhash
static constexpr u32 THREADS = 512;

// Derived
static constexpr u32 BUCKBITS = DIGITBITS - RB;          // 12
static constexpr u32 NBUCKETS = 1u << BUCKBITS;          // 4096
static constexpr u32 BUCKMASK = NBUCKETS - 1;
static constexpr u32 SLOTBITS = RB + 2;                  // 10
static constexpr u32 SLOTRANGE = 1u << SLOTBITS;         // 1024
static constexpr u32 NSLOTS = SM_PARAM;                  // 640
static constexpr u32 SLOTMASK = SLOTRANGE - 1;
static constexpr u32 NRESTS = 1u << RB;                  // 256
static constexpr u32 RESTMASK = NRESTS - 1;
static constexpr u32 RB8_NSLOTS = 640;
static constexpr u32 RB8_NSLOTS_LD = 624;

static constexpr u32 MAXREALSOLS = 9;

// ---- Slot data structures (must match djezo's equi_miner.cu lines 133-148) ----
struct __align__(32) slot {
    u32 hash[8];
};

struct __align__(16) slotsmall {
    u32 hash[4];
};

struct __align__(8) slottiny {
    u32 hash[2];
};

struct scontainerreal {
    u32 sols[MAXREALSOLS][512];
    u32 nsols;
};

// ---- Top-level state struct (matches djezo's `equi<RB,SM>`) ----
//
// IMPORTANT: This MUST be byte-compatible with djezo's `equi<8, 640>`
// for Phase 0 byte-for-byte comparison to work.
struct equi_state {
    slot      round0trees[4096][RB8_NSLOTS];
    slot      trees[1][NBUCKETS][NSLOTS];

    struct {
        slotsmall treessmall[NSLOTS];
        slottiny  treestiny[NSLOTS];
    } round2trees[NBUCKETS];

    struct {
        slotsmall treessmall[NSLOTS];
        slottiny  treestiny[NSLOTS];
    } round3trees[NBUCKETS];

    slotsmall treessmall[4][NBUCKETS][NSLOTS];
    slottiny  treestiny[1][4096][RB8_NSLOTS_LD];
    u32       round4bidandsids[NBUCKETS][NSLOTS];

    union {
        u64 blake_h[8];
        u32 blake_h32[16];
    };

    struct {
        u32 nslots8[4096];
        u32 nslots0[4096];
        u32 nslots[9][NBUCKETS];
        scontainerreal srealcont;
    } edata;
};

// ---- Tree-node bit packing (matches djezo's `tree` struct, lines 64-91) ----
//
// XINTREE format: bid_s0_s1_x = (((bucketid << SLOTBITS) | s0) << SLOTBITS | s1) << RB | xhash
struct tree_node {
    u32 bid_s0_s1_x;

    __device__ tree_node() : bid_s0_s1_x(0) {}
    __device__ tree_node(u32 idx) : bid_s0_s1_x(idx) {}
    __device__ tree_node(u32 idx, u32 xh) : bid_s0_s1_x((idx << RB) | xh) {}
    __device__ tree_node(u32 bid, u32 s0, u32 s1, u32 xh)
        : bid_s0_s1_x((((((bid << SLOTBITS) | s0) << SLOTBITS) | s1) << RB) | xh) {}

    __device__ u32 getindex() const { return bid_s0_s1_x >> RB; }
    __device__ u32 bucketid() const { return bid_s0_s1_x >> (RB + 2 * SLOTBITS); }
    __device__ u32 slotid0()  const { return (bid_s0_s1_x >> (RB + SLOTBITS)) & SLOTMASK; }
    __device__ u32 slotid1()  const { return (bid_s0_s1_x >> RB) & SLOTMASK; }
    __device__ u32 xhash()    const { return bid_s0_s1_x & RESTMASK; }
};

} // namespace bw
