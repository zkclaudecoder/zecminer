// cuda_blackwell.cpp — solver wrapper class for MinerFactory.
//
// Phase 0: minimal — context lifecycle and stub solve() that runs digit_first
// and digit_1 only. Returns no solutions yet.

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <algorithm>
#include "cuda_runtime.h"

#include "cuda_blackwell.hpp"
#include "eq_context.hpp"
#include "../cpu_tromp/blake2/blake2.h"

// CPU-side duplicate check + pair-sort, matching djezo's post-processing
static int compu32_bw(const void* pa, const void* pb) {
    uint32_t a = *(uint32_t*)pa, b = *(uint32_t*)pb;
    return a < b ? -1 : a == b ? 0 : +1;
}
static bool duped_bw(const uint32_t* prf) {
    uint32_t sort[512];
    std::memcpy(sort, prf, sizeof(sort));
    std::qsort(sort, 512, sizeof(uint32_t), &compu32_bw);
    for (uint32_t i = 1; i < 512; ++i)
        if (sort[i] <= sort[i - 1]) return true;
    return false;
}
static void sort_pair_bw(uint32_t* a, uint32_t len) {
    uint32_t* b = a + len;
    uint32_t tmp, need_sort = 0;
    for (uint32_t i = 0; i < len; ++i) {
        if (need_sort || a[i] > b[i]) {
            need_sort = 1;
            tmp = a[i]; a[i] = b[i]; b[i] = tmp;
        } else if (a[i] < b[i]) {
            return;   // already sorted at first differing key — early exit
        }
    }
}

// Match djezo/bw Equihash parameters (MUST stay in sync with bucket_layout.hpp)
static constexpr uint32_t WN_ = 200;
static constexpr uint32_t WK_ = 9;
static constexpr uint32_t HASHOUT_ = (512 / WN_) * WN_ / 8;  // 50
static constexpr uint32_t HASHESPERBLAKE_ = 512 / WN_;        // 2
static constexpr uint32_t DIGITBITS_ = WN_ / (WK_ + 1);       // 20

// Port of djezo's setheader(). Given the job's header bytes and a nonce,
// produces the blake2b_state that our digit_first_kernel folds the block index
// into on the device.
static void bw_setheader(blake2b_state* ctx,
                         const char* header, uint32_t header_len,
                         const char* nce, uint32_t nonce_len) {
    uint32_t le_N = WN_;
    uint32_t le_K = WK_;
    unsigned char personal[] = "ZcashPoW01230123";
    std::memcpy(personal + 8, &le_N, 4);
    std::memcpy(personal + 12, &le_K, 4);
    blake2b_param P[1];
    P->digest_length = HASHOUT_;
    P->key_length = 0;
    P->fanout = 1;
    P->depth = 1;
    P->leaf_length = 0;
    P->node_offset = 0;
    P->node_depth = 0;
    P->inner_length = 0;
    std::memset(P->reserved, 0, sizeof(P->reserved));
    std::memset(P->salt,     0, sizeof(P->salt));
    std::memcpy(P->personal, (const uint8_t*)personal, 16);
    blake2b_init_param(ctx, P);
    blake2b_update(ctx, (const unsigned char*)header, header_len);
    blake2b_update(ctx, (const unsigned char*)nce,    nonce_len);
}

// Host-side emulation of CUDA's __byte_perm intrinsic.
// __byte_perm(a, b, s): output byte i = selector nibble s.n[i] which picks
// a.byte[0..3] (selectors 0..3) or b.byte[0..3] (selectors 4..7).
static inline uint32_t byte_perm_host(uint32_t a, uint32_t b, uint32_t s) {
    uint8_t ab[8] = {
        (uint8_t)(a & 0xff), (uint8_t)((a >> 8) & 0xff),
        (uint8_t)((a >> 16) & 0xff), (uint8_t)((a >> 24) & 0xff),
        (uint8_t)(b & 0xff), (uint8_t)((b >> 8) & 0xff),
        (uint8_t)((b >> 16) & 0xff), (uint8_t)((b >> 24) & 0xff)
    };
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        r |= ((uint32_t)ab[(s >> (i * 4)) & 0xf]) << (i * 8);
    return r;
}

// ---- CPU-side Equihash verification (port of djezo's verify_solution) ----
enum bw_verify_code { BW_POW_OK, BW_POW_DUPLICATE, BW_POW_OUT_OF_ORDER, BW_POW_NONZERO_XOR };

static void bw_genhash(blake2b_state* ctx, uint32_t idx, unsigned char* hash) {
    blake2b_state state = *ctx;
    uint32_t leb = idx / HASHESPERBLAKE_;
    blake2b_update(&state, (unsigned char*)&leb, sizeof(uint32_t));
    unsigned char blakehash[HASHOUT_];
    blake2b_final(&state, blakehash, HASHOUT_);
    std::memcpy(hash, blakehash + (idx % HASHESPERBLAKE_) * (WN_ / 8), WN_ / 8);
}

static int bw_xorfail_level = -1;

static int bw_verifyrec(blake2b_state* ctx, uint32_t* indices, unsigned char* hash, int r) {
    if (r == 0) {
        bw_genhash(ctx, *indices, hash);
        return BW_POW_OK;
    }
    uint32_t* indices1 = indices + (1u << (r - 1));
    if (*indices >= *indices1) return BW_POW_OUT_OF_ORDER;
    unsigned char h0[WN_ / 8], h1[WN_ / 8];
    int v0 = bw_verifyrec(ctx, indices, h0, r - 1);
    if (v0 != BW_POW_OK) return v0;
    int v1 = bw_verifyrec(ctx, indices1, h1, r - 1);
    if (v1 != BW_POW_OK) return v1;
    for (uint32_t i = 0; i < WN_ / 8; ++i) hash[i] = h0[i] ^ h1[i];
    int i, b = r * (int)DIGITBITS_;
    for (i = 0; i < b / 8; ++i) {
        if (hash[i]) { if (bw_xorfail_level < 0) bw_xorfail_level = r; return BW_POW_NONZERO_XOR; }
    }
    if ((b % 8) && (hash[i] >> (8 - (b % 8)))) {
        if (bw_xorfail_level < 0) bw_xorfail_level = r;
        return BW_POW_NONZERO_XOR;
    }
    return BW_POW_OK;
}

static int bw_verify_solution(uint32_t* indices,
                              const char* header, uint32_t header_len,
                              const char* nce, uint32_t nonce_len) {
    blake2b_state ctx;
    bw_setheader(&ctx, header, header_len, nce, nonce_len);
    unsigned char hash[WN_ / 8];
    return bw_verifyrec(&ctx, indices, hash, WK_);
}

// ===== eq_cuda_context_blackwell =====

eq_cuda_context_blackwell_interface::~eq_cuda_context_blackwell_interface() = default;

// SolContainer layout must match GPU edata.srealcont. Shared across solve() and ctor.
static constexpr uint32_t MAXREALSOLS_HOST = 9;
struct SolContainer {
    uint32_t sols[MAXREALSOLS_HOST][512];
    uint32_t nsols;
};

eq_cuda_context_blackwell::eq_cuda_context_blackwell(int id) : device_id(id), device_eq(nullptr), stream(0) {
    checkCudaErrorsBW(cudaSetDevice(device_id));
    checkCudaErrorsBW(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    checkCudaErrorsBW(cudaStreamCreate(&stream));

    if (cudaMalloc((void**)&device_eq, sizeof(bw::equi_state)) != cudaSuccess)
        throw std::runtime_error("CUDA(blackwell): failed to alloc equi_state");

    // Pinned host buffers for graph-captured memcpys. Addresses must be stable.
    checkCudaErrorsBW(cudaHostAlloc((void**)&pinned_blake_h, sizeof(uint64_t) * 8,
                                    cudaHostAllocDefault));
    checkCudaErrorsBW(cudaHostAlloc(&pinned_sols, sizeof(SolContainer),
                                    cudaHostAllocDefault));

    // L2 persisting window on the edata region (~200KB: nslots arrays + srealcont).
    // Measured neutral-to-slightly-negative on Blackwell RTX 5090 — 96MB L2 already
    // keeps this hot data cached; marking it persisting just displaces other data.
    // Opt-in only (BW_USE_L2_PERSIST=1) for future tuning.
    if (std::getenv("BW_USE_L2_PERSIST")) {
        int max_persist_size = 0, max_window_size = 0;
        cudaDeviceGetAttribute(&max_persist_size, cudaDevAttrMaxPersistingL2CacheSize, device_id);
        cudaDeviceGetAttribute(&max_window_size, cudaDevAttrMaxAccessPolicyWindowSize, device_id);
        if (max_persist_size > 0 && max_window_size > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)max_persist_size);
            size_t edata_bytes = sizeof(((bw::equi_state*)0)->edata);
            size_t num_bytes = edata_bytes < (size_t)max_window_size
                               ? edata_bytes : (size_t)max_window_size;
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr  = &device_eq->edata;
            attr.accessPolicyWindow.num_bytes = num_bytes;
            attr.accessPolicyWindow.hitRatio  = 1.0f;
            attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        }
    }
}

eq_cuda_context_blackwell::~eq_cuda_context_blackwell() {
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (pinned_blake_h) cudaFreeHost(pinned_blake_h);
    if (pinned_sols) cudaFreeHost(pinned_sols);
    if (device_eq) cudaFree(device_eq);
    if (stream) cudaStreamDestroy(stream);
}

void eq_cuda_context_blackwell::solve(
    const char* header, unsigned int header_len,
    const char* nonce, unsigned int nonce_len,
    std::function<bool()> /*cancelf*/,
    std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
    std::function<void(void)> hashdonef)
{
    blake2b_state ctx;
    bw_setheader(&ctx, header, header_len, nonce, nonce_len);
    // Stage blake mid-state into pinned buffer — stable address for graph capture.
    std::memcpy(pinned_blake_h, &ctx.h, sizeof(uint64_t) * 8);

    // Any diagnostic env var forces non-graph path (they inject extra memcpys).
    static const bool any_diag =
        std::getenv("DJEZO_BW_VERBOSE") || std::getenv("EQUI_DUMP_SETXOR") ||
        std::getenv("BW_VALIDATE_ROUND1") || std::getenv("BW_VALIDATE_ROUND2") ||
        std::getenv("BW_VALIDATE_ROUND3") || std::getenv("BW_VALIDATE_ROUND4") ||
        std::getenv("BW_VALIDATE_ROUND5") || std::getenv("BW_VALIDATE_ROUND6") ||
        std::getenv("BW_VALIDATE_ROUND7") || std::getenv("BW_VALIDATE_ROUND8") ||
        std::getenv("BW_HOST_WALK") || std::getenv("BW_PACK_UNIQUENESS");
    // CUDA graph capture: opt-in. Measured slower than per-launch on 8-instance
    // Blackwell (-15%) — graphs force strict ordering and prevent the driver
    // from overlapping launches across iterations. Keep as a knob for future
    // work (may be worth revisiting with persistent kernels or upgraded driver).
    static const bool use_graph = std::getenv("BW_USE_GRAPH") != nullptr;

    if (!any_diag && use_graph) {
        // --- Fast path: CUDA graph capture + replay ---
        if (graph_exec == nullptr) {
            checkCudaErrorsBW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            checkCudaErrorsBW(cudaMemcpyAsync(&device_eq->blake_h, pinned_blake_h,
                                               sizeof(uint64_t) * 8,
                                               cudaMemcpyHostToDevice, stream));
            checkCudaErrorsBW(cudaMemsetAsync(&device_eq->edata, 0,
                                               sizeof(device_eq->edata), stream));
            bw::launch_digit_first(device_eq, 0, stream);
            bw::launch_digit_1(device_eq, stream);
            bw::launch_digit_2(device_eq, stream);
            bw::launch_digit_3(device_eq, stream);
            bw::launch_digit_4(device_eq, stream);
            bw::launch_digit_5(device_eq, stream);
            bw::launch_digit_6(device_eq, stream);
            bw::launch_digit_7(device_eq, stream);
            bw::launch_digit_8(device_eq, stream);
            bw::launch_digit_last_wdc(device_eq, stream);
            checkCudaErrorsBW(cudaMemcpyAsync(pinned_sols, &device_eq->edata.srealcont,
                                               sizeof(SolContainer),
                                               cudaMemcpyDeviceToHost, stream));
            cudaGraph_t graph;
            checkCudaErrorsBW(cudaStreamEndCapture(stream, &graph));
            checkCudaErrorsBW(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
            cudaGraphDestroy(graph);
        }
        checkCudaErrorsBW(cudaGraphLaunch(graph_exec, stream));
        checkCudaErrorsBW(cudaStreamSynchronize(stream));

        // Solutions already in pinned_sols. Jump to post-processing.
        goto post_process;
    }

    // --- Slow path: diagnostic-compatible, per-launch ---
    checkCudaErrorsBW(cudaMemcpyAsync(&device_eq->blake_h, pinned_blake_h,
                                       sizeof(uint64_t) * 8,
                                       cudaMemcpyHostToDevice, stream));
    checkCudaErrorsBW(cudaMemsetAsync(&device_eq->edata, 0,
                                       sizeof(device_eq->edata), stream));

    // BW_PROFILE: time each kernel via cudaEvent. Adds overhead so default-off.
    static const bool do_profile = std::getenv("BW_PROFILE") != nullptr;
    if (do_profile) {
        static cudaEvent_t ev[11];
        static bool ev_init = false;
        if (!ev_init) { for (int i = 0; i < 11; ++i) cudaEventCreate(&ev[i]); ev_init = true; }
        cudaEventRecord(ev[0], stream);
        bw::launch_digit_first(device_eq, 0, stream); cudaEventRecord(ev[1], stream);
        bw::launch_digit_1(device_eq, stream);        cudaEventRecord(ev[2], stream);
        bw::launch_digit_2(device_eq, stream);        cudaEventRecord(ev[3], stream);
        bw::launch_digit_3(device_eq, stream);        cudaEventRecord(ev[4], stream);
        bw::launch_digit_4(device_eq, stream);        cudaEventRecord(ev[5], stream);
        bw::launch_digit_5(device_eq, stream);        cudaEventRecord(ev[6], stream);
        bw::launch_digit_6(device_eq, stream);        cudaEventRecord(ev[7], stream);
        bw::launch_digit_7(device_eq, stream);        cudaEventRecord(ev[8], stream);
        bw::launch_digit_8(device_eq, stream);        cudaEventRecord(ev[9], stream);
        bw::launch_digit_last_wdc(device_eq, stream); cudaEventRecord(ev[10], stream);
        cudaStreamSynchronize(stream);
        static int prof_count = 0;
        static double ms_total[10] = {0};
        for (int i = 0; i < 10; ++i) {
            float ms = 0; cudaEventElapsedTime(&ms, ev[i], ev[i+1]);
            ms_total[i] += ms;
        }
        if (++prof_count >= 100) {
            const char* names[10] = {"digit_first", "digit_1", "digit_2", "digit_3",
                                     "digit_4", "digit_5", "digit_6", "digit_7",
                                     "digit_8", "digit_last_wdc"};
            std::fprintf(stderr, "[BW-PROFILE] avg of %d iters (ms/kernel):\n", prof_count);
            for (int i = 0; i < 10; ++i)
                std::fprintf(stderr, "  %-15s %.4f ms\n", names[i], ms_total[i] / prof_count);
            prof_count = 0;
            for (int i = 0; i < 10; ++i) ms_total[i] = 0;
        }
    } else {
        bw::launch_digit_first(device_eq, /*nonce_suffix=*/0, stream);
        bw::launch_digit_1(device_eq, stream);
        bw::launch_digit_2(device_eq, stream);
        bw::launch_digit_3(device_eq, stream);
        bw::launch_digit_4(device_eq, stream);
        bw::launch_digit_5(device_eq, stream);
        bw::launch_digit_6(device_eq, stream);
        bw::launch_digit_7(device_eq, stream);
        bw::launch_digit_8(device_eq, stream);
        bw::launch_digit_last_wdc(device_eq, stream);
    }

    static const bool verbose = std::getenv("DJEZO_BW_VERBOSE") != nullptr;
    if (verbose) {
        // Pull back round-1..7 nslots + round-8 nslots8 for sanity checks
        uint32_t ns[7][bw::NBUCKETS];
        for (int r = 1; r <= 7; ++r) {
            checkCudaErrorsBW(cudaMemcpyAsync(ns[r - 1], &device_eq->edata.nslots[r],
                                               sizeof(ns[r - 1]), cudaMemcpyDeviceToHost, stream));
        }
        uint32_t ns8[4096];
        checkCudaErrorsBW(cudaMemcpyAsync(ns8, &device_eq->edata.nslots8,
                                           sizeof(ns8), cudaMemcpyDeviceToHost, stream));
        checkCudaErrorsBW(cudaStreamSynchronize(stream));

        for (int r = 1; r <= 7; ++r) {
            uint64_t total = 0, min_v = ~0ULL, max_v = 0, nonempty = 0;
            for (uint32_t i = 0; i < bw::NBUCKETS; ++i) {
                total += ns[r - 1][i];
                if (ns[r - 1][i] < min_v) min_v = ns[r - 1][i];
                if (ns[r - 1][i] > max_v) max_v = ns[r - 1][i];
                if (ns[r - 1][i] > 0) ++nonempty;
            }
            std::cerr << "[BW] round-" << r << " nslots: total=" << total
                      << " mean=" << (double)total / bw::NBUCKETS
                      << " min=" << min_v
                      << " max=" << max_v
                      << " nonempty=" << nonempty << "/" << bw::NBUCKETS
                      << std::endl;
        }
        uint64_t total = 0, min_v = ~0ULL, max_v = 0, nonempty = 0;
        for (uint32_t i = 0; i < 4096; ++i) {
            total += ns8[i];
            if (ns8[i] < min_v) min_v = ns8[i];
            if (ns8[i] > max_v) max_v = ns8[i];
            if (ns8[i] > 0) ++nonempty;
        }
        std::cerr << "[BW] round-8 nslots8: total=" << total
                  << " mean=" << (double)total / 4096
                  << " min=" << min_v
                  << " max=" << max_v
                  << " nonempty=" << nonempty << "/4096"
                  << std::endl;
    } else {
        checkCudaErrorsBW(cudaStreamSynchronize(stream));
    }

    // BW_VALIDATE_ROUND1: for each round-1 output slot, decode its pack,
    // look up the source pair in round0trees, XOR their content, and compare
    // with the stored content. This is a direct check of digit_1's correctness
    // independent of all atomic-ordering noise.
    if (std::getenv("BW_VALIDATE_ROUND1")) {
        using bw::u32; using bw::slot; using bw::NBUCKETS; using bw::NSLOTS;
        using bw::RB8_NSLOTS;
        std::vector<slot> r0(4096 * RB8_NSLOTS);
        cudaMemcpyAsync(r0.data(), &device_eq->round0trees[0][0],
                        r0.size() * sizeof(slot), cudaMemcpyDeviceToHost, stream);
        std::vector<slot> r1(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r1.data(), &device_eq->trees[0][0][0],
                        r1.size() * sizeof(slot), cudaMemcpyDeviceToHost, stream);
        u32 hnslots1[NBUCKETS];
        cudaMemcpyAsync(hnslots1, &device_eq->edata.nslots[1], sizeof(hnslots1),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        u32 ok = 0, bad_content = 0, bad_hr = 0, checked = 0;
        for (u32 b = 0; b < NBUCKETS && checked < 500; ++b) {
            u32 ns = hnslots1[b]; if (ns > NSLOTS) ns = NSLOTS;
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                const slot& out = r1[b * NSLOTS + s];
                u32 pack = out.hash[6];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;   // encoder s0 field
                u32 src_p   =  pack        & 0x3FFu;   // encoder s1 field
                if (src_si >= RB8_NSLOTS || src_p >= RB8_NSLOTS) { ++bad_content; ++checked; continue; }
                const slot& a = r0[src_bid * RB8_NSLOTS + src_si];
                const slot& c = r0[src_bid * RB8_NSLOTS + src_p];
                u32 e0 = a.hash[0] ^ c.hash[0];  // xors[0]
                u32 e1 = a.hash[1] ^ c.hash[1];  // xors[1]
                u32 e2 = a.hash[2] ^ c.hash[2];  // xors[2]
                u32 e3 = a.hash[3] ^ c.hash[3];  // xors[3]
                u32 e4 = a.hash[4] ^ c.hash[4];  // xors[4]
                u32 e5 = a.hash[5] ^ c.hash[5];  // xors[5]
                // expected layout: hash[0..5] = {xors[1..5], xors[0]}
                bool content_ok =
                    out.hash[0] == e1 && out.hash[1] == e2 && out.hash[2] == e3 &&
                    out.hash[3] == e4 && out.hash[4] == e5 && out.hash[5] == e0;
                // hr collision: bits 20-27 of e0 (=xors[0]) must be zero.
                bool hr_ok = ((e0 >> 20) & 0xFFu) == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R1VAL] checked=%u ok=%u bad_content=%u bad_hr=%u\n",
                     checked, ok, bad_content, bad_hr);
    }

    // BW_VALIDATE_ROUND2: for each round-2 output slot, decode its pack,
    // look up the pair in round-1 (trees[0]), XOR their content, compare.
    if (std::getenv("BW_VALIDATE_ROUND2")) {
        using bw::u32; using bw::slot; using bw::slotsmall; using bw::slottiny;
        using bw::NBUCKETS; using bw::NSLOTS;
        std::vector<slot> r1(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r1.data(), &device_eq->trees[0][0][0],
                        r1.size() * sizeof(slot), cudaMemcpyDeviceToHost, stream);
        std::vector<uint8_t> r2buf(NBUCKETS * (NSLOTS * sizeof(slotsmall) + NSLOTS * sizeof(slottiny)));
        cudaMemcpyAsync(r2buf.data(), &device_eq->round2trees[0],
                        r2buf.size(), cudaMemcpyDeviceToHost, stream);
        u32 hnslots2[NBUCKETS], hnslots1[NBUCKETS];
        cudaMemcpyAsync(hnslots1, &device_eq->edata.nslots[1], sizeof(hnslots1),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hnslots2, &device_eq->edata.nslots[2], sizeof(hnslots2),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        const size_t r2stride = NSLOTS * sizeof(slotsmall) + NSLOTS * sizeof(slottiny);
        u32 ok = 0, bad_content = 0, bad_hr = 0, bad_range = 0, checked = 0;
        for (u32 b = 0; b < NBUCKETS && checked < 500; ++b) {
            u32 ns = hnslots2[b]; if (ns > NSLOTS) ns = NSLOTS;
            const slotsmall* sm = (const slotsmall*)(r2buf.data() + b * r2stride);
            const slottiny*  tn = (const slottiny*)(r2buf.data() + b * r2stride + NSLOTS * sizeof(slotsmall));
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                u32 pack = tn[s].hash[1];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;
                u32 src_p   =  pack        & 0x3FFu;
                u32 src_bucket_size = hnslots1[src_bid];
                if (src_bucket_size > NSLOTS) src_bucket_size = NSLOTS;
                if (src_si >= src_bucket_size || src_p >= src_bucket_size) {
                    ++bad_range; ++checked; continue;
                }
                const slot& a = r1[src_bid * NSLOTS + src_si];
                const slot& c = r1[src_bid * NSLOTS + src_p];
                // Expected: sm.hash[0..3] = XOR of a,c hash[0..3]; tn.hash[0] = XOR hash[4]
                bool content_ok =
                    sm[s].hash[0] == (a.hash[0] ^ c.hash[0]) &&
                    sm[s].hash[1] == (a.hash[1] ^ c.hash[1]) &&
                    sm[s].hash[2] == (a.hash[2] ^ c.hash[2]) &&
                    sm[s].hash[3] == (a.hash[3] ^ c.hash[3]) &&
                    tn[s].hash[0] == (a.hash[4] ^ c.hash[4]);
                // hr collision invariant: r1 hash[5] low byte matched, so XOR low byte = 0
                bool hr_ok = ((a.hash[5] ^ c.hash[5]) & 0xFFu) == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R2VAL] checked=%u ok=%u bad_content=%u bad_hr=%u bad_range=%u\n",
                     checked, ok, bad_content, bad_hr, bad_range);
    }

    // BW_VALIDATE_ROUND3: validate round-3 output against round-2 source slots.
    if (std::getenv("BW_VALIDATE_ROUND3")) {
        using bw::u32; using bw::slotsmall; using bw::slottiny;
        using bw::NBUCKETS; using bw::NSLOTS;
        const size_t rstride = NSLOTS * sizeof(slotsmall) + NSLOTS * sizeof(slottiny);
        std::vector<uint8_t> r2buf(NBUCKETS * rstride), r3buf(NBUCKETS * rstride);
        cudaMemcpyAsync(r2buf.data(), &device_eq->round2trees[0], r2buf.size(), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r3buf.data(), &device_eq->round3trees[0], r3buf.size(), cudaMemcpyDeviceToHost, stream);
        u32 hnslots2[NBUCKETS], hnslots3[NBUCKETS];
        cudaMemcpyAsync(hnslots2, &device_eq->edata.nslots[2], sizeof(hnslots2), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hnslots3, &device_eq->edata.nslots[3], sizeof(hnslots3), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        u32 ok = 0, bad_content = 0, bad_hr = 0, bad_range = 0, checked = 0;
        for (u32 b = 0; b < NBUCKETS && checked < 500; ++b) {
            u32 ns = hnslots3[b]; if (ns > NSLOTS) ns = NSLOTS;
            const slotsmall* sm = (const slotsmall*)(r3buf.data() + b * rstride);
            const slottiny*  tn = (const slottiny*)(r3buf.data() + b * rstride + NSLOTS * sizeof(slotsmall));
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                u32 pack = tn[s].hash[1];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;
                u32 src_p   =  pack        & 0x3FFu;
                u32 src_bs = hnslots2[src_bid]; if (src_bs > NSLOTS) src_bs = NSLOTS;
                if (src_si >= src_bs || src_p >= src_bs) { ++bad_range; ++checked; continue; }
                const slotsmall* src_sm = (const slotsmall*)(r2buf.data() + src_bid * rstride);
                const slottiny*  src_tn = (const slottiny*)(r2buf.data() + src_bid * rstride + NSLOTS * sizeof(slotsmall));
                // XOR sources
                u32 e0 = src_sm[src_si].hash[0] ^ src_sm[src_p].hash[0];
                u32 e1 = src_sm[src_si].hash[1] ^ src_sm[src_p].hash[1];
                u32 e2 = src_sm[src_si].hash[2] ^ src_sm[src_p].hash[2];
                u32 e3 = src_sm[src_si].hash[3] ^ src_sm[src_p].hash[3];
                u32 e4 = src_tn[src_si].hash[0] ^ src_tn[src_p].hash[0];
                u32 bexor = byte_perm_host(e0, e1, 0x2107);
                bool content_ok =
                    sm[s].hash[0] == e1 && sm[s].hash[1] == e2 &&
                    sm[s].hash[2] == e3 && sm[s].hash[3] == e4 &&
                    tn[s].hash[0] == bexor;
                // hr in digit_3 = bits 12-19 of r2.sm.h0, so (e0 >> 12) & 0xFF == 0
                bool hr_ok = ((e0 >> 12) & 0xFFu) == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R3VAL] checked=%u ok=%u bad_content=%u bad_hr=%u bad_range=%u\n",
                     checked, ok, bad_content, bad_hr, bad_range);
    }

    // BW_VALIDATE_ROUND4: round-4 output in treessmall[3] + round4bidandsids; input is round3trees.
    if (std::getenv("BW_VALIDATE_ROUND4")) {
        using bw::u32; using bw::slotsmall; using bw::slottiny;
        using bw::NBUCKETS; using bw::NSLOTS;
        const size_t r3stride = NSLOTS * sizeof(slotsmall) + NSLOTS * sizeof(slottiny);
        std::vector<uint8_t> r3buf(NBUCKETS * r3stride);
        cudaMemcpyAsync(r3buf.data(), &device_eq->round3trees[0], r3buf.size(), cudaMemcpyDeviceToHost, stream);
        std::vector<slotsmall> r4sm(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r4sm.data(), &device_eq->treessmall[3][0][0],
                        r4sm.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        std::vector<u32> r4pack(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r4pack.data(), &device_eq->round4bidandsids[0][0],
                        r4pack.size() * sizeof(u32), cudaMemcpyDeviceToHost, stream);
        u32 hnslots3[NBUCKETS], hnslots4[NBUCKETS];
        cudaMemcpyAsync(hnslots3, &device_eq->edata.nslots[3], sizeof(hnslots3), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hnslots4, &device_eq->edata.nslots[4], sizeof(hnslots4), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        u32 ok = 0, bad_content = 0, bad_hr = 0, bad_range = 0, checked = 0;
        for (u32 b = 0; b < NBUCKETS && checked < 500; ++b) {
            u32 ns = hnslots4[b]; if (ns > NSLOTS) ns = NSLOTS;
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                u32 pack = r4pack[b * NSLOTS + s];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;
                u32 src_p   =  pack        & 0x3FFu;
                u32 src_bs = hnslots3[src_bid]; if (src_bs > NSLOTS) src_bs = NSLOTS;
                if (src_si >= src_bs || src_p >= src_bs) { ++bad_range; ++checked; continue; }
                const slotsmall* src_sm = (const slotsmall*)(r3buf.data() + src_bid * r3stride);
                const slottiny*  src_tn = (const slottiny*)(r3buf.data() + src_bid * r3stride + NSLOTS * sizeof(slotsmall));
                // Expected: xors[0..3] = XOR of src_sm hash[0..3]
                u32 e0 = src_sm[src_si].hash[0] ^ src_sm[src_p].hash[0];
                u32 e1 = src_sm[src_si].hash[1] ^ src_sm[src_p].hash[1];
                u32 e2 = src_sm[src_si].hash[2] ^ src_sm[src_p].hash[2];
                u32 e3 = src_sm[src_si].hash[3] ^ src_sm[src_p].hash[3];
                const slotsmall& out = r4sm[b * NSLOTS + s];
                bool content_ok = out.hash[0] == e0 && out.hash[1] == e1 &&
                                  out.hash[2] == e2 && out.hash[3] == e3;
                // hr in digit_4 = src.tn.hash[0] & RESTMASK (which is bexor). After XOR, low 8 bits of bexor = 0.
                u32 hr_xor = (src_tn[src_si].hash[0] ^ src_tn[src_p].hash[0]) & 0xFFu;
                bool hr_ok = hr_xor == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R4VAL] checked=%u ok=%u bad_content=%u bad_hr=%u bad_range=%u\n",
                     checked, ok, bad_content, bad_hr, bad_range);
    }

    // BW_VALIDATE_ROUND5: input = treessmall[3], output = treessmall[2].
    // Round-5 output: {xors[1], xors[2], xors[3], pack}. pack is in hash[3].
    if (std::getenv("BW_VALIDATE_ROUND5")) {
        using bw::u32; using bw::slotsmall; using bw::NBUCKETS; using bw::NSLOTS;
        std::vector<slotsmall> r4(NBUCKETS * NSLOTS), r5(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r4.data(), &device_eq->treessmall[3][0][0],
                        r4.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r5.data(), &device_eq->treessmall[2][0][0],
                        r5.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        u32 hnslots4[NBUCKETS], hnslots5[NBUCKETS];
        cudaMemcpyAsync(hnslots4, &device_eq->edata.nslots[4], sizeof(hnslots4), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hnslots5, &device_eq->edata.nslots[5], sizeof(hnslots5), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        u32 ok = 0, bad_content = 0, bad_hr = 0, bad_range = 0, checked = 0;
        for (u32 b = 0; b < NBUCKETS && checked < 500; ++b) {
            u32 ns = hnslots5[b]; if (ns > NSLOTS) ns = NSLOTS;
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                u32 pack = r5[b * NSLOTS + s].hash[3];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;
                u32 src_p   =  pack        & 0x3FFu;
                u32 src_bs = hnslots4[src_bid]; if (src_bs > NSLOTS) src_bs = NSLOTS;
                if (src_si >= src_bs || src_p >= src_bs) { ++bad_range; ++checked; continue; }
                const slotsmall& a = r4[src_bid * NSLOTS + src_si];
                const slotsmall& c = r4[src_bid * NSLOTS + src_p];
                u32 e0 = a.hash[0] ^ c.hash[0];
                u32 e1 = a.hash[1] ^ c.hash[1];
                u32 e2 = a.hash[2] ^ c.hash[2];
                u32 e3 = a.hash[3] ^ c.hash[3];
                const slotsmall& out = r5[b * NSLOTS + s];
                bool content_ok = out.hash[0] == e1 && out.hash[1] == e2 && out.hash[2] == e3;
                // hr in digit_5: (tt.x >> 16) & RESTMASK -> (e0 >> 16) & 0xFF == 0
                bool hr_ok = ((e0 >> 16) & 0xFFu) == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R5VAL] checked=%u ok=%u bad_content=%u bad_hr=%u bad_range=%u\n",
                     checked, ok, bad_content, bad_hr, bad_range);
    }

    // BW_VALIDATE_ROUND6: input = treessmall[2], output = treessmall[0].
    // Round-6 output: {xors[1], xors[2], bexor, pack}. bexor = byte_perm(xors[0], xors[1], 0x1076).
    if (std::getenv("BW_VALIDATE_ROUND6")) {
        using bw::u32; using bw::slotsmall; using bw::NBUCKETS; using bw::NSLOTS;
        std::vector<slotsmall> r5(NBUCKETS * NSLOTS), r6(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r5.data(), &device_eq->treessmall[2][0][0],
                        r5.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r6.data(), &device_eq->treessmall[0][0][0],
                        r6.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        u32 hnslots5[NBUCKETS], hnslots6[NBUCKETS];
        cudaMemcpyAsync(hnslots5, &device_eq->edata.nslots[5], sizeof(hnslots5), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hnslots6, &device_eq->edata.nslots[6], sizeof(hnslots6), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        u32 ok = 0, bad_content = 0, bad_hr = 0, bad_range = 0, checked = 0;
        for (u32 b = 0; b < NBUCKETS && checked < 500; ++b) {
            u32 ns = hnslots6[b]; if (ns > NSLOTS) ns = NSLOTS;
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                u32 pack = r6[b * NSLOTS + s].hash[3];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;
                u32 src_p   =  pack        & 0x3FFu;
                u32 src_bs = hnslots5[src_bid]; if (src_bs > NSLOTS) src_bs = NSLOTS;
                if (src_si >= src_bs || src_p >= src_bs) { ++bad_range; ++checked; continue; }
                const slotsmall& a = r5[src_bid * NSLOTS + src_si];
                const slotsmall& c = r5[src_bid * NSLOTS + src_p];
                // Round-5 input: hash[0..2] are content (xors[1..3] from r5); hash[3] is pack (ignored).
                u32 e0 = a.hash[0] ^ c.hash[0];
                u32 e1 = a.hash[1] ^ c.hash[1];
                u32 e2 = a.hash[2] ^ c.hash[2];
                u32 bexor = byte_perm_host(e0, e1, 0x1076);
                const slotsmall& out = r6[b * NSLOTS + s];
                bool content_ok = out.hash[0] == e1 && out.hash[1] == e2 && out.hash[2] == bexor;
                bool hr_ok = ((e0 >> 16) & 0xFFu) == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R6VAL] checked=%u ok=%u bad_content=%u bad_hr=%u bad_range=%u\n",
                     checked, ok, bad_content, bad_hr, bad_range);
    }

    // BW_VALIDATE_ROUND7: input = treessmall[0], output = treessmall[1].
    // Round-7 output: {xors[0], xors[1], pack, 0}.
    if (std::getenv("BW_VALIDATE_ROUND7")) {
        using bw::u32; using bw::slotsmall; using bw::NBUCKETS; using bw::NSLOTS;
        std::vector<slotsmall> r6(NBUCKETS * NSLOTS), r7(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r6.data(), &device_eq->treessmall[0][0][0],
                        r6.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r7.data(), &device_eq->treessmall[1][0][0],
                        r7.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        u32 hnslots6[NBUCKETS], hnslots7[NBUCKETS];
        cudaMemcpyAsync(hnslots6, &device_eq->edata.nslots[6], sizeof(hnslots6), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hnslots7, &device_eq->edata.nslots[7], sizeof(hnslots7), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        u32 ok = 0, bad_content = 0, bad_hr = 0, bad_range = 0, checked = 0;
        for (u32 b = 0; b < NBUCKETS && checked < 500; ++b) {
            u32 ns = hnslots7[b]; if (ns > NSLOTS) ns = NSLOTS;
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                u32 pack = r7[b * NSLOTS + s].hash[2];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;
                u32 src_p   =  pack        & 0x3FFu;
                u32 src_bs = hnslots6[src_bid]; if (src_bs > NSLOTS) src_bs = NSLOTS;
                if (src_si >= src_bs || src_p >= src_bs) { ++bad_range; ++checked; continue; }
                const slotsmall& a = r6[src_bid * NSLOTS + src_si];
                const slotsmall& c = r6[src_bid * NSLOTS + src_p];
                u32 e0 = a.hash[0] ^ c.hash[0];
                u32 e1 = a.hash[1] ^ c.hash[1];
                u32 e2 = a.hash[2] ^ c.hash[2];
                const slotsmall& out = r7[b * NSLOTS + s];
                bool content_ok = out.hash[0] == e0 && out.hash[1] == e1 && out.hash[3] == 0;
                // hr in digit_7: (tt.z >> 12) & RESTMASK -> (e2 >> 12) & 0xFF == 0
                bool hr_ok = ((e2 >> 12) & 0xFFu) == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R7VAL] checked=%u ok=%u bad_content=%u bad_hr=%u bad_range=%u\n",
                     checked, ok, bad_content, bad_hr, bad_range);
    }

    // BW_VALIDATE_ROUND8: input = treessmall[1], output = treestiny[0].
    // Round-8 output: {xors[1], pack}. Uses nslots8 (4096 entries).
    if (std::getenv("BW_VALIDATE_ROUND8")) {
        using bw::u32; using bw::slotsmall; using bw::slottiny;
        using bw::NBUCKETS; using bw::NSLOTS; using bw::RB8_NSLOTS_LD;
        std::vector<slotsmall> r7(NBUCKETS * NSLOTS);
        std::vector<slottiny> r8(4096 * RB8_NSLOTS_LD);
        cudaMemcpyAsync(r7.data(), &device_eq->treessmall[1][0][0],
                        r7.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r8.data(), &device_eq->treestiny[0][0][0],
                        r8.size() * sizeof(slottiny), cudaMemcpyDeviceToHost, stream);
        u32 hnslots7[NBUCKETS], hn8[4096];
        cudaMemcpyAsync(hnslots7, &device_eq->edata.nslots[7], sizeof(hnslots7), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hn8, &device_eq->edata.nslots8, sizeof(hn8), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        u32 ok = 0, bad_content = 0, bad_hr = 0, bad_range = 0, checked = 0;
        for (u32 b = 0; b < 4096 && checked < 500; ++b) {
            u32 ns = hn8[b]; if (ns > RB8_NSLOTS_LD) ns = RB8_NSLOTS_LD;
            for (u32 s = 0; s < ns && checked < 500; ++s) {
                u32 pack = r8[b * RB8_NSLOTS_LD + s].hash[1];
                u32 src_bid = (pack >> 20) & 0xFFFu;
                u32 src_si  = (pack >> 10) & 0x3FFu;
                u32 src_p   =  pack        & 0x3FFu;
                u32 src_bs = hnslots7[src_bid]; if (src_bs > NSLOTS) src_bs = NSLOTS;
                if (src_si >= src_bs || src_p >= src_bs) { ++bad_range; ++checked; continue; }
                const slotsmall& a = r7[src_bid * NSLOTS + src_si];
                const slotsmall& c = r7[src_bid * NSLOTS + src_p];
                u32 e0 = a.hash[0] ^ c.hash[0];
                u32 e1 = a.hash[1] ^ c.hash[1];
                const slottiny& out = r8[b * RB8_NSLOTS_LD + s];
                bool content_ok = out.hash[0] == e1;
                // hr in digit_8: (tt.x >> 8) & RESTMASK -> (e0 >> 8) & 0xFF == 0
                bool hr_ok = ((e0 >> 8) & 0xFFu) == 0;
                if (!content_ok) ++bad_content;
                else if (!hr_ok) ++bad_hr;
                else ++ok;
                ++checked;
            }
        }
        std::fprintf(stderr, "[BW-R8VAL] checked=%u ok=%u bad_content=%u bad_hr=%u bad_range=%u\n",
                     checked, ok, bad_content, bad_hr, bad_range);
    }

    // BW_PACK_UNIQUENESS: across each round's output, count total valid slots
    // vs distinct pack values. If distinct < total, a kernel is writing the
    // same (bucket, si, p) tuple to multiple output slots — that would cause
    // subtree collapse in walk-back and explain 100% duped candidates.
    if (std::getenv("BW_PACK_UNIQUENESS")) {
        using bw::u32; using bw::slot; using bw::slotsmall; using bw::slottiny;
        using bw::NBUCKETS; using bw::NSLOTS; using bw::RB8_NSLOTS_LD;
        auto count_unique = [](std::vector<u32>& v) -> std::pair<size_t, size_t> {
            size_t total = v.size();
            std::sort(v.begin(), v.end());
            size_t distinct = 0;
            for (size_t k = 0; k < v.size(); ++k)
                if (k == 0 || v[k] != v[k-1]) ++distinct;
            return {total, distinct};
        };
        const size_t r23stride = NSLOTS * sizeof(slotsmall) + NSLOTS * sizeof(slottiny);

        // Fetch everything we need.
        u32 hnslots[9][NBUCKETS], hn8[4096];
        for (int r = 1; r <= 7; ++r)
            cudaMemcpyAsync(hnslots[r], &device_eq->edata.nslots[r], sizeof(hnslots[r]),
                            cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hn8, &device_eq->edata.nslots8, sizeof(hn8),
                        cudaMemcpyDeviceToHost, stream);
        std::vector<slot> r1buf(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r1buf.data(), &device_eq->trees[0][0][0],
                        r1buf.size() * sizeof(slot), cudaMemcpyDeviceToHost, stream);
        std::vector<uint8_t> r2buf(NBUCKETS * r23stride), r3buf(NBUCKETS * r23stride);
        cudaMemcpyAsync(r2buf.data(), &device_eq->round2trees[0], r2buf.size(),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r3buf.data(), &device_eq->round3trees[0], r3buf.size(),
                        cudaMemcpyDeviceToHost, stream);
        std::vector<u32> r4pack(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r4pack.data(), &device_eq->round4bidandsids[0][0],
                        r4pack.size() * sizeof(u32), cudaMemcpyDeviceToHost, stream);
        std::vector<slotsmall> r5(NBUCKETS * NSLOTS), r6(NBUCKETS * NSLOTS), r7(NBUCKETS * NSLOTS);
        cudaMemcpyAsync(r5.data(), &device_eq->treessmall[2][0][0],
                        r5.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r6.data(), &device_eq->treessmall[0][0][0],
                        r6.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r7.data(), &device_eq->treessmall[1][0][0],
                        r7.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        std::vector<slottiny> r8buf(4096 * RB8_NSLOTS_LD);
        cudaMemcpyAsync(r8buf.data(), &device_eq->treestiny[0][0][0],
                        r8buf.size() * sizeof(slottiny), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        auto gather_r23 = [&](const std::vector<uint8_t>& buf, int r) {
            std::vector<u32> v;
            for (u32 b = 0; b < NBUCKETS; ++b) {
                u32 ns = hnslots[r][b]; if (ns > NSLOTS) ns = NSLOTS;
                const slottiny* tn = (const slottiny*)(buf.data() + b * r23stride + NSLOTS * sizeof(slotsmall));
                for (u32 s = 0; s < ns; ++s) v.push_back(tn[s].hash[1]);
            }
            return v;
        };
        auto gather_sm = [&](const std::vector<slotsmall>& buf, int r, int field) {
            std::vector<u32> v;
            for (u32 b = 0; b < NBUCKETS; ++b) {
                u32 ns = hnslots[r][b]; if (ns > NSLOTS) ns = NSLOTS;
                for (u32 s = 0; s < ns; ++s) v.push_back(buf[b * NSLOTS + s].hash[field]);
            }
            return v;
        };

        struct R { int r; std::vector<u32> v; };
        std::vector<R> rounds;
        // Round 1: trees[0][b][s].hash[6]
        {
            std::vector<u32> v;
            for (u32 b = 0; b < NBUCKETS; ++b) {
                u32 ns = hnslots[1][b]; if (ns > NSLOTS) ns = NSLOTS;
                for (u32 s = 0; s < ns; ++s) v.push_back(r1buf[b * NSLOTS + s].hash[6]);
            }
            rounds.push_back({1, std::move(v)});
        }
        rounds.push_back({2, gather_r23(r2buf, 2)});
        rounds.push_back({3, gather_r23(r3buf, 3)});
        // Round 4: round4bidandsids[b][s]
        {
            std::vector<u32> v;
            for (u32 b = 0; b < NBUCKETS; ++b) {
                u32 ns = hnslots[4][b]; if (ns > NSLOTS) ns = NSLOTS;
                for (u32 s = 0; s < ns; ++s) v.push_back(r4pack[b * NSLOTS + s]);
            }
            rounds.push_back({4, std::move(v)});
        }
        rounds.push_back({5, gather_sm(r5, 5, 3)});
        rounds.push_back({6, gather_sm(r6, 6, 3)});
        rounds.push_back({7, gather_sm(r7, 7, 2)});
        // Round 8: treestiny[0][b][s].hash[1]
        {
            std::vector<u32> v;
            for (u32 b = 0; b < 4096; ++b) {
                u32 ns = hn8[b]; if (ns > RB8_NSLOTS_LD) ns = RB8_NSLOTS_LD;
                for (u32 s = 0; s < ns; ++s) v.push_back(r8buf[b * RB8_NSLOTS_LD + s].hash[1]);
            }
            rounds.push_back({8, std::move(v)});
        }

        for (auto& rd : rounds) {
            auto [total, distinct] = count_unique(rd.v);
            std::fprintf(stderr, "[BW-PACKUNIQ] round-%d total=%zu distinct=%zu dup=%zu (%.4f%%)\n",
                         rd.r, total, distinct, total - distinct,
                         100.0 * double(total - distinct) / double(total ? total : 1));
        }
    }

    // BW_HOST_WALK: do the entire tree walk-back on the host using validated
    // on-device tree data. If this produces CPU-verifying solutions but the
    // GPU-produced solutions don't, the bug is purely in digit_last_wdc.
    if (std::getenv("BW_HOST_WALK")) {
        using bw::u32; using bw::slot; using bw::slotsmall; using bw::slottiny;
        using bw::NBUCKETS; using bw::NSLOTS; using bw::RB8_NSLOTS; using bw::RB8_NSLOTS_LD;

        // Pull back every round's data.
        std::vector<slot> r0(4096 * RB8_NSLOTS);
        std::vector<slot> r1(NBUCKETS * NSLOTS);
        const size_t r23stride = NSLOTS * sizeof(slotsmall) + NSLOTS * sizeof(slottiny);
        std::vector<uint8_t> r2(NBUCKETS * r23stride), r3(NBUCKETS * r23stride);
        std::vector<slotsmall> r4sm(NBUCKETS * NSLOTS);
        std::vector<u32> r4pack(NBUCKETS * NSLOTS);
        std::vector<slotsmall> r5(NBUCKETS * NSLOTS), r6(NBUCKETS * NSLOTS), r7(NBUCKETS * NSLOTS);
        std::vector<slottiny> r8(4096 * RB8_NSLOTS_LD);
        u32 hn8[4096];

        cudaMemcpyAsync(r0.data(), &device_eq->round0trees[0][0], r0.size() * sizeof(slot), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r1.data(), &device_eq->trees[0][0][0], r1.size() * sizeof(slot), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r2.data(), &device_eq->round2trees[0], r2.size(), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r3.data(), &device_eq->round3trees[0], r3.size(), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r4sm.data(), &device_eq->treessmall[3][0][0], r4sm.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r4pack.data(), &device_eq->round4bidandsids[0][0], r4pack.size() * sizeof(u32), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r5.data(), &device_eq->treessmall[2][0][0], r5.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r6.data(), &device_eq->treessmall[0][0][0], r6.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r7.data(), &device_eq->treessmall[1][0][0], r7.size() * sizeof(slotsmall), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(r8.data(), &device_eq->treestiny[0][0][0], r8.size() * sizeof(slottiny), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(hn8, &device_eq->edata.nslots8, sizeof(hn8), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        auto get_bid = [](u32 p) { return (p >> 20) & 0xFFFu; };
        auto get_s1  = [](u32 p) { return (p >> 10) & 0x3FFu; };  // encoder s0 field (was "me")
        auto get_s0  = [](u32 p) { return  p        & 0x3FFu; };  // encoder s1 field (was "other")

        // Recursive walk. Given a pack and its level, produce 2^level leaf indices.
        // Level 9 = top. Level 0 = single leaf.
        // Given a pack at `lvl` (1..9), fetch the child pack that the slot at
        // (bid, slot_idx) points to in the round-(lvl-1) data. lvl=2 returns the
        // round-1 slot's hash[6] which is a round-0 pack; lvl=1 is handled by
        // caller (reads r0 leaf directly).
        auto fetch_child = [&](int lvl, u32 bid, u32 slot_idx, u32& out_pack) -> bool {
            if (bid >= NBUCKETS) return false;
            switch (lvl) {
                case 9:
                    if (slot_idx >= RB8_NSLOTS_LD) return false;
                    out_pack = r8[bid * RB8_NSLOTS_LD + slot_idx].hash[1]; return true;
                case 8:
                    if (slot_idx >= NSLOTS) return false;
                    out_pack = r7[bid * NSLOTS + slot_idx].hash[2]; return true;
                case 7:
                    if (slot_idx >= NSLOTS) return false;
                    out_pack = r6[bid * NSLOTS + slot_idx].hash[3]; return true;
                case 6:
                    if (slot_idx >= NSLOTS) return false;
                    out_pack = r5[bid * NSLOTS + slot_idx].hash[3]; return true;
                case 5:
                    if (slot_idx >= NSLOTS) return false;
                    out_pack = r4pack[bid * NSLOTS + slot_idx]; return true;
                case 4: {
                    if (slot_idx >= NSLOTS) return false;
                    const slottiny* tn = (const slottiny*)(r3.data() + bid * r23stride + NSLOTS * sizeof(slotsmall));
                    out_pack = tn[slot_idx].hash[1]; return true;
                }
                case 3: {
                    if (slot_idx >= NSLOTS) return false;
                    const slottiny* tn = (const slottiny*)(r2.data() + bid * r23stride + NSLOTS * sizeof(slotsmall));
                    out_pack = tn[slot_idx].hash[1]; return true;
                }
                case 2:
                    if (slot_idx >= NSLOTS) return false;
                    out_pack = r1[bid * NSLOTS + slot_idx].hash[6]; return true;
                default: return false;
            }
        };

        // Iterative walk: BFS-style. top_pack is a level-8 pack (stored in round-8
        // output). It encodes 2 round-7 positions. Collect 256 leaves beneath it.
        auto walk = [&](u32 top_pack, uint32_t* out_leaves) -> bool {
            std::vector<u32> current, next;
            current.push_back(top_pack);
            // At lvl iteration, `current` holds level-lvl packs. Each pack points to
            // 2 children; we read their round-(lvl-1) slots' pack fields to get
            // level-(lvl-1) packs in next. After lvl=2, next holds level-1 packs.
            for (int lvl = 8; lvl >= 2; --lvl) {
                next.clear();
                for (u32 pack : current) {
                    u32 bid = get_bid(pack);
                    u32 si  = get_s1(pack);
                    u32 p   = get_s0(pack);
                    u32 c1, c2;
                    if (!fetch_child(lvl, bid, si, c1)) return false;
                    if (!fetch_child(lvl, bid, p,  c2)) return false;
                    next.push_back(c1);
                    next.push_back(c2);
                }
                current.swap(next);
            }
            // current now has 128 level-1 packs. Each encodes 2 round-0 leaves.
            if (current.size() != 128) return false;
            for (size_t k = 0; k < current.size(); ++k) {
                u32 pack = current[k];
                u32 bid = get_bid(pack);
                u32 si  = get_s1(pack);
                u32 p   = get_s0(pack);
                if (bid >= 4096 || si >= RB8_NSLOTS || p >= RB8_NSLOTS) return false;
                out_leaves[2*k + 0] = r0[bid * RB8_NSLOTS + si].hash[7];
                out_leaves[2*k + 1] = r0[bid * RB8_NSLOTS + p].hash[7];
            }
            return true;
        };

        // Find pairs in round-8 that share hash[0] (content). Walk them, verify.
        u32 tried = 0, walk_fail = 0, walk_ok = 0, verify_ok = 0, verify_fail = 0, dup_filtered = 0;
        const u32 tried_max = 5000;
        for (u32 b = 0; b < 4096 && tried < tried_max; ++b) {
            u32 ns = hn8[b]; if (ns > RB8_NSLOTS_LD) ns = RB8_NSLOTS_LD;
            for (u32 i = 0; i < ns && tried < tried_max; ++i) {
                for (u32 j = i + 1; j < ns && tried < tried_max; ++j) {
                    if (r8[b * RB8_NSLOTS_LD + i].hash[0] != r8[b * RB8_NSLOTS_LD + j].hash[0]) continue;
                    u32 pi = r8[b * RB8_NSLOTS_LD + i].hash[1];
                    u32 pj = r8[b * RB8_NSLOTS_LD + j].hash[1];
                    std::vector<uint32_t> leaves(512, 0);
                    bool ok1 = walk(pi, leaves.data());
                    bool ok2 = walk(pj, leaves.data() + 256);
                    if (!ok1 || !ok2) { ++walk_fail; ++tried; continue; }
                    ++walk_ok;
                    // Skip candidates with duplicate leaves (GPU's CHECK_DUP filter).
                    {
                        // Check left half (from pi) and right half (from pj) separately.
                        auto distinct_count = [](const uint32_t* arr, size_t n) {
                            std::vector<uint32_t> s(arr, arr + n);
                            std::sort(s.begin(), s.end());
                            size_t d = 0;
                            for (size_t k = 0; k < s.size(); ++k)
                                if (k == 0 || s[k] != s[k-1]) ++d;
                            return d;
                        };
                        size_t left_distinct  = distinct_count(leaves.data(),       256);
                        size_t right_distinct = distinct_count(leaves.data() + 256, 256);
                        size_t total_distinct = distinct_count(leaves.data(),       512);
                        if (false /*tried < 5*/) {
                            // Inspect the level-7 packs that pi and pj point to.
                            auto get_bid_fn = [](u32 p) { return (p >> 20) & 0xFFFu; };
                            auto get_s1_fn  = [](u32 p) { return (p >> 10) & 0x3FFu; };
                            auto get_s0_fn  = [](u32 p) { return  p        & 0x3FFu; };
                            u32 pi_bid = get_bid_fn(pi), pi_si = get_s1_fn(pi), pi_p = get_s0_fn(pi);
                            u32 pj_bid = get_bid_fn(pj), pj_si = get_s1_fn(pj), pj_p = get_s0_fn(pj);
                            u32 pi_c1 = r7[pi_bid * NSLOTS + pi_si].hash[2];
                            u32 pi_c2 = r7[pi_bid * NSLOTS + pi_p ].hash[2];
                            u32 pj_c1 = r7[pj_bid * NSLOTS + pj_si].hash[2];
                            u32 pj_c2 = r7[pj_bid * NSLOTS + pj_p ].hash[2];
                            std::fprintf(stderr,
                                "  pi_decode: bid=%u si=%u p=%u   r7.hash[2]: c1=%08x c2=%08x\n"
                                "  pj_decode: bid=%u si=%u p=%u   r7.hash[2]: c1=%08x c2=%08x\n",
                                pi_bid, pi_si, pi_p, pi_c1, pi_c2,
                                pj_bid, pj_si, pj_p, pj_c1, pj_c2);
                            std::fprintf(stderr, "[BW-HOSTWALK-DBG] b=%u i=%u j=%u pi=%08x pj=%08x left_d=%zu right_d=%zu total_d=%zu\n",
                                         b, i, j, pi, pj, left_distinct, right_distinct, total_distinct);
                            // Also dump first few leaves of each half
                            std::fprintf(stderr, "  left[0..7]:");
                            for (int k = 0; k < 8; ++k) std::fprintf(stderr, " %x", leaves[k]);
                            std::fprintf(stderr, "\n  right[0..7]:");
                            for (int k = 0; k < 8; ++k) std::fprintf(stderr, " %x", leaves[256 + k]);
                            std::fprintf(stderr, "\n");
                        }
                        if (total_distinct != 512) { ++dup_filtered; ++tried; continue; }
                    }
                    // Sort pairs like verify expects.
                    for (uint32_t level = 0; level < 9; ++level)
                        for (uint32_t k = 0; k < (1u << 9); k += (2u << level))
                            sort_pair_bw(&leaves[k], 1u << level);
                    bw_xorfail_level = -1;
                    int vrc = bw_verify_solution(leaves.data(), header, header_len, nonce, nonce_len);
                    if (vrc == BW_POW_OK) ++verify_ok;
                    else {
                        ++verify_fail;
                        std::fprintf(stderr, "[BW-HOSTWALK] fail vrc=%d xorlvl=%d pair=(b=%u,i=%u,j=%u)\n",
                                     vrc, bw_xorfail_level, b, i, j);
                    }
                    ++tried;
                }
            }
        }
        std::fprintf(stderr, "[BW-HOSTWALK] tried=%u walk_ok=%u walk_fail=%u dup=%u verify_ok=%u verify_fail=%u\n",
                     tried, walk_ok, walk_fail, dup_filtered, verify_ok, verify_fail);
        std::fflush(stderr);
    }

    // EQUI_DUMP_SETXOR: per-round set-XOR of all valid slot contents. Matches
    // the diagnostic in cuda_djezo — output must be byte-identical if the two
    // solvers are producing the same hash tables.
    static const bool dump_setxor = std::getenv("EQUI_DUMP_SETXOR") != nullptr;
    if (dump_setxor) {
        using bw::u32; using bw::NBUCKETS; using bw::NSLOTS; using bw::RB8_NSLOTS_LD;
        using bw::slotsmall; using bw::slottiny;

        u32 hnslots[8][NBUCKETS];
        for (int r = 1; r <= 7; ++r)
            cudaMemcpyAsync(hnslots[r], &device_eq->edata.nslots[r],
                            sizeof(hnslots[r]), cudaMemcpyDeviceToHost, stream);
        u32 hn8[4096];
        cudaMemcpyAsync(hn8, &device_eq->edata.nslots8, sizeof(hn8),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        using bw::slot; using bw::RB8_NSLOTS;

        // round-0: round0trees[4096][RB8_NSLOTS] (slot = 8 u32s); count nslots0[4096]
        {
            u32 hn0[4096];
            cudaMemcpyAsync(hn0, &device_eq->edata.nslots0, sizeof(hn0),
                            cudaMemcpyDeviceToHost, stream);
            std::vector<slot> buf(4096 * RB8_NSLOTS);
            cudaMemcpyAsync(buf.data(), &device_eq->round0trees[0][0],
                            buf.size() * sizeof(slot),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            u32 sx[8] = {0};
            for (u32 b = 0; b < 4096; ++b) {
                u32 ns = hn0[b]; if (ns > RB8_NSLOTS) ns = RB8_NSLOTS;
                for (u32 s = 0; s < ns; ++s) {
                    const slot& x = buf[b * RB8_NSLOTS + s];
                    for (int i = 0; i < 8; ++i) sx[i] ^= x.hash[i];
                }
            }
            std::fprintf(stderr, "[BW] round-0 xor: sl={%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x}\n",
                         sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6], sx[7]);
        }

        // round-1: trees[0][NBUCKETS][NSLOTS] (slot = 8 u32s); count nslots[1][NB]
        // Only XOR buckets safely below NSLOTS (non-overflow) for a stable signature.
        // hash[6] is pack (order-dependent), hash[7] is always 0.
        const u32 safe_cap = NSLOTS - 10;
        {
            std::vector<slot> buf(NBUCKETS * NSLOTS);
            cudaMemcpyAsync(buf.data(), &device_eq->trees[0][0][0],
                            buf.size() * sizeof(slot),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            u32 sx[8] = {0};
            u32 included = 0, overflowed = 0;
            for (u32 b = 0; b < NBUCKETS; ++b) {
                u32 ns = hnslots[1][b];
                if (ns > safe_cap) { ++overflowed; continue; }
                ++included;
                for (u32 s = 0; s < ns; ++s) {
                    const slot& x = buf[b * NSLOTS + s];
                    for (int i = 0; i < 8; ++i) sx[i] ^= x.hash[i];
                }
            }
            std::fprintf(stderr, "[BW] round-1 xor(safe %u/%u skip=%u): content=[%08x %08x %08x %08x %08x %08x] pack=%08x z=%08x\n",
                         included, NBUCKETS, overflowed, sx[0], sx[1], sx[2], sx[3], sx[4], sx[5], sx[6], sx[7]);
        }

        const size_t r23stride = NSLOTS * sizeof(slotsmall) + NSLOTS * sizeof(slottiny);
        auto xor_r23 = [&](const void* dev_base, int r) {
            std::vector<uint8_t> buf(NBUCKETS * r23stride);
            cudaMemcpyAsync(buf.data(), dev_base, buf.size(),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            u32 sx[4] = {0,0,0,0}, tx_content = 0, tx_pack = 0;
            u32 included = 0, overflowed = 0;
            for (u32 b = 0; b < NBUCKETS; ++b) {
                u32 ns = hnslots[r][b];
                if (ns > safe_cap) { ++overflowed; continue; }
                ++included;
                const u32* sm_ptr = (const u32*)(buf.data() + b * r23stride);
                const u32* tn_ptr = (const u32*)(buf.data() + b * r23stride + NSLOTS * sizeof(slotsmall));
                for (u32 s = 0; s < ns; ++s) {
                    sx[0] ^= sm_ptr[s*4+0]; sx[1] ^= sm_ptr[s*4+1];
                    sx[2] ^= sm_ptr[s*4+2]; sx[3] ^= sm_ptr[s*4+3];
                    tx_content ^= tn_ptr[s*2+0];
                    tx_pack    ^= tn_ptr[s*2+1];
                }
            }
            std::fprintf(stderr, "[BW] round-%d xor(safe %u/%u skip=%u): sm={%08x,%08x,%08x,%08x} tn_content=%08x tn_pack=%08x\n",
                         r, included, NBUCKETS, overflowed, sx[0], sx[1], sx[2], sx[3], tx_content, tx_pack);
        };
        xor_r23(&device_eq->round2trees[0], 2);
        xor_r23(&device_eq->round3trees[0], 3);

        auto xor_sm = [&](int ts_idx, int r) {
            std::vector<slotsmall> buf(NBUCKETS * NSLOTS);
            cudaMemcpyAsync(buf.data(), &device_eq->treessmall[ts_idx][0][0],
                            buf.size() * sizeof(slotsmall),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            u32 sx[4] = {0,0,0,0};
            u32 included = 0, overflowed = 0;
            for (u32 b = 0; b < NBUCKETS; ++b) {
                u32 ns = hnslots[r][b];
                if (ns > safe_cap) { ++overflowed; continue; }
                ++included;
                for (u32 s = 0; s < ns; ++s) {
                    const slotsmall& x = buf[b * NSLOTS + s];
                    sx[0] ^= x.hash[0]; sx[1] ^= x.hash[1];
                    sx[2] ^= x.hash[2]; sx[3] ^= x.hash[3];
                }
            }
            std::fprintf(stderr, "[BW] round-%d xor(safe %u/%u skip=%u): sm={%08x,%08x,%08x,%08x}\n",
                         r, included, NBUCKETS, overflowed, sx[0], sx[1], sx[2], sx[3]);
        };
        xor_sm(0, 4); xor_sm(1, 5); xor_sm(2, 6); xor_sm(3, 7);

        {
            std::vector<slottiny> buf(4096 * RB8_NSLOTS_LD);
            cudaMemcpyAsync(buf.data(), &device_eq->treestiny[0][0][0],
                            buf.size() * sizeof(slottiny),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            const u32 r8_safe_cap = RB8_NSLOTS_LD - 10;
            u32 tx_content = 0, tx_pack = 0;
            u32 included = 0, overflowed = 0;
            for (u32 b = 0; b < 4096; ++b) {
                u32 ns = hn8[b];
                if (ns > r8_safe_cap) { ++overflowed; continue; }
                ++included;
                for (u32 s = 0; s < ns; ++s) {
                    const slottiny& x = buf[b * RB8_NSLOTS_LD + s];
                    tx_content ^= x.hash[0];
                    tx_pack    ^= x.hash[1];
                }
            }
            std::fprintf(stderr, "[BW] round-8 xor(safe %u/4096 skip=%u): tn_content=%08x tn_pack=%08x\n",
                         included, overflowed, tx_content, tx_pack);
        }
    }

    // Slow path: explicit copy of solutions container from device to host.
    {
        cudaError_t err = cudaMemcpyAsync(pinned_sols, &device_eq->edata.srealcont,
                                           sizeof(SolContainer), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            cudaGetLastError();
            hashdonef();
            return;
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            cudaGetLastError();
            hashdonef();
            return;
        }
    }

post_process:
    // Solutions now live in pinned_sols (either via graph or slow-path memcpy).
    SolContainer& cont = *(SolContainer*)pinned_sols;

    // Post-process solutions (matches djezo's equi_miner.cu:2109-2126)
    static const bool verify_log = std::getenv("BW_VERIFY_LOG") != nullptr;
    uint32_t emitted = 0, rejected_dup = 0, rejected_verify[4] = {0,0,0,0};
    uint32_t xorfail_by_level[10] = {0};
    for (uint32_t s = 0; s < cont.nsols && s < MAXREALSOLS_HOST; ++s) {
        if (duped_bw(cont.sols[s])) { ++rejected_dup; continue; }
        for (uint32_t level = 0; level < 9; ++level)
            for (uint32_t i = 0; i < (1u << 9); i += (2u << level))
                sort_pair_bw(&cont.sols[s][i], 1u << level);

        bw_xorfail_level = -1;
        int vrc = bw_verify_solution(cont.sols[s], header, header_len, nonce, nonce_len);
        if (vrc != BW_POW_OK) {
            if (vrc >= 0 && vrc < 4) ++rejected_verify[vrc];
            if (vrc == BW_POW_NONZERO_XOR && bw_xorfail_level >= 0 && bw_xorfail_level <= 9)
                ++xorfail_by_level[bw_xorfail_level];
            // Dump first failing solution's indices for hand analysis
            static bool dumped_one = false;
            if (!dumped_one && std::getenv("BW_DUMP_FAIL_SOL")) {
                dumped_one = true;
                std::fprintf(stderr, "[BW-FAILSOL] xorfail_level=%d vrc=%d\n", bw_xorfail_level, vrc);
                for (uint32_t i = 0; i < 512; ++i) {
                    std::fprintf(stderr, "%08x%c", cont.sols[s][i],
                                 ((i + 1) % 8 == 0) ? '\n' : ' ');
                }
            }
            continue;
        }

        std::vector<uint32_t> idx(bw::PROOFSIZE);
        for (uint32_t i = 0; i < bw::PROOFSIZE; ++i)
            idx[i] = cont.sols[s][i];
        solutionf(idx, 20 /*DIGITBITS*/, nullptr);
        ++emitted;
    }
    if (verify_log && cont.nsols > 0) {
        std::fprintf(stderr,
            "[BW-VERIFY] nsols=%u dup=%u ordfail=%u xorfail=%u ok=%u "
            "xorfail_by_level={%u %u %u %u %u %u %u %u %u}\n",
            cont.nsols, rejected_dup, rejected_verify[BW_POW_OUT_OF_ORDER],
            rejected_verify[BW_POW_NONZERO_XOR], emitted,
            xorfail_by_level[1], xorfail_by_level[2], xorfail_by_level[3],
            xorfail_by_level[4], xorfail_by_level[5], xorfail_by_level[6],
            xorfail_by_level[7], xorfail_by_level[8], xorfail_by_level[9]);
    }

    hashdonef();
}

// ===== cuda_blackwell wrapper =====

cuda_blackwell::cuda_blackwell(int /*platf_id*/, int dev_id) : threadsperblock(0), blocks(0), device_id(dev_id), context(nullptr) {
    getinfo(0, dev_id, m_gpu_name, m_sm_count, m_version);

    int major = 0, minor = 0;
    auto n = m_version.find('.');
    if (n != std::string::npos) {
        major = std::atoi(m_version.substr(0, n).c_str());
        minor = std::atoi(m_version.substr(n + 1).c_str());
    }
    if (major < 9) {
        throw std::runtime_error(
            "cuda_blackwell solver requires SM 9.0+ (Hopper or Blackwell). "
            "Use -cv 0 (djezo) or -cv 1 (tromp) for older GPUs.");
    }
    (void)minor;
}

std::string cuda_blackwell::getdevinfo() {
    return m_gpu_name + " (#" + std::to_string(device_id) + ") [BLACKWELL phase0]";
}

int cuda_blackwell::getcount() {
    int n = 0;
    auto err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return 0;
    return n;
}

void cuda_blackwell::getinfo(int /*platf_id*/, int d_id, std::string& gpu_name, int& sm_count, std::string& version) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, d_id) != cudaSuccess)
        throw std::runtime_error("cuda_blackwell: cudaGetDeviceProperties failed");
    gpu_name = p.name;
    sm_count = p.multiProcessorCount;
    version = std::to_string(p.major) + "." + std::to_string(p.minor);
}

void cuda_blackwell::start(cuda_blackwell& device_context) {
    device_context.context = new eq_cuda_context_blackwell(device_context.device_id);
}

void cuda_blackwell::stop(cuda_blackwell& device_context) {
    delete device_context.context;
    device_context.context = nullptr;
}

void cuda_blackwell::solve(const char* header, unsigned int header_len,
                           const char* nonce, unsigned int nonce_len,
                           std::function<bool()> cancelf,
                           std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
                           std::function<void(void)> hashdonef,
                           cuda_blackwell& device_context) {
    if (device_context.context) {
        device_context.context->solve(header, header_len, nonce, nonce_len,
                                       cancelf, solutionf, hashdonef);
    } else {
        hashdonef();
    }
}
