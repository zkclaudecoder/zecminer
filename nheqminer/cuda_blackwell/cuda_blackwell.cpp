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
#include "cuda_runtime.h"

#include "cuda_blackwell.hpp"
#include "eq_context.hpp"
#include "../cpu_tromp/blake2/blake2.h"

// Match djezo/bw Equihash parameters (MUST stay in sync with bucket_layout.hpp)
static constexpr uint32_t WN_ = 200;
static constexpr uint32_t WK_ = 9;
static constexpr uint32_t HASHOUT_ = (512 / WN_) * WN_ / 8;  // 50

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

// ===== eq_cuda_context_blackwell =====

eq_cuda_context_blackwell_interface::~eq_cuda_context_blackwell_interface() = default;

eq_cuda_context_blackwell::eq_cuda_context_blackwell(int id) : device_id(id), device_eq(nullptr), stream(0) {
    checkCudaErrorsBW(cudaSetDevice(device_id));
    checkCudaErrorsBW(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    checkCudaErrorsBW(cudaStreamCreate(&stream));

    if (cudaMalloc((void**)&device_eq, sizeof(bw::equi_state)) != cudaSuccess)
        throw std::runtime_error("CUDA(blackwell): failed to alloc equi_state");
}

eq_cuda_context_blackwell::~eq_cuda_context_blackwell() {
    if (device_eq) cudaFree(device_eq);
    if (stream) cudaStreamDestroy(stream);
}

void eq_cuda_context_blackwell::solve(
    const char* header, unsigned int header_len,
    const char* nonce, unsigned int nonce_len,
    std::function<bool()> /*cancelf*/,
    std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> /*solutionf*/,
    std::function<void(void)> hashdonef)
{
    // Phase 0: run digit_first + digit_1 and report bucket stats. No solutions
    // are emitted yet (digit_2..digit_8 and digit_last_wdc still unimplemented).
    // Use DJEZO_BW_VERBOSE=1 in env to see the per-iteration stats for debugging.

    blake2b_state ctx;
    bw_setheader(&ctx, header, header_len, nonce, nonce_len);

    checkCudaErrorsBW(cudaMemcpyAsync(&device_eq->blake_h, &ctx.h,
                                       sizeof(uint64_t) * 8,
                                       cudaMemcpyHostToDevice, stream));
    checkCudaErrorsBW(cudaMemsetAsync(&device_eq->edata, 0,
                                       sizeof(device_eq->edata), stream));

    bw::launch_digit_first(device_eq, /*nonce_suffix=*/0, stream);
    bw::launch_digit_1(device_eq, stream);
    bw::launch_digit_2(device_eq, stream);
    bw::launch_digit_3(device_eq, stream);
    bw::launch_digit_4(device_eq, stream);
    bw::launch_digit_5(device_eq, stream);
    bw::launch_digit_6(device_eq, stream);
    bw::launch_digit_7(device_eq, stream);
    bw::launch_digit_8(device_eq, stream);

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
