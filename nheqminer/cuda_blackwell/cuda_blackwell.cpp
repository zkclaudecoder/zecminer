// cuda_blackwell.cpp — solver wrapper class for MinerFactory.
//
// Phase 0: minimal — context lifecycle and stub solve() that runs digit_first
// and digit_1 only. Returns no solutions yet.

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include "cuda_runtime.h"

#include "cuda_blackwell.hpp"
#include "eq_context.hpp"

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
    const char* /*header*/, unsigned int /*header_len*/,
    const char* /*nonce*/, unsigned int /*nonce_len*/,
    std::function<bool()> /*cancelf*/,
    std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> /*solutionf*/,
    std::function<void(void)> hashdonef)
{
    // TODO(phase0): implement the actual solve loop:
    //   1. setheader(...) on host  -> blake_h state
    //   2. cudaMemcpy blake_h -> device_eq->blake_h
    //   3. cudaMemset device_eq->edata
    //   4. launch_digit_first(device_eq, 0, stream)
    //   5. launch_digit_1(device_eq, stream)        // currently a no-op
    //   6. cudaStreamSynchronize(stream)
    //
    // Phase 1+ adds: digit_2..digit_8, digit_last_wdc, solution extraction,
    //               solutionf() callback per valid solution found.
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
