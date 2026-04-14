#pragma once
//
// Host-side eq_cuda_context for the cuda_blackwell solver.
// Parallels cuda_djezo/eqcuda.hpp.
//
// PHASE 0: minimal — just allocate the equi_state on device, expose solve()
// that runs digit_first + digit_1, and free.
//
// LATER: full 9-round pipeline with persistent kernel.

#include <functional>
#include <vector>
#include <stdint.h>
#include <stdexcept>
#include <string>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_blackwell.hpp"
#include "bucket_layout.hpp"

#ifdef WIN32
#define _SNPRINTF _snprintf
#else
#include <stdio.h>
#define _SNPRINTF snprintf
#endif

#define checkCudaErrorsBW(call)                                          \
do {                                                                     \
    cudaError_t err = call;                                              \
    if (cudaSuccess != err) {                                            \
        char errorBuff[512];                                             \
        _SNPRINTF(errorBuff, sizeof(errorBuff) - 1,                      \
            "CUDA(blackwell) error '%s' in func '%s' line %d",           \
            cudaGetErrorString(err), __FUNCTION__, __LINE__);            \
        throw std::runtime_error(errorBuff);                             \
    }                                                                    \
} while (0)

namespace bw {

// Forward declaration of host-launchable kernels (defined in digit_kernels.cu)
// `header` is the prepared blake2b mid-state; `nonce` is full 32-byte nonce.
void launch_digit_first(equi_state* device_eq, uint32_t nonce, cudaStream_t stream);
void launch_digit_1(equi_state* device_eq, cudaStream_t stream);

} // namespace bw

// The interface that MinerFactory binds to. Mirrors djezo's eq_cuda_context_interface.
struct eq_cuda_context_blackwell_interface {
    virtual ~eq_cuda_context_blackwell_interface();
    virtual void solve(const char* header, unsigned int header_len,
                       const char* nonce, unsigned int nonce_len,
                       std::function<bool()> cancelf,
                       std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
                       std::function<void(void)> hashdonef) = 0;
};

struct eq_cuda_context_blackwell : public eq_cuda_context_blackwell_interface {
    int device_id;
    bw::equi_state* device_eq;
    cudaStream_t stream;

    explicit eq_cuda_context_blackwell(int id);
    ~eq_cuda_context_blackwell() override;

    void solve(const char* header, unsigned int header_len,
               const char* nonce, unsigned int nonce_len,
               std::function<bool()> cancelf,
               std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
               std::function<void(void)> hashdonef) override;
};
