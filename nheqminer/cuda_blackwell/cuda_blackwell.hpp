#pragma once
//
// Solver wrapper that MinerFactory instantiates. Parallels cuda_djezo.hpp.
//
// PHASE 0 SCOPE: The solver is registered in MinerFactory but currently only
// runs digit_first + digit_1 (for the byte-for-byte comparison test against
// djezo). Calling solve() will return zero solutions until Phase 1 lands the
// remaining rounds.

#include <functional>
#include <string>
#include <vector>
#include <stdint.h>

#ifdef _LIB
#define DLL_CUDA_BLACKWELL __declspec(dllexport)
#else
#define DLL_CUDA_BLACKWELL
#endif

struct eq_cuda_context_blackwell_interface;

struct DLL_CUDA_BLACKWELL cuda_blackwell {
    int threadsperblock;
    int blocks;
    int device_id;
    eq_cuda_context_blackwell_interface* context;

    cuda_blackwell(int platf_id, int dev_id);

    std::string getdevinfo();

    static int getcount();

    static void getinfo(int platf_id, int d_id, std::string& gpu_name,
                        int& sm_count, std::string& version);

    static void start(cuda_blackwell& device_context);
    static void stop(cuda_blackwell& device_context);

    static void solve(const char* tequihash_header,
                      unsigned int tequihash_header_len,
                      const char* nonce,
                      unsigned int nonce_len,
                      std::function<bool()> cancelf,
                      std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf,
                      std::function<void(void)> hashdonef,
                      cuda_blackwell& device_context);

    std::string getname() { return "CUDA-BLACKWELL"; }

private:
    std::string m_gpu_name;
    std::string m_version;
    int m_sm_count;
};
