#pragma once
//
// cp.async + mbarrier helpers used by the producer/consumer warp-specialized
// digit kernels.
//
// Wraps the libcudacxx primitives so the kernel code stays readable.
//
// References:
//  - CUDA Programming Guide §7.27 "Asynchronous Data Copies"
//  - <cuda/pipeline> in libcudacxx
//  - PTX ISA cp.async, cp.async.bulk, mbarrier.*

#include <cuda/pipeline>
#include <cuda/barrier>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace bw {
namespace cg = cooperative_groups;

// ---- Pipeline depth ----
// 2 stages = double-buffered (one being filled, one being drained).
// 3 stages = small win on long-latency loads at cost of more shared memory.
// Tune in Phase 3.
static constexpr int kPipelineDepth = 2;

// ---- cp.async-style group memcpy ----
//
// Copies `nbytes` from `src` (global) to `dst` (shared) cooperatively across
// all threads in `group`. Returns immediately; caller MUST wait on the
// associated barrier before reading `dst`.
//
// On Blackwell, the compiler picks cp.async or cp.async.bulk depending on
// alignment / size. We don't need to choose manually.
template <typename Group>
__device__ __forceinline__
void async_copy_global_to_shared(const Group& group,
                                 void* __restrict__ dst,
                                 const void* __restrict__ src,
                                 size_t nbytes,
                                 cuda::barrier<cuda::thread_scope_block>& bar) {
    cuda::memcpy_async(group, dst, src, nbytes, bar);
}

// ---- Block-scope barrier ----
// Caller initializes `bar` once via init() before first use.
struct BlockBarrier {
    cuda::barrier<cuda::thread_scope_block>* bar;

    __device__ __forceinline__ void arrive_and_wait() {
        bar->arrive_and_wait();
    }

    __device__ __forceinline__ auto arrive() {
        return bar->arrive();
    }

    __device__ __forceinline__ void wait(typename cuda::barrier<cuda::thread_scope_block>::arrival_token&& tok) {
        bar->wait(std::move(tok));
    }
};

} // namespace bw
