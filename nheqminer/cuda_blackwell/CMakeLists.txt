set(EXECUTABLE cuda_blackwell)

find_package(CUDAToolkit REQUIRED)

# This solver uses cp.async (SM 8.0+), cp.async.bulk and clusters (SM 9.0+),
# and is tuned primarily for Blackwell (SM 12.0). It is NOT supported on older GPUs.
# The build deliberately compiles for SM 12.0 only — fall back to cuda_djezo for
# older architectures via runtime detection in MinerFactory.
if(COMPUTE AND (COMPUTE GREATER 0))
    set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE})
else()
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# Phase 0 prototype kernels (only blake2b + digit_1 to validate architecture).
# blake2b is a .cuh header (inline device functions) so digit_kernels.cu can
# call into it without relocatable-device-code linking.
file(GLOB SRC_LIST
    cuda_blackwell.cpp
    digit_kernels.cu
)
file(GLOB HEADERS
    cuda_blackwell.hpp
    eq_context.hpp
    bucket_layout.hpp
    blake2b.cuh
    pipeline.hpp
)

# Modern CUDA features need higher optimization and relaxed PTX
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES --ptxas-options=-v -Xptxas=-dlcm=ca -Xptxas=-dscm=cs -O3")

# Enable extended lambda support for cooperative_groups + libcudacxx pipeline
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(..)

add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
target_link_libraries(${EXECUTABLE} CUDA::cudart CUDA::cuda_driver)

message("-- cuda_blackwell architectures: ${CMAKE_CUDA_ARCHITECTURES}")

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
