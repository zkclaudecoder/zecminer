set(EXECUTABLE cuda_tromp)

find_package(CUDAToolkit REQUIRED)

file(GLOB SRC_LIST
    cuda_tromp.cpp
    equi_miner.cu )
file(GLOB HEADERS
    cuda_tromp.hpp
    eqcuda.hpp
    )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math -lineinfo -O3")

add_definitions(-DHIST)

# Set GPU architecture targets
if(COMPUTE AND (COMPUTE GREATER 0))
    set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE})
else()
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(..)

add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})

# cuda_tromp.cpp uses CUDA runtime API (cudaGetDeviceCount, etc.) - ensure it can find CUDA headers
target_include_directories(${EXECUTABLE} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE} CUDA::cudart)

message("-- CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
