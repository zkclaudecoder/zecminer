set(EXECUTABLE cuda_djezo)

find_package(CUDAToolkit REQUIRED)

file(GLOB SRC_LIST
    cuda_djezo.cpp
    equi_miner.cu )
file(GLOB HEADERS
    cuda_djezo.hpp
    eqcuda.hpp
    )

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FORCE_INLINES --ptxas-options=-v -Xptxas=-dlcm=ca -Xptxas=-dscm=cs -O3")

# Set GPU architecture targets
# Use COMPUTE variable if provided, otherwise target common modern architectures
if(COMPUTE AND (COMPUTE GREATER 0))
    set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE})
else()
    # Target SM 5.0 (Maxwell) through SM 12.0 (Blackwell)
    # PTX for 8.9 provides forward compatibility via JIT for newer architectures
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(..)

add_library(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADERS})
target_link_libraries(${EXECUTABLE} CUDA::cudart CUDA::cuda_driver)

message("-- CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )
