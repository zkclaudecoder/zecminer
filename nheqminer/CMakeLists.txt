cmake_minimum_required(VERSION 3.18)
project(nheqminer LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

## Enable solvers here
#### CPU solvers
option(USE_CPU_TROMP "USE CPU_TROMP" OFF)
option(USE_CPU_XENONCAT "USE CPU_XENONCAT" OFF)  # Disabled: requires Linux asm objects
#### GPU solvers
option(USE_CUDA_TROMP "USE CUDA_TROMP" OFF)
option(USE_CUDA_DJEZO "USE CUDA_DJEZO" ON)
#### experimental — modern Blackwell-tuned solver (Phase 0 prototype)
option(USE_CUDA_BLACKWELL "USE CUDA_BLACKWELL (experimental, SM 9.0+)" OFF)

## Add solver definitions
if (USE_CPU_TROMP)
    add_definitions(-DUSE_CPU_TROMP)
    message("-- USE_CPU_TROMP DEFINED")
endif()
if (USE_CPU_XENONCAT)
    add_definitions(-DUSE_CPU_XENONCAT)
    message("-- USE_CPU_XENONCAT DEFINED")
endif()
if (USE_CUDA_TROMP)
    add_definitions(-DUSE_CUDA_TROMP)
    message("-- USE_CUDA_TROMP DEFINED")
endif()
if (USE_CUDA_DJEZO)
    add_definitions(-DUSE_CUDA_DJEZO)
    message("-- USE_CUDA_DJEZO DEFINED")
endif()
if (USE_CUDA_BLACKWELL)
    add_definitions(-DUSE_CUDA_BLACKWELL)
    message("-- USE_CUDA_BLACKWELL DEFINED")
endif()


########
# LINUX
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64 -msse2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -msse2")
    add_definitions(-O2)
endif()

# MSVC
if(MSVC)
    add_definitions(/D_CRT_SECURE_NO_WARNINGS /D_WINSOCK_DEPRECATED_NO_WARNINGS)
    add_definitions(/D_WIN32_WINNT=0x0A00)  # Target Windows 10
    add_definitions(/DBOOST_BIND_GLOBAL_PLACEHOLDERS)
    add_definitions(/D_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

# Common
include_directories(${nheqminer_SOURCE_DIR}/nheqminer)

# BOOST
set(Boost_USE_STATIC_LIBS ON)
if(WIN32)
    # On Windows with vcpkg, don't force .a suffix
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()
find_package(Boost REQUIRED COMPONENTS system log_setup log date_time filesystem thread)

if (Boost_FOUND)
  include_directories (SYSTEM ${Boost_INCLUDE_DIR})
  link_directories (${Boost_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "Boost not found. Install via vcpkg: vcpkg install boost:x64-windows")
endif ()

include_directories(${CMAKE_CURRENT_BINARY_DIR}/../)

set(SOURCE_FILES
    # sources
    nheqminer/amount.cpp
    nheqminer/api.cpp
    nheqminer/arith_uint256.cpp
    nheqminer/crypto/sha256.cpp
    nheqminer/json/json_spirit_reader.cpp
    nheqminer/json/json_spirit_value.cpp
    nheqminer/json/json_spirit_writer.cpp
    nheqminer/libstratum/ZcashStratum.cpp
    nheqminer/main.cpp
    nheqminer/primitives/block.cpp
    nheqminer/speed.cpp
    nheqminer/uint256.cpp
    nheqminer/utilstrencodings.cpp
    # headers
    nheqminer/amount.h
    nheqminer/api.hpp
    nheqminer/arith_uint256.h
    nheqminer/crypto/sha256.h
    nheqminer/hash.h
    nheqminer/json/json_spirit.h
    nheqminer/json/json_spirit_error_position.h
    nheqminer/json/json_spirit_reader.h
    nheqminer/json/json_spirit_reader_template.h
    nheqminer/json/json_spirit_stream_reader.h
    nheqminer/json/json_spirit_utils.h
    nheqminer/json/json_spirit_value.h
    nheqminer/json/json_spirit_writer.h
    nheqminer/json/json_spirit_writer_template.h
    nheqminer/libstratum/StratumClient.cpp
    nheqminer/libstratum/StratumClient.h
    nheqminer/libstratum/ZcashStratum.cpp
    nheqminer/libstratum/ZcashStratum.h
    nheqminer/primitives/block.h
    nheqminer/primitives/transaction.h
    nheqminer/script/script.h
    nheqminer/serialize.h
    nheqminer/speed.hpp
    nheqminer/streams.h
    nheqminer/support/allocators/zeroafterfree.h
    nheqminer/tinyformat.h
    nheqminer/uint252.h
    nheqminer/uint256.h
    nheqminer/utilstrencodings.h
    nheqminer/version.h
    nheqminer/zcash/JoinSplit.hpp
    nheqminer/zcash/NoteEncryption.hpp
    nheqminer/zcash/Proof.hpp
    nheqminer/zcash/Zcash.h
    nheqminer/SolverStub.h

    nheqminer/AvailableSolvers.h
    nheqminer/ISolver.h
    nheqminer/Solver.h
    nheqminer/MinerFactory.h
    nheqminer/MinerFactory.cpp

    # blake shared
    blake2/blake2bx.cpp
    blake2/blake2.h
    blake2/blake2b-load-sse2.h
    blake2/blake2b-load-sse41.h
    blake2/blake2b-round.h
    blake2/blake2-config.h
    blake2/blake2-impl.h
    blake2/blake2-round.h
    )

set(LIBS ${LIBS} ${Boost_LIBRARIES})

message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")
message("-- LIBS: ${LIBS}")

if (USE_CPU_TROMP)
    add_subdirectory(cpu_tromp)
endif()
if (USE_CPU_XENONCAT)
    add_subdirectory(cpu_xenoncat)
endif()
if (USE_CUDA_TROMP)
    add_subdirectory(cuda_tromp)
endif()
if (USE_CUDA_DJEZO)
    add_subdirectory(cuda_djezo)
endif()
if (USE_CUDA_BLACKWELL)
    add_subdirectory(cuda_blackwell)
endif()

ADD_EXECUTABLE(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${LIBS} )

if(WIN32)
    target_link_libraries(${PROJECT_NAME} ws2_32)
endif()

# link solver libs
if (USE_CPU_TROMP)
    target_link_libraries(${PROJECT_NAME} cpu_tromp)
endif()
if (USE_CPU_XENONCAT)
    add_library ( xenoncat_avx1 SHARED IMPORTED GLOBAL )
    set_target_properties ( xenoncat_avx1 PROPERTIES IMPORTED_LOCATION "../nheqminer/cpu_xenoncat/asm_linux/equihash_avx1.o" )
    add_library ( xenoncat_avx2 SHARED IMPORTED GLOBAL )
    set_target_properties ( xenoncat_avx2 PROPERTIES IMPORTED_LOCATION "../nheqminer/cpu_xenoncat/asm_linux/equihash_avx2.o" )
    target_link_libraries(${PROJECT_NAME} cpu_xenoncat xenoncat_avx1 xenoncat_avx2)
endif()
if (USE_CUDA_TROMP)
    target_link_libraries(${PROJECT_NAME} cuda_tromp)
endif()
if (USE_CUDA_DJEZO)
    target_link_libraries(${PROJECT_NAME} cuda_djezo)
endif()
if (USE_CUDA_BLACKWELL)
    target_link_libraries(${PROJECT_NAME} cuda_blackwell)
endif()
