# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Zcash GPU miner based on nheqminer (Equihash 200,9 solver). Targets NVIDIA RTX 5090 (Blackwell, SM 12.0) for pool mining via Stratum protocol.

## Architecture

- `nheqminer/` — cloned from nicehash/nheqminer, modernized for CUDA 12.x
  - `nheqminer/nheqminer/main.cpp` — entry point, CLI arg parsing, mining loop
  - `nheqminer/nheqminer/libstratum/` — Stratum protocol client (pool communication)
  - `nheqminer/nheqminer/MinerFactory.cpp` — creates solver instances per device
  - `nheqminer/cuda_djezo/` — primary CUDA Equihash solver (djeZo, faster)
  - `nheqminer/cuda_tromp/` — alternative CUDA Equihash solver (tromp)
  - `nheqminer/blake2/` — BLAKE2b hash (used by Equihash)
  - `nheqminer/3rdparty/` — bundled old Boost (not used; we link system Boost via vcpkg)

## Modernization Changes Made

Original nheqminer targeted CUDA 8 / SM 5.0. The following changes were applied:
- CMakeLists.txt files rewritten for modern CMake 3.18+ with `enable_language(CUDA)` and `CMAKE_CUDA_ARCHITECTURES`
- GPU architecture targets updated: SM 5.0 through SM 12.0 (Blackwell)
- Deprecated CUDA warp intrinsics replaced: `__shfl()` -> `__shfl_sync()`, `__any()` -> `__any_sync()`
- Removed deprecated `device_functions_decls.h` and `sm_32_intrinsics.h` includes
- CPU xenoncat solver disabled (requires Linux assembly objects)
- C++ standard set to C++17
- Added ws2_32 link for Windows networking

## Prerequisites

1. **Visual Studio 2022** — "Desktop development with C++" workload
2. **CUDA Toolkit 12.8+** — from https://developer.nvidia.com/cuda-downloads
3. **vcpkg + Boost**:
   ```
   git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
   C:\vcpkg\bootstrap-vcpkg.bat
   C:\vcpkg\vcpkg install boost:x64-windows
   ```

## Build

```bat
build.bat
```

Or manually:
```bat
cd nheqminer
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
    -DUSE_CUDA_DJEZO=ON
cmake --build . --config Release
```

To target only your specific GPU (faster compile):
```bat
cmake .. -DCOMPUTE=120   # RTX 5090 only
```

## Run

Edit `mine.bat` with your ZEC wallet address, then:
```bat
mine.bat
```

CLI flags:
- `-l <pool>` — Stratum server:port
- `-u <wallet>.<worker>` — ZEC address with worker name
- `-cd <n>` — CUDA device index (0 = first GPU)
- `-cv <n>` — CUDA solver (0 = djeZo, 1 = tromp)
- `-t 0` — disable CPU mining
- `-cb <n>` — CUDA blocks
- `-ct <n>` — CUDA threads per block
- `-b` — benchmark mode
