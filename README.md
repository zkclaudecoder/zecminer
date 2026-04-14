# zecminer

Zcash Equihash 200,9 GPU miner based on [nheqminer](https://github.com/nicehash/nheqminer), modernized for **NVIDIA Blackwell (SM 12.0)** and CUDA 13.2+.

Achieves **~1000+ Sols/s** on RTX 5090 with the djezo solver.

## Quick Start

1. Install prerequisites (see [Building](#building))
2. Run `autotune.bat` to benchmark your GPU and find optimal settings
3. Edit `mine.bat` with your ZEC wallet address
4. Run `mine.bat` to start mining

## Requirements

- **GPU**: NVIDIA with SM 7.5+ (Turing, Ampere, Ada, Blackwell)
- **OS**: Windows 10/11
- **CUDA Toolkit**: 12.8+ (13.2 recommended)
- **Visual Studio 2022**: with "Desktop development with C++" workload
- **Boost**: via vcpkg (`vcpkg install boost-system boost-log boost-date-time boost-filesystem boost-thread boost-asio boost-array boost-bind boost-signals2 boost-circular-buffer:x64-windows`)

## Building

```bat
:: Install vcpkg and Boost (one-time)
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install boost-system boost-log boost-date-time boost-filesystem boost-thread boost-asio boost-array boost-bind boost-signals2 boost-circular-buffer:x64-windows

:: Build the miner
build.bat
```

Or use `configure.bat` which runs cmake + msbuild with the correct environment.

Build takes ~30 minutes due to CUDA kernel compilation for SM 120 (Blackwell).

## Files

| File | Purpose |
|------|---------|
| `autotune.bat` | Auto-detect GPU and benchmark all configurations to find optimal settings |
| `mine.bat` | Start mining with the best configuration (edit wallet address first) |
| `build.bat` | Build nheqminer from source |
| `configure.bat` | CMake configure + build (used by build.bat) |
| `architecture.md` | Detailed GPU solver architecture and bottleneck analysis |

## Usage

```bat
:: Auto-tune (finds best config for your GPU)
autotune.bat
autotune.bat -Quick                          :: Faster, fewer tests
autotune.bat -Wallet <your_zec_address>      :: Also validates on pool

:: Manual mining
nheqminer\build\Release\nheqminer.exe -l <pool>:<port> -u <wallet>.<worker> -p x -cd 0 0 0 -cv 0 -cb 300 300 300 -t 0

:: Benchmark
nheqminer\build\Release\nheqminer.exe -b 2000 -cd 0 -cv 0 -t 0
```

### Key flags

| Flag | Description |
|------|-------------|
| `-l <host:port>` | Pool address |
| `-u <wallet>.<worker>` | ZEC address and worker name |
| `-cd 0 [0 0]` | CUDA device(s) — repeat for multi-instance |
| `-cv 0` | Solver: 0 = djeZo (fast), 1 = tromp (compatible) |
| `-cb <n> [n n]` | Blocks per instance |
| `-ct <n> [n n]` | Threads per block |
| `-t 0` | CPU threads (0 = disabled) |
| `-b <n>` | Benchmark mode (n iterations) |
| `-ci` | Show CUDA GPU info |

## Solvers

| Solver | Speed (RTX 5090 Laptop) | Speed (RTX 5090 Desktop) | Notes |
|--------|------------------------|--------------------------|-------|
| **djezo** (`-cv 0`) | ~1000 Sols/s | **~2300 Sols/s** | Fastest. Auto-selects CONFIG_MODE_2 on Blackwell. |
| tromp (`-cv 1`) | ~179 Sols/s | ~350 Sols/s | Simpler fallback solver. |

Set `DJEZO_MODE1=1` environment variable to force legacy CONFIG_MODE_1 if needed.

## What was changed from original nheqminer

- **CONFIG_MODE_2 on Blackwell (~21% gain)**: Auto-selects smaller bucket config (SM=640 vs 1248) on SM 12.0+. Smaller shared memory footprint allows more blocks per SM on 170-SM GPUs.
- **Blackwell warp sync fix**: Added `__syncwarp()` barriers in djezo's `digit_last_wdc` kernel — Blackwell's independent thread scheduling caused stale shared memory reads during solution reconstruction
- **CUDA API modernization**: `__shfl()` -> `__shfl_sync()`, `__any()` -> `__any_sync()`
- **Removed deprecated headers**: `device_functions_decls.h`, `sm_32_intrinsics.h`
- **Removed `cudaDeviceReset()`**: Corrupted multi-instance contexts on modern GPUs
- **Boost 1.90 compatibility**: `io_service` -> `io_context`, modern resolver API, added missing headers
- **CMake modernization**: `enable_language(CUDA)`, `CMAKE_CUDA_ARCHITECTURES`, `find_package(CUDAToolkit)`
- **Windows fixes**: ws2_32 linking, pthread removal, MSVC compatibility
- **Cache config**: `cudaFuncCachePreferNone` for Blackwell L1/shared partitioning

## License

Original nheqminer: MIT (tromp solver) + GPL 3.0 (djezo solver). See `nheqminer/LICENSE_MIT`.
