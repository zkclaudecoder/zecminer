@echo off
REM ============================================
REM  Build nheqminer with CUDA support
REM ============================================
REM
REM Prerequisites:
REM   1. Visual Studio 2022 with "Desktop development with C++" workload
REM   2. CUDA Toolkit 12.8+ (https://developer.nvidia.com/cuda-downloads)
REM   3. vcpkg with Boost:
REM        git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
REM        C:\vcpkg\bootstrap-vcpkg.bat
REM        C:\vcpkg\vcpkg install boost:x64-windows
REM

set VCPKG_ROOT=C:\vcpkg

echo === Configuring nheqminer ===
cd nheqminer

if not exist build mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake ^
    -DUSE_CUDA_DJEZO=ON ^
    -DUSE_CUDA_TROMP=OFF ^
    -DUSE_CPU_TROMP=OFF ^
    -DUSE_CPU_XENONCAT=OFF

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo CMAKE CONFIGURATION FAILED
    echo Make sure Visual Studio 2022, CUDA Toolkit, and vcpkg with Boost are installed.
    pause
    exit /b 1
)

echo.
echo === Building nheqminer ===
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo === Build successful! ===
echo Binary: %CD%\Release\nheqminer.exe
echo.
echo Edit mine.bat with your ZEC wallet address, then run it to start mining.

cd ..\..
pause
