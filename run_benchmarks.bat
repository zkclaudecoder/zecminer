@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%

set OUT=C:\Users\sdmur\Documents\zecminer\benchmark_raw.txt
set EXE=C:\Users\sdmur\Documents\zecminer\nheqminer\build\Release\nheqminer.exe

cd /d C:\Users\sdmur\Documents\zecminer\nheqminer\build

echo === Rebuilding === > %OUT%
cmake --build . --config Release >> %OUT% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo BUILD FAILED >> %OUT%
    exit /b 1
)
echo BUILD SUCCESSFUL >> %OUT%

echo === Test 1: TPB=32 blocks=574 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 32 -cb 574 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 2: TPB=64 blocks=574 (baseline) === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 64 -cb 574 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 3: TPB=128 blocks=574 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 128 -cb 574 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 4: TPB=256 blocks=574 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 256 -cb 574 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 5: TPB=64 blocks=246 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 64 -cb 246 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 6: TPB=64 blocks=410 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 64 -cb 410 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 7: TPB=64 blocks=820 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 64 -cb 820 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 8: TPB=128 blocks=246 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 128 -cb 246 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 9: TPB=128 blocks=410 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 128 -cb 410 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 10: TPB=128 blocks=820 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 128 -cb 820 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 11: TPB=32 blocks=820 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 32 -cb 820 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 12: TPB=32 blocks=1640 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 32 -cb 1640 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 13: TPB=256 blocks=246 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 256 -cb 246 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 14: TPB=256 blocks=164 === >> %OUT%
%EXE% -b 2000 -cd 0 -cv 1 -ct 256 -cb 164 -t 0 >> %OUT% 2>&1
timeout /t 20 /nobreak > nul

echo === Test 15: Multi-instance cd=0,0,0 cb=200,200,200 === >> %OUT%
%EXE% -b 2000 -cd 0 0 0 -cv 1 -cb 200 200 200 -t 0 >> %OUT% 2>&1

echo === ALL TESTS COMPLETE === >> %OUT%
