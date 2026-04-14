@echo off
REM ============================================
REM  Zcash GPU Miner
REM  Blackwell auto-selects CONFIG_MODE_2 for +21%% gain
REM  RTX 5090 Desktop: ~2300 Sols/s @ 5 instances
REM  RTX 5090 Laptop:  ~1000 Sols/s @ 3 instances
REM ============================================

set WALLET=tmRhAwek1qaG3bqy4W8nih9NQsycYrLuV4n
set WORKER=rig1
set POOL=pool.tazminer.com:3333
set PASS=x

REM Tune instance count based on your GPU (run autotune.bat to find optimal):
REM   RTX 5090 Desktop (170 SMs): 5 instances
REM   RTX 5090 Laptop  (82 SMs):  3 instances
REM   Smaller GPUs: 1-2 instances

echo Starting nheqminer...
echo Pool: %POOL%
echo Wallet: %WALLET%.%WORKER%
echo.

"%~dp0nheqminer\build\Release\nheqminer.exe" -l %POOL% -u %WALLET%.%WORKER% -p %PASS% -cd 0 0 0 0 0 -cv 0 -t 0

pause
