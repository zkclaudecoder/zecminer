@echo off
REM ============================================
REM  Zcash GPU Miner - Auto-tuned Configuration
REM  GPU: NVIDIA GeForce RTX 5090 Laptop GPU
REM  Solver: djeZo x3 instances
REM  Benchmark: 1018.72 Sols/s
REM ============================================

set WALLET=YOUR_ZEC_ADDRESS_HERE
set WORKER=rig1
set POOL=pool.tazminer.com:3333
set PASS=x

echo Starting nheqminer (djeZo x3, 1019 Sols/s)...
echo Pool: %POOL%
echo Wallet: %WALLET%.%WORKER%
echo.

nheqminer\build\Release\nheqminer.exe -l %POOL% -u %WALLET%.%WORKER% -p %PASS% -cd 0 0 0 -cv 0 -cb 500 500 500 -t 0

pause
