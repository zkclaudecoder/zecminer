@echo off
REM ============================================
REM  Zcash GPU Miner
REM  Edit WALLET below with your ZEC address
REM ============================================

set WALLET=YOUR_ZEC_ADDRESS_HERE
set WORKER=rig1
set POOL=pool.tazminer.com:3333
set PASS=x

echo Starting nheqminer (djeZo x3 instances)...
echo Pool: %POOL%
echo Wallet: %WALLET%.%WORKER%
echo.
echo Run autotune.bat first to find optimal settings for your GPU.
echo.

"%~dp0nheqminer.exe" -l %POOL% -u %WALLET%.%WORKER% -p %PASS% -cd 0 0 0 -cv 0 -cb 300 300 300 -t 0

pause
