@echo off
REM =====================================================================
REM  Zcash (Equihash 200,9) GPU Mining — Launch Templates
REM  Generated: April 2026
REM =====================================================================
REM
REM  IMPORTANT: GPU mining ZEC is NOT profitable in 2026 due to ASIC
REM  dominance. This is for hobby/learning/testing purposes only.
REM
REM  Instructions:
REM    1. Set your wallet address and worker name below
REM    2. Choose a pool by uncommenting the desired POOL line
REM    3. Uncomment ONE miner launch section at the bottom
REM    4. Ensure the miner executable path is correct
REM =====================================================================

REM === WALLET CONFIGURATION ===
REM Your Zcash transparent address (starts with t1...)
set WALLET=YOUR_ZEC_T_ADDRESS_HERE

REM Worker name (identifier for this machine)
set WORKER=rig1

REM Password (usually just 'x' for most pools)
set PASS=x

REM CUDA device index (0 = first GPU)
set CUDA_DEVICE=0

REM === POOL SELECTION (uncomment one) ===

REM -- Flypool (1% fee, PPLNS, customizable min payout) --
REM set POOL=stratum+tcp://zec.flypool.org:3333
REM set POOL=stratum+tcp://us1-zec.flypool.org:3333
REM set POOL=stratum+ssl://zec.flypool.org:3443

REM -- 2Miners (0.5% fee PPLNS, 0.1 ZEC min payout, every 2h) --
set POOL=stratum+tcp://zec.2miners.com:1010
REM set POOL=stratum+tcp://us-zec.2miners.com:1010

REM -- NanoPool (1% fee, 0.01 ZEC min payout) --
REM set POOL=stratum+tcp://zec-eu1.nanopool.org:6666
REM set POOL=stratum+tcp://zec-us-east1.nanopool.org:6666

REM -- F2Pool --
REM set POOL=stratum+tcp://zec.f2pool.com:3357

echo =====================================================================
echo  Zcash GPU Miner
echo  Pool:   %POOL%
echo  Wallet: %WALLET%.%WORKER%
echo  Device: %CUDA_DEVICE%
echo =====================================================================
echo.

REM =====================================================================
REM  MINER 1: nheqminer (open source, 0% fee)
REM  - Equihash 200,9: YES
REM  - RTX 5090: Requires modernized CUDA kernels (this project)
REM  - Download: https://github.com/nicehash/nheqminer (then modernize)
REM =====================================================================
nheqminer\build\Release\nheqminer.exe -l %POOL% -u %WALLET%.%WORKER% -p %PASS% -cd %CUDA_DEVICE% -cv 0 -t 0
goto :end

REM =====================================================================
REM  MINER 2: Funakoshi Miner (closed source, 0% fee)
REM  - Equihash 200,9: YES
REM  - RTX 5090: UNKNOWN — may need CUDA update for SM 12.0
REM  - Download: https://github.com/funakoshi2718/funakoshi-miner
REM  - NOTE: Check if a version supporting Blackwell/SM 12.0 exists
REM =====================================================================
REM funakoshi.exe -l %POOL% -u %WALLET%.%WORKER% -p %PASS% -d %CUDA_DEVICE%
REM goto :end

REM =====================================================================
REM  MINER 3: GMiner (closed source, 2% fee)
REM  - Equihash 200,9: UNCERTAIN — may still have residual support
REM  - RTX 5090: Likely supported (modern CUDA)
REM  - Download: https://github.com/develsoftware/GMinerRelease/releases
REM  - NOTE: Test with --algo equihash200_9 flag; may not work
REM =====================================================================
REM miner.exe --algo equihash200_9 --server %POOL% --user %WALLET%.%WORKER% --pass %PASS% --cuda_devices %CUDA_DEVICE%
REM goto :end

REM =====================================================================
REM  MINER 4: miniZ (closed source, 2% fee)
REM  - Equihash 200,9: NO — does NOT support 200,9
REM  - Supports: 144,5 / 192,7 / 210,9 / 125,4 / 150,5 / 96,5
REM  - RTX 5090: YES
REM  - Download: https://github.com/miniZ-miner/miniZ/releases
REM  - USE FOR: Bitcoin Gold, Flux/ZelCash, Beam, Ycash, etc.
REM =====================================================================
REM miniZ.exe --url %WALLET%.%WORKER%@zec.flypool.org:3333 --pass %PASS% --cuda-devices %CUDA_DEVICE%
REM goto :end

REM =====================================================================
REM  MINER 5: lolMiner (closed source, 1-2% fee)
REM  - Equihash 200,9: NO — does NOT support 200,9
REM  - Supports: 144,5 / 192,7 and Beam/Cuckoo variants
REM  - RTX 5090: YES
REM  - Download: https://github.com/Lolliedieb/lolMiner-releases/releases
REM  - USE FOR: Beam, Bitcoin Gold, and other Equihash-variant coins
REM =====================================================================
REM lolMiner.exe --algo EQUI144_5 --pool %POOL% --user %WALLET%.%WORKER% --pass %PASS% --devices %CUDA_DEVICE%
REM goto :end

REM =====================================================================
REM  NOTES ON EWBF AND OTHERS:
REM
REM  EWBF Miner: ABANDONED (~2019). Will not run on RTX 5090.
REM  Optiminer:  ABANDONED (~2018). Linux/AMD only.
REM  SilentArmy: ABANDONED (~2017). OpenCL only.
REM  Cryptknocker: Status unknown. Check if still available.
REM
REM  For profitable GPU mining on RTX 5090, consider:
REM  - Flux (Equihash 125,4) with miniZ or lolMiner
REM  - Bitcoin Gold (Equihash 144,5) with miniZ, GMiner, or lolMiner
REM  - Ravencoin (KawPow) with GMiner or lolMiner
REM  - Ergo (Autolykos) with lolMiner
REM =====================================================================

:end
pause
