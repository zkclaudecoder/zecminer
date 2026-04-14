@echo off
REM Phase 0 sanity test for cuda_blackwell solver.
REM Expects build already done via build_blackwell.bat.
REM
REM What to look for in output:
REM   [BW] round-1 nslots: total=~1M  mean=~250  nonempty=4096/4096
REM Anything near those numbers = digit_first + digit_1 are producing
REM plausible output. Zeros everywhere = bug.

setlocal
set DJEZO_BW_VERBOSE=1
set NHEQ=C:\Users\sdmur\Documents\zecminer\nheqminer\build\Release\nheqminer.exe

echo === BLACKWELL SOLVER (-cv 2) ===
"%NHEQ%" -b 5 -cd 0 -cv 2 -t 0 2>&1
echo.
echo === DJEZO SOLVER BASELINE (-cv 0) ===
"%NHEQ%" -b 5 -cd 0 -cv 0 -t 0 2>&1
