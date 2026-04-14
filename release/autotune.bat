@echo off
REM ============================================
REM  nheqminer Auto-Tuner
REM  Benchmarks all configurations and finds
REM  the optimal settings for your GPU
REM ============================================
powershell -ExecutionPolicy Bypass -File "%~dp0autotune.ps1" %*
pause
