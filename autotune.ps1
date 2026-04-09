##############################################################################
#  nheqminer Auto-Tuner
#  Detects GPU, benchmarks configurations, finds optimal settings
##############################################################################

param(
    [string]$MinerPath = "",
    [string]$Pool = "pool.tazminer.com:3333",
    [string]$Wallet = "",
    [string]$Worker = "rig1",
    [int]$BenchIterations = 1000,
    [int]$CooldownSeconds = 15,
    [switch]$Quick,
    [switch]$SkipPoolTest
)

$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- Find miner binary ---
if (-not $MinerPath) {
    $candidates = @(
        "$scriptDir\nheqminer\build\Release\nheqminer.exe",
        "$scriptDir\nheqminer.exe",
        "$scriptDir\build\Release\nheqminer.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $MinerPath = $c; break }
    }
}
if (-not $MinerPath -or -not (Test-Path $MinerPath)) {
    Write-Host "ERROR: nheqminer.exe not found." -ForegroundColor Red
    Write-Host "  Place this script in the zecminer directory, or use -MinerPath <path>"
    exit 1
}
$MinerPath = (Resolve-Path $MinerPath).Path
Write-Host "Miner: $MinerPath" -ForegroundColor Cyan

# --- Detect GPU ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  GPU DETECTION" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

try {
    $smiOutput = & nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,power.limit --format=csv,noheader 2>&1
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed" }
} catch {
    Write-Host "ERROR: nvidia-smi not found. Is an NVIDIA GPU installed?" -ForegroundColor Red
    exit 1
}

$gpuLines = $smiOutput -split "`n" | Where-Object { $_.Trim() -ne "" }
$gpuCount = $gpuLines.Count

Write-Host "Found $gpuCount GPU(s):" -ForegroundColor Green
$gpuInfos = @()
foreach ($line in $gpuLines) {
    $parts = $line -split "," | ForEach-Object { $_.Trim() }
    $info = @{
        Name = $parts[0]
        ComputeCap = $parts[1]
        MemoryMB = [int]($parts[2] -replace '[^\d]','')
        Driver = $parts[3]
        PowerLimit = $parts[4]
    }
    $gpuInfos += $info
    Write-Host "  $($info.Name) | SM $($info.ComputeCap) | $($info.MemoryMB) MB | Driver $($info.Driver) | TDP $($info.PowerLimit)" -ForegroundColor White
}

# Get SM count from miner's CUDA info
Write-Host ""
Write-Host "Detecting SM count..." -ForegroundColor Gray
$cudaInfo = & $MinerPath -ci 2>&1 | Out-String
$smMatch = [regex]::Match($cudaInfo, 'SM count:\s*(\d+)')
$smCount = if ($smMatch.Success) { [int]$smMatch.Groups[1].Value } else { 0 }
if ($smCount -gt 0) {
    Write-Host "  SM count: $smCount" -ForegroundColor White
} else {
    Write-Host "  Could not detect SM count, using defaults" -ForegroundColor Yellow
    $smCount = 82  # fallback
}

# --- Determine test configurations ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  BENCHMARK PLAN" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

if ($Quick) {
    $BenchIterations = 500
    $CooldownSeconds = 10
    Write-Host "Quick mode: $BenchIterations iterations, ${CooldownSeconds}s cooldown" -ForegroundColor Gray
} else {
    Write-Host "Full mode: $BenchIterations iterations, ${CooldownSeconds}s cooldown" -ForegroundColor Gray
}

# Solver configurations to test
$tests = @()

# Phase 1: Single instance, both solvers, default settings
$tests += @{ Name = "djezo-1x-default";  Solver = 0; Instances = 1; Blocks = @(); TPB = @() }
$tests += @{ Name = "tromp-1x-default";  Solver = 1; Instances = 1; Blocks = @(); TPB = @() }

# Phase 2: djezo multi-instance
$tests += @{ Name = "djezo-2x-default";  Solver = 0; Instances = 2; Blocks = @(); TPB = @() }
$tests += @{ Name = "djezo-3x-default";  Solver = 0; Instances = 3; Blocks = @(); TPB = @() }
$tests += @{ Name = "djezo-4x-default";  Solver = 0; Instances = 4; Blocks = @(); TPB = @() }

# Phase 3: djezo block tuning (3 instances)
$blockValues = @(200, 250, 300, 350, 400, 500)
foreach ($b in $blockValues) {
    $tests += @{ Name = "djezo-3x-cb$b"; Solver = 0; Instances = 3; Blocks = @($b, $b, $b); TPB = @() }
}

# Phase 4: tromp multi-instance
$tests += @{ Name = "tromp-2x-default";  Solver = 1; Instances = 2; Blocks = @(); TPB = @() }
$tests += @{ Name = "tromp-3x-default";  Solver = 1; Instances = 3; Blocks = @(); TPB = @() }

# Phase 5: djezo TPB tuning (single instance)
$tests += @{ Name = "djezo-1x-tpb128"; Solver = 0; Instances = 1; Blocks = @(); TPB = @(128) }
$tests += @{ Name = "djezo-1x-tpb256"; Solver = 0; Instances = 1; Blocks = @(); TPB = @(256) }

# Phase 6: Best multi-instance with TPB tuning
$tests += @{ Name = "djezo-3x-cb300-tpb128"; Solver = 0; Instances = 3; Blocks = @(300,300,300); TPB = @(128,128,128) }

if ($Quick) {
    # In quick mode, skip some tests
    $tests = $tests | Where-Object { $_.Name -notmatch "tpb|tromp-(2|3)x" }
}

$totalTests = $tests.Count
Write-Host "Will run $totalTests benchmark configurations" -ForegroundColor White
Write-Host ""

# --- Run benchmarks ---
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  RUNNING BENCHMARKS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

$results = @()
$testNum = 0

foreach ($test in $tests) {
    $testNum++

    # Build command line
    $cdArgs = (1..$test.Instances | ForEach-Object { "0" }) -join " "
    $args = "-b $BenchIterations -cd $cdArgs -cv $($test.Solver) -t 0"

    if ($test.Blocks.Count -gt 0) {
        $args += " -cb " + ($test.Blocks -join " ")
    }
    if ($test.TPB.Count -gt 0) {
        $args += " -ct " + ($test.TPB -join " ")
    }

    # Get GPU temp
    $temp = (& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>&1).Trim()

    Write-Host ""
    Write-Host "[$testNum/$totalTests] $($test.Name)" -ForegroundColor Cyan
    Write-Host "  Command: nheqminer.exe $args" -ForegroundColor Gray
    Write-Host "  GPU Temp: ${temp}C" -ForegroundColor Gray

    # Run benchmark
    $output = & $MinerPath $args.Split(' ') 2>&1 | Out-String

    # Parse results
    $solsMatch = [regex]::Match($output, 'Speed:\s+([\d.]+)\s+Sols/s')
    $ipsMatch = [regex]::Match($output, 'Speed:\s+([\d.]+)\s+I/s')
    $solsFoundMatch = [regex]::Match($output, 'Total solutions found:\s+(\d+)')

    $solsPerSec = if ($solsMatch.Success) { [double]$solsMatch.Groups[1].Value } else { 0 }
    $ipsVal = if ($ipsMatch.Success) { [double]$ipsMatch.Groups[1].Value } else { 0 }
    $solsFound = if ($solsFoundMatch.Success) { [int]$solsFoundMatch.Groups[1].Value } else { 0 }

    $result = @{
        Name = $test.Name
        Solver = $test.Solver
        Instances = $test.Instances
        Blocks = $test.Blocks
        TPB = $test.TPB
        SolsPerSec = $solsPerSec
        IPerSec = $ipsVal
        SolutionsFound = $solsFound
        StartTemp = $temp
        Args = $args
    }
    $results += $result

    if ($solsPerSec -gt 0) {
        Write-Host "  Result: $solsPerSec Sols/s | $ipsVal I/s | $solsFound solutions" -ForegroundColor Green
    } else {
        Write-Host "  Result: FAILED (0 Sols/s)" -ForegroundColor Red
    }

    # Cooldown
    if ($testNum -lt $totalTests) {
        Write-Host "  Cooling down ${CooldownSeconds}s..." -ForegroundColor DarkGray
        Start-Sleep -Seconds $CooldownSeconds
    }
}

# --- Find best configuration ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  RESULTS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

$sorted = $results | Where-Object { $_.SolsPerSec -gt 0 } | Sort-Object -Property SolsPerSec -Descending
$best = $sorted[0]

# Print results table
Write-Host ""
Write-Host ("{0,-30} {1,10} {2,10} {3,10}" -f "Configuration", "Sols/s", "I/s", "Solutions") -ForegroundColor White
Write-Host ("{0,-30} {1,10} {2,10} {3,10}" -f "-------------", "------", "---", "---------") -ForegroundColor Gray

foreach ($r in $sorted) {
    $marker = if ($r.Name -eq $best.Name) { " <-- BEST" } else { "" }
    $color = if ($r.Name -eq $best.Name) { "Green" } else { "White" }
    Write-Host ("{0,-30} {1,10:F1} {2,10:F1} {3,10}" -f $r.Name, $r.SolsPerSec, $r.IPerSec, $r.SolutionsFound) -ForegroundColor $color -NoNewline
    if ($marker) { Write-Host $marker -ForegroundColor Green } else { Write-Host "" }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  OPTIMAL CONFIGURATION" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Best: $($best.Name)" -ForegroundColor Green
Write-Host "  Sols/s: $($best.SolsPerSec)" -ForegroundColor Green
Write-Host "  Command: nheqminer.exe $($best.Args -replace '-b \d+','-l <pool>')" -ForegroundColor White

# --- Pool validation (optional) ---
if (-not $SkipPoolTest -and $Wallet -and $best.SolsPerSec -gt 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "  POOL VALIDATION (30 seconds)" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    $cdArgs = (1..$best.Instances | ForEach-Object { "0" }) -join " "
    $poolArgs = "-l $Pool -u $Wallet.$Worker -p x -cd $cdArgs -cv $($best.Solver) -t 0"
    if ($best.Blocks.Count -gt 0) { $poolArgs += " -cb " + ($best.Blocks -join " ") }
    if ($best.TPB.Count -gt 0) { $poolArgs += " -ct " + ($best.TPB -join " ") }

    Write-Host "  Connecting to $Pool for 30 seconds..." -ForegroundColor Gray

    $job = Start-Process -FilePath $MinerPath -ArgumentList $poolArgs.Split(' ') -RedirectStandardOutput "$scriptDir\pool_test.log" -RedirectStandardError "$scriptDir\pool_test_err.log" -PassThru -NoNewWindow
    Start-Sleep -Seconds 30

    if (-not $job.HasExited) { Stop-Process -Id $job.Id -Force 2>$null }
    Start-Sleep -Seconds 2

    if (Test-Path "$scriptDir\pool_test.log") {
        $poolLog = Get-Content "$scriptDir\pool_test.log" -Raw
        $accepted = ([regex]::Matches($poolLog, 'Accepted share')).Count
        $rejected = ([regex]::Matches($poolLog, 'Rejected share')).Count
        $total = $accepted + $rejected

        if ($accepted -gt 0) {
            Write-Host "  Pool test PASSED: $accepted/$total shares accepted" -ForegroundColor Green
        } elseif ($total -gt 0) {
            Write-Host "  Pool test FAILED: 0/$total shares accepted" -ForegroundColor Red
        } else {
            Write-Host "  Pool test: no shares submitted (may need longer test)" -ForegroundColor Yellow
        }
        Remove-Item "$scriptDir\pool_test.log" -Force 2>$null
        Remove-Item "$scriptDir\pool_test_err.log" -Force 2>$null
    }
}

# --- Generate mine.bat ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  GENERATING mine.bat" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

$solverName = if ($best.Solver -eq 0) { "djeZo" } else { "tromp" }
$cdLine = (1..$best.Instances | ForEach-Object { "0" }) -join " "
$cbLine = if ($best.Blocks.Count -gt 0) { " -cb " + ($best.Blocks -join " ") } else { "" }
$ctLine = if ($best.TPB.Count -gt 0) { " -ct " + ($best.TPB -join " ") } else { "" }
$walletLine = if ($Wallet) { $Wallet } else { "YOUR_ZEC_ADDRESS_HERE" }

$batContent = @"
@echo off
REM ============================================
REM  Zcash GPU Miner - Auto-tuned Configuration
REM  GPU: $($gpuInfos[0].Name)
REM  Solver: $solverName x$($best.Instances) instances
REM  Benchmark: $($best.SolsPerSec) Sols/s
REM ============================================

set WALLET=$walletLine
set WORKER=$Worker
set POOL=$Pool
set PASS=x

echo Starting nheqminer ($solverName x$($best.Instances), $([math]::Round($best.SolsPerSec)) Sols/s)...
echo Pool: %POOL%
echo Wallet: %WALLET%.%WORKER%
echo.

nheqminer\build\Release\nheqminer.exe -l %POOL% -u %WALLET%.%WORKER% -p %PASS% -cd $cdLine -cv $($best.Solver)$cbLine$ctLine -t 0

pause
"@

$batContent | Out-File -FilePath "$scriptDir\mine.bat" -Encoding ascii
Write-Host "  Written to: $scriptDir\mine.bat" -ForegroundColor Green

# --- Save full results ---
$reportPath = "$scriptDir\autotune_results.md"
$report = "# Auto-Tune Results`n`n"
$report += "**GPU:** $($gpuInfos[0].Name) (SM $($gpuInfos[0].ComputeCap), $($gpuInfos[0].MemoryMB) MB)`n"
$report += "**Driver:** $($gpuInfos[0].Driver)`n"
$report += "**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm')`n"
$report += "**Best:** $($best.Name) at $($best.SolsPerSec) Sols/s`n`n"
$report += "| Configuration | Sols/s | I/s | Solutions |`n"
$report += "|---|---|---|---|`n"
foreach ($r in $sorted) {
    $mark = if ($r.Name -eq $best.Name) { " **" } else { "" }
    $report += "| $($r.Name)$mark | $($r.SolsPerSec) | $($r.IPerSec) | $($r.SolutionsFound) |`n"
}
$report += "`n**Optimal command:**`n``````"
$report += "`nnheqminer.exe -l <pool> -u <wallet>.<worker> -p x -cd $cdLine -cv $($best.Solver)$cbLine$ctLine -t 0"
$report += "`n``````"

$report | Out-File -FilePath $reportPath -Encoding utf8
Write-Host "  Full results: $reportPath" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  DONE" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Best config: $($best.Name) = $($best.SolsPerSec) Sols/s" -ForegroundColor Green
Write-Host "  Run mine.bat to start mining!" -ForegroundColor White
Write-Host ""
