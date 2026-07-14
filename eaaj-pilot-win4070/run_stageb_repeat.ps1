# Run exactly one isolated Stage-B seed x checkpoint repeat on the RTX 4070.
param(
    [Parameter(Mandatory = $true)][ValidateSet(43, 44)][int]$Seed,
    [Parameter(Mandatory = $true)][ValidateSet(0, 25, 50, 100, 200)][int]$Checkpoint,
    [string]$SourceRun = "D:\algoverse\eaaj-pilot\outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Pilot = Join-Path $Repo "eaaj-pilot"
$PythonExe = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "eaaj-win4070 Python missing: $PythonExe" }

$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$MutexName = "Local\EAAJ_Win4070_StageB_Repeat_e9b0b52aab6c"
$Mutex = [System.Threading.Mutex]::new($false, $MutexName)
$MutexAcquired = $false
try {
    $MutexAcquired = $Mutex.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    $MutexAcquired = $true
}
if (-not $MutexAcquired) {
    $Mutex.Dispose()
    throw "another Stage-B repeat wrapper is active; do not run two trainers concurrently"
}

if (-not ("Win32Kernel.PowerState" -as [type])) {
    Add-Type -Namespace Win32Kernel -Name PowerState -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$Power = "Win32Kernel.PowerState" -as [type]
[void]$Power::SetThreadExecutionState([uint32]2147483649)

$LogDir = Join-Path $Here "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$GpuLog = Join-Path $LogDir "gpu_${Stamp}_stageb_seed${Seed}_ckpt${Checkpoint}.csv"
$GpuJob = Start-Job -ScriptBlock {
    param($OutFile)
    "timestamp,name,utilization.gpu [%],memory.used [MiB],memory.total [MiB],temperature.gpu,power.draw [W],clocks.sm [MHz]" |
        Set-Content -LiteralPath $OutFile -Encoding utf8
    while ($true) {
        & nvidia-smi "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm" `
            --format=csv,noheader,nounits | Add-Content -LiteralPath $OutFile -Encoding utf8
        Start-Sleep -Seconds 60
    }
} -ArgumentList $GpuLog

$Started = Get-Date
$OutDir = Join-Path $SourceRun "adaptation_repeats\seed-$Seed\ckpt-$Checkpoint"
$AttemptId = [guid]::NewGuid().ToString("N")
$AttemptMarker = Join-Path $OutDir ".repeat_attempt.json"
$StatusPath = Join-Path $LogDir "stageb_${Stamp}_${AttemptId}_status.json"
$RunMode = $null
$Failure = $null
$Captured = @()

function Test-AttemptOwnership {
    if (-not (Test-Path -LiteralPath $AttemptMarker)) { return $false }
    try {
        $Marker = Get-Content -LiteralPath $AttemptMarker -Raw | ConvertFrom-Json
        return $Marker.attempt_id -eq $AttemptId
    } catch {
        return $false
    }
}

try {
    $CliArgs = @(
        (Join-Path $Pilot "scripts\run_stageb_seed_repeat.py"),
        "--source-run", $SourceRun,
        "--seed", "$Seed",
        "--checkpoint", "$Checkpoint",
        "--attempt-id", $AttemptId,
        "--status-path", $StatusPath
    )
    & $PythonExe @CliArgs 2>&1 | Tee-Object -Variable Captured
    if ($LASTEXITCODE -ne 0) { throw "Stage-B repeat runner exited with code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $StatusPath)) {
        throw "runner status is missing: $StatusPath"
    }
    $RunStatus = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
    if ($RunStatus.attempt_id -ne $AttemptId) {
        throw "runner status attempt ID does not match $AttemptId"
    }
    $RunMode = $RunStatus.mode
    if ($RunMode -eq "new_attempt" -and -not (Test-AttemptOwnership)) {
        throw "runner did not establish ownership of $OutDir for attempt $AttemptId"
    }
    if ($RunMode -notin @("new_attempt", "existing_valid")) {
        throw "unexpected runner mode: $RunMode"
    }
    if ($RunMode -eq "new_attempt") {
        $TelemetryReady = $false
        for ($i = 0; $i -lt 10; $i++) {
            if ((Test-Path -LiteralPath $GpuLog) -and
                ((Get-Content -LiteralPath $GpuLog).Count -gt 1)) {
                $TelemetryReady = $true
                break
            }
            Start-Sleep -Milliseconds 500
        }
        if ($GpuJob.State -ne "Running") {
            Receive-Job $GpuJob -ErrorAction SilentlyContinue | Write-Warning
            throw "GPU telemetry job stopped unexpectedly with state $($GpuJob.State)"
        }
        if (-not $TelemetryReady) {
            throw "GPU telemetry is missing or contains no samples: $GpuLog"
        }
    }
} catch {
    $Failure = $_
} finally {
    Stop-Job $GpuJob -ErrorAction SilentlyContinue
    Remove-Job $GpuJob -Force -ErrorAction SilentlyContinue
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    try {
        $Duration = (Get-Date) - $Started

    $OwnsOutDir = Test-AttemptOwnership
    if ($Failure -and $OwnsOutDir -and (Test-Path -LiteralPath $OutDir)) {
        $Reason = "run"
        $Text = (($Captured | Out-String) + " " + $Failure)
        if (Test-Path -LiteralPath (Join-Path $OutDir "safety_stop.json")) { $Reason = "safety" }
        elseif (Test-Path -LiteralPath (Join-Path $OutDir "baseline_mismatch.json")) { $Reason = "baseline" }
        elseif ($Text -match "out of memory|CUDA OOM") { $Reason = "oom" }
        $FailedDir = Join-Path (Split-Path -Parent $OutDir) "ckpt-${Checkpoint}_failed_${Reason}_${Stamp}"
        Move-Item -LiteralPath $OutDir -Destination $FailedDir
        $OutDir = $FailedDir
    }
    $CanAttachTelemetry = (($RunMode -eq "new_attempt") -and (-not $Failure)) -or
        ($Failure -and $OwnsOutDir)
    if ($CanAttachTelemetry -and (Test-Path -LiteralPath $OutDir)) {
        Copy-Item -LiteralPath $GpuLog -Destination (Join-Path $OutDir (Split-Path -Leaf $GpuLog)) -Force
    }
    if (-not $Failure -and $OwnsOutDir -and
        (Test-Path -LiteralPath $AttemptMarker)) {
        Remove-Item -LiteralPath $AttemptMarker -Force
    }
    $Status = if ($Failure) { "failed" } elseif ($RunMode -eq "existing_valid") {
        "validated_existing"
    } else { "complete" }
    $Minutes = [math]::Round($Duration.TotalMinutes, 1)
    $ComputeLog = Join-Path $Pilot "compute_log.md"
    $Row = "| $(Get-Date -Format 'yyyy-MM-dd HH:mm') | Stage-B repeat seed $Seed ckpt $Checkpoint | RTX 4070 Laptop | $Minutes min | $Status; telemetry ``$(Split-Path -Leaf $GpuLog)`` |"
    Add-Content -LiteralPath $ComputeLog -Value $Row -Encoding utf8
        Write-Host "Stage-B seed $Seed ckpt ${Checkpoint}: $Status in $Minutes min; telemetry $GpuLog"
    } finally {
        if ($MutexAcquired) { [void]$Mutex.ReleaseMutex() }
        $Mutex.Dispose()
    }
}

if ($Failure) { throw $Failure }
