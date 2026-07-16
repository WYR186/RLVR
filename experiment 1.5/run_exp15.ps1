# Phase wrapper for experiment 1.5 on the Windows RTX 4070 Laptop box.
#
#   powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 1
#   powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase 3 -AdaptCheckpoint 500 -AdaptSeed 43
#   powershell -ExecutionPolicy Bypass -File "experiment 1.5\run_exp15.ps1" -Phase all -Smoke
#
# What it adds around "experiment 1.5/run_exp1_5.py --backend cuda":
#   - keep-awake while training
#   - nvidia-smi CSV telemetry every 60 s (compute-accounting evidence)
#   - full console transcript to experiment 1.5\logs\ (crash evidence)
#   - tolerates native stderr noise (triton/bitsandbytes warnings) and judges
#     success by the process exit code only — the 2026-07-14 seed-43 lesson.
# All phases are resumable: re-run the same command after any interruption.

param(
    [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3", "4", "all")]
    [string]$Phase,
    [int]$AdaptCheckpoint = -1,
    [int]$AdaptSeed = -1,
    [switch]$KeepTrainerDirs,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
# PowerShell 7.3+: never promote native stderr lines to errors.
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path      # experiment 1.5
$Repo = Split-Path -Parent $Here
$Pilot = Join-Path $Repo "eaaj-pilot"
$RepoCondaPy = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
$Win4070Venv = Join-Path $Repo "eaaj-pilot-win4070\.venv\Scripts\python.exe"
if ($env:EAAJ_PYTHON -and (Test-Path $env:EAAJ_PYTHON)) {
    $PythonExe = $env:EAAJ_PYTHON
} elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
    $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
} elseif (Test-Path $RepoCondaPy) {
    $PythonExe = $RepoCondaPy
} elseif (Test-Path $Win4070Venv) {
    $PythonExe = $Win4070Venv
} else {
    throw "no Python environment found - use the same env as the pilot v2 runs (setup_win4070.ps1)"
}

$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# keep-awake (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
if (-not ("Win32Kernel.PowerState" -as [type])) {
    Add-Type -Namespace Win32Kernel -Name PowerState -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$Power = "Win32Kernel.PowerState" -as [type]
[void]$Power::SetThreadExecutionState([uint32]2147483649)

# GPU telemetry + console transcript
$LogDir = Join-Path $Here "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Suffix = "exp15_phase$Phase"
if ($AdaptCheckpoint -ge 0) { $Suffix += "_ck$AdaptCheckpoint" }
if ($AdaptSeed -ge 0) { $Suffix += "_s$AdaptSeed" }
if ($Smoke) { $Suffix += "_smoke" }
$GpuLog = Join-Path $LogDir "gpu_${Stamp}_${Suffix}.csv"
$RunLog = Join-Path $LogDir "run_${Stamp}_${Suffix}.log"
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
try {
    $CliArgs = @((Join-Path $Here "run_exp1_5.py"),
                 "--phase", $Phase, "--backend", "cuda")
    if ($AdaptCheckpoint -ge 0) { $CliArgs += @("--adapt-checkpoint", "$AdaptCheckpoint") }
    if ($AdaptSeed -ge 0) { $CliArgs += @("--adapt-seed", "$AdaptSeed") }
    if ($KeepTrainerDirs) { $CliArgs += "--keep-trainer-dirs" }
    if ($Smoke) { $CliArgs += "--smoke" }

    # PowerShell 5 wraps native stderr lines as NativeCommandError records.
    # Preserve them in the transcript, but trust the exit code rather than
    # turning harmless warnings into a wrapper failure (2026-07-14 lesson).
    $SavedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PythonExe @CliArgs 2>&1 | Tee-Object -LiteralPath $RunLog
        $RunnerExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $SavedErrorActionPreference
    }
    if ($RunnerExitCode -ne 0) { throw "exp1.5 runner exited with code $RunnerExitCode (transcript: $RunLog)" }
} finally {
    Stop-Job $GpuJob -ErrorAction SilentlyContinue
    Remove-Job $GpuJob -Force -ErrorAction SilentlyContinue
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    $Mins = [math]::Round(((Get-Date) - $Started).TotalMinutes, 1)
    Write-Host ""
    Write-Host "exp1.5 phase $Phase wall time: $Mins min"
    Write-Host "GPU telemetry: $GpuLog"
    Write-Host "Transcript:    $RunLog"
    Write-Host "Next: copy the telemetry CSV into the run dir's telemetry\ folder,"
    Write-Host "append a row to eaaj-pilot/compute_log.md, and commit artifacts"
    Write-Host "(see WIN4070_EXP15_GUIDE.md sections 9-10)."
}
