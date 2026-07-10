# Phase wrapper for the cuda stratum on the Windows RTX 4070 Laptop box.
#
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 1
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 3 -AdaptCheckpoint 100
#
# What it adds around eaaj-pilot/scripts/run_local_pipeline.py --backend cuda:
#   - keep-awake while training (caffeinate analog; display may still sleep)
#   - nvidia-smi CSV telemetry every 60 s (local analog of the Colab
#     GPU-dashboard screenshots required by compute accounting)
#   - refuses phases 2-4 when ACTIVE_RUN.txt does not point at the cuda
#     stratum (e.g. fresh clone where phase 1 has not run yet)
# All phases are resumable: re-run the same command after any interruption.

param(
    [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3", "4", "all")]
    [string]$Phase,
    [int]$AdaptCheckpoint = -1
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Pilot = Join-Path $Repo "eaaj-pilot"
$RepoCondaPy = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
$VenvPy = Join-Path $Here ".venv\Scripts\python.exe"
if ($env:EAAJ_PYTHON -and (Test-Path $env:EAAJ_PYTHON)) {
    $PythonExe = $env:EAAJ_PYTHON
} elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
    $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
} elseif (Test-Path $RepoCondaPy) {
    $PythonExe = $RepoCondaPy
} elseif (Test-Path $VenvPy) {
    $PythonExe = $VenvPy
} else {
    throw "no Python environment found - run setup_win4070.ps1, or create D:\algoverse\.conda\envs\eaaj-win4070"
}

$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

# Phases 2-4 follow outputs/ACTIVE_RUN.txt (machine-local, untracked). Refuse
# to run them against a non-cuda run dir - that would mix backend strata.
if ($Phase -ne "1" -and $Phase -ne "all") {
    $Marker = Join-Path $Pilot "outputs\ACTIVE_RUN.txt"
    if (-not (Test-Path $Marker)) {
        throw "outputs/ACTIVE_RUN.txt missing - run '-Phase 1' (or 'all') on this machine first"
    }
    $Active = (Get-Content $Marker -Raw).Trim()
    if ($Active -notlike "*local_cuda_grpo_gsm8k_*") {
        throw "ACTIVE_RUN points at '$Active', not the cuda stratum - run '-Phase 1' here first"
    }
}

# keep-awake (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
if (-not ("Win32Kernel.PowerState" -as [type])) {
    Add-Type -Namespace Win32Kernel -Name PowerState -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$Power = "Win32Kernel.PowerState" -as [type]
[void]$Power::SetThreadExecutionState([uint32]2147483649)

# GPU telemetry (compute-accounting evidence)
$LogDir = Join-Path $Here "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$GpuLog = Join-Path $LogDir "gpu_${Stamp}_phase$Phase.csv"
$GpuJob = Start-Job -ScriptBlock {
    param($OutFile)
    "timestamp,name,utilization.gpu [%],memory.used [MiB],memory.total [MiB],temperature.gpu,power.draw [W],clocks.sm [MHz]" |
        Set-Content -Path $OutFile -Encoding utf8
    while ($true) {
        & nvidia-smi "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm" `
            --format=csv,noheader,nounits | Add-Content -Path $OutFile -Encoding utf8
        Start-Sleep -Seconds 60
    }
} -ArgumentList $GpuLog

$Started = Get-Date
try {
    $CliArgs = @((Join-Path $Pilot "scripts\run_local_pipeline.py"),
                 "--phase", $Phase, "--backend", "cuda")
    if ($AdaptCheckpoint -ge 0) { $CliArgs += @("--adapt-checkpoint", "$AdaptCheckpoint") }
    & $PythonExe @CliArgs
    if ($LASTEXITCODE -ne 0) { throw "pipeline exited with code $LASTEXITCODE" }
} finally {
    Stop-Job $GpuJob -ErrorAction SilentlyContinue
    Remove-Job $GpuJob -Force -ErrorAction SilentlyContinue
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    $Mins = [math]::Round(((Get-Date) - $Started).TotalMinutes, 1)
    Write-Host ""
    Write-Host "Phase $Phase wall time: $Mins min | GPU telemetry: $GpuLog"
    Write-Host "Now append a row to eaaj-pilot/compute_log.md (date, GPU, phase, duration, telemetry file) and commit the run artifacts."
}
