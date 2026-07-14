# Short real-GPU hard-gate smoke with keep-awake and durable telemetry.
param([ValidateSet(2, 3, 4, 5)][int]$Updates = 2)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Pilot = Join-Path $Repo "eaaj-pilot"
$PythonExe = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
$SourceRun = Join-Path $Pilot "outputs\local_cuda_grpo_gsm8k_e9b0b52aab6c"
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Python missing: $PythonExe" }

$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if (-not ("Win32Kernel.PowerState" -as [type])) {
    Add-Type -Namespace Win32Kernel -Name PowerState -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$Power = "Win32Kernel.PowerState" -as [type]
[void]$Power::SetThreadExecutionState([uint32]2147483649)

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $Pilot "outputs\smoke_stageb_repeat_$Stamp"
$LogDir = Join-Path $Here "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$GpuLog = Join-Path $LogDir "gpu_${Stamp}_stageb_smoke.csv"
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
    & $PythonExe (Join-Path $Pilot "scripts\smoke_stageb_repeat.py") `
        --source-run $SourceRun --updates $Updates --out-dir $OutDir
    if ($LASTEXITCODE -ne 0) { throw "Stage-B smoke exited with code $LASTEXITCODE" }
} finally {
    Stop-Job $GpuJob -ErrorAction SilentlyContinue
    Remove-Job $GpuJob -Force -ErrorAction SilentlyContinue
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    if (Test-Path -LiteralPath $OutDir) {
        Copy-Item -LiteralPath $GpuLog -Destination (Join-Path $OutDir (Split-Path -Leaf $GpuLog)) -Force
    }
    $Mins = [math]::Round(((Get-Date) - $Started).TotalMinutes, 1)
    Write-Host "Stage-B smoke wall time: $Mins min; telemetry $GpuLog"
}
