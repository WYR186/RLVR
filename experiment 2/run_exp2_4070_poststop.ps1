param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("audit", "zero-shot", "q", "phase2-finalize", "stageb-cell", "stageb-finalize", "status")]
    [string]$Action,
    [string]$ConfigPath = "exp2_config_4070_instruct_v8_poststop.json",
    [int]$Checkpoint = -1,
    [int]$Seed = -1
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Python = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
$Runner = Join-Path $Here "run_exp2_4070_poststop.py"
$Config = if ([IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $Here $ConfigPath
}
foreach ($Path in @($Python, $Runner, $Config)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required path missing: $Path" }
}

$ArgsList = @($Runner, "--action", $Action, "--config", $Config)
if ($Checkpoint -ge 0) { $ArgsList += @("--checkpoint", "$Checkpoint") }
if ($Seed -ge 0) { $ArgsList += @("--seed", "$Seed") }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$GpuActions = @("zero-shot", "q", "stageb-cell")
if ($Action -notin $GpuActions) {
    & $Python @ArgsList
    exit $LASTEXITCODE
}

if (-not ("Win32Kernel.Exp2PostStopPowerState" -as [type])) {
    Add-Type -Namespace Win32Kernel -Name Exp2PostStopPowerState -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$Power = "Win32Kernel.Exp2PostStopPowerState" -as [type]
[void]$Power::SetThreadExecutionState([uint32]2147483649)

$LogDir = Join-Path $Here "logs_4070_poststop"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Cell = if ($Checkpoint -ge 0) { "_ckpt${Checkpoint}" } else { "" }
if ($Seed -ge 0) { $Cell += "_seed${Seed}" }
$GpuLog = Join-Path $LogDir "gpu_${Stamp}_${Action}${Cell}.csv"
$RunLog = Join-Path $LogDir "run_${Stamp}_${Action}${Cell}.log"
$GpuJob = Start-Job -ScriptBlock {
    param($OutFile)
    "timestamp,name,utilization.gpu [%],memory.used [MiB],memory.total [MiB],temperature.gpu,power.draw [W]" |
        Set-Content -LiteralPath $OutFile -Encoding utf8
    while ($true) {
        & nvidia-smi "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw" `
            --format=csv,noheader,nounits | Add-Content -LiteralPath $OutFile -Encoding utf8
        Start-Sleep -Seconds 30
    }
} -ArgumentList $GpuLog

try {
    $ErrorActionPreference = "Continue"
    & $Python @ArgsList 2>&1 | Tee-Object -LiteralPath $RunLog
    $Code = $LASTEXITCODE
    if ($Code -ne 0) { throw "exp2 post-stop $Action exited with code $Code" }
} finally {
    Stop-Job $GpuJob -ErrorAction SilentlyContinue
    Remove-Job $GpuJob -Force -ErrorAction SilentlyContinue
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    Write-Host "GPU telemetry: $GpuLog"
    Write-Host "Transcript:    $RunLog"
}
