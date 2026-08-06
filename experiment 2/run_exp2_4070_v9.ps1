param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("contract", "prepare", "smoke", "stage1", "status")]
    [string]$Action,
    [string]$ConfigPath = "D:\algoverse\experiment 2\exp2_config_4070_instruct_v9.json"
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$Repo = "D:\algoverse"
$Here = "D:\algoverse\experiment 2"
$Python = "D:\algoverse\.conda\envs\eaaj-win4070\python.exe"
$Runner = "D:\algoverse\experiment 2\run_exp2_4070_v9.py"
$Config = if ([IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath
} else {
    Join-Path $Here $ConfigPath
}
Set-Location -LiteralPath $Repo
foreach ($Path in @($Python, $Runner, $Config)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required v9 path missing: $Path"
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$ArgsList = @($Runner, "--action", $Action, "--config", $Config)

if ($Action -in @("contract", "prepare", "status")) {
    & $Python @ArgsList
    exit $LASTEXITCODE
}

if (-not ("Win32Kernel.Exp2V9PowerState" -as [type])) {
    Add-Type -Namespace Win32Kernel -Name Exp2V9PowerState -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$Power = "Win32Kernel.Exp2V9PowerState" -as [type]
[void]$Power::SetThreadExecutionState([uint32]2147483649)

$LogDir = "D:\algoverse\experiment 2\logs_4070_v9"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ConfigStem = [IO.Path]::GetFileNameWithoutExtension($Config)
$GpuLog = Join-Path $LogDir "gpu_${Stamp}_exp2_v9_${Action}.csv"
$RunLog = Join-Path $LogDir "run_${Stamp}_${ConfigStem}_${Action}.log"
foreach ($Path in @($GpuLog, $RunLog)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite v9 log: $Path"
    }
}
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
    if ($Code -ne 0) {
        throw "exp2 v9 $Action exited with code $Code"
    }
}
finally {
    Stop-Job $GpuJob -ErrorAction SilentlyContinue
    Remove-Job $GpuJob -Force -ErrorAction SilentlyContinue
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    Write-Host "GPU telemetry: $GpuLog"
    Write-Host "Transcript:    $RunLog"
}
