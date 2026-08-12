param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("prepare", "smoke", "stage1", "status")]
    [string]$Action,
    [string]$ConfigPath = "exp2_config_4070.json"
)

$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Python = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$Config = if ([IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $Here $ConfigPath }
if (-not (Test-Path -LiteralPath $Config)) { throw "Config missing: $Config" }

if ($Action -eq "prepare" -or $Action -eq "status") {
    & $Python (Join-Path $Here "run_exp2_4070.py") --action $Action --config $Config
    exit $LASTEXITCODE
}

if (-not ("Win32Kernel.Exp2PowerState" -as [type])) {
    Add-Type -Namespace Win32Kernel -Name Exp2PowerState -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$Power = "Win32Kernel.Exp2PowerState" -as [type]
[void]$Power::SetThreadExecutionState([uint32]2147483649)

$LogDir = Join-Path $Here "logs_4070"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$GpuLog = Join-Path $LogDir "gpu_${Stamp}_exp2_4070_${Action}.csv"
$ConfigStem = [IO.Path]::GetFileNameWithoutExtension($Config)
$RunLog = Join-Path $LogDir "run_${Stamp}_exp2_${ConfigStem}_${Action}.log"
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
    & $Python (Join-Path $Here "run_exp2_4070.py") --action $Action --config $Config 2>&1 |
        Tee-Object -LiteralPath $RunLog
    $Code = $LASTEXITCODE
    if ($Code -ne 0) { throw "exp2 4070 $Action exited with code $Code" }
} finally {
    Stop-Job $GpuJob -ErrorAction SilentlyContinue
    Remove-Job $GpuJob -Force -ErrorAction SilentlyContinue
    [void]$Power::SetThreadExecutionState([uint32]2147483648)
    Write-Host "GPU telemetry: $GpuLog"
    Write-Host "Transcript:    $RunLog"
}
