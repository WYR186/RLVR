# Recover Experiment 1.5 v3 from the Phase-2 dtype mismatch, then continue
# through gated Phase 3. This script never runs Phase 1 or Phase 4.

param(
    [switch]$Phase2Only
)

$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$RunDir = Join-Path $Repo "eaaj-pilot\outputs\exp15_cuda_grpo_gsm8k_c7cc7a1d02d9"
$Config = Join-Path $Here "exp1_5_config_v3_float32_measurement.json"
$Wrapper = Join-Path $Here "run_exp15.ps1"
$Gates = Join-Path $Here "exp15_gates.py"
$Python = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
$LogDir = Join-Path $Here "logs"
$TelemetryDir = Join-Path $RunDir "telemetry"
$RecoveryLog = Join-Path $RunDir "postgate_recovery.jsonl"
$ConfigStem = [IO.Path]::GetFileNameWithoutExtension($Config)
$Float16Archive = Join-Path $RunDir "measurements_float16_v3_gate_stop_20260717"
$Float16Marker = Join-Path $RunDir "phase2_complete_float16_v3_gate_stop_20260717.json"
$Measurements = Join-Path $RunDir "measurements"
$Phase2Marker = Join-Path $RunDir "phase2_complete.json"
$RequiredCheckpoints = @(0, 25, 50, 100, 200, 300, 400, 500)

function Write-RecoveryEvent {
    param([string]$Event, [string]$Status, [hashtable]$Details = @{})
    $row = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        event = $Event
        status = $Status
        details = $Details
    }
    $row | ConvertTo-Json -Compress -Depth 5 |
        Add-Content -LiteralPath $RecoveryLog -Encoding utf8
}

function Copy-RecoveryLogs {
    param([datetime]$Since)
    New-Item -ItemType Directory -Force -Path $TelemetryDir | Out-Null
    Get-ChildItem -LiteralPath $LogDir -File |
        Where-Object {
            $_.LastWriteTime -ge $Since.AddSeconds(-2) -and
            $_.Name -like "*_$ConfigStem.*"
        } |
        ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination $TelemetryDir -Force
        }
}

function Invoke-Gate {
    param([string]$Name)
    & $Python $Gates $Name --config $Config
    $code = $LASTEXITCODE
    $status = if ($code -eq 0) { "pass" } else { "stop" }
    Write-RecoveryEvent -Event "gate:$Name" -Status $status -Details @{ exit_code = $code }
    if ($code -ne 0) {
        throw "Gate '$Name' did not PASS (exit $code). No later phase was started."
    }
}

function Invoke-RecordedPhase {
    param([string]$Phase, [string[]]$ExtraArgs = @())
    $started = Get-Date
    Write-RecoveryEvent -Event "phase:$Phase" -Status "started" -Details @{ args = $ExtraArgs }
    $phaseCommandArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Wrapper,
                          "-Phase", $Phase, "-ConfigPath", $Config) + $ExtraArgs
    try {
        & powershell.exe @phaseCommandArgs
        if ($LASTEXITCODE -ne 0) {
            throw "run_exp15.ps1 Phase $Phase exited with code $LASTEXITCODE"
        }
        Write-RecoveryEvent -Event "phase:$Phase" -Status "complete" -Details @{ args = $ExtraArgs }
    } catch {
        Write-RecoveryEvent -Event "phase:$Phase" -Status "failed" -Details @{
            args = $ExtraArgs
            error = $_.Exception.Message
        }
        throw
    } finally {
        Copy-RecoveryLogs -Since $started
    }
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
if (-not (Test-Path -LiteralPath $Config)) { throw "Recovery config missing: $Config" }
if (-not (Test-Path -LiteralPath (Join-Path $RunDir "phase1_complete.json"))) {
    throw "v3 Phase 1 is not complete: $RunDir"
}

$gpuProcesses = @(
    & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null |
        Where-Object { $_ -and $_.Trim() -and $_ -notmatch '\[N/A\]' }
)
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed; refusing to start" }
if ($gpuProcesses.Count -gt 0) {
    throw "CUDA compute device is busy: $($gpuProcesses -join '; ')"
}

foreach ($checkpoint in $RequiredCheckpoints) {
    $configPath = Join-Path $RunDir "ckpt-$checkpoint\config.json"
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "required v3 checkpoint is missing: $configPath"
    }
}

New-Item -ItemType Directory -Force -Path $TelemetryDir | Out-Null
Invoke-Gate -Name "rundir"

if (-not (Test-Path -LiteralPath $Float16Archive)) {
    if (-not (Test-Path -LiteralPath $Measurements)) {
        throw "neither canonical measurements nor the float16 archive exists"
    }
    $oldMetric = Get-Content -LiteralPath (Join-Path $Measurements "metrics_ckpt0.json") -Raw |
        ConvertFrom-Json
    if ($oldMetric.measurement_contract.model_dtype -ne "torch.float16") {
        throw "refusing to archive unexpected measurement dtype: $($oldMetric.measurement_contract.model_dtype)"
    }
    Move-Item -LiteralPath $Measurements -Destination $Float16Archive
    if (Test-Path -LiteralPath $Phase2Marker) {
        Move-Item -LiteralPath $Phase2Marker -Destination $Float16Marker
    }
    Write-RecoveryEvent -Event "archive_float16_phase2" -Status "complete" -Details @{
        measurements = $Float16Archive
        marker = $Float16Marker
    }
} else {
    $archivedMetric = Get-Content -LiteralPath (Join-Path $Float16Archive "metrics_ckpt0.json") -Raw |
        ConvertFrom-Json
    if ($archivedMetric.measurement_contract.model_dtype -ne "torch.float16") {
        throw "float16 evidence archive has an unexpected dtype"
    }
}

$phase2Ready = $false
if ((Test-Path -LiteralPath $Phase2Marker) -and
        (Test-Path -LiteralPath (Join-Path $Measurements "metrics_ckpt0.json"))) {
    $currentMetric = Get-Content -LiteralPath (Join-Path $Measurements "metrics_ckpt0.json") -Raw |
        ConvertFrom-Json
    $phase2Ready = $currentMetric.measurement_contract.model_dtype -eq "torch.float32"
}
if (-not $phase2Ready) {
    Invoke-RecordedPhase -Phase "2"
}
Invoke-Gate -Name "ckpt0"

if ($Phase2Only) {
    Write-RecoveryEvent -Event "recovery" -Status "phase2_only_complete"
    Write-Host "Float32 Phase 2 and ckpt-0 gate completed. Phase 3 was not requested."
    exit 0
}

$BridgeSummary = Join-Path $RunDir "adaptation_seed42\ckpt-0\summary.json"
if (-not (Test-Path -LiteralPath $BridgeSummary)) {
    Invoke-RecordedPhase -Phase "3" -ExtraArgs @("-AdaptCheckpoint", "0", "-AdaptSeed", "42")
}
Invoke-Gate -Name "bridge"

Invoke-RecordedPhase -Phase "3"
if (-not (Test-Path -LiteralPath (Join-Path $RunDir "phase3_complete.json"))) {
    throw "Phase 3 returned without phase3_complete.json"
}

Write-RecoveryEvent -Event "recovery" -Status "complete" -Details @{ phase4_run = $false }
Write-Host "Experiment 1.5 recovery complete through Phase 3. Phase 4 was not run."
