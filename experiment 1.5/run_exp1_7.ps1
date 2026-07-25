# Orchestrator for the three replicated Stage-A trajectories in experiment 1.7.
# Each underlying phase remains resumable and is recorded by run_exp15.ps1.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("stagea", "measure", "probe", "status", "expand")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Wrapper = Join-Path $Here "run_exp15.ps1"
$Gates = Join-Path $Here "exp15_gates.py"
$Exp17Gate = Join-Path $Here "exp1_7_gate_eval.py"
$Python = Join-Path $Repo ".conda\envs\eaaj-win4070\python.exe"
$Configs = @(
    (Join-Path $Here "exp1_7_config_seed42.json"),
    (Join-Path $Here "exp1_7_config_seed43.json"),
    (Join-Path $Here "exp1_7_config_seed44.json")
)

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment missing: $Python"
}

function Invoke-Phase {
    param(
        [string]$Config,
        [string]$Phase,
        [int]$AdaptSeed = -1
    )
    $args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Wrapper,
              "-Phase", $Phase, "-ConfigPath", $Config)
    if ($AdaptSeed -ge 0) {
        $args += @("-AdaptSeed", "$AdaptSeed")
    }
    & powershell.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "phase $Phase failed for $([IO.Path]::GetFileName($Config))"
    }
}

function Get-RunDir {
    param([string]$Config)
    $cfg = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
    return Join-Path $Repo ("eaaj-pilot\outputs\" + $cfg.expected_run_name)
}

if ($Action -eq "status") {
    & $Python $Exp17Gate
    exit $LASTEXITCODE
}

if ($Action -eq "stagea") {
    foreach ($config in $Configs) {
        & $Python $Gates rundir --config $config
        if ($LASTEXITCODE -ne 0) {
            throw "run-dir gate failed for $config"
        }
        Invoke-Phase -Config $config -Phase "1"
    }
    Write-Host "All three Stage-A launch/resume calls completed."
    exit 0
}

if ($Action -eq "measure") {
    foreach ($config in $Configs) {
        Invoke-Phase -Config $config -Phase "2"
        & $Python $Gates ckpt0 --config $config
        if ($LASTEXITCODE -ne 0) {
            throw "ckpt-0 measurement gate failed for $config"
        }
    }
    Write-Host "All available Stage-A trajectories were measured."
    exit 0
}

if ($Action -eq "probe") {
    foreach ($config in $Configs) {
        $runDir = Get-RunDir -Config $config
        if (-not (Test-Path -LiteralPath (Join-Path $runDir "ckpt-500\config.json"))) {
            Write-Host "Skipping endpoint probe for incomplete Stage-A: $runDir"
            continue
        }
        Invoke-Phase -Config $config -Phase "3" -AdaptSeed 42
    }
    & $Python $Exp17Gate
    exit $LASTEXITCODE
}

if ($Action -eq "expand") {
    & $Python $Exp17Gate
    if ($LASTEXITCODE -ne 0) {
        throw "exp1.7 gate did not print EXPAND; expansion is locked"
    }
    foreach ($config in $Configs) {
        $runDir = Get-RunDir -Config $config
        if (-not (Test-Path -LiteralPath (Join-Path $runDir "ckpt-500\config.json"))) {
            Write-Host "Skipping expansion for incomplete Stage-A: $runDir"
            continue
        }
        Invoke-Phase -Config $config -Phase "3"
    }
    Write-Host "Expanded endpoint replication completed for all trajectories."
    exit 0
}

throw "unsupported action: $Action"
