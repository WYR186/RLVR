param(
    [int]$StageBSupervisorPid = 27412,
    [string]$ConfigPath = "D:\algoverse\experiment 2\exp2_config_4070_instruct_v9.json"
)

$ErrorActionPreference = "Stop"
$Repo = "D:\algoverse"
$PostStopRoot = "D:\algoverse\eaaj-pilot\outputs\exp2_4070_v8_poststop_72f5cbd815e5"
$StageBComplete = Join-Path $PostStopRoot "stage_b_complete.json"
$V9Wrapper = "D:\algoverse\experiment 2\run_exp2_4070_v9.ps1"
$V9SmokeRoot = "D:\algoverse\experiment 2\smoke_outputs_4070_instruct_v9"
$V9Marker = "D:\algoverse\experiment 2\data\exp2_4070_instruct_v9_phase0_smoke_complete.json"
$V9SplitIdentity = "D:\algoverse\experiment 2\data\exp2_4070_instruct_v9_phase0_split_identity.json"
$LogDir = "D:\algoverse\experiment 2\logs_4070_v9"

Set-Location -LiteralPath $Repo
foreach ($Path in @($ConfigPath, $V9Wrapper, $PostStopRoot)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required v9 handoff path missing: $Path"
    }
}

$Mutex = [Threading.Mutex]::new($false, "Local\AlgoverseExp2V9AfterStageB")
if (-not $Mutex.WaitOne(0)) {
    $Mutex.Dispose()
    throw "Another v9-after-Stage-B handoff is already active"
}
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$HandoffLog = Join-Path $LogDir "v9_after_stageb_${Stamp}.log"
if (Test-Path -LiteralPath $HandoffLog) {
    throw "Refusing to overwrite v9 handoff log: $HandoffLog"
}

function Write-HandoffEvent([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $HandoffLog -Value $Line -Encoding utf8
    Write-Host $Line
}

function Get-ExperimentPython {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(.exe)?$' -and
        ($_.CommandLine -like '*run_exp2_4070_poststop.py*' -or
         $_.CommandLine -like '*run_exp2_4070_v9.py*' -or
         $_.CommandLine -like '*run_exp2_4070.py*')
    })
}

function Assert-V9SmokeComplete {
    foreach ($Stage in @("a", "b")) {
        $StageRoot = Join-Path $V9SmokeRoot "stage_$Stage"
        foreach ($Required in @(
            (Join-Path $StageRoot "smoke_complete.json"),
            (Join-Path $StageRoot "sparse_reward_preflight.json"),
            (Join-Path $StageRoot "trainer\checkpoint-2\config.json")
        )) {
            if (-not (Test-Path -LiteralPath $Required)) {
                throw "v9 smoke evidence missing: $Required"
            }
        }
        foreach ($Forbidden in @(
            (Join-Path $StageRoot "safety_stop.json"),
            (Join-Path $StageRoot "smoke_gate_failure.json")
        )) {
            if (Test-Path -LiteralPath $Forbidden) {
                throw "v9 smoke failure evidence exists: $Forbidden"
            }
        }
    }
    $Preflight = Get-Content -LiteralPath (
        Join-Path $V9SmokeRoot "stage_a\sparse_reward_preflight.json") -Raw |
        ConvertFrom-Json
    if ($Preflight.gate_pass -ne $true -or
        [int]$Preflight.n_prompts -ne 16 -or
        [int]$Preflight.num_generations -ne 8 -or
        [int]$Preflight.combined_groups_with_reward_variance -lt 2) {
        throw "v9 Stage-A preflight failed the registered contract"
    }
    if (-not (Test-Path -LiteralPath $V9Marker)) {
        throw "v9 Phase-0 completion marker missing"
    }
}

try {
    Write-HandoffEvent "handoff waiting for truncated Stage B to complete"
    while (-not (Test-Path -LiteralPath $StageBComplete)) {
        if (-not (Get-Process -Id $StageBSupervisorPid -ErrorAction SilentlyContinue)) {
            throw "Stage-B supervisor exited without stage_b_complete.json"
        }
        Start-Sleep -Seconds 60
    }
    Write-HandoffEvent "stage_b_complete.json observed; waiting for all experiment Python processes to exit"
    while (@(Get-ExperimentPython).Count -gt 0) {
        Start-Sleep -Seconds 30
    }
    Start-Sleep -Seconds 10
    if (Test-Path -LiteralPath (Join-Path $PostStopRoot "stage_b.lock")) {
        throw "Stage-B lock remains after all experiment Python processes exited"
    }
    $Drive = Get-PSDrive -Name D
    if ($Drive.Free -lt 50GB) {
        throw "D: free space is below the 50 GB v9 Phase-0 floor"
    }
    $Gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used `
        --format=csv,noheader,nounits) -split ',\s*'
    if ([int]$Gpu[0] -gt 5 -or [int]$Gpu[1] -gt 500) {
        throw "GPU is not idle after Stage B; refusing v9 Phase 0"
    }
    if ((Test-Path -LiteralPath $V9SmokeRoot) -or
        (Test-Path -LiteralPath $V9Marker)) {
        throw "v9 smoke namespace already exists; refusing automatic overwrite or retry"
    }

    Write-HandoffEvent "running read-only v9 contract check"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $V9Wrapper `
        -Action contract -ConfigPath $ConfigPath
    if ($LASTEXITCODE -ne 0) { throw "v9 contract check failed" }

    Write-HandoffEvent "running v9 prepare and exact split-identity gate"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $V9Wrapper `
        -Action prepare -ConfigPath $ConfigPath
    if ($LASTEXITCODE -ne 0) { throw "v9 prepare failed" }
    if (-not (Test-Path -LiteralPath $V9SplitIdentity)) {
        throw "v9 split-identity marker missing"
    }
    $Identity = Get-Content -LiteralPath $V9SplitIdentity -Raw | ConvertFrom-Json
    if ($Identity.status -ne "exact_id_match") {
        throw "v9 split identity did not pass"
    }

    Write-HandoffEvent "launching registered v9 two-stage CUDA smoke"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $V9Wrapper `
        -Action smoke -ConfigPath $ConfigPath
    if ($LASTEXITCODE -ne 0) { throw "v9 Phase-0 smoke failed" }
    Assert-V9SmokeComplete
    Write-HandoffEvent "v9 Phase 0 passed; formal Stage A intentionally NOT launched pending review"
}
catch {
    Write-HandoffEvent "STOPPED: $($_.Exception.Message)"
    throw
}
finally {
    try { $Mutex.ReleaseMutex() } catch { }
    $Mutex.Dispose()
}
