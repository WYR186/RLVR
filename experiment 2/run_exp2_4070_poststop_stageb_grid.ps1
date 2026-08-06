param(
    [string]$ConfigPath = "D:\algoverse\experiment 2\exp2_config_4070_instruct_v8_poststop.json"
)

$ErrorActionPreference = "Stop"
$Repo = "D:\algoverse"
$Wrapper = "D:\algoverse\experiment 2\run_exp2_4070_poststop.ps1"
$Output = "D:\algoverse\eaaj-pilot\outputs\exp2_4070_v8_poststop_72f5cbd815e5"
$LogDir = "D:\algoverse\experiment 2\logs_4070_poststop"

Set-Location -LiteralPath $Repo
foreach ($Path in @($ConfigPath, $Wrapper, $Output, (Join-Path $Output "phase2_complete.json"))) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required Stage-B grid path missing: $Path"
    }
}

$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$ExpectedCheckpoints = @(0, 50, 100)
$ExpectedSeeds = @(42, 43, 44)
if ((Compare-Object @($Config.stage_b.checkpoints) $ExpectedCheckpoints) -or
    (Compare-Object @($Config.stage_b.seeds) $ExpectedSeeds) -or
    [int]$Config.stage_b.budget_updates -ne 50) {
    throw "Refusing to supervise an unexpected Stage-B grid"
}

$Mutex = [Threading.Mutex]::new($false, "Local\AlgoverseExp2V8PostStopStageBGrid")
if (-not $Mutex.WaitOne(0)) {
    $Mutex.Dispose()
    throw "Another Stage-B grid supervisor is already active"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SupervisorLog = Join-Path $LogDir "stageb_grid_supervisor_${Stamp}.log"
if (Test-Path -LiteralPath $SupervisorLog) {
    throw "Refusing to overwrite supervisor log: $SupervisorLog"
}

function Write-SupervisorEvent([string]$Message) {
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $SupervisorLog -Value $Line -Encoding utf8
    Write-Host $Line
}

function Get-PostStopPython {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(.exe)?$' -and
        $_.CommandLine -like '*run_exp2_4070_poststop.py*'
    })
}

function Wait-ForCurrentCell {
    $SawProcess = $false
    while ($true) {
        $Active = @(Get-PostStopPython)
        if ($Active.Count -eq 0) { break }
        $SawProcess = $true
        $Description = ($Active | ForEach-Object { "pid=$($_.ProcessId)" }) -join ","
        Write-SupervisorEvent "waiting for active post-stop process ($Description)"
        Start-Sleep -Seconds 60
    }
    if ($SawProcess) {
        Start-Sleep -Seconds 5
    }
}

function Assert-CellComplete([int]$Checkpoint, [int]$Seed) {
    $Cell = Join-Path $Output "stage_b\seed-$Seed\ckpt-$Checkpoint"
    $Summary = Join-Path $Cell "summary.json"
    $SafetyStop = Join-Path $Cell "safety_stop.json"
    if (Test-Path -LiteralPath $SafetyStop) {
        throw "Stage-B safety stop exists for ckpt=$Checkpoint seed=$Seed"
    }
    if (-not (Test-Path -LiteralPath $Summary)) {
        throw "Stage-B summary missing for ckpt=$Checkpoint seed=$Seed; refusing automatic retry"
    }
    $CurvePath = Join-Path $Cell "codeio_eval_curve.jsonl"
    $SentinelPath = Join-Path $Cell "update_sentinel.jsonl"
    $DashboardPath = Join-Path $Cell "dashboard.jsonl"
    foreach ($Required in @($CurvePath, $SentinelPath, $DashboardPath)) {
        if (-not (Test-Path -LiteralPath $Required)) {
            throw "Stage-B evidence missing: $Required"
        }
    }
    $Value = Get-Content -LiteralPath $Summary -Raw | ConvertFrom-Json
    if ($Value.completion_status -ne "complete" -or
        [int]$Value.actual_updates -ne 50 -or
        [int]$Value.checkpoint -ne $Checkpoint -or
        [int]$Value.seed -ne $Seed) {
        throw "Stage-B summary failed supervisor validation for ckpt=$Checkpoint seed=$Seed"
    }
    $CurveSteps = @(Get-Content -LiteralPath $CurvePath | Where-Object { $_.Trim() } |
        ForEach-Object { [int](($_ | ConvertFrom-Json).step) })
    if ((Compare-Object $CurveSteps @(0, 10, 20, 30, 40, 50)) -or $CurveSteps.Count -ne 6) {
        throw "Stage-B eval curve failed supervisor validation for ckpt=$Checkpoint seed=$Seed"
    }
    $Sentinels = @(Get-Content -LiteralPath $SentinelPath | Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json })
    $SentinelSteps = @($Sentinels | ForEach-Object { [int]$_.step })
    if ((Compare-Object $SentinelSteps @(10, 20, 30, 40, 50)) -or
        $SentinelSteps.Count -ne 5 -or
        @($Sentinels | Where-Object { $_.updates_effective -ne $true }).Count -ne 0) {
        throw "Stage-B update sentinel failed supervisor validation for ckpt=$Checkpoint seed=$Seed"
    }
    $DashboardSteps = @(Get-Content -LiteralPath $DashboardPath | Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $null -ne $_.loss -or $null -ne $_.grad_norm } |
        ForEach-Object { [int]$_.step } | Sort-Object -Unique)
    if ((Compare-Object $DashboardSteps @(1..50)) -or $DashboardSteps.Count -ne 50) {
        throw "Stage-B dashboard failed supervisor validation for ckpt=$Checkpoint seed=$Seed"
    }
}

try {
    Write-SupervisorEvent "supervisor started; repo=$Repo"
    Wait-ForCurrentCell

    foreach ($Seed in $ExpectedSeeds) {
        foreach ($Checkpoint in $ExpectedCheckpoints) {
            $Cell = Join-Path $Output "stage_b\seed-$Seed\ckpt-$Checkpoint"
            $Summary = Join-Path $Cell "summary.json"
            if (Test-Path -LiteralPath $Summary) {
                Assert-CellComplete -Checkpoint $Checkpoint -Seed $Seed
                Write-SupervisorEvent "validated existing complete cell ckpt=$Checkpoint seed=$Seed"
                continue
            }

            if (Test-Path -LiteralPath (Join-Path $Output "stage_b.lock")) {
                throw "stage_b.lock exists without an active post-stop Python process"
            }
            $PartialFiles = @(Get-ChildItem -LiteralPath $Cell -Force -ErrorAction SilentlyContinue)
            if ($PartialFiles.Count -gt 0) {
                throw "partial Stage-B cell exists for ckpt=$Checkpoint seed=$Seed; refusing overwrite or retry"
            }
            $Drive = Get-PSDrive -Name D
            if ($Drive.Free -lt 50GB) {
                throw "D: free space is below the 50 GB supervisor floor"
            }
            if (@(Get-PostStopPython).Count -ne 0) {
                throw "parallel post-stop Python process appeared before cell launch"
            }

            Write-SupervisorEvent "launching ckpt=$Checkpoint seed=$Seed"
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Wrapper `
                -Action stageb-cell -ConfigPath $ConfigPath `
                -Checkpoint $Checkpoint -Seed $Seed
            $ExitCode = $LASTEXITCODE
            if ($ExitCode -ne 0) {
                throw "Stage-B wrapper exited $ExitCode for ckpt=$Checkpoint seed=$Seed"
            }
            Assert-CellComplete -Checkpoint $Checkpoint -Seed $Seed
            Write-SupervisorEvent "completed ckpt=$Checkpoint seed=$Seed"
        }
    }

    Write-SupervisorEvent "all nine cells validated; finalizing truncated Stage B"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Wrapper `
        -Action stageb-finalize -ConfigPath $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-B finalize exited $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Output "stage_b_complete.json"))) {
        throw "Stage-B completion marker missing after finalize"
    }
    Write-SupervisorEvent "stage_b_complete.json validated"
}
catch {
    Write-SupervisorEvent "STOPPED: $($_.Exception.Message)"
    throw
}
finally {
    try { $Mutex.ReleaseMutex() } catch { }
    $Mutex.Dispose()
}
