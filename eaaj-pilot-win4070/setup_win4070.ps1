# One-shot environment setup for the eaaj pilot cuda stratum on the Windows
# RTX 4070 Laptop box. Idempotent: safe to re-run.
#
#   powershell -ExecutionPolicy Bypass -File setup_win4070.ps1
#
# Steps: NVIDIA driver check -> git long paths -> venv (3.13/3.12/3.11) ->
# torch cu128 -> pinned deps -> CUDA sanity -> prefetch pinned model/datasets
# -> unit tests -> environment preflight.

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
$Pilot = Join-Path $Repo "eaaj-pilot"
$VenvDir = Join-Path $Here ".venv"
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host "== eaaj win4070 setup =="
Write-Host "repo:  $Repo"

# 0) NVIDIA driver present?
try {
    $smi = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    Write-Host "GPU:   $smi"
} catch {
    throw "nvidia-smi not found - install/update the NVIDIA driver first (latest Game Ready/Studio, >= 560.xx recommended)"
}

# 1) long paths (HF cache paths can exceed 260 chars)
& git -C $Repo config core.longpaths true

# 2) venv - prefer 3.13 (matches the macOS run manifests), then 3.12, 3.11
if (-not (Test-Path $VenvPy)) {
    $created = $false
    foreach ($v in @("-3.13", "-3.12", "-3.11")) {
        try {
            & py $v -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "creating venv with py $v"
                & py $v -m venv $VenvDir
                $created = $true
                break
            }
        } catch { }
    }
    if (-not $created) {
        try {
            & python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                & python -m venv $VenvDir
                $created = $true
            }
        } catch { }
    }
    if (-not $created) {
        throw "No Python 3.11+ found. Install from python.org (keep the 'py launcher' option checked)."
    }
}
& $VenvPy -m pip install --upgrade pip

# 3) torch CUDA build first (its own index), then the pinned shared stack.
#    If the cu128 index cannot resolve on this driver, retry with cu126 and
#    log the substitution in eaaj-pilot/compute_log.md.
& $VenvPy -m pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "torch cu128 install failed - try the cu126 index (see comment above)" }
& $VenvPy -m pip install -r (Join-Path $Here "requirements-win4070.txt")
if ($LASTEXITCODE -ne 0) { throw "pinned dependency install failed" }

# 4) CUDA visible from torch?
& $VenvPy -c "import torch; assert torch.cuda.is_available(), 'torch sees no CUDA device'; print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "torch cannot see the GPU (driver/build mismatch)" }

# 5) prefetch pinned model + datasets (phase 1 loads with local_files_only)
& $VenvPy (Join-Path $Here "scripts\prefetch_assets.py")
if ($LASTEXITCODE -ne 0) { throw "prefetch failed" }

# 6) unit tests (metrics + reward parsing; CPU, fast)
Push-Location $Pilot
& $VenvPy -m pytest tests -q
$TestExit = $LASTEXITCODE
Pop-Location
if ($TestExit -ne 0) { throw "unit tests failed - do not run the pipeline" }

# 7) environment preflight (no GPU load yet)
& $VenvPy (Join-Path $Here "scripts\win_preflight.py")
if ($LASTEXITCODE -ne 0) { throw "preflight reported FAILs - fix them before running" }

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  1. .venv\Scripts\python.exe scripts\win_preflight.py --grpo-probe-small"
Write-Host "  2. .venv\Scripts\python.exe scripts\win_preflight.py --grpo-probe"
Write-Host "  3. powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 1"
