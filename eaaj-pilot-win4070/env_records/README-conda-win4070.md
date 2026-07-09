# Windows 4070 Conda Environment Record

Date: 2026-07-09
Repo: D:\algoverse
Environment name: eaaj-win4070
Environment prefix: D:\algoverse\.conda\envs\eaaj-win4070
Conda install: D:\algoverse\.tools\miniconda3
Conda version: conda 26.5.3
Channels used for Python: conda-forge only, with --override-channels.
CUDA run dir: D:\algoverse\eaaj-pilot\outputs\local_cuda_grpo_gsm8k_6a075c15808e

## CUDA Torch Deviation

The Windows plan requested torch==2.12.* from the cu128 PyTorch index. That exact version was not available from the cu128 index during setup. pip reported available cu128 builds up to 2.11.0+cu128, so the environment uses torch==2.11.0+cu128 instead of torch==2.12.*. This is recorded as an execution-environment deviation; CUDA, the RTX 4070 Laptop GPU, and bf16 support all passed preflight.

## Core Versions

```text
python=3.13.14
torch=2.11.0+cu128
torch_cuda=12.8
cuda_available=True
gpu=NVIDIA GeForce RTX 4070 Laptop GPU
bf16=True
trl=1.6.0
transformers=5.13.0
datasets=5.0.0
accelerate=1.14.0
pytest=8.4.2
numpy=2.3.5
scipy=1.16.3
pandas=2.3.3
matplotlib=3.10.6
huggingface-hub=1.22.0
tokenizers=0.22.2
safetensors=0.8.0
```

## Verification

- pip check: No broken requirements found.
- pytest: 45 passed, 14 warnings.
- Windows preflight: 15 ok, 0 failed; one warning that HKLM LongPathsEnabled is 0.
- Pinned model/datasets prefetched: Qwen/Qwen2.5-0.5B, openai/gsm8k, ChilleD/SVAMP.
- Small CUDA GRPO probe: passed; one update took 2.84s, peak reserved VRAM was 3.812 GiB.
- Original full CUDA GRPO probe (micro-batch 8 x grad-accum 8): speed passed at 52.77s/update, but VRAM gate failed with 11.865 GiB peak reserved.
- Fallback full CUDA GRPO probe (micro-batch 4 x grad-accum 16): passed; one update took 31.97s, peak reserved VRAM was 6.941 GiB. This is the formal cuda profile.

## Record Files

- environment-conda-win4070.yml: conda env export --no-builds
- conda-list-conda-win4070.txt: conda list
- pip-freeze-conda-win4070.txt: pip freeze --all
- preflight-conda-win4070.txt: scripts/win_preflight.py output
- small-grpo-probe-conda-win4070.jsonl: small CUDA GRPO probe result
- full-grpo-probes-conda-win4070.jsonl: original and fallback full CUDA GRPO probe results

## Use

```powershell
cd D:\algoverse\eaaj-pilot-win4070
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -Phase 1
```

`run_pipeline.ps1` now auto-detects this repo-local conda environment. You can
also override the interpreter explicitly with `EAAJ_PYTHON`.

For non-activated commands, call the interpreter directly:

```powershell
D:\algoverse\.conda\envs\eaaj-win4070\python.exe scripts\win_preflight.py --grpo-probe-small
```
