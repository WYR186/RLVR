"""Build a self-contained Colab notebook for the exp2 7B track.

Packs the repo files the pipeline needs into a gzip+base64 blob embedded in the
notebook, so Colab needs no GitHub token and no clone. Everything else about
notebook 00 is preserved verbatim -- only the clone cell is replaced.
"""
import base64
import gzip
import io
import json
import tarfile
from pathlib import Path

REPO = Path("/Users/ipanda/Documents/algoverse")
EXP2 = REPO / "experiment 2"
OUT = REPO / "experiment 2" / "colab" / "00_phase0_selfcontained.ipynb"

# (source path on disk, path inside the archive)
MEMBERS = []
for p in sorted((EXP2 / "src").glob("*.py")):
    MEMBERS.append((p, f"experiment 2/src/{p.name}"))
for p in sorted((EXP2 / "vendor").rglob("*.py")):
    if "__pycache__" in p.parts:
        continue
    MEMBERS.append((p, f"experiment 2/{p.relative_to(EXP2)}"))
for name in ("exp2_colab_config_mvp.json", "exp2_colab_config.json",
             "requirements.txt"):
    MEMBERS.append((EXP2 / name, f"experiment 2/{name}"))
MEMBERS.append((EXP2 / "data" / "guru_schema_audit.json",
                "experiment 2/data/guru_schema_audit.json"))
for name in ("metrics.py", "callbacks.py"):
    MEMBERS.append((REPO / "eaaj-pilot" / "src" / name,
                    f"eaaj-pilot/src/{name}"))

missing = [str(s) for s, _ in MEMBERS if not s.is_file()]
if missing:
    raise SystemExit("missing source files:\n  " + "\n  ".join(missing))

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w") as tar:
    for src, arc in MEMBERS:
        tar.add(src, arcname=arc)
raw = buf.getvalue()
blob = base64.b64encode(gzip.compress(raw, 9)).decode()
print(f"packed {len(MEMBERS)} files: {len(raw)/1024:.0f} KiB raw "
      f"-> {len(blob)/1024:.0f} KiB base64")

# wrap the base64 so the notebook source stays readable
WRAP = 120
blob_lines = "\n".join(blob[i:i + WRAP] for i in range(0, len(blob), WRAP))

nb = json.loads((EXP2 / "colab" / "00_setup_schema_audit.ipynb").read_text())


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


GPU_GATE = '''#@title 1 GPU gate - refuse to continue on unsuitable hardware
import torch

assert torch.cuda.is_available(), "No GPU. Runtime -> Change runtime type -> A100"

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
bf16 = torch.cuda.is_bf16_supported()

print(f"GPU            : {name}")
print(f"compute cap    : {cap[0]}.{cap[1]}")
print(f"total memory   : {total_gb:.1f} GiB")
print(f"bf16 supported : {bf16}")
print(f"torch          : {torch.__version__}")

problems = []
if cap[0] < 8:
    problems.append(
        f"Architecture too old (cap {cap[0]}.{cap[1]}). The recipe uses bfloat16, "
        f"which needs Ampere (8.0) or newer. T4/V100 will not work.")
if not bf16:
    problems.append("This device has no native bf16 support.")
if total_gb < 35:
    problems.append(
        f"Only {total_gb:.1f} GiB of VRAM. Qwen2.5-7B in bf16 is ~15 GiB of weights "
        f"before any group-8 generation state. Measured evidence from 2026-08-16: "
        f"the 0.5B track at this same group-8 geometry already needed ~27 GiB and "
        f"OOM'd on a 22 GiB L4. Use A100.")

if problems:
    print("\\nUnsuitable GPU:")
    for p in problems:
        print("  -", p)
    raise SystemExit("GPU gate failed")
print("\\nGPU gate passed")
'''

UNPACK = '''#@title 2 Unpack embedded source (no GitHub token required)
# This notebook carries the repo files it needs as a gzip+base64 blob instead of
# cloning the private repo. Same trick the v9 4070 probe used on 2026-08-16, for
# the same reason: the private-repo clone needs a PAT, and getting that PAT's
# fine-grained permissions right repeatedly failed (and leaked the token into
# cell output twice before it was sanitized). No token means neither failure mode
# can happen. The layout below reproduces the sibling-directory structure
# src/pipeline.py depends on (EXP2_ROOT/.. must contain eaaj-pilot/src).
import base64, gzip, io, os, sys, json, tarfile
from pathlib import Path

_B64 = """
{BLOB}
"""

REPO_DIR = "/content/RLVR"
os.makedirs(REPO_DIR, exist_ok=True)
_raw = gzip.decompress(base64.b64decode("".join(_B64.split())))
with tarfile.open(fileobj=io.BytesIO(_raw), mode="r") as _tar:
    _tar.extractall(REPO_DIR)

EXP2_DIR = f"{REPO_DIR}/experiment 2"
os.makedirs(f"{EXP2_DIR}/data", exist_ok=True)
for _p in sorted(Path(REPO_DIR).rglob("*")):
    if _p.is_file():
        print(" ", _p.relative_to(REPO_DIR))
print("\\nunpacked into", REPO_DIR)
'''.replace("{BLOB}", blob_lines)

INSTALL = '''#@title 3 Install pinned dependencies (2-4 min)
# Pins come from experiment 2/requirements.txt unchanged. torch is deliberately
# NOT reinstalled -- Colab's preinstalled CUDA build is kept.
!pip install -q -r "/content/RLVR/experiment 2/requirements.txt"
print("\\nInstall finished.")
'''

ENVCHECK = '''#@title 4 Environment check - versions must match the pinned manifest
import importlib.metadata as md
import sys
import torch

EXPECTED = {"trl": "1.6.0", "transformers": "5.13.0", "datasets": "5.0.0",
            "accelerate": "1.14.0", "peft": "0.15.2", "bitsandbytes": "0.49.2",
            "pylatexenc": "2.10"}

print("python", ".".join(map(str, sys.version_info[:3])))
print("torch ", torch.__version__, f"(CUDA {torch.version.cuda})  <- Colab preinstalled")
print()
mismatched = []
for pkg, want in EXPECTED.items():
    try:
        got = md.version(pkg)
    except Exception:
        got = "MISSING"
    if got != want:
        mismatched.append(pkg)
    print(f"  {'ok ' if got == want else 'BAD'} {pkg:<14} want {want:<10} got {got}")

if mismatched:
    print("\\nVersion mismatch:", ", ".join(mismatched))
    print("TRL version drift can silently change GRPO semantics. Report before trusting results.")
else:
    print("\\nAll pinned packages match the manifest.")

# Import the bundled modules now, in-process, so a missing dependency surfaces in
# seconds rather than after the 7B download.
sys.path.insert(0, "/content/RLVR/experiment 2")
import src.guru_data, src.guru_reward, src.pipeline  # noqa: F401
from vendor.reasoning360_reward_score import codeio, naive_dapo  # noqa: F401
print("Import smoke test passed: bundled data/reward/pipeline modules load cleanly.")
'''

PREWARM = '''#@title 5 Pre-download the model and dataset (7B is ~15 GB - several minutes)
# Not strictly required (unlike the 4070 track, this code does not force
# local_files_only), but downloading here isolates a network failure from a
# training failure, and gives one clean progress bar instead of a stall in the
# middle of Phase 0.
import json, subprocess, sys, time

CFG = json.load(open("/content/RLVR/experiment 2/exp2_colab_config_mvp.json"))
MODEL_ID = CFG["model_id"]
MODEL_REVISION = CFG["model_revision"]
DS_SOURCE = CFG["dataset"]["source"]
DS_REVISION = CFG["dataset"]["revision"]
print(f"model  : {MODEL_ID} @ {MODEL_REVISION}")
print(f"dataset: {DS_SOURCE} @ {DS_REVISION}")

code = "from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM\\n"
code += "from huggingface_hub import snapshot_download\\n"
code += f"AutoTokenizer.from_pretrained({MODEL_ID!r}, revision={MODEL_REVISION!r})\\n"
code += f"AutoConfig.from_pretrained({MODEL_ID!r}, revision={MODEL_REVISION!r})\\n"
code += f"snapshot_download({MODEL_ID!r}, revision={MODEL_REVISION!r})\\n"
code += (f"snapshot_download({DS_SOURCE!r}, repo_type='dataset', "
         f"revision={DS_REVISION!r})\\n")

for attempt in range(1, 6):
    print(f"Attempt {attempt}/5 ...", flush=True)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        print("Download successful. Model weights and dataset are cached.")
        break
    print("Failed:", (r.stderr or "").strip().splitlines()[-1:] or "(no stderr)")
    if attempt < 5:
        time.sleep(15)
else:
    raise SystemExit("Could not download from Hugging Face after 5 attempts.")
'''

# The replacement for notebook 00's cell 1: same globals, no clone, no token.
SETUP = '''import json, os, sys
from pathlib import Path

# Replaces the original notebook's private-repo clone. The source is already on
# disk from cell 2; everything below defines exactly the same globals the rest of
# this notebook expects, so the pre-registered cells that follow are unmodified.
REPO_DIR = "/content/RLVR"
EXP2_DIR = f"{REPO_DIR}/experiment 2"
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)  # only this one goes on sys.path - pipeline.py
# reaches eaaj-pilot/src by explicit file path internally, avoiding a
# top-level `src` package-name collision between the two sibling dirs
# (see experiment 2/src/pipeline.py's module docstring).

import src.guru_data as guru_data
import src.guru_reward as guru_reward
import src.pipeline as pipeline

# MVP scope fork - see EXPERIMENT_2_COLAB_MVP_AMENDMENT.md. This is the ONE
# knob: switch it back to 'exp2_colab_config.json' if and only if the Phase-0
# promotion gate passes. Never switch it mid-run.
CONFIG_NAME = 'exp2_colab_config_mvp.json'
CONFIG = json.load(open(f'{EXP2_DIR}/{CONFIG_NAME}'))
DATA_DIR = Path(EXP2_DIR) / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID, MODEL_REVISION = CONFIG['model_id'], CONFIG['model_revision']
DATASET_REVISION = CONFIG['dataset']['revision']
print('config loaded:', CONFIG['experiment'])
print('status       :', CONFIG['status'])
print('model        :', MODEL_ID, '| variant:', CONFIG.get('model_variant'))
print('stage_a      : max_steps', CONFIG['stage_a']['max_steps'],
      '| checkpoints', CONFIG['stage_a']['checkpoint_steps'],
      '| group', CONFIG['stage_a']['num_generations'])
'''

HEADER = '''# exp2 7B Phase 0 - self-contained (no GitHub token, no Drive mount)

Generated 2026-08-16 from `experiment 2/colab/00_setup_schema_audit.ipynb`.
**Only the clone cell was replaced**; every pre-registered Phase-0 cell below is
byte-for-byte the committed version.

Config actually loaded: `exp2_colab_config_mvp.json` (MVP scope fork registered
2026-08-16 - model stays Qwen2.5-7B, scope is cut on the update axis).

## How to run
1. Runtime -> Change runtime type -> **A100 GPU** -> Save
2. Runtime -> Run all
3. Walk away. Phase 0 on a 7B base model is expected to take well over an hour,
   most of it generation.

This runs Phase 0 only (contract re-verification, token audit, split freeze,
Gate C0 memory calibration, sparse-reward preflight, 2-update smoke). **It does
not start Stage A.**

## Why this exists instead of the normal notebook 00

`00_setup_schema_audit.ipynb` clones the private repo with a PAT from Colab
Secrets. That PAT is **broken** — verified 2026-08-16, the clone fails with
`remote: Write access to repository not granted` / HTTP 403. Colab's own GitHub
integration (OAuth) reads the private repo fine, so this notebook is opened
through that instead and carries its source inline, touching no PAT anywhere.

**The embedded blob is a SNAPSHOT.** If `experiment 2/src/`,
`experiment 2/vendor/`, either config, `requirements.txt`, or
`eaaj-pilot/src/{metrics,callbacks}.py` changes, this notebook is stale.
Regenerate it with `scripts/build_7b_selfcontained.py` and re-commit; do not
hand-edit the blob. Delete this notebook once the PAT is fixed.

## Deviation logged up front
The config registers "L4 first, escalate on Gate C0". This notebook's GPU gate
requires >= 35 GiB and therefore goes straight to A100. Reason: on 2026-08-16 the
0.5B track at this identical group-8 geometry was measured needing ~27 GiB and
OOM'd on a 22 GiB L4. A 7B base cannot fit where 0.5B did not. Trying L4 first
would spend a full model-download cycle to learn something already measured,
which defeats the "cheaper compute-unit draw" rationale L4-first was registered
for.
'''

cells = [md(HEADER), code(GPU_GATE), code(UNPACK), code(INSTALL),
         code(ENVCHECK), code(PREWARM), md("## Phase 0 - pre-registered cells below are unmodified"),
         code(SETUP)]
# keep notebook 00's cells from index 2 onward (0 = its markdown header,
# 1 = the clone cell we just replaced)
cells.extend(nb["cells"][2:])

nb["cells"] = cells
nb["metadata"].setdefault("accelerator", "GPU")
nb["metadata"].setdefault("colab", {})["provenance"] = []
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KiB, {len(cells)} cells)")
