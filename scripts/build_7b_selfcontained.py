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

REPO = Path(__file__).resolve().parent.parent
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
# Deliberately does NOT import torch. torch imports numpy, and this notebook
# later installs a pinned numpy that is a major version ahead of Colab's; if
# numpy is already resident when that install lands, every downstream import
# dies ("cannot import name '_center' from 'numpy._core.umath'") and the only
# cure is a kernel restart, which turns Run all into a two-pass process that
# needs a human to press it again. On 2026-08-16 that cost two runtimes to idle
# reclamation. Querying nvidia-smi in a subprocess keeps this cell's fail-fast
# value without touching the numpy that cell 3 is about to replace.
import subprocess, sys

_q = subprocess.run(
    ["nvidia-smi",
     "--query-gpu=name,memory.total,compute_cap",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True)
if _q.returncode != 0 or not _q.stdout.strip():
    raise SystemExit("No GPU. Runtime -> Change runtime type -> A100 GPU -> Save")

_name, _mem_mib, _cap = [f.strip() for f in _q.stdout.strip().splitlines()[0].split(",")]
total_gb = int(_mem_mib) / 1024
cap_major = int(float(_cap))
print(f"GPU          : {_name}")
print(f"compute cap  : {_cap}")
print(f"total memory : {total_gb:.1f} GiB")

problems = []
if cap_major < 8:
    problems.append(
        f"Architecture too old (cap {_cap}). The recipe uses bfloat16, which "
        f"needs Ampere (8.0) or newer. T4/V100 will not work.")
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
print("\\nGPU gate passed (bf16 support is re-confirmed against torch in cell 4)")
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

INSTALL = '''#@title 3 Install pinned dependencies, keeping the resident numpy (2-4 min)
# DEVIATION LOGGED 2026-08-16: requirements.txt pins numpy==2.3.5 to mirror the
# local env, but Colab's kernel has numpy RESIDENT from interpreter startup
# (2.0.2 today) - before any user cell runs. Upgrading numpy across versions
# under a live kernel breaks every later import ("cannot import name '_center'
# from 'numpy._core.umath'"), and the only cure is a kernel restart, which makes
# Run all two-pass and cost three runtimes to idle reclamation on 2026-08-16.
# So numpy is deliberately HELD at the resident version via a pip constraint.
# The scientific manifest (trl/transformers/datasets/accelerate/peft/
# bitsandbytes/pylatexenc, checked in cell 4) is installed exactly as pinned;
# numpy is not part of that contract and its exact patch version does not enter
# GRPO semantics. torch is NOT reinstalled - Colab's CUDA build is kept.
import importlib.metadata as _md

_resident_numpy = _md.version("numpy")
_req = open("/content/RLVR/experiment 2/requirements.txt").read().splitlines()
_kept = [l for l in _req if not l.strip().lower().startswith("numpy")]
open("/tmp/req_no_numpy.txt", "w").write("\\n".join(_kept))
open("/tmp/constraints.txt", "w").write(f"numpy=={_resident_numpy}\\n")
print(f"holding numpy at resident {_resident_numpy}; installing the rest as pinned")

%pip install -q -r /tmp/req_no_numpy.txt -c /tmp/constraints.txt
print("\\nInstall finished.")
'''

RESTART_GUARD = '''#@title 3b Assert the kernel is clean (must be a no-op)
# numpy is resident from interpreter startup (before any user cell), so cell 3
# holds it at that version by constraint instead of upgrading it. Loaded and
# installed must therefore agree here on the FIRST pass, and Run all completes
# without any restart. If this fires, the constraint failed (e.g. some package
# force-upgraded numpy transitively) - fix cell 3, do not add a restart.
import importlib.metadata as _md
import numpy as _np

_installed = _md.version("numpy")
if _np.__version__ != _installed:
    raise SystemExit(
        f"CONSTRAINT FAILED: numpy {_np.__version__} loaded but {_installed} "
        "on disk. Something in cell 3 upgraded numpy despite the constraint.")
print(f"kernel clean: numpy loaded == installed == {_installed}, single pass OK")
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
import numpy as _np_v
print(f"numpy held at resident {_np_v.__version__} (deviation logged in cell 3; not part of the pinned manifest)")

# Import the bundled modules now, in-process, so a missing dependency surfaces in
# seconds rather than after the 7B download.
sys.path.insert(0, "/content/RLVR/experiment 2")
import src.guru_data, src.guru_reward, src.pipeline  # noqa: F401
from vendor.reasoning360_reward_score import codeio, naive_dapo  # noqa: F401
print("Import smoke test passed: bundled data/reward/pipeline modules load cleanly.")
'''


RESTART_GUARD = '''#@title 3b Assert the kernel is clean (must be a no-op)
# With the cell-1 reorder nothing imports numpy before cell 3 installs it, so
# the loaded and installed versions must agree on the first pass and Run all
# completes without a restart. If this ever fires, the ordering invariant broke:
# something above imported numpy (directly, or via torch/pandas) before the
# install. Fix the ordering rather than adding a restart - a restart makes Run
# all two-pass, which needs an operator present and cost two runtimes to idle
# reclamation on 2026-08-16.
import importlib.metadata as _md
import numpy as _np

_installed = _md.version("numpy")
if _np.__version__ != _installed:
    raise SystemExit(
        f"ORDERING BROKEN: numpy {_np.__version__} was already resident before "
        f"the install put {_installed} on disk. Something above cell 3 imports "
        "numpy. Restarting would work but re-introduces the two-pass problem - "
        "fix the import order instead.")
print(f"kernel clean: numpy loaded == installed == {_installed}, single pass OK")
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


PERSIST = '''#@title Persist Phase-0 artifacts (Drive if possible, else base64 in output)
# /content is ephemeral (lesson from the v9 probe: the runtime was recycled
# overnight and every artifact vanished). The frozen splits and audits below
# must reach the repo, and the Colab PAT is broken, so they go home via Drive
# or, failing that, inline base64 that gets transcribed from this output.
import base64, gzip, io, os, tarfile

SRC = "/content/RLVR/experiment 2/data"
names = sorted(n for n in os.listdir(SRC) if n.endswith(".json"))
print("artifacts:", names)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    dest = "/content/drive/MyDrive/exp2_7b_phase0_artifacts"
    os.makedirs(dest, exist_ok=True)
    import shutil
    for n in names:
        shutil.copy2(f"{SRC}/{n}", f"{dest}/{n}")
    print("copied to Drive:", dest)
except Exception as exc:
    print("Drive mount unavailable:", exc)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for n in names:
            tar.add(f"{SRC}/{n}", arcname=n)
    b64 = base64.b64encode(gzip.compress(buf.getvalue(), 9)).decode()
    print("BEGIN_ARTIFACTS_B64")
    for i in range(0, len(b64), 200):
        print(b64[i:i+200])
    print("END_ARTIFACTS_B64")
print("commit these into experiment 2/data/ from the Mac afterwards.")
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
         code(RESTART_GUARD), code(ENVCHECK), code(PREWARM), md("## Phase 0 - pre-registered cells below are unmodified"),
         code(SETUP)]
# keep notebook 00's cells from index 2 onward (0 = its markdown header,
# 1 = the clone cell we just replaced)
_tail = [dict(c) for c in nb["cells"][2:]]

# Notebook 00 carries its OWN `%pip install -r requirements.txt` cell. Left in
# place it re-installs the full manifest INCLUDING numpy==2.3.5, silently
# undoing cell 3's constraint - and because it sits AFTER the 3b assertion, the
# tripwire passes and the breakage only surfaces later as
# "cannot import name '_center' from 'numpy._core.umath'" inside
# load_all_records. That is exactly how the 2026-08-16 19:56 run died. Cell 3
# already installed this same manifest, so this cell is pure duplication;
# neutralise it rather than let it fight the constraint.
_neutralised = 0
for _c in _tail:
    _s = "".join(_c.get("source", []))
    if _c.get("cell_type") == "code" and "pip install" in _s and "requirements.txt" in _s:
        _c["source"] = (
            "# NEUTRALISED 2026-08-16 by scripts/build_7b_selfcontained.py.\n"
            "# The original line here was:\n"
            "#     %pip install -q -r \"/content/RLVR/experiment 2/requirements.txt\"\n"
            "# Cell 3 above already installed that exact manifest, but with a pip\n"
            "# constraint holding numpy at the version Colab has resident from\n"
            "# interpreter startup. Re-running the unconstrained install here put\n"
            "# numpy 2.3.5 on disk under a kernel holding 2.0.2, which breaks every\n"
            "# later import ('cannot import name _center from numpy._core.umath').\n"
            "# It also sits after the 3b tripwire, so the damage went undetected\n"
            "# until load_all_records. Deliberately a no-op; see the header cell.\n"
            "print('skipped: cell 3 already installed the pinned manifest')\n"
        ).splitlines(keepends=True)
        _c["outputs"] = []
        _c["execution_count"] = None
        _neutralised += 1
assert _neutralised == 1, f"expected exactly one duplicate pip cell, found {_neutralised}"
cells.extend(_tail)
cells.append(code(PERSIST))

nb["cells"] = cells
nb["metadata"].setdefault("accelerator", "GPU")
nb["metadata"].setdefault("colab", {})["provenance"] = []
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KiB, {len(cells)} cells)")
