"""Build the self-contained Colab notebook for the E1 metric re-measurement sweep.

E1 (`experiment 2/SPEC_E1_METRIC_REMEASUREMENT.md`) re-reads the three Stage-A
adapters already in Drive. It needs no training and no team decision, but it
does need the repo source inside the Colab runtime — and the PAT that
`colab/02_*.ipynb` clones with is confirmed broken (HTTP 403). So this follows
the proven pattern of `build_7b_selfcontained.py`: setup cells 1-6 of
`colab/00_phase0_selfcontained.ipynb` reused verbatim, with cell 2's source blob
rebuilt from the current tree so it carries `src/e1_sweep.py` and the E1 driver.

Regenerate and re-commit whenever `experiment 2/src/`, the E1 driver, the
instruct config, `requirements.txt`, or `eaaj-pilot/src/{metrics,callbacks}.py`
changes. Do not hand-edit the blob.

    python scripts/build_e1_notebook.py
"""
import base64
import gzip
import io
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP2 = REPO / "experiment 2"
SOURCE_NB = EXP2 / "colab" / "00_phase0_selfcontained.ipynb"
OUT = EXP2 / "colab" / "05_e1_metric_sweep_selfcontained.ipynb"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# 1. pack the source the runtime needs
# ---------------------------------------------------------------------------

MEMBERS = []
for p in sorted((EXP2 / "src").glob("*.py")):
    MEMBERS.append((p, f"experiment 2/src/{p.name}"))
for p in sorted((EXP2 / "vendor").rglob("*.py")):
    if "__pycache__" in p.parts:
        continue
    MEMBERS.append((p, f"experiment 2/{p.relative_to(EXP2)}"))
for name in ("exp2_colab_config_mvp_instruct.json", "requirements.txt"):
    MEMBERS.append((EXP2 / name, f"experiment 2/{name}"))
MEMBERS.append((EXP2 / "data" / "guru_schema_audit.json",
                "experiment 2/data/guru_schema_audit.json"))
for name in ("00_restore_from_drive.py", "04_e1_metric_sweep.py"):
    MEMBERS.append((EXP2 / "drivers" / name, f"experiment 2/drivers/{name}"))
for name in ("metrics.py", "callbacks.py"):
    MEMBERS.append((REPO / "eaaj-pilot" / "src" / name, f"eaaj-pilot/src/{name}"))

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
WRAP = 120
blob_lines = "\n".join(blob[i:i + WRAP] for i in range(0, len(blob), WRAP))


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


# ---------------------------------------------------------------------------
# 2. reuse the proven setup cells
# ---------------------------------------------------------------------------

src_nb = json.loads(SOURCE_NB.read_text())
# cells 1-6 of notebook 00: GPU gate, unpack, pinned install, numpy assert,
# env check, model download. Cell 2 (unpack) and cell 6 (download) are replaced;
# the rest are carried over byte-for-byte so their hard-won fixes survive.
gpu_gate = src_nb["cells"][1]
pinned_install = src_nb["cells"][3]
numpy_assert = src_nb["cells"][4]
env_check = src_nb["cells"][5]

unpack_src = json.loads(SOURCE_NB.read_text())["cells"][2]["source"]
unpack_text = "".join(unpack_src)
head, _, tail = unpack_text.partition('_B64 = """')
_old_blob, _, rest = tail.partition('"""')
unpack = code(f'{head}_B64 = """\n{blob_lines}\n"""{rest}')

DOWNLOAD = f'''#@title 5 Pre-download {MODEL_ID} (~15 GB - several minutes)
# E1 measures the INSTRUCT checkpoints, so this downloads the Instruct model,
# not the base model notebook 00 pulls. Isolating the download from the
# measurement gives one clean progress bar instead of a stall mid-sweep.
import subprocess, sys, time

MODEL_ID = {MODEL_ID!r}
print("model:", MODEL_ID)
code = (
    "from transformers import AutoTokenizer, AutoConfig\\n"
    "from huggingface_hub import snapshot_download\\n"
    f"AutoTokenizer.from_pretrained({{MODEL_ID!r}})\\n"
    f"AutoConfig.from_pretrained({{MODEL_ID!r}})\\n"
    f"snapshot_download({{MODEL_ID!r}})\\n")

for attempt in range(1, 6):
    print(f"Attempt {{attempt}}/5 ...", flush=True)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode == 0:
        print("Download successful.")
        break
    print("Failed:", (r.stderr or "").strip().splitlines()[-1:] or "(no stderr)")
    if attempt < 5:
        time.sleep(15)
else:
    raise SystemExit("Could not download from Hugging Face after 5 attempts.")
'''

RESTORE = '''#@title 6 Restore config, splits and the three Stage-A adapters from Drive
# Mount IN-PROCESS first. The restore driver runs as a subprocess, and
# google.colab.drive.mount() cannot complete its interactive authorization from
# a subprocess - it talks to the notebook frontend over the kernel's comms
# channel, which a child process does not have. Mounting here means the driver
# finds /content/drive/MyDrive already present and skips its own mount call.
# This popup is the one manual step in the notebook.
from google.colab import drive
drive.mount("/content/drive")

# Verbatim driver, hash-verified: a stale or corrupted Drive copy fails loudly
# rather than silently changing the experiment.
import subprocess, sys
r = subprocess.run([sys.executable, "/content/RLVR/experiment 2/drivers/00_restore_from_drive.py"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr)
    raise SystemExit("restore failed")
'''

DRY = '''#@title 7 Pre-flight: provenance + probe, no GPU work
# Fails fast on a bad config hash or split hash BEFORE the model is loaded,
# so a provenance problem costs seconds instead of an A100-hour.
import sys
sys.path.insert(0, "/content/RLVR/experiment 2")
sys.argv = ["04_e1_metric_sweep.py"]
import importlib.util
spec = importlib.util.spec_from_file_location(
    "e1driver", "/content/RLVR/experiment 2/drivers/04_e1_metric_sweep.py")
e1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e1)

config, splits = e1.load_provenance()
prompts = e1.probe_prompts(config, splits)
print("probe ready:", len(prompts), "prompts")
print("layers:", config["measurement"]["layers"],
      "| batch:", config["measurement"]["batch_size"])
'''

SWEEP = '''#@title 8 Run E1 steps 1-3 (reference-arm gate, then the sweep)
# Step 1 is the GATE: if the reference arm does not reproduce
# FINDING_Q_METRICS_7B_INSTRUCT.md exactly, this raises and nothing downstream
# runs (spec section 6 gate 1). Steps 2-3 involve no generation.
#
# Detached so an idle-reclaim of the browser tab does not kill the run; the log
# is tailed in the next cell.
import subprocess, sys
LOG = "/content/e1_sweep.log"
proc = subprocess.Popen(
    [sys.executable, "/content/RLVR/experiment 2/drivers/04_e1_metric_sweep.py"],
    stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
print("launched pid", proc.pid, "->", LOG)
'''

TAIL = '''#@title 9 Follow the run
!tail -n 60 /content/e1_sweep.log
'''

BACKUP = '''#@title 10 Copy E1 outputs back to Drive
# The sweep's JSONs, per-unit score vectors and summary.csv are small. Copying
# them to Drive means a VM recycle does not cost the A100-hour that produced them.
import shutil
from pathlib import Path
RUN = "exp2_colab_guru_math7b_instruct_group8_e33527592dd9"
src = Path("/content/outputs") / RUN / "measurements" / "e1_sweep"
dst = Path("/content/drive/MyDrive/eaaj-exp2-checkpoints/e1_sweep")
if not src.is_dir():
    raise SystemExit(f"nothing at {src} yet - has the sweep finished?")
shutil.copytree(src, dst, dirs_exist_ok=True)
print("copied to", dst)
print(sorted(p.name for p in dst.iterdir()))
'''

HEADER = f'''# E1 - metric re-measurement sweep (self-contained, no GitHub token)

Implements `experiment 2/SPEC_E1_METRIC_REMEASUREMENT.md`. Generated by
`scripts/build_e1_notebook.py` from the current tree.

**What this does.** Re-measures Q on the three Stage-A adapters already in Drive
under a grid of alternative operationalizations (V1-V6). **No training, no
Stage B, no dependency on anyone else.** Estimated ~1-2 A100-hours.

**What it does not do.** It cannot test RQ1 - Delta-R is flat across these three
checkpoints however Q is measured. E1 is about whether *the detector* works.

## How to run
1. Runtime -> Change runtime type -> **A100 GPU** -> Save
2. Run cells 1-6 (setup + Drive restore). Mounting Drive asks for authorization.
3. Run cell 7 (pre-flight - fails fast on a provenance problem, costs seconds)
4. Run cell 8, then re-run cell 9 to follow the log.
5. Run cell 10 to copy results back to Drive.

## The gate
Step 1 re-runs the **unchanged reference arm** and must reproduce
`FINDING_Q_METRICS_7B_INSTRUCT.md` exactly (erank delta < 1e-4, dormant_frac
0.0 everywhere). Two independent ckpt-0 passes previously agreed bit for bit, so
any drift is a real environment change. If the gate fails the driver **stops** -
do not accept and note it.

## Why the source is embedded
The PAT that `colab/02_transfer_T_and_qmetrics.ipynb` clones with is confirmed
broken (HTTP 403, `Write access to repository not granted`). This notebook
carries the source it needs as a blob instead, same as
`00_phase0_selfcontained.ipynb`. **The blob is a SNAPSHOT** - regenerate with
`scripts/build_e1_notebook.py` if `experiment 2/src/`, the driver, the instruct
config, or `eaaj-pilot/src/*.py` changes.

Model: `{MODEL_ID}`, LoRA r=16 alpha=32, base bf16 / adapter fp32.
'''

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "machine_shape": "hm"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": [
        md(HEADER), gpu_gate, unpack, pinned_install, numpy_assert, env_check,
        code(DOWNLOAD), code(RESTORE), code(DRY), code(SWEEP), code(TAIL),
        code(BACKUP),
    ],
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KiB, {len(nb['cells'])} cells)")
