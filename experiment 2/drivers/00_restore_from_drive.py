"""Rebuild a Colab runtime from Drive after the VM is recycled.

VERBATIM as executed 2026-08-19. Not part of the science — Colab reclaimed the VM
once mid-run, and this is what got the session back in ~15 minutes without asking
the operator to re-upload anything.

Order matters: run cells 1-6 of colab/00_phase0_selfcontained.ipynb FIRST (GPU gate,
unpack the embedded source blob, pinned installs, numpy single-pass assert, env
check, model download). Do NOT re-run cells 8+ — those are the Phase-0 gates and
cost ~30 minutes. Then run this.

Everything restored is hash-verified before use, so a corrupted or stale Drive copy
fails loudly instead of silently changing the experiment.
"""
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

if not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/eaaj-exp2-checkpoints")
EXP2 = Path("/content/RLVR/experiment 2")
(EXP2 / "data").mkdir(parents=True, exist_ok=True)


def restore_cfg(name, want):
    s = (DRIVE / name).read_text()
    h = hashlib.sha256(s.encode()).hexdigest()[:12]
    assert h == want, name + " hash mismatch: " + h
    (EXP2 / name).write_text(s)
    print("  restored", name, h)


restore_cfg("exp2_colab_config_mvp_instruct.json", "e33527592dd9")
restore_cfg("exp2_colab_config_mvp_instruct_stageb_v2.json", "bd99ddd2817f")

# Restoring the frozen splits rather than regenerating them saves several minutes
# AND removes any chance of a silent divergence.
splits = json.loads((DRIVE / "exp2_colab_splits_instruct.json").read_text())
sha16 = lambda L: hashlib.sha256(
    json.dumps(sorted(L), separators=(",", ":")).encode()).hexdigest()[:16]
EXPECT = {"stage_a_train_ids": ("a06fe6b80e7d40ca", 54251),
          "stage_b_train_ids": ("df8623cf009e0690", 1132),
          "stage_b_eval_ids": ("8ee975c7089dc72a", 300),
          "probe_stage_a_topup_ids": ("1e61252e7b54793e", 4096)}
for k, (exp, n) in EXPECT.items():
    assert sha16(splits[k]) == exp and len(splits[k]) == n, k
(EXP2 / "data" / "exp2_colab_splits_instruct.json").write_text(
    json.dumps(splits, indent=1))
print("  splits restored and hash-verified (no regeneration needed)")

dst = Path("/content/ckpts")
dst.mkdir(exist_ok=True)
for f in ["02_ckpt-0.tar.gz", "03_ckpt-50.tar.gz", "04_ckpt-100.tar.gz"]:
    with tarfile.open(DRIVE / f) as t:
        t.extractall(dst)
print("  checkpoints unpacked:", sorted(p.name for p in dst.iterdir()))

# Carry back any Stage-B arm that already finished, so it is skipped on relaunch.
RUN = "exp2_colab_guru_math7b_instruct_group8_e33527592dd9"
OUT = Path("/content/outputs") / RUN / "stage_b_v2"
src = DRIVE / "stage_b_v2"
if src.is_dir():
    for arm in sorted(src.iterdir()):
        if (arm / "summary.json").exists():
            shutil.copytree(arm, OUT / arm.name, dirs_exist_ok=True)
            print("  arm already complete, will be skipped:", arm.name)
print("RESTORE OK")
