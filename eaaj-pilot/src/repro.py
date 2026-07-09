"""Reproducibility helpers shared by the Colab notebooks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TRACKED_INPUTS = (
    "pilot_config.json",
    "requirements.txt",
    "LOCAL_EXPERIMENT_PLAN.md",
    "01_grpo_gsm8k.ipynb",
    "02_measure_Q.ipynb",
    "03_svamp_adaptation.ipynb",
    "04_analysis.ipynb",
    "src/metrics.py",
    "src/reward.py",
    "src/data.py",
    "src/callbacks.py",
    "src/evaluate.py",
    "src/adaptation.py",
    "src/analysis.py",
    "src/mps_compat.py",
    "src/preflight.py",
    "src/repro.py",
    "scripts/run_local_pipeline.py",
    "data/gsm8k_splits.json",
    "data/svamp_splits.json",
    "data/probe_set_ids.json",
)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_manifest(project_dir, config: dict) -> dict:
    project_dir = Path(project_dir)
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        gpu = ""
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir,
        capture_output=True, text=True).stdout.strip()
    hashes = {p: sha256_file(project_dir / p) for p in TRACKED_INPUTS}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": gpu or None,
        "git_sha": git_sha or None,
        "versions": {name: _version(name) for name in (
            "torch", "trl", "transformers", "datasets", "accelerate",
            "bitsandbytes", "numpy", "scipy", "pandas", "matplotlib")},
        "tracked_input_sha256": hashes,
    }


def set_active_run(project_dir, run_dir) -> None:
    project_dir, run_dir = Path(project_dir), Path(run_dir)
    marker = project_dir / "outputs" / "ACTIVE_RUN.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(run_dir.relative_to(project_dir)) + "\n")


def get_active_run(project_dir) -> Path:
    project_dir = Path(project_dir)
    marker = project_dir / "outputs" / "ACTIVE_RUN.txt"
    if not marker.exists():
        raise FileNotFoundError(
            "outputs/ACTIVE_RUN.txt is missing; run notebook 01 first or set it explicitly")
    run_dir = project_dir / marker.read_text().strip()
    if not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"active run is incomplete: {run_dir}")
    return run_dir
