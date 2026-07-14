"""Stage-B seed-repeat contracts for the Windows RTX 4070 stratum."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .adaptation import validate_adaptation_completion


CHECKPOINTS = (0, 25, 50, 100, 200)
EXPECTED_BASELINES = {0: 0.53, 25: 0.51, 50: 0.56, 100: 0.55, 200: 0.54}
EXPECTED_CONFIG_HASH = "e9b0b52aab6c"


def frozen_repeat_recipe() -> dict:
    """Return the preregistered Windows-v2 Stage-B repeat recipe."""
    return {
        "task": "SVAMP",
        "algorithm": "GRPO",
        "train_questions": 256,
        "eval_questions": 100,
        "budget_updates": 50,
        "eval_every": 10,
        "learning_rate": 1e-6,
        "beta": 0.0,
        "temperature": 0.7,
        "top_p": 1.0,
        "num_generations": 8,
        "max_prompt_length": 512,
        "max_completion_length": 512,
        "dtype": "float32",
        "autocast_dtype": "bfloat16",
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repeat_output_dir(source_run, seed: int, checkpoint: int) -> Path:
    return (Path(source_run) / "adaptation_repeats" / f"seed-{seed}"
            / f"ckpt-{checkpoint}")


def validate_source_run(source_run, checkpoint: int) -> tuple[dict, dict]:
    source_run = Path(source_run).resolve()
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"checkpoint must be one of {list(CHECKPOINTS)}")
    config_path = source_run / "config.json"
    manifest_path = source_run / "manifest.json"
    if not config_path.exists() or not manifest_path.exists():
        raise RuntimeError("source config.json and manifest.json are required")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != EXPECTED_CONFIG_HASH:
        raise RuntimeError(
            f"source config hash {manifest.get('config_hash')} != {EXPECTED_CONFIG_HASH}")
    if not str(manifest.get("platform", "")).startswith("Windows"):
        raise RuntimeError("source manifest is not from the Windows stratum")
    if "RTX 4070" not in str(manifest.get("gpu", "")):
        raise RuntimeError("source manifest GPU is not an RTX 4070")
    execution = config.get("execution", {})
    expected_execution = {
        "device": "cuda",
        "dtype": "float32",
        "autocast_dtype": "bfloat16",
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 16,
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            raise RuntimeError(
                f"source execution {key}={execution.get(key)!r}; expected {expected!r}")
    weights = source_run / f"ckpt-{checkpoint}" / "model.safetensors"
    if not weights.exists():
        raise RuntimeError(f"source checkpoint weights missing: {weights}")
    return config, manifest


def validate_repeat_summary(summary: dict, seed: int, checkpoint_path,
                            recipe: dict, expected_acc_before: float) -> dict:
    """Prove a completed artifact belongs to the requested repeat stratum."""
    expected = {
        "seed": seed,
        "task": recipe["task"],
        "train_questions": recipe["train_questions"],
        "eval_questions": recipe["eval_questions"],
        "algo": recipe["algorithm"].lower(),
        "budget_updates": recipe["budget_updates"],
        "requested_updates": recipe["budget_updates"],
        "actual_updates": recipe["budget_updates"],
        "completion_status": "complete",
        "learning_rate": recipe["learning_rate"],
        "beta": recipe["beta"],
        "temperature": recipe["temperature"],
        "top_p": recipe["top_p"],
        "num_generations": recipe["num_generations"],
        "per_device_train_batch_size": recipe["per_device_train_batch_size"],
        "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
        "max_prompt_length": recipe["max_prompt_length"],
        "max_completion_length": recipe["max_completion_length"],
        "device": "cuda",
        "dtype": recipe["dtype"],
        "autocast_dtype": recipe["autocast_dtype"],
        "optim": recipe["optim"],
        "gradient_checkpointing": recipe["gradient_checkpointing"],
        "bf16": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(
                f"repeat summary {key}={summary.get(key)!r}; expected {value!r}")
    actual_checkpoint = Path(str(summary.get("checkpoint", ""))).resolve()
    expected_checkpoint = Path(checkpoint_path).resolve()
    if actual_checkpoint != expected_checkpoint:
        raise RuntimeError(
            f"repeat summary checkpoint {actual_checkpoint} != {expected_checkpoint}")
    if abs(float(summary.get("acc_before", math.nan)) - expected_acc_before) > 1e-12:
        raise RuntimeError(
            f"repeat baseline {summary.get('acc_before')!r} != {expected_acc_before}")
    for key in ("acc_before", "acc_after", "delta_acc", "wall_seconds"):
        if not math.isfinite(float(summary.get(key, math.nan))):
            raise RuntimeError(f"repeat summary has non-finite {key}")
    return summary


def validate_repeat_telemetry(out_dir, seed: int, checkpoint_path) -> Path:
    """Require an attached telemetry CSV with at least one GPU sample."""
    checkpoint_name = Path(checkpoint_path).name
    try:
        checkpoint = int(checkpoint_name.removeprefix("ckpt-"))
    except ValueError as exc:
        raise RuntimeError(f"invalid checkpoint path: {checkpoint_path}") from exc
    pattern = f"gpu_*_stageb_seed{seed}_ckpt{checkpoint}.csv"
    for path in sorted(Path(out_dir).glob(pattern)):
        with path.open(encoding="utf-8-sig") as handle:
            if sum(1 for line in handle if line.strip()) > 1:
                return path
    raise RuntimeError(
        f"attached Stage-B telemetry is missing or header-only: {pattern}")


def validate_repeat_directory(out_dir, budget_updates: int = 50,
                              eval_every: int = 10, *, seed: int | None = None,
                              checkpoint_path=None, recipe: dict | None = None,
                              expected_acc_before: float | None = None,
                              allow_attempt_id: str | None = None,
                              require_attached_telemetry: bool = True) -> dict | None:
    """Return a complete summary, or reject any partial/failed collision."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return None
    marker_path = out_dir / ".repeat_attempt.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if allow_attempt_id is None:
            raise RuntimeError(
                f"repeat directory is not finalized; attempt marker exists: {marker_path}")
        if marker.get("attempt_id") != allow_attempt_id:
            raise RuntimeError("repeat attempt marker belongs to another invocation")
    if (out_dir / "summary.json").exists():
        summary = validate_adaptation_completion(out_dir, budget_updates, eval_every)
        identity_args = (seed, checkpoint_path, recipe, expected_acc_before)
        if any(value is not None for value in identity_args):
            if any(value is None for value in identity_args):
                raise ValueError("all repeat identity validation arguments are required")
            validate_repeat_summary(
                summary, seed, checkpoint_path, recipe, expected_acc_before)
            if allow_attempt_id is None and require_attached_telemetry:
                validate_repeat_telemetry(out_dir, seed, checkpoint_path)
        return summary
    if any(out_dir.iterdir()):
        raise RuntimeError(
            f"partial or failed repeat artifacts already exist in {out_dir}; "
            "move them to a timestamped failed-attempt directory before retrying")
    return None


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) \
                and code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_repeat_lock(source_run) -> Path:
    """Atomically prevent concurrent trainers across all Stage-B repeats."""
    import atexit

    lock = Path(source_run).resolve() / "stageb_repeat.lock"
    pid = os.getpid()
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                other = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                other = None
            if other is not None and _pid_alive(other):
                raise RuntimeError(
                    f"another Stage-B repeat runner (pid {other}) is active; "
                    "do not run two trainers concurrently")
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(pid))

        def release() -> None:
            try:
                owner = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                return
            if owner == pid:
                lock.unlink(missing_ok=True)

        atexit.register(release)
        return lock
    raise RuntimeError(f"could not acquire Stage-B repeat lock: {lock}")


def runtime_versions() -> dict:
    names = ("torch", "trl", "transformers", "bitsandbytes", "datasets",
             "accelerate")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_runtime_against_source(source_manifest: dict, gpu_name: str) -> dict:
    """Reject environment drift before writing a repeat in the same stratum."""
    if "RTX 4070" not in gpu_name:
        raise RuntimeError(f"expected RTX 4070, found {gpu_name}")
    current = runtime_versions()
    expected = source_manifest.get("versions", {})
    mismatches = {
        name: {"expected": expected.get(name), "actual": version}
        for name, version in current.items()
        if expected.get(name) != version
    }
    if mismatches:
        raise RuntimeError(f"runtime package drift from source manifest: {mismatches}")
    return current


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def ensure_repeat_manifest(source_run, seed: int, recipe: dict,
                           repo: Path, gpu_name: str) -> dict:
    source_run = Path(source_run).resolve()
    root = source_run / "adaptation_repeats" / f"seed-{seed}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "repeat_manifest.json"
    source_files = {
        name: sha256_file(source_run / name)
        for name in ("config.json", "manifest.json")
    }
    proposed = {
        "created_unix": time.time(),
        "source_run": str(source_run),
        "source_sha256": source_files,
        "source_config_hash": EXPECTED_CONFIG_HASH,
        "git_sha": _git_sha(repo),
        "seed": seed,
        "recipe": recipe,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu_name,
        "versions": runtime_versions(),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("source_sha256", "source_config_hash", "seed", "recipe"):
            if existing.get(key) != proposed.get(key):
                raise RuntimeError(f"repeat manifest mismatch for {key}")
        # Before the first scientifically complete checkpoint, allow an
        # engineering-only retry to refresh provenance (for example, a
        # wrapper fix after a pre-training failure). Once any canonical
        # checkpoint is complete, the seed stratum stays pinned.
        has_complete_checkpoint = any(
            (root / f"ckpt-{checkpoint}" / "summary.json").exists()
            for checkpoint in CHECKPOINTS)
        if not has_complete_checkpoint and existing.get("git_sha") != proposed["git_sha"]:
            history = list(existing.get("git_sha_history", []))
            old_sha = existing.get("git_sha")
            if old_sha and old_sha not in history:
                history.append(old_sha)
            existing.update({
                "git_sha": proposed["git_sha"],
                "git_sha_history": history,
                "python": proposed["python"],
                "platform": proposed["platform"],
                "gpu": proposed["gpu"],
                "versions": proposed["versions"],
                "refreshed_unix": time.time(),
            })
            path.write_text(json.dumps(existing, indent=1), encoding="utf-8")
        return existing
    path.write_text(json.dumps(proposed, indent=1), encoding="utf-8")
    return proposed
