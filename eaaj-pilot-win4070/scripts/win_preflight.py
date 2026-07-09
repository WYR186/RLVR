#!/usr/bin/env python3
"""Windows RTX 4070 Laptop preflight for the eaaj pilot cuda stratum.

Three levels, all read-only with respect to scientific artifacts:

  (no flag)            environment checks only (seconds, no GPU load)
  --grpo-probe-small   tiny GRPO update on CUDA (~2-3 min): stack contract
  --grpo-probe         ONE update at the current cuda profile geometry
                       (micro-batch x grad-accum x 8 generations x 512
                       completion tokens, bf16 + gradient checkpointing).
                       This is the VRAM/throughput go/no-go before the
                       200-update spend.
                       Optional --per-device-batch/--grad-accum probe the
                       pre-declared VRAM ladder while keeping the effective
                       64-completion update fixed.

Probe outputs are plumbing (like eaaj-pilot/scripts/dry_run_metrics.py),
never results: they train a throwaway model copy for one step in a scratch
dir under logs/ (gitignored) with a non-scientific seed. Exit code 0 = pass.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # eaaj-pilot-win4070/
REPO = HERE.parent
PILOT = REPO / "eaaj-pilot"
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(PILOT / "scripts"))

OK, WARN, FAIL = "ok", "warn", "FAIL"
RESULTS: list = []


def check(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    suffix = f" - {detail}" if detail else ""
    print(f"[{status:>4}] {name}{suffix}")


def env_checks() -> None:
    import platform

    check("platform", OK if sys.platform == "win32" else WARN,
          f"{platform.platform()} (win32 expected on the 4070 laptop)")
    check("python", OK if (3, 11) <= sys.version_info[:2] <= (3, 13) else WARN,
          platform.python_version())

    pilot_cfg = json.loads((PILOT / "pilot_config.json").read_text())

    try:
        import torch
    except ImportError:
        check("torch", FAIL, "not installed - run setup_win4070.ps1 first")
        return
    check("torch", OK, f"{torch.__version__} (cuda build: {torch.version.cuda})")

    if not torch.cuda.is_available():
        check("cuda", FAIL,
              "torch.cuda.is_available() is False - driver missing/too old, "
              "or a CPU-only torch build was installed")
        return
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / 1024 ** 3
    check("gpu", OK if "4070" in name else WARN, f"{name}, {vram_gib:.1f} GiB VRAM")
    check("vram >= 7.5 GiB", OK if vram_gib >= 7.5 else FAIL, f"{vram_gib:.2f} GiB")
    check("bf16 support", OK if torch.cuda.is_bf16_supported() else FAIL,
          "required by the cuda profile (mirrors the notebook-01 Colab recipe)")
    free_b, total_b = torch.cuda.mem_get_info()
    check("free VRAM at start", OK if free_b / 1024 ** 3 >= 6.5 else WARN,
          f"{free_b / 1024 ** 3:.2f} GiB free of {total_b / 1024 ** 3:.2f} "
          "(close other GPU apps / move display to the iGPU if low)")

    free_gb = shutil.disk_usage(REPO).free / 1000 ** 3
    check("free disk >= 80 GB", OK if free_gb >= 80 else WARN, f"{free_gb:.0f} GB free")

    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32),
                        ("dwMemoryLoad", ctypes.c_uint32)] + [
                (n, ctypes.c_uint64) for n in (
                    "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                    "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                    "ullAvailExtendedVirtual")]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        ram_gib = stat.ullTotalPhys / 1024 ** 3
        check("RAM >= 16 GiB", OK if ram_gib >= 15 else WARN, f"{ram_gib:.0f} GiB")

        try:
            import winreg
            with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
                value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
            check("long paths enabled", OK if value == 1 else WARN,
                  "set HKLM\\...\\FileSystem\\LongPathsEnabled=1 if HF cache "
                  "paths ever error")
        except OSError:
            check("long paths enabled", WARN, "could not read registry")

        in_onedrive = "onedrive" in str(REPO).lower()
        check("repo outside OneDrive", WARN if in_onedrive else OK,
              str(REPO) if in_onedrive else "")

    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(
            pilot_cfg["model_id"], revision=pilot_cfg["model_revision"],
            local_files_only=True)
        check("model cached", OK,
              f"{pilot_cfg['model_id']} @ {pilot_cfg['model_revision'][:8]}")
    except Exception:
        check("model cached", FAIL, "run scripts/prefetch_assets.py first")

    for fname in ("gsm8k_splits.json", "svamp_splits.json", "probe_set_ids.json"):
        path = PILOT / "data" / fname
        check(f"data/{fname}", OK if path.exists() else FAIL,
              "" if path.exists() else "frozen split file missing from checkout")


def grpo_probe(full: bool, per_device_batch: int | None = None,
               grad_accum: int | None = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from src.data import gsm8k_grpo_dataset
    from src.reward import exact_answer_reward

    pilot = json.loads((PILOT / "pilot_config.json").read_text())
    stage_a = pilot["stage_a"]
    try:
        from run_local_pipeline import EXECUTION_PROFILES
        cuda_profile = EXECUTION_PROFILES["cuda"]
    except Exception:
        cuda_profile = {}
    # Rehearse exactly what the formal cuda profile will do: master-weight
    # dtype, autocast dtype, and optimizer all come from the profile (v2 =
    # fp32 master + bf16 autocast + paged_adamw_8bit; pure-bf16 params made
    # v1 a no-op run - see ../WIN4070_RUN_ANALYSIS.md).
    master_dtype_name = cuda_profile.get("dtype", "float32")
    autocast_dtype_name = cuda_profile.get("autocast_dtype", master_dtype_name)
    optim = cuda_profile.get("optim")
    if full:
        default_batch = cuda_profile.get(
            "per_device_train_batch_size", stage_a["per_device_train_batch_size"])
        default_accum = cuda_profile.get(
            "gradient_accumulation_steps", stage_a["gradient_accumulation_steps"])
        per_device_batch = per_device_batch or default_batch
        grad_accum = grad_accum or default_accum
        geometry = dict(
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            num_generations=stage_a["num_generations"],
            max_completion_length=stage_a["max_completion_length"])
    else:
        geometry = dict(per_device_train_batch_size=4,
                        gradient_accumulation_steps=2,
                        num_generations=4, max_completion_length=64)
    if full and (per_device_batch != default_batch or grad_accum != default_accum):
        label = f"full-geometry-mb{per_device_batch}-ga{grad_accum}"
    else:
        label = "full-geometry" if full else "small"
    scratch = HERE / "logs" / f"_grpo_probe_{label}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    print(f"\n== GRPO probe ({label}): one update, master={master_dtype_name} "
          f"autocast={autocast_dtype_name} optim={optim or 'trainer-default'} "
          "+ gradient checkpointing ==")
    set_seed(20260708)  # probe-only seed; probe artifacts are never results
    tokenizer = AutoTokenizer.from_pretrained(
        pilot["model_id"], revision=pilot["model_revision"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtypes = {"float32": torch.float32, "float16": torch.float16,
              "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        pilot["model_id"], revision=pilot["model_revision"],
        dtype=dtypes[master_dtype_name], local_files_only=True).to("cuda")

    args = GRPOConfig(
        output_dir=str(scratch / "trainer"), seed=20260708, max_steps=1,
        learning_rate=stage_a["learning_rate"], beta=stage_a["beta"],
        temperature=stage_a["temperature"], top_p=stage_a["top_p"],
        bf16=(autocast_dtype_name == "bfloat16"),
        fp16=(autocast_dtype_name == "float16"),
        **({"optim": optim} if optim else {}),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none", **geometry)
    trainer = GRPOTrainer(model=model, args=args,
                          train_dataset=gsm8k_grpo_dataset(),
                          reward_funcs=exact_answer_reward,
                          processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OOM at the", label, "geometry. Fallback ladder "
              "(log every rung you take in compute_log.md + notebook header):")
        print("  1. free VRAM: close GPU apps, move the display to the iGPU")
        print("  2. micro-batch 2 x grad-accum 32 (same 64-completion update)")
        print("  3. stop and flag to the team - do not change scientific knobs")
        raise
    wall = time.time() - t0
    row = {
        "probe": label,
        "wall_seconds_one_update": round(wall, 2),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "geometry": geometry,
        "master_dtype": master_dtype_name, "autocast_dtype": autocast_dtype_name,
        "optim": optim or "trainer-default", "gradient_checkpointing": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "unix_time": time.time(),
    }
    (HERE / "logs").mkdir(exist_ok=True)
    with (HERE / "logs" / "probe_results.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    if full:
        total_updates = pilot["stage_a"]["max_steps"] + \
            len(pilot["stage_a"]["checkpoint_steps"]) * pilot["adaptation"]["budget_updates"]
        print(f"\nExtrapolation: ~{wall:.0f} s/update -> ~"
              f"{wall * total_updates / 3600:.1f} h of pure training for "
              f"Phase 1 + Phase 3 ({total_updates} updates), plus eval and "
              "measurement overhead. First updates run slower until CUDA "
              "kernels/caches warm up; calibrate on the first 5 real updates.")
    shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--grpo-probe", action="store_true",
                       help="one update at the full pre-registered geometry")
    group.add_argument("--grpo-probe-small", action="store_true",
                       help="one tiny update (fast stack check)")
    parser.add_argument("--per-device-batch", type=int,
                        help="full-geometry fallback ladder micro-batch")
    parser.add_argument("--grad-accum", type=int,
                        help="full-geometry fallback ladder gradient accumulation")
    args = parser.parse_args()
    if (args.per_device_batch or args.grad_accum) and not args.grpo_probe:
        parser.error("--per-device-batch/--grad-accum only apply to --grpo-probe")

    env_checks()
    fails = [r for r in RESULTS if r[1] == FAIL]
    print(f"\nenv checks: {len(RESULTS) - len(fails)} ok, {len(fails)} failed")
    if fails:
        sys.exit(1)
    if args.grpo_probe or args.grpo_probe_small:
        grpo_probe(full=args.grpo_probe, per_device_batch=args.per_device_batch,
                   grad_accum=args.grad_accum)


if __name__ == "__main__":
    main()
