"""Short real-GPU smoke for the Stage-B completion and telemetry path."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.adaptation import run_fixed_budget_adaptation  # noqa: E402
from src.repeats import (EXPECTED_BASELINES, validate_runtime_against_source,  # noqa: E402
                         validate_source_run)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--updates", type=int, choices=(2, 3, 4, 5), default=2)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    import torch

    source_run = args.source_run.resolve()
    config, manifest = validate_source_run(source_run, 0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage-B smoke")
    validate_runtime_against_source(manifest, torch.cuda.get_device_name(0))
    execution = config["execution"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir.resolve() if args.out_dir else
               PROJECT / "outputs" / f"smoke_stageb_repeat_{stamp}")
    summary = run_fixed_budget_adaptation(
        checkpoint_path=source_run / "ckpt-0",
        out_dir=out_dir,
        budget_updates=args.updates,
        eval_every=1,
        seed=943,
        learning_rate=1e-6,
        num_generations=8,
        per_device_batch=4,
        grad_accum=16,
        beta=0.0,
        temperature=0.7,
        top_p=1.0,
        max_prompt_length=512,
        max_completion_length=512,
        bf16=False,
        device="cuda",
        dtype_name=execution["dtype"],
        autocast_dtype_name=execution["autocast_dtype"],
        optim=execution["optim"],
        gradient_checkpointing=execution["gradient_checkpointing"],
        save_steps=args.updates,
        expected_acc_before=EXPECTED_BASELINES[0],
    )
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
