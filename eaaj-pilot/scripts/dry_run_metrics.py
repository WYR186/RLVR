"""Phase 0 dry run (briefing §6): metric functions exercised on the real
base model with 8 probe prompts. Run locally (CPU/MPS) or on Colab T4.

Usage: python scripts/dry_run_metrics.py [n_prompts]
Writes outputs/dry_run_metrics.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_probe_prompts
from src.metrics import checkpoint_q_metrics

ROOT = Path(__file__).resolve().parent.parent
PILOT = json.loads((ROOT / "pilot_config.json").read_text())
MODEL = PILOT["model_id"]
REVISION = PILOT["model_revision"]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
print(f"device={device} dtype={dtype}")

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REVISION, dtype=dtype).to(device)

prompts = load_probe_prompts()[:n]
t0 = time.time()
metrics = checkpoint_q_metrics(model, tokenizer, prompts, batch_size=4)
metrics["model"] = MODEL
metrics["model_revision"] = REVISION
metrics["purpose"] = "plumbing_only_not_scientific_result"
metrics["device"] = device
metrics["wall_seconds"] = time.time() - t0

out = Path(__file__).resolve().parent.parent / "outputs" / "dry_run_metrics.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(metrics, indent=1))
print(json.dumps({k: v for k, v in metrics["per_layer"].items()}, indent=1)[:2000])
print(f"wrote {out} in {metrics['wall_seconds']:.1f}s")
