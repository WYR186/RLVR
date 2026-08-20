# Where the model weights are

The three Stage-A LoRA adapters are **not** in the package — 154 MB each, and nothing
in `VERIFY.py` or any figure needs them. You only need them to re-run training or to
re-measure Q on a checkpoint.

## What exists

| checkpoint | contents | size |
|---|---|---|
| `ckpt-0` | `adapter_model.safetensors`, `adapter_config.json`, `README.md` | 154.1 MB |
| `ckpt-50` | the above + `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | 165.0 MB |
| `ckpt-100` | same as ckpt-50 | 165.0 MB |

`ckpt-0` is the adapter saved before the first update. LoRA initialises `B` to zero,
so it is the identity — attaching it is equivalent to running the bare base model.
It is kept explicitly rather than reconstructed, so every arm loads a checkpoint the
same way and nothing depends on "None means base model" behaving as expected.

Adapter config, identical for all three:

```json
{"peft_type": "LORA", "task_type": "CAUSAL_LM",
 "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
 "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
 "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"]}
```

## How to get them

**Google Drive**, in `MyDrive/eaaj-exp2-checkpoints/`:

```
02_ckpt-0.tar.gz      31.4 MB
03_ckpt-50.tar.gz    151.5 MB
04_ckpt-100.tar.gz   151.5 MB
```

That folder also holds the two configs, the frozen splits, the completion-length
measurement, and a mirror of every Stage-B arm — enough to rebuild a Colab runtime
from scratch (see `drivers/00_restore_from_drive.py`).

The Drive folder is private. **Ask Aaron for access** rather than assuming it is
shared; it is a personal Drive, not a team one. Filenames are numbered because the
adapters were originally recovered from a Colab runtime part by part.

## How to load one

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "path/to/ckpt-100")
```

`src/pipeline.py:build_peft_model` does exactly this and then upcasts the LoRA
parameters to fp32 — use it rather than the snippet above if you want to reproduce
the run's numerics, since the adapter dtype affects them.

## One caveat if you re-measure Q

`measurement_contract` in every `metrics_ckpt*.json` records the exact conditions
these numbers were produced under: eval mode, bf16, last-non-padding-token pooling,
mean-abs-over-non-padding for dormancy, 512 max prompt tokens, float32 accumulator,
float64 SVD, layers [5, 14, 26], 4096 probe prompts. Effective-rank magnitudes are
sensitive to the probe count in particular. Numbers measured under a different
contract are not comparable to these ones —
`CROSS_RUN_NOTE_7B_VS_05B.md` §6.2 is about exactly this problem.
