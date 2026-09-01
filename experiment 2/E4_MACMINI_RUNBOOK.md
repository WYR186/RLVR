# E4 runbook — for the agent on the Mac mini

You are running **E4, detector calibration**. Read
`SPEC_E4_DETECTOR_CALIBRATION.md` before touching anything; this file is the
operational half only. `CLAUDE.md` at the repo root governs, and its hard
constraints are not negotiable.

**Your machine has 16 GB.** That is the fact the whole plan is shaped around:
a 7B model in bf16 is 15.2 GB of weights and does **not** fit. You run the
CPU-side arms and the entire 0.5B scale. The 7B scale is handed to a Colab
A100 session afterwards, by a human, using the same code you will have
already proven green.

---

## 0. The one-line goal

E1 showed Q is flat across our three checkpoints. Nobody knows whether 0.73%
is a big number or a small one. You are building the ruler that says so.

---

## 1. Do not do these

- **Do not quantize the model to make 7B fit.** 4-bit or 8-bit perturbs
  exactly the activation statistics this experiment measures. If something
  does not fit, it goes to Colab. This is not a performance trade-off.
- **Do not re-render the probe per model.** Step 2 freezes the text once.
  Every later step reads that file. Re-rendering would confound a weight
  difference with a prompt difference and silently invalidate Arm R.
- **Do not "fix" a failing gate by loosening it.** Gates P1, R1, W1, A1, A2,
  A3 stop the run on purpose. If one fires, write down what it said and stop.
- **Do not write causal language into any result.** Not "instruction tuning
  reduces plasticity", not "RLVR reduces the ability to learn". See
  `SPEC_E4_DETECTOR_CALIBRATION.md` §1 for the exact framing this project is
  held to.
- **Do not commit `.safetensors`.** They are gitignored for a reason.

---

## 2. Setup

```bash
cd <repo>/experiment\ 2
python3 -m venv .venv-e4 && ./.venv-e4/bin/pip install -U pip
./.venv-e4/bin/pip install torch transformers safetensors numpy pyarrow huggingface_hub datasets pytest psutil
```

Verify before anything else — this must be **green**, not "green except
environment errors":

```bash
./.venv-e4/bin/python -m pytest tests/test_e4_calibration.py -q
```

Expected: `30 passed`. If any test errors on a missing module, install it;
the suite is written so a bare machine can run all of it.

Disk you will need: ~2 GB for the 0.5B pair, ~3 GB for the GURU parquet,
plus **15.2 GB** if you do the optional Arm W base-norm half. Check with
`df -h .` first.

---

## 3. Step 1 — freeze the probe (ALREADY DONE — skip unless re-deriving)

**The frozen probe is committed.** `outputs/probe_frozen.json` (2.1 MB) and
`outputs/probe_manifest.json` are in git, so every machine gets the
byte-identical probe without the ~3 GB GURU download. Verify and move on:

```bash
python3 -c "import json,hashlib;p=json.load(open('../outputs/e4/probe_frozen.json'))['prompts'];h=hashlib.sha256();[ (h.update(x.encode()),h.update(b'\x00')) for x in p];print(len(p), h.hexdigest())"
```

Expect `4096 8bc2b4066c892c0ff6ac69e9a64846557c1d586c2ee6e162d3b831abbaecd265`,
matching `rendered_text_sha256` in the manifest. Gate P1 already passed on the
packaging machine (probe ids `1e61252e7b54793e`, n=4096, not truncated).

Only re-run the driver below if you are deliberately re-deriving the probe from
the dataset.

### Re-deriving it (minutes)

```bash
./.venv-e4/bin/python drivers/06_e4_freeze_probe.py --out ../outputs/e4
```

This downloads the GURU parquet on first run and renders the 4096 frozen
probe prompts through the Qwen2.5-7B-Instruct chat template.

**Gate P1** must print `probe id gate OK: 1e61252e7b54793e, n=4096`. If it
does not, stop — you do not have E1's probe and nothing you measure would be
comparable to E1.

Record `rendered_text_sha256` from the output. The Colab session must see the
same value.

---

## 4. Step 2 — Arm W, the LoRA's dose in weight space

The three adapter `.safetensors` ship in the handoff bundle (see §8), not in
git. Unpack them first, then:

```bash
./.venv-e4/bin/python drivers/07_e4_weight_dose.py \
  --adapters ../eaaj-pilot/outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/stage_a \
  --out ../outputs/e4 --skip-base-norms
```

That gives `||ΔW||_F` per module in seconds and needs no model download. It
records `relative_dose: not_run` with a reason, which is a legitimate
intermediate state, not a failure.

**To complete the ratio** you need the base weight norms, which means the
15.2 GB Instruct download — disk only, never fully in RAM:

```bash
./.venv-e4/bin/python drivers/07_e4_weight_dose.py \
  --adapters <same path> --out ../outputs/e4 --download
```

**Gate W1:** `ckpt-0` must report a dose of exactly `0.0`. LoRA initialises
`B = 0`, so the pre-update adapter is the identity. A nonzero value means you
read the wrong checkpoint — stop.

If disk is tight, run `--skip-base-norms` now and leave the ratio for the
Colab session, which downloads the model anyway.

---

## 5. Step 3 — E4-small, Arms R and N (1–3 hours)

```bash
./.venv-e4/bin/python drivers/08_e4_ruler.py \
  --scale small --probe ../outputs/e4/probe_frozen.json \
  --out ../outputs/e4_small --arms R N --device auto
```

`--device auto` picks MPS on Apple Silicon. Each arm writes its own JSON and
is **skipped if that JSON already exists**, so if the run is interrupted,
re-run the identical command and it resumes.

**Wrap it in `caffeinate -dimsu`.** Learned the hard way on 2026-08-31: the
packaging machine lost power mid-run and slept. The process survived, but its
MPS context did not — it sat in state `SN` at 0.0% CPU with its resident set
collapsed from 14 GB to 0.7 GB and made no progress for 93 minutes. It does not
recover; kill it and re-run. No completed arm was damaged (every one of the
seven artifacts on disk parsed and verified), so the only cost was wall clock —
but a wedged run looks exactly like a slow one, so check `ps -o stat=,%cpu=`
and the log mtime rather than assuming it is still working.

**Gate R1** must print `tokenizer identity gate OK` with
`"n_mismatched": 0, "identical_vocab": true`. If it fails, stop and report —
the two models would differ in input as well as weights.

Runtime control: it is 8 probe passes of 4096 prompts. If a pass looks like
it will take more than ~30 minutes, cut the probe instead of the arms:

```bash
./.venv-e4/bin/python drivers/06_e4_freeze_probe.py --out ../outputs/e4_n1024 --n-probe 1024
```

1024 still exceeds the 0.5B hidden size (896), so the spectrum is not
rank-truncated. Anything below 896 is, and the manifest will say so. Never
mix probe sizes between arms — the audit fails on it (Gate A1), correctly.

A useful sanity check, free: passing the same model to both
`--base-model` and `--instruct-model` must make Arm R report exactly
`0.0000%`. That is the pipeline's self-test.

---

## 6. Step 4 — audit

```bash
./.venv-e4/bin/python drivers/09_audit_e4_artifacts.py --dir ../outputs/e4_small
```

It recomputes the ruler table from the raw per-arm records and checks every
gate in `SPEC_E4_DETECTOR_CALIBRATION.md` §6. It must print
`E4 ARTIFACT AUDIT PASS`. **Any assertion failure is a real finding — report
it, do not patch around it.**

---

## 7. What to report back

Post these, and nothing interpretive beyond them:

1. The audit's `erank change vs R_instruct` table.
2. The ladder table: requested dose → achieved dose → max |Δerank|.
3. Arm W's aggregate relative dose for ckpt-50 and ckpt-100.
4. Where E1's **+0.7303%** falls on the ladder — between which two rungs.
5. Every gate that fired, verbatim.
6. Wall-clock per arm, and the machine's RAM (the audit records both).

The single question this answers: **is our Stage-A dose above or below the
dose at which Q starts to register?** If below, the "your intervention was
under-dosed" objection is answered with a number. If Q is flat even at a
destructive dose, that is the stronger negative result. Both are wins; do not
push the data toward either.

**Reference numbers from the validation run** on Qwen2.5-0.5B, CPU float32,
n=1024, layers 4/12/22 — your MPS numbers will differ slightly and that is
expected, but the shape should hold:

| requested dose | achieved | max \|Δerank\| | at layer |
|---|---|---|---|
| 1e-3 | 9.999853e-04 | 0.1114% | 4 |
| 1e-2 | 9.999770e-03 | 1.7703% | 12 |
| 3e-2 | 2.999943e-02 | 3.2035% | 12 |
| 1e-1 | 9.999856e-02 | 19.0537% | 12 |

Monotone, and every rung hit its requested dose to under 0.01% relative error.
E1's 7B LoRA moved erank by at most **0.7303%**, which at this scale falls
between the 1e-3 and 1e-2 rungs. Whether our *actual* dose also falls there is
exactly what Arm W decides — and it is the whole point of the experiment, so
do not pre-empt it.

Arm W on the real adapters (validated on the packaging machine, adapter side
only): 196 LoRA modules, `||ΔW||_F` = **0** for ckpt-0 (Gate W1 holds on real
data), **0.678072** for ckpt-50, **0.727543** for ckpt-100. The ratio still
needs the base-weight norms.

---

## 8. What is in the handoff bundle

`E4_HANDOFF.tar.gz` contains the three Stage-A adapters (~460 MB) that are
gitignored. Everything else — spec, code, drivers, tests, this runbook —
comes from `git pull`. Unpack with:

```bash
tar -xzf E4_HANDOFF.tar.gz -C eaaj-pilot/outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/
```

Verify against the SHA-256 printed in `E4_HANDOFF.sha256` before use.

---

## 9. Escalating to the 7B scale

Do not attempt this on the Mac mini. When E4-small is green, the same command
with `--scale large` runs on a Colab A100 High-RAM session. Two things that
are easy to forget and are hard constraints:

- **Take the GPU-dashboard screenshots** — balance before, resources panel,
  balance after, disconnect confirmation. `compute_log.md` records that *no*
  exp2 session has ever had them. This one starts.
- **Disconnect and delete the runtime the moment the run finishes.** 5.6
  units were burned idle after a previous run completed.

Budget: ~94 units remain. E4-large is ~7–10.
