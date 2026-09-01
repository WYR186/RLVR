# Arm A on the RTX 4070 — the last experiment E4 needs

**For:** the agent on the Windows RTX 4070
**Why this one:** it is the only remaining measurement that puts dose, noise
response and *real* update response in a single frame — at the one scale where
the outcome side actually moved (exp1.5 v3: adaptability −8.56 pp while
erank_L12 moved −0.976%).
**Cost:** 2–4 passes of a 0.5B model. Minutes on this hardware. No units.
**Read first:** `FINDING_E4_DETECTOR_CALIBRATION.md` §5 — it states exactly
which comparison is missing and why the tempting number cannot be computed yet.

---

## 0. What is missing, precisely

`outputs/e4_small/` already has, all in one frame:

- **Arm W** — exp1.5 v3's dose: ckpt-0 `0.0`, ckpt-500 `7.179374e-04`
- **Arm N** — isotropic noise at that exact dose, 3 directions:
  0.0321% / 0.0079% / 0.0301%, mean **0.0234%**

What is *not* in that frame is exp1.5 v3's **own** response. Its published
−0.976% was measured on **its own GSM8K probe at n_probe = 512**, which is below
the 896-dim hidden size and therefore sample-truncated, while the ladder uses
the 4096-prompt GURU probe. Different probe, different size, truncated spectrum.
**Those numbers must not be divided by one another.** This run fixes that by
measuring the checkpoints themselves under E4-small's contract.

---

## 1. The one thing that is easy to get wrong

**exp1.5 v3 was trained from `Qwen/Qwen2.5-0.5B` (base), revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` — the same model and revision as
`R_base` in `outputs/e4_small/`.**

So every Arm-A number is compared against **`R_base`**, never `R_instruct`.
`10_e4_report.py` defaults to `R_instruct`; you must pass `--reference R_base`.

This also gives a free identity gate: **`A_ckpt0` must reproduce `R_base` to
exactly `0.000000%`.** ckpt-0 is the pre-update snapshot of the same weights.
Anything else means the wrong checkpoint, the wrong revision, or a dtype
mismatch — stop and report rather than continuing.

(Verified end to end on the packaging machine by saving Qwen2.5-0.5B as a
synthetic ckpt-0 and measuring it: `0.000000%`.)

---

## 2. Setup

```bash
git pull
```

`08_e4_ruler.py` gained `--checkpoints LABEL=PATH` for **full-parameter**
snapshots. It loads them with `AutoModelForCausalLM`, unlike `--adapters`,
which is for LoRA and would fail here.

Reuse the environment and the frozen probe from the E4-small run. The probe is
committed, so nothing needs re-deriving.

---

## 3. Run it

Set `RUN` to the exp1.5 v3 run directory (the one holding `ckpt-0 … ckpt-500`
with their `.safetensors`):

```bash
python drivers/08_e4_ruler.py --scale small --probe ../outputs/e4_small/probe_frozen.json --out ../outputs/e4_small --device auto --dtype float32 --batch-size 8 --spectra-only --checkpoints ckpt0=$RUN/ckpt-0 ckpt100=$RUN/ckpt-100 ckpt500=$RUN/ckpt-500
```

Notes that matter:

- `--out ../outputs/e4_small` is deliberate: writing into the **same directory**
  as the ladder is what puts Arm A in the same frame. Existing arms are skipped,
  not recomputed.
- `--dtype float32` and `--batch-size 8` match what the E4-small arms used. The
  audit fails if any arm disagrees on dtype, probe size or layers — that check
  exists precisely to stop a mismatched Arm A from being compared.
- `--spectra-only` matches the other arms too.
- Do **not** pass `--arms`; omitting it skips R and N, which are already done.
- If the 8 GiB device OOMs, drop to `--batch-size 4`. That is an execution
  parameter and changes no measured quantity.

Adding `ckpt100` is worth the extra pass: its dose (`4.876857e-04`) is close to
the 7B Stage-A dose (`5.459591e-04`), which makes the two scales comparable *as
doses* even though their erank levels are not.

---

## 4. Check it

```bash
python drivers/09_audit_e4_artifacts.py --dir ../outputs/e4_small --require-arm-w
python drivers/10_e4_report.py --dir ../outputs/e4_small --reference R_base
```

Gates that must hold:

| gate | expected |
|---|---|
| `A_ckpt0` vs `R_base` | **exactly 0.000000%** |
| tokenizer identity | passes, or the run says which tokenizer it used and why |
| gated-MLP hook check | max abs err 0.0 |
| dtype / probe / layers | identical across every arm in the directory |

---

## 5. What to report

1. `A_ckpt0`, `A_ckpt100`, `A_ckpt500` max \|Δerank\| **against `R_base`**.
2. The comparison this whole run exists for, at matched dose `7.179374e-04`:
   **`A_ckpt500` response vs Arm N's 0.0234% mean (range 0.0079–0.0321%).**
3. Every gate result, verbatim, including any that failed.

Then the question is answerable in one frame: **does a real, structured,
gradient-derived update move the activation spectrum more per unit weight norm
than isotropic noise of the same size?**

Both answers are useful, and neither should be reached for:

- **Yes, clearly larger.** Effective rank is sensitive to *what kind* of weight
  change occurred, not only how big it was — so the isotropic ladder understates
  the detector for real training, and every threshold in
  `FINDING_E4_DETECTOR_CALIBRATION.md` is a conservative bound.
- **No, comparable.** The dose threshold is the whole story, and our regime is
  simply too small an intervention to test any of this. That is the cleaner
  power-analysis result.

## 6. What this still will not license

exp1.5 v3's −8.56 pp adaptability drop and its erank behaviour are measured on
different probes and remain separate observations. This run makes the *dose* and
the *spectral response* commensurable; it does not link either to adaptability.
Do not write a causal chain from weight dose to adaptability loss.
