"""E4 — detector calibration: how large an intervention does Q actually need?

Implements `SPEC_E4_DETECTOR_CALIBRATION.md`. Nothing here re-opens E1: the
measurement contract, the frozen probe and the reductions are E1's, reused
verbatim through `e1_sweep`. E4 only changes *what model* is measured.

E1 established that Q is flat across the three rank-16 Stage-A checkpoints.
"Flat" is a statement without a scale until something known-large is measured
on the same ruler. E4 supplies the scale along one axis that can be stated in
a number — relative Frobenius weight change:

  Arm W   the Stage-A LoRA's own dose        ||BA*(alpha/r)||_F / ||W_0||_F
  Arm R   a known-large intervention          Q(base) vs Q(Instruct)
  Arm N   a reference perturbation ladder     Q(Instruct + noise at dose d)

Arm N's noise is scaled to hit an exact relative Frobenius dose, so Arm W's
number lands on Arm N's x-axis and the three arms compose into one figure.

What this module is careful NOT to claim: isotropic Gaussian noise is not an
RLVR update, and base->Instruct is not a controlled dose. Arm N is a
sensitivity curve for the *detector*, and Arm R is an order-of-magnitude
reference point, not a causal statement about plasticity. Both are labelled
that way in every record they write.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# The LoRA target modules of the Stage-A adapters (WEIGHTS.md). Arm N perturbs
# exactly this set so its dose axis is the same quantity Arm W measures.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")

# Arm N dose ladder, relative Frobenius. Spans the region where the Stage-A
# LoRA dose is expected to fall through to a clearly destructive perturbation.
DOSE_LADDER = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

NOISE_SEED = 42


# ---------------------------------------------------------------------------
# Arm W — the Stage-A LoRA's dose in weight space
# ---------------------------------------------------------------------------

def lora_scaling(adapter_config: dict) -> float:
    """PEFT's LoRA scaling alpha/r (or alpha/sqrt(r) when rslora is set)."""
    r = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    if adapter_config.get("use_rslora"):
        return alpha / (r ** 0.5)
    return alpha / r


def _base_key_for(lora_key: str) -> str | None:
    """Map a PEFT adapter tensor name onto its base-model weight name.

    PEFT writes e.g.
      base_model.model.model.layers.0.mlp.up_proj.lora_A.weight
    against a base weight named
      model.layers.0.mlp.up_proj.weight
    """
    if ".lora_A" not in lora_key and ".lora_B" not in lora_key:
        return None
    stem = lora_key.split(".lora_")[0]
    prefix = "base_model.model."
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return f"{stem}.weight"


def lora_delta_norms(adapter_dir, *, scaling: float | None = None) -> dict:
    """Frobenius norm of each module's effective weight update ||B A * s||_F.

    Streams tensor by tensor through safetensors so a 16 GB machine never holds
    more than one A/B pair. Returns {base_weight_name: {"delta_fro": float,
    "shape": [...], "r": int}}.
    """
    from safetensors import safe_open

    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if scaling is None:
        scaling = lora_scaling(cfg)

    path = adapter_dir / "adapter_model.safetensors"
    out: dict = {}
    with safe_open(str(path), framework="np") as f:
        keys = list(f.keys())
        a_keys = {_base_key_for(k): k for k in keys if ".lora_A" in k}
        b_keys = {_base_key_for(k): k for k in keys if ".lora_B" in k}
        for base_key in sorted(set(a_keys) & set(b_keys)):
            A = np.asarray(f.get_tensor(a_keys[base_key]), dtype=np.float64)
            B = np.asarray(f.get_tensor(b_keys[base_key]), dtype=np.float64)
            # W_eff = W_0 + s * B @ A  (PEFT convention: A is r x in, B is out x r)
            delta = (B @ A) * scaling
            out[base_key] = {
                "delta_fro": float(np.linalg.norm(delta)),
                "shape": [int(x) for x in delta.shape],
                "r": int(cfg["r"]),
            }
    return out


def full_delta_norms(ckpt_files, base_files, *,
                     target_modules=TARGET_MODULES) -> dict:
    """||W_ckpt - W_0||_F per module for a FULL-PARAMETER checkpoint.

    `lora_delta_norms` only covers adapters. Runs like exp1.5 v3 train every
    parameter, so their dose is a direct weight difference against the base
    snapshot. Both sides are streamed a tensor at a time and matched by name,
    so an 8 GiB machine can do this for a 0.5B model without loading either
    model whole.

    Only weights whose name ends in one of `target_modules` are compared, so
    the dose is measured over the same parameter set as Arm N perturbs and as
    `lora_delta_norms` reports. Returns the same shape as `lora_delta_norms`,
    which means `relative_dose` consumes either one unchanged.
    """
    from safetensors import safe_open

    def index(files):
        loc = {}
        for path in files:
            with safe_open(str(path), framework="np") as f:
                for key in f.keys():
                    loc[key] = path
        return loc

    ckpt_loc = index(ckpt_files)
    base_loc = index(base_files)

    def wanted(key: str) -> bool:
        stem = key[:-len(".weight")] if key.endswith(".weight") else key
        return any(stem.endswith(t) for t in target_modules)

    out: dict = {}
    for key in sorted(k for k in ckpt_loc if wanted(k)):
        if key not in base_loc:
            continue
        with safe_open(str(ckpt_loc[key]), framework="np") as fc:
            W = np.asarray(fc.get_tensor(key), dtype=np.float64)
        with safe_open(str(base_loc[key]), framework="np") as fb:
            W0 = np.asarray(fb.get_tensor(key), dtype=np.float64)
        if W.shape != W0.shape:
            raise ValueError(
                f"{key}: checkpoint shape {W.shape} != base {W0.shape}")
        out[key] = {
            "delta_fro": float(np.linalg.norm(W - W0)),
            "shape": [int(x) for x in W.shape],
            "r": None,
        }
    if not out:
        raise ValueError(
            "no target-module weights matched between the checkpoint and the "
            f"base snapshot; target_modules={list(target_modules)}")
    return out


def base_weight_norms(weight_files, wanted: set[str] | None = None) -> dict:
    """||W_0||_F for each base weight, streamed one tensor at a time.

    `weight_files` is an iterable of `.safetensors` shard paths. Only tensors in
    `wanted` are read, so Arm W never materializes the 15 GB model.
    """
    from safetensors import safe_open

    out: dict = {}
    for path in weight_files:
        with safe_open(str(path), framework="np") as f:
            for key in f.keys():
                if wanted is not None and key not in wanted:
                    continue
                W = np.asarray(f.get_tensor(key), dtype=np.float64)
                out[key] = {"base_fro": float(np.linalg.norm(W)),
                            "shape": [int(x) for x in W.shape]}
    return out


def relative_dose(delta_norms: dict, base_norms: dict) -> dict:
    """Per-module and aggregate relative Frobenius dose ||dW||_F / ||W_0||_F.

    The aggregate is the norm ratio over the concatenation of every targeted
    module, i.e. sqrt(sum ||dW_m||^2) / sqrt(sum ||W_m||^2) — not the mean of
    per-module ratios, which would weight a tiny module equally with a large
    one. Both are reported; the aggregate is the one Arm N's ladder matches.
    """
    per_module = {}
    num = 0.0
    den = 0.0
    missing = []
    for key, d in sorted(delta_norms.items()):
        b = base_norms.get(key)
        if b is None:
            missing.append(key)
            continue
        ratio = d["delta_fro"] / b["base_fro"] if b["base_fro"] > 0 else float("nan")
        per_module[key] = {
            "delta_fro": d["delta_fro"],
            "base_fro": b["base_fro"],
            "relative_dose": ratio,
            "shape": d["shape"],
        }
        num += d["delta_fro"] ** 2
        den += b["base_fro"] ** 2
    aggregate = (num ** 0.5) / (den ** 0.5) if den > 0 else float("nan")
    ratios = [v["relative_dose"] for v in per_module.values()
              if np.isfinite(v["relative_dose"])]
    return {
        "per_module": per_module,
        "aggregate_relative_dose": aggregate,
        "mean_per_module_relative_dose": float(np.mean(ratios)) if ratios else float("nan"),
        "max_per_module_relative_dose": float(np.max(ratios)) if ratios else float("nan"),
        "min_per_module_relative_dose": float(np.min(ratios)) if ratios else float("nan"),
        "n_modules": len(per_module),
        "modules_missing_from_base": missing,
    }


# ---------------------------------------------------------------------------
# Arm N — the reference perturbation ladder
# ---------------------------------------------------------------------------

def _module_seed(global_seed: int, dose: float, name: str) -> int:
    """Deterministic per-(dose, module) seed so a ladder rung is reproducible."""
    h = hashlib.sha256(f"{global_seed}|{dose:.12g}|{name}".encode()).hexdigest()
    return int(h[:8], 16)


def perturb_model_(model, dose: float, *, seed: int = NOISE_SEED,
                   target_modules=TARGET_MODULES) -> dict:
    """Add Gaussian noise at an EXACT relative Frobenius dose, in place.

    For each targeted Linear weight W:  W <- W + dose * (||W||_F / ||G||_F) * G,
    G ~ N(0, 1) of the same shape. The rescaling makes ||dW||_F / ||W||_F equal
    to `dose` by construction, so the ladder's x-axis is the same quantity
    `relative_dose` reports for the LoRA.

    The achieved dose is measured back off the actual tensors (in float32,
    before the dtype cast) and returned, so a bf16 rounding surprise shows up in
    the record instead of being assumed away.

    Returns a record of what was perturbed. The model is modified IN PLACE —
    reload it before measuring the next rung.
    """
    import torch

    touched = []
    num = 0.0
    den = 0.0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not any(name.endswith(t) for t in target_modules):
                continue
            W = getattr(module, "weight", None)
            if W is None or W.ndim != 2:
                continue
            # .clone() is load-bearing: `.to(float32)` on a float32
            # parameter returns the SAME tensor, so without the copy `w32`
            # aliases W.data and the achieved-dose measurement below would
            # compare the perturbed weight against itself and report 0.
            w32 = W.detach().to(torch.float32).clone()
            w_norm = float(torch.linalg.norm(w32))
            if w_norm <= 0:
                continue
            g = torch.Generator(device="cpu").manual_seed(
                _module_seed(seed, dose, name))
            G = torch.randn(w32.shape, generator=g, dtype=torch.float32)
            g_norm = float(torch.linalg.norm(G))
            if g_norm <= 0:
                continue
            dW = G.to(w32.device) * (dose * w_norm / g_norm)
            new = w32 + dW
            W.data.copy_(new.to(W.dtype))
            # Measure what the parameter ACTUALLY moved by, post dtype cast.
            achieved = float(torch.linalg.norm(
                W.detach().to(torch.float32) - w32))
            touched.append({"module": name, "shape": list(w32.shape),
                            "base_fro": w_norm, "delta_fro": achieved,
                            "relative_dose": achieved / w_norm})
            num += achieved ** 2
            den += w_norm ** 2
            del w32, G, dW, new

    return {
        "requested_dose": float(dose),
        "achieved_aggregate_dose": (num ** 0.5) / (den ** 0.5) if den > 0 else float("nan"),
        "n_modules_perturbed": len(touched),
        "seed": int(seed),
        "target_modules": list(target_modules),
        "noise_family": "isotropic Gaussian, rescaled to an exact relative "
                        "Frobenius dose per module",
        "per_module": touched,
        "caveat": ("A reference perturbation, NOT a model of an RLVR update. "
                   "It calibrates what weight change size the detector responds "
                   "to; it says nothing about whether RLVR moves weights that "
                   "way."),
    }


# ---------------------------------------------------------------------------
# assembly — the ruler table
# ---------------------------------------------------------------------------

def erank_by_layer(record: dict, tensor: str = "resid", pooling: str = "last") -> dict:
    """{layer:int -> erank} out of one `build_variant_records` output."""
    out = {}
    for key, entry in record.get("spectra", {}).items():
        if entry.get("tensor") != tensor or entry.get("pooling") != pooling:
            continue
        out[int(key.rsplit("/layer", 1)[1])] = float(entry["erank"])
    return out


def relative_change(reference: dict, other: dict) -> dict:
    """Per-layer signed relative erank change, other vs reference, in percent."""
    return {l: (other[l] - reference[l]) / reference[l] * 100.0
            for l in sorted(set(reference) & set(other))
            if reference[l] > 0}


def ruler_table(arms: dict, reference_arm: str) -> dict:
    """Compare every arm's erank profile against `reference_arm`.

    `arms` maps arm label -> variant record. Returns per-arm per-layer relative
    change plus the max absolute change, which is the number E1's 0.53%/0.72%
    is placed against.
    """
    if reference_arm not in arms:
        raise KeyError(f"reference arm {reference_arm!r} not among {sorted(arms)}")
    ref = erank_by_layer(arms[reference_arm])
    if not ref:
        raise ValueError(f"reference arm {reference_arm!r} has no resid/last spectra")
    table = {}
    for label, rec in sorted(arms.items()):
        if label == reference_arm:
            continue
        rel = relative_change(ref, erank_by_layer(rec))
        if not rel:
            table[label] = {"per_layer_pct": {}, "status": "no overlapping layers"}
            continue
        peak = max(rel, key=lambda l: abs(rel[l]))
        table[label] = {
            "per_layer_pct": {str(k): v for k, v in sorted(rel.items())},
            "max_abs_change_pct": abs(rel[peak]),
            "max_abs_change_layer": peak,
            "signed_change_at_max_pct": rel[peak],
        }
    return {"reference_arm": reference_arm,
            "reference_erank_by_layer": {str(k): v for k, v in sorted(ref.items())},
            "arms": table}


def dormancy_by_layer(record: dict, tensor: str = "down_in",
                      pooling: str = "mean", tau: str = "0.025") -> dict:
    """{layer:int -> dormant fraction} at one (tensor, pooling, tau) cell."""
    out = {}
    for key, entry in record.get("dormancy", {}).items():
        t, layer = key.split("/layer")
        if t != tensor or pooling not in entry:
            continue
        curve = entry[pooling].get("dormant_frac_by_tau", {})
        if tau in curve:
            out[int(layer)] = float(curve[tau])
    return out


def platform_reproduction_delta(measured: dict, published: dict) -> dict:
    """Compare this platform's bare-model eranks with E1's published values.

    NOT a pass/fail gate for E4: E1's 1e-4 tolerance is a same-hardware,
    same-kernel statement, and a different accelerator will not reproduce it.
    What this records is how much of erank's value is kernel/precision
    dependent, which bounds how portable the metric is. A sanity ceiling is
    applied by the caller — a large delta means the wrong model or probe, not a
    precision effect.
    """
    rows = {}
    for layer_key, want in published.items():
        layer = int(layer_key.replace("layer", ""))
        got = measured.get(layer)
        if got is None:
            rows[layer_key] = {"error": "layer missing from measurement"}
            continue
        exp = float(want["erank"] if isinstance(want, dict) else want)
        rows[layer_key] = {
            "published": exp,
            "measured": got,
            "abs_delta": abs(got - exp),
            "rel_delta_pct": (got - exp) / exp * 100.0 if exp > 0 else float("nan"),
        }
    finite = [abs(r["rel_delta_pct"]) for r in rows.values()
              if "rel_delta_pct" in r and np.isfinite(r["rel_delta_pct"])]
    return {
        "per_layer": rows,
        "max_abs_rel_delta_pct": max(finite) if finite else float("nan"),
        "interpretation": ("Cross-platform erank drift. E1's 1e-4 gate is a "
                           "same-hardware statement and is NOT expected to hold "
                           "across accelerators; this number is reported, not "
                           "gated on."),
    }
