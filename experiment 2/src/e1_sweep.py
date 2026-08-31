"""E1 — metric re-measurement sweep over alternative operationalizations of Q.

Implements `SPEC_E1_METRIC_REMEASUREMENT.md`. The reference arm stays in
`eaaj-pilot/src/metrics.py` and is NOT touched; everything here is additive.

The six variants are all reductions over the same forward passes:

  V2  dormancy pooling   mean / per-token / max / per-prompt-median
  V3  dormancy tensor    down_proj input (ref) / gate_pre / gate_post / up
  V4  tau sweep          27-point grid + full per-unit score vectors
  V5  sensitivity        all layers (a), probe-size prefixes (b), pooling (c)
  V6  erank tensor       residual (ref) / mlp post-activation / gate_post
  V1  probe distribution prompt+continuation, in `e1_generate_probe`

Spec §6 gate 5 ("no silent variant substitution"): every reduction records what
it changed into `measurement_contract`, and a variant that cannot be computed is
recorded as not-run with a reason rather than replaced by a lookalike.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Spec §3 V4: logarithmic 1e-4 .. 1.0, 25 points, plus the two registered taus.
TAU_GRID = tuple(sorted(set(
    [float(t) for t in np.logspace(-4, 0, 25)] + [0.025, 0.1]
)))

# Dormancy tensors (spec §3 V3). "down_in" is the reference arm's tensor.
DORMANT_TENSORS = ("down_in", "gate_pre", "gate_post", "up")

# Tensors carrying a hidden-state spectrum (spec §3 V6).
SPECTRUM_TENSORS = ("resid", "down_in", "gate_post")

# Spec §3 V5b: nested prefixes of the frozen probe, in stored order.
PROBE_PREFIXES = (512, 1024, 2048, 4096)

POOLINGS = ("mean", "per_token", "max", "per_prompt_median")


# ---------------------------------------------------------------------------
# accumulators
# ---------------------------------------------------------------------------

class _DormancyAccumulator:
    """Streaming reductions of |h| for one (layer, tensor) over the probe.

    Holds every V2 pooling at once so a single forward pass serves all of them:
      mean              running sum over non-pad tokens / token count
      max               running elementwise max over non-pad tokens
      per_token         histogram of per-(unit,token) scores over TAU_GRID
      per_prompt_median needs the per-prompt vectors, so those are retained
                        (float32, n_prompts x H) only when `keep_per_prompt`
    """

    def __init__(self, hidden_size: int, taus, keep_per_prompt: bool):
        import torch

        self.hidden_size = hidden_size
        self.taus = np.asarray(taus, dtype=np.float64)
        self.keep_per_prompt = keep_per_prompt
        self._abssum = torch.zeros(hidden_size, dtype=torch.float64)
        self._absmax = torch.zeros(hidden_size, dtype=torch.float64)
        self._tok_count = 0
        self._tok_hist = torch.zeros(len(self.taus) + 1, dtype=torch.float64)
        self._tok_scored = 0
        self._per_prompt: list = []

    def update(self, act: "object", token_mask: "object", prompt_index: "object"):
        """act: (B, T, H) activations. token_mask: (B, T) bool, True = real token."""
        import torch

        x = act.abs().float()
        flat = x[token_mask]                       # (N_tok, H)
        if flat.numel() == 0:
            return
        self._abssum += flat.sum(dim=0).double().cpu()
        self._absmax = torch.maximum(self._absmax, flat.max(dim=0).values.double().cpu())
        self._tok_count += int(flat.shape[0])

        # V2a — score each (unit, token) against that token's own layer mean.
        denom = flat.mean(dim=1, keepdim=True)
        live = denom.squeeze(-1) > 0
        if bool(live.any()):
            s_tok = flat[live] / denom[live]
            edges = torch.as_tensor(self.taus, dtype=s_tok.dtype, device=s_tok.device)
            # right=True gives boundaries[i-1] <= v < boundaries[i], so the
            # cumulative counts below are strictly `s < tau` — a score exactly
            # equal to tau must not count as dormant.
            idx = torch.bucketize(s_tok.reshape(-1), edges, right=True)
            self._tok_hist += torch.bincount(
                idx, minlength=len(self.taus) + 1).double().cpu()
            self._tok_scored += int(s_tok.shape[0]) * self.hidden_size

        # V2c — one vector per prompt (mean |h| over that prompt's real tokens).
        if self.keep_per_prompt:
            counts = token_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            per_prompt = (x * token_mask.unsqueeze(-1)).sum(dim=1) / counts
            self._per_prompt.append(per_prompt.cpu().numpy().astype(np.float32))
        del prompt_index

    # -- finalizers ---------------------------------------------------------

    def per_unit_scores(self) -> dict:
        """Per-unit dormancy score vectors s_i, one per V2 pooling.

        Normalization is ReDo's: s_i = stat_i / mean_j stat_j, applied to
        whichever per-unit statistic the pooling produced.
        """
        out = {}
        if self._tok_count:
            out["mean"] = _normalize(self._abssum.numpy() / self._tok_count)
            out["max"] = _normalize(self._absmax.numpy())
        if self.keep_per_prompt and self._per_prompt:
            stacked = np.concatenate(self._per_prompt, axis=0)
            out["per_prompt_median"] = _normalize(np.median(stacked, axis=0))
        return out

    def per_token_dormant_frac(self) -> dict:
        """V2a: fraction of (unit, token) pairs scoring below each tau."""
        if not self._tok_scored:
            return {}
        hist = self._tok_hist.numpy()
        below = np.cumsum(hist)[:len(self.taus)]
        return {_tau_key(t): float(b / self._tok_scored)
                for t, b in zip(self.taus, below)}


def _verify_gated_mlp(hook_check: dict, down_in, torch) -> None:
    """Assert act_fn(gate) * up reproduces the down_proj input.

    If a transformers version reorders Qwen2MLP, shares one activation module
    across blocks, or fuses the gate multiply, the V3/V6c hooks would quietly
    capture something other than the tensor the spec names. Spec §6 gate 5 says
    a variant that cannot be measured as specified is recorded as not-run, never
    silently substituted — so this raises instead of degrading.
    """
    cache = hook_check["cache"]
    if "gate_post" not in cache or "up" not in cache:
        raise RuntimeError(
            "gated-MLP hook check could not run: the act_fn/up_proj hooks did "
            "not fire before down_proj, so this architecture does not have the "
            "gate/up/down structure the V3 and V6c variants are defined on.")
    lhs = cache["gate_post"] * cache["up"]
    err = float((lhs - down_in).abs().max())
    hook_check["verified"] = True
    hook_check["max_abs_err"] = err
    tol = 1e-2 if lhs.dtype in (torch.bfloat16, torch.float16) else 1e-5
    if not (err <= tol):
        raise RuntimeError(
            "gated-MLP hook check failed: act_fn(gate)*up did not reproduce the "
            f"down_proj input (max abs err {err:.3e} > tol {tol:.0e}). The V3/V6 "
            "tensors would not be what the spec names — refusing to record a "
            "substituted measurement (spec §6 gate 5).")


def _normalize(stat: np.ndarray) -> np.ndarray:
    stat = np.asarray(stat, dtype=np.float64)
    denom = stat.mean()
    if denom <= 0:
        return np.zeros_like(stat)
    return stat / denom


def _tau_key(tau: float) -> str:
    return f"{tau:.6g}"


def dormant_frac_by_tau(scores: np.ndarray, taus=TAU_GRID) -> dict:
    return {_tau_key(t): float((scores < t).mean()) for t in taus}


def score_summary(scores: np.ndarray) -> dict:
    """Spec §3 V4 deliverable 3 — distance between the distribution and tau."""
    return {
        "dormant_score_min": float(np.min(scores)),
        "dormant_score_p1": float(np.percentile(scores, 1)),
        "dormant_score_p5": float(np.percentile(scores, 5)),
        "dormant_score_median": float(np.median(scores)),
    }


# ---------------------------------------------------------------------------
# forward-pass collection
# ---------------------------------------------------------------------------

def collect_e1_activations(model, tokenizer, prompts, layers,
                           batch_size: int = 16,
                           max_length: int = 512,
                           device=None,
                           taus=TAU_GRID,
                           spectrum_layers=None,
                           depth_profile_layers=None,
                           per_prompt_layers=None,
                           full_variant_layers=None,
                           continuation_starts=None,
                           verify_gated_mlp: bool = True):
    """One instrumented sweep: every V2/V3/V5a/V5c/V6 reduction at once.

    `layers` get the cheap streaming dormancy accumulators (V2/V3) — pass all
    28 for V5a. `spectrum_layers` (default: `layers`) additionally retain the
    pooled activation matrices the V6/V5b/V5c spectra need, which cost
    n_prompts x H floats per tensor and so are normally the three reference
    layers only. `per_prompt_layers` (default: `spectrum_layers`) additionally
    retain per-prompt vectors for the V2c median.

    `depth_profile_layers` retain only residual/last matrices.  This is V5a's
    all-block effective-rank profile: it deliberately avoids silently limiting
    the spectrum to the three reference layers, without paying for V5b/V5c/V6
    matrices at all 28 blocks.

    `continuation_starts` (V1) restricts every pooling to token positions at or
    after the given per-prompt index, so the prompt half of a prompt+completion
    sequence contributes nothing. Omit it for the prompt-only reference probe.

    Returns (pooled, dormancy, meta).
    """
    import torch

    layers = tuple(layers)
    spectrum_layers = tuple(layers if spectrum_layers is None else spectrum_layers)
    depth_profile_layers = tuple(() if depth_profile_layers is None
                                 else depth_profile_layers)
    per_prompt_layers = tuple(
        spectrum_layers if per_prompt_layers is None else per_prompt_layers)
    full_variant_layers = tuple(
        layers if full_variant_layers is None else full_variant_layers)

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    decoder_layers = model.model.layers
    n_layers = len(decoder_layers)
    for l in layers:
        if not (0 <= l < n_layers):
            raise ValueError(f"layer {l} out of range for {n_layers}-layer model")
    hidden_size = model.config.hidden_size
    intermediate = model.config.intermediate_size

    # Which layers get the full V2 x V3 cross-product. Layers outside this set
    # (V5a's depth profile) get only the reference dormancy tensor, because
    # retaining four MLP tensors at 28 layers is tens of GB of activations.
    full_layers = tuple(l for l in layers if l in set(full_variant_layers))
    tensors_for = {l: (DORMANT_TENSORS if l in full_layers else ("down_in",))
                   for l in layers}

    dormancy = {
        (t, l): _DormancyAccumulator(
            intermediate, taus, keep_per_prompt=(l in per_prompt_layers))
        for l in layers for t in tensors_for[l]
    }
    pooled_chunks: dict = {}
    for l in spectrum_layers:
        pooled_chunks[("resid", "last", l)] = []
        pooled_chunks[("resid", "mean", l)] = []   # V5c
        pooled_chunks[("down_in", "last", l)] = []  # V6b
        pooled_chunks[("gate_post", "last", l)] = []  # V6c
    for l in depth_profile_layers:
        pooled_chunks.setdefault(("resid", "last", l), [])

    # Per-batch state, set before each forward so the hooks can reduce inline
    # and free immediately instead of holding every layer's activations until
    # the pass ends (28 layers x 4 tensors would not fit on an A100).
    state: dict = {"m": None, "last_idx": None, "rows": None, "counts": None}
    hook_check = {"verified": False, "max_abs_err": None, "cache": {}}

    def _reduce(tensor_name, l, act):
        if (tensor_name, l) in dormancy:
            dormancy[(tensor_name, l)].update(act, state["m"], state["rows"])
        key = (tensor_name, "last", l)
        if key in pooled_chunks:
            v = act[state["rows"], state["last_idx"]]
            pooled_chunks[key].append(v.float().cpu().numpy())
        key_mean = (tensor_name, "mean", l)
        if key_mean in pooled_chunks:
            v = ((act.float() * state["m"].unsqueeze(-1)).sum(dim=1)
                 / state["counts"])
            pooled_chunks[key_mean].append(v.float().cpu().numpy())

    def _make_hook(tensor_name, l, pre=False):
        def hook(module, args, output=None):
            act = args[0] if pre else output
            act = (act[0] if isinstance(act, tuple) else act).detach()
            if (verify_gated_mlp and not hook_check["verified"]
                    and l == layers[0] and tensor_name in ("gate_post", "up")):
                hook_check["cache"][tensor_name] = act
            if verify_gated_mlp and not hook_check["verified"] \
                    and l == layers[0] and tensor_name == "down_in":
                _verify_gated_mlp(hook_check, act, torch)
            _reduce(tensor_name, l, act)
        return hook

    # Register only the hooks something downstream actually consumes, so a
    # 28-layer V5a pass does not pay for tensors nobody reduces.
    handles = []
    for l in layers:
        mlp = decoder_layers[l].mlp
        verify_here = verify_gated_mlp and l == layers[0]
        wanted = set()
        if l in spectrum_layers or l in depth_profile_layers:
            wanted |= {"resid", "down_in", "gate_post"}
        if l in depth_profile_layers and l not in spectrum_layers:
            wanted -= {"down_in", "gate_post"}
        wanted |= set(tensors_for[l])
        if verify_here:
            wanted |= {"gate_post", "up", "down_in"}
        if "resid" in wanted:
            handles.append(decoder_layers[l].register_forward_hook(
                _make_hook("resid", l)))
        if "gate_pre" in wanted:
            handles.append(mlp.gate_proj.register_forward_hook(
                _make_hook("gate_pre", l)))
        if "gate_post" in wanted:
            handles.append(mlp.act_fn.register_forward_hook(
                _make_hook("gate_post", l)))
        if "up" in wanted:
            handles.append(mlp.up_proj.register_forward_hook(_make_hook("up", l)))
        if "down_in" in wanted:
            handles.append(mlp.down_proj.register_forward_pre_hook(
                _make_hook("down_in", l, pre=True)))

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    n_seen = 0
    n_empty_pool = 0
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(device)
                mask = enc["attention_mask"]

                positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
                m = mask.bool()
                if continuation_starts is not None:
                    starts = torch.as_tensor(
                        continuation_starts[i:i + batch_size],
                        device=mask.device).unsqueeze(1)
                    m = m & (positions >= starts)
                    n_empty_pool += int((~m.any(dim=1)).sum())

                state["m"] = m
                state["rows"] = torch.arange(mask.shape[0], device=mask.device)
                state["last_idx"] = (positions * mask).max(dim=1).values
                state["counts"] = m.sum(dim=1, keepdim=True).clamp(min=1).float()

                model(**enc)   # hooks reduce inline; nothing is retained
                hook_check["cache"].clear()
                n_seen += len(batch)
    finally:
        tokenizer.padding_side = old_padding_side
        for h in handles:
            h.remove()

    pooled = {k: np.concatenate(v, axis=0) for k, v in pooled_chunks.items()}
    meta = {
        "n_probe": n_seen,
        "layers": list(layers),
        "spectrum_layers": list(spectrum_layers),
        "depth_profile_layers": list(depth_profile_layers),
        "per_prompt_layers": list(per_prompt_layers),
        "full_variant_layers": list(full_layers),
        "dormancy_tensors_by_layer": {str(l): list(tensors_for[l]) for l in layers},
        "hidden_size": hidden_size,
        "intermediate_size": intermediate,
        "gated_mlp_hook_check_max_abs_err": hook_check["max_abs_err"],
        "pooling_restricted_to_continuation": continuation_starts is not None,
        "n_sequences_with_empty_pool": n_empty_pool,
    }
    return pooled, dormancy, meta


def continuation_start_indices(tokenizer, prompts, sequences=None,
                               max_length: int = 512) -> list[int]:
    """First token position of each continuation in the concatenated sequence.

    Taken as the length of the common token prefix between the prompt alone and
    the full prompt+continuation, NOT as the prompt's own token count: BPE can
    merge across the boundary, so the two disagree by a token whenever the join
    forms a new merge. Using the divergence point keeps a boundary token that
    is partly prompt on the continuation side, which is the conservative
    direction — it never lets a purely-prompt token into a pooling that the
    contract says covers continuation tokens only.
    """
    old = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        starts = []
        for i, p in enumerate(prompts):
            p_ids = tokenizer(p, truncation=True, max_length=max_length)["input_ids"]
            if sequences is None:
                starts.append(len(p_ids))
                continue
            f_ids = tokenizer(sequences[i], truncation=True,
                              max_length=max_length)["input_ids"]
            k = 0
            while k < len(p_ids) and k < len(f_ids) and p_ids[k] == f_ids[k]:
                k += 1
            starts.append(k)
        return starts
    finally:
        tokenizer.padding_side = old


# ---------------------------------------------------------------------------
# variant assembly
# ---------------------------------------------------------------------------

def build_variant_records(pooled, dormancy, meta, spectrum_metrics, anisotropy_metrics,
                          taus=TAU_GRID, probe_prefixes=PROBE_PREFIXES,
                          score_dir: Path | None = None,
                          checkpoint=None, variant_label: str | None = None) -> dict:
    """Turn one sweep's accumulators into the per-variant records of spec §5.

    `spectrum_metrics` / `anisotropy_metrics` are the unchanged pure-numpy
    functions from `eaaj-pilot/src/metrics.py`, injected so this module stays
    importable without the pilot tree.
    """
    n_probe = meta["n_probe"]
    hidden_size = meta["hidden_size"]
    records: dict = {}

    # -- V2 x V3: dormancy pooling x dormancy tensor ------------------------
    dorm: dict = {}
    for (tensor, layer), acc in dormancy.items():
        scores = acc.per_unit_scores()
        entry = {}
        for pooling, s in scores.items():
            path = None
            if score_dir is not None:
                score_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"{variant_label}_" if variant_label else ""
                name = (f"{prefix}{tensor}_{pooling}_ckpt{checkpoint}"
                        f"_layer{layer}.npy")
                np.save(score_dir / name, s.astype(np.float64))
                path = f"scores/{name}"
            entry[pooling] = {
                "dormant_frac_by_tau": dormant_frac_by_tau(s, taus),
                **score_summary(s),
                "dormant_score_vector_path": path,
            }
        per_token = acc.per_token_dormant_frac()
        if per_token:
            # V2a has no per-unit vector by construction: it scores
            # (unit, token) pairs, so only the fraction curve exists.
            entry["per_token"] = {
                "dormant_frac_by_tau": per_token,
                "dormant_score_vector_path": None,
                "note": "scored per (unit, token) pair; no per-unit vector exists",
            }
        dorm[f"{tensor}/layer{layer}"] = entry
    records["dormancy"] = dorm

    # -- V6 x V5b x V5c: spectra --------------------------------------------
    spectra: dict = {}
    for (tensor, pooling, layer), A in pooled.items():
        d = A.shape[1]
        variant = {"resid/last": "V6a", "down_in/last": "V6b",
                   "gate_post/last": "V6c"}.get(f"{tensor}/{pooling}")
        entry = {
            "variant": variant or f"{tensor}/{pooling}",
            "tensor": tensor,
            "pooling": pooling,
            "dim": int(d),
            "sample_truncated": bool(n_probe < d),
        }
        entry.update(spectrum_metrics(A, center=True))
        entry.update(anisotropy_metrics(A))
        # V5b — nested prefixes of the same matrix, no resampling.
        prefixes = {}
        for n in probe_prefixes:
            if n > A.shape[0]:
                continue
            sub = spectrum_metrics(A[:n], center=True)
            prefixes[str(n)] = {
                "erank": sub["erank"],
                "erank_norm": sub["erank_norm"],
                "participation_ratio": sub["participation_ratio"],
                "sample_truncated": bool(n < d),
            }
        entry["probe_size_sweep"] = prefixes
        spectra[f"{tensor}/{pooling}/layer{layer}"] = entry
    records["spectra"] = spectra

    records["meta"] = {
        **meta,
        "hidden_size": hidden_size,
        "taus": [float(t) for t in taus],
    }
    return records


def measurement_contract(*, model_dtype: str, max_length: int, batch_size: int,
                         n_probe: int, layers, overrides: dict | None = None) -> dict:
    """Spec §2 contract block, with this variant's overrides applied.

    Spec §6 gate 2: a variant whose JSON does not record what it changed is not
    usable evidence, so the overrides are recorded explicitly as well as merged.
    """
    base = {
        "model_eval": True,
        "model_dtype": model_dtype,
        "hidden_pooling": "last_non_padding_token",
        "dormant_pooling": "mean_abs_over_all_non_padding_tokens",
        "dormant_tensor": "down_proj input == act_fn(gate_proj(x)) * up_proj(x)",
        "dormant_score": "s_i = E|h_i| / mean_j E|h_j|",
        "max_prompt_tokens": max_length,
        "activation_accumulator": "float32",
        "svd_dtype": "float64",
        "spectrum_centering": True,
        "n_probe": n_probe,
        "layers": list(layers),
        "batch_size": batch_size,
        "taus": [float(t) for t in TAU_GRID],
    }
    contract = dict(base)
    contract.update(overrides or {})
    contract["_overrides_vs_reference_arm"] = dict(overrides or {})
    return contract


def write_summary_csv(records_by_ckpt: dict, path) -> Path:
    """One row per (variant, checkpoint, layer, tau) — spec §5."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "checkpoint", "tensor", "pooling", "layer",
                    "tau", "dormant_frac", "erank", "sample_truncated"])
        for ckpt, rec in sorted(records_by_ckpt.items(), key=lambda kv: str(kv[0])):
            for key, entry in sorted(rec.get("dormancy", {}).items()):
                tensor, layer = key.split("/layer")
                for pooling, block in sorted(entry.items()):
                    for tau, frac in sorted(block["dormant_frac_by_tau"].items(),
                                            key=lambda kv: float(kv[0])):
                        w.writerow([f"V2:{pooling}/V3:{tensor}", ckpt, tensor,
                                    pooling, layer, tau, frac, "", ""])
            for key, entry in sorted(rec.get("spectra", {}).items()):
                layer = key.rsplit("/layer", 1)[1]
                w.writerow([entry["variant"], ckpt, entry["tensor"],
                            entry["pooling"], layer, "", entry["erank"],
                            entry["sample_truncated"]])
    return path


# ---------------------------------------------------------------------------
# V1 — probe distribution: prompt-only -> prompt + continuation
# ---------------------------------------------------------------------------

V1_N_PROMPTS = 512
V1_MAX_NEW_TOKENS = 256


def generate_continuations(model, tokenizer, prompts, *, max_new_tokens=V1_MAX_NEW_TOKENS,
                           batch_size: int = 8, max_length: int = 512, device=None) -> list[str]:
    """Greedy continuations for the V1 probe.

    `do_sample=False` so no seed enters the measurement (spec §3 V1 settings):
    the probe stays a frozen artifact that a later session can reproduce.
    """
    import torch

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    old = tokenizer.padding_side
    tokenizer.padding_side = "left"   # generation requires left padding
    out: list[str] = []
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(device)
                gen = model.generate(
                    **enc, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                out.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = old
    return out


def build_v1_probe(prompts, continuations) -> list[str]:
    """Concatenate prompt + frozen continuation into the V1 probe sequences."""
    if len(prompts) != len(continuations):
        raise ValueError(
            f"{len(prompts)} prompts but {len(continuations)} continuations")
    return [p + c for p, c in zip(prompts, continuations)]


def v1_contract_overrides(*, on_policy: bool, n_probe: int, hidden_size: int) -> dict:
    """The contract deltas and the two caveats V1 must carry (spec §3 V1)."""
    return {
        "probe_distribution": ("prompt + own continuation (on-policy)"
                               if on_policy else
                               "prompt + ckpt-0 continuation (frozen)"),
        "hidden_pooling": "last_continuation_token",
        "dormant_pooling": "mean_abs_over_continuation_tokens",
        "generation": {"do_sample": False, "max_new_tokens": V1_MAX_NEW_TOKENS},
        "comparable_across_checkpoints": not on_policy,
        "caveats": [
            ("V1b is on-policy: the input distribution differs per checkpoint, "
             "so it violates the comparability contract. Never plot it on the "
             "same series as V1a or the reference arm.")
            if on_policy else
            ("V1a holds the token sequences fixed across checkpoints, so it "
             "keeps the frozen-probe contract."),
            (f"n_probe={n_probe} < hidden dim {hidden_size}: V1 erank is "
             "sample-truncated and its LEVEL is not comparable to the n=4096 "
             "reference arm. Within-V1 across-checkpoint only; see V5b."),
        ],
    }


def check_reference_arm(measured: dict, reference: dict, tol: float = 1e-4) -> dict:
    """Spec §6 gate 1 — the reference arm must reproduce §2 exactly.

    `reference` maps "layer5" -> {"erank": ..., "dormant_score_min": ...}.
    Returns a verdict dict; the caller stops the sweep when `passed` is False.
    """
    deltas = {}
    passed = True
    for layer_key, expected in reference.items():
        got = measured["per_layer"].get(layer_key)
        if got is None:
            deltas[layer_key] = {"error": "layer missing from measurement"}
            passed = False
            continue
        d = abs(float(got["erank"]) - float(expected["erank"]))
        row = {"erank_expected": float(expected["erank"]),
               "erank_measured": float(got["erank"]),
               "erank_delta": d,
               "erank_ok": bool(d < tol)}
        dormant_nonzero = [k for k, v in got.items()
                           if k.startswith("dormant_frac_tau") and float(v) != 0.0]
        row["dormant_frac_all_zero"] = not dormant_nonzero
        if dormant_nonzero:
            row["dormant_frac_nonzero_keys"] = dormant_nonzero
        passed = passed and row["erank_ok"] and row["dormant_frac_all_zero"]
        deltas[layer_key] = row
    return {"passed": bool(passed), "tol": tol, "per_layer": deltas}
