"""Plasticity (Q) metrics for the eaaj RLVR pilot.

Implements proposal §7 / briefing §5:
  - effective rank (primary Q) + normalized erank, participation ratio,
    top-k variance share, centered & uncentered anisotropy
  - dormant-neuron fraction (secondary Q) at tau in {0.025, 0.1}
  - weight-norm per layer group (cheap auxiliary)

Comparability contract (proposal §8 "probe set fixed"): every metric is
computed on the frozen probe set, model in eval mode, fixed dtype, fixed
layers, so values are comparable across checkpoints. Leakage rule
(proposal §6): each checkpoint is measured standalone — nothing here
normalizes across a whole run.

Pure-numpy functions (`spectrum_metrics`, `anisotropy_metrics`,
`dormant_metrics`) are unit-tested on synthetic inputs; torch is only
required for the activation collectors.
"""
from __future__ import annotations

import numpy as np

# Default probe layers for Qwen2.5-0.5B (24 decoder blocks, 0-indexed):
# early / middle / late per briefing §5. "Layer L" means the hidden state
# OUTPUT of decoder block L, i.e. outputs.hidden_states[L + 1].
DEFAULT_LAYERS = (4, 12, 22)
DORMANT_TAUS = (0.025, 0.1)
TOPK_SHARES = (1, 8, 32)


# ---------------------------------------------------------------------------
# spectrum metrics (effective rank & friends) — pure numpy
# ---------------------------------------------------------------------------

def spectrum_metrics(A: np.ndarray, center: bool = True,
                     topk: tuple[int, ...] = TOPK_SHARES) -> dict:
    """Spectral capacity metrics of an activation matrix A (n_samples, d).

    Returns effective rank exp(-sum p_i log p_i) with p_i = sigma_i / sum(sigma),
    its /d normalization, participation ratio (sum lam)^2 / sum(lam^2) on the
    covariance eigenvalues lam = sigma^2, and top-k variance shares.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"expected 2-D activation matrix, got shape {A.shape}")
    n, d = A.shape
    if center:
        A = A - A.mean(axis=0, keepdims=True)
    sigma = np.linalg.svd(A, compute_uv=False)
    total = sigma.sum()
    if total <= 0:  # all-zero matrix: no active dimensions
        out = {"erank": 0.0, "erank_norm": 0.0, "participation_ratio": 0.0}
        for k in topk:
            out[f"top{k}_var_share"] = float("nan")
        return out

    p = sigma / total
    p_nz = p[p > 0]
    erank = float(np.exp(-(p_nz * np.log(p_nz)).sum()))

    lam = sigma ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())

    out = {
        "erank": erank,
        "erank_norm": erank / d,
        "participation_ratio": pr,
    }
    var_total = lam.sum()
    for k in topk:
        out[f"top{k}_var_share"] = float(lam[:min(k, len(lam))].sum() / var_total)
    return out


def anisotropy_metrics(A: np.ndarray) -> dict:
    """Mean pairwise cosine similarity of rows, centered AND uncentered.

    Reported as a pair (briefing §5) to rule out mean-shift artifacts: a high
    uncentered anisotropy with low centered anisotropy indicates a common-mean
    offset rather than genuine directional collapse.
    """
    A = np.asarray(A, dtype=np.float64)

    def _mean_pairwise_cos(X: np.ndarray) -> float:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        valid = norms.squeeze(-1) > 0
        R = X[valid] / norms[valid]
        n = R.shape[0]
        if n < 2:
            return float("nan")
        G = R @ R.T
        # mean of off-diagonal entries
        return float((G.sum() - np.trace(G)) / (n * (n - 1)))

    return {
        "anisotropy_uncentered": _mean_pairwise_cos(A),
        "anisotropy_centered": _mean_pairwise_cos(A - A.mean(axis=0, keepdims=True)),
    }


# ---------------------------------------------------------------------------
# dormant neurons — pure numpy
# ---------------------------------------------------------------------------

def dormant_metrics(abs_mean: np.ndarray,
                    taus: tuple[float, ...] = DORMANT_TAUS) -> dict:
    """Dormant-neuron fractions from per-unit mean absolute activations.

    abs_mean[i] = E_x |h_i(x)| over the probe set (MLP post-activation units).
    Score s_i = abs_mean_i / mean_j(abs_mean_j); unit i is dormant iff
    s_i < tau (Sokar et al., ReDo 2023). Reported at every tau in `taus`.
    """
    abs_mean = np.asarray(abs_mean, dtype=np.float64)
    if abs_mean.ndim != 1:
        raise ValueError(f"expected 1-D per-unit vector, got shape {abs_mean.shape}")
    denom = abs_mean.mean()
    out = {}
    if denom <= 0:  # entire layer silent -> everything dormant
        for tau in taus:
            out[f"dormant_frac_tau{tau}"] = 1.0
        out["dormant_score_min"] = 0.0
        out["dormant_score_median"] = 0.0
        return out
    s = abs_mean / denom
    for tau in taus:
        out[f"dormant_frac_tau{tau}"] = float((s < tau).mean())
    out["dormant_score_min"] = float(s.min())
    out["dormant_score_median"] = float(np.median(s))
    return out


# ---------------------------------------------------------------------------
# torch collectors (lazy import so numpy-only environments can run the above)
# ---------------------------------------------------------------------------

def collect_probe_activations(model, tokenizer, prompts,
                              layers: tuple[int, ...] = DEFAULT_LAYERS,
                              batch_size: int = 8,
                              max_length: int = 512,
                              device=None):
    """Prompt-only forward passes over the frozen probe set, in eval mode.

    Returns
      hidden:      {layer: (n_prompts, d) float32 array} last-token hidden
                   states (output of decoder block `layer`)
      mlp_absmean: {layer: (intermediate_size,) float32 array} mean |h| of the
                   MLP post-activation units, averaged over all non-pad token
                   positions of all prompts.

    "MLP post-activation" for Qwen2's gated MLP = the input to down_proj,
    i.e. act_fn(gate_proj(x)) * up_proj(x) — the units ReDo-style dormancy is
    scored on. Captured with a forward pre-hook on down_proj.
    """
    import torch

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    decoder_layers = model.model.layers
    n_layers = len(decoder_layers)
    for l in layers:
        if not (0 <= l < n_layers):
            raise ValueError(f"layer {l} out of range for {n_layers}-layer model")

    hidden_chunks = {l: [] for l in layers}
    abssum = {}   # layer -> running sum of |h| over valid tokens
    tok_count = 0
    captured_hidden = {}
    captured_mlp = {}

    def make_hidden_hook(l):
        def hook(module, args, output):
            # Decoder blocks return either a tensor or a tuple whose first
            # item is the residual-stream hidden state.
            captured_hidden[l] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    def make_mlp_hook(l):
        def hook(module, args):
            captured_mlp[l] = args[0].detach()
        return hook

    # Hook only the three requested residual blocks rather than asking the
    # model to retain hidden states from all 24 blocks. This materially lowers
    # Phase-2 peak memory without changing the measured activations.
    handles = []
    for l in layers:
        handles.append(decoder_layers[l].register_forward_hook(make_hidden_hook(l)))
        handles.append(decoder_layers[l].mlp.down_proj.register_forward_pre_hook(make_mlp_hook(l)))

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length).to(device)
                model(**enc)
                mask = enc["attention_mask"]  # (B, T)

                # last-token hidden state per prompt at each probe layer
                positions = torch.arange(mask.shape[1], device=device).unsqueeze(0)
                last_idx = (positions * mask).max(dim=1).values
                rows = torch.arange(mask.shape[0], device=device)
                for l in layers:
                    h = captured_hidden.pop(l)  # (B, T, d)
                    hidden_chunks[l].append(
                        h[rows, last_idx].float().cpu().numpy())

                # dormant-score accumulators over all non-pad tokens
                m = mask.bool()
                tok_count += int(m.sum())
                for l in layers:
                    post = captured_mlp.pop(l)         # (B, T, H_int)
                    a = post.abs()[m].sum(dim=0)       # (H_int,)
                    abssum[l] = abssum.get(l, 0) + a.float().cpu()
    finally:
        tokenizer.padding_side = old_padding_side
        for h in handles:
            h.remove()

    hidden = {l: np.concatenate(hidden_chunks[l], axis=0) for l in layers}
    mlp_absmean = {l: (abssum[l] / tok_count).numpy() for l in layers}
    return hidden, mlp_absmean


def weight_norm_by_group(model) -> dict:
    """L2 norm of parameters per coarse layer group (cheap auxiliary Q).

    Groups: embeddings, lm_head, and per-block attn/mlp. Weight-norm *growth*
    across checkpoints is the signal (Nikishin et al. 2022); comparing these
    dicts between checkpoints gives it.
    """
    import torch

    groups: dict[str, float] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "embed_tokens" in name:
                key = "embed"
            elif "lm_head" in name:
                key = "lm_head"
            elif ".layers." in name:
                block = name.split(".layers.")[1].split(".")[0]
                part = "attn" if "attn" in name else ("mlp" if "mlp" in name else "norm")
                key = f"layer{block}.{part}"
            else:
                key = "other"
            groups[key] = groups.get(key, 0.0) + float((p.detach().float() ** 2).sum())
    return {k: float(np.sqrt(v)) for k, v in groups.items()}


def checkpoint_q_metrics(model, tokenizer, prompts,
                         layers: tuple[int, ...] = DEFAULT_LAYERS,
                         batch_size: int = 8,
                         max_length: int = 512) -> dict:
    """All per-checkpoint Q metrics in one call -> JSON-serializable dict."""
    hidden, mlp_absmean = collect_probe_activations(
        model, tokenizer, prompts, layers=layers,
        batch_size=batch_size, max_length=max_length)

    per_layer = {}
    for l in layers:
        entry = {}
        entry.update(spectrum_metrics(hidden[l], center=True))
        entry.update(anisotropy_metrics(hidden[l]))
        entry.update(dormant_metrics(mlp_absmean[l]))
        per_layer[f"layer{l}"] = entry

    return {
        "n_probe": len(prompts),
        "layers": list(layers),
        "measurement_contract": {
            "model_eval": True,
            "model_dtype": str(next(model.parameters()).dtype),
            "hidden_pooling": "last_non_padding_token",
            "dormant_pooling": "mean_abs_over_all_non_padding_tokens",
            "max_prompt_tokens": max_length,
            "activation_accumulator_dtype": "float32",
            "svd_dtype": "float64",
        },
        "per_layer": per_layer,
        "weight_norms": weight_norm_by_group(model),
    }
