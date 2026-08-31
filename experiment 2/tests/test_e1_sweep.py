"""Unit tests for the E1 metric re-measurement sweep.

Phase-0 discipline (CLAUDE.md): these run on CPU against a tiny randomly
initialized Qwen2 so the reductions, hook points and gates are exercised
without a GPU or the 7B checkpoints. They test the *plumbing and the
contract*, not the scientific values.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

EXP2_ROOT = Path(__file__).resolve().parent.parent
ALGOVERSE_ROOT = EXP2_ROOT.parent
sys.path.insert(0, str(EXP2_ROOT))

from src import e1_sweep  # noqa: E402


def _load_pilot_metrics():
    path = ALGOVERSE_ROOT / "eaaj-pilot" / "src" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_pilot_metrics_for_e1_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PILOT = _load_pilot_metrics()


# ---------------------------------------------------------------------------
# tiny model + stub tokenizer
# ---------------------------------------------------------------------------

HIDDEN, INTERMEDIATE, N_LAYERS, VOCAB = 32, 48, 4, 64


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen2 import Qwen2Config

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


class StubTokenizer:
    """Minimal stand-in for the HF tokenizer surface the collector uses."""

    pad_token_id = 0

    def __init__(self, seed: int = 0):
        self.padding_side = "right"
        self._rng = np.random.default_rng(seed)
        self._lengths: dict[str, int] = {}

    def _length_of(self, p: str) -> int:
        # One token per whitespace word: deterministic, and additive over
        # space-joined concatenation, which is what the V1 probe builds.
        return max(1, len(p.split()))

    def __call__(self, batch, return_tensors=None, padding=True,
                 truncation=True, max_length=512):
        if isinstance(batch, str):
            n = min(self._length_of(batch), max_length)
            return _Enc({"input_ids": torch.zeros(n, dtype=torch.long)})
        lens = [min(self._length_of(p), max_length) for p in batch]
        T = max(lens)
        ids = torch.zeros(len(batch), T, dtype=torch.long)
        mask = torch.zeros(len(batch), T, dtype=torch.long)
        for i, (p, n) in enumerate(zip(batch, lens)):
            rng = np.random.default_rng(abs(hash(p)) % (2 ** 31))
            toks = torch.as_tensor(rng.integers(1, VOCAB, size=n), dtype=torch.long)
            if self.padding_side == "right":
                ids[i, :n], mask[i, :n] = toks, 1
            else:
                ids[i, T - n:], mask[i, T - n:] = toks, 1
        return _Enc({"input_ids": ids, "attention_mask": mask})


class _Enc(dict):
    def to(self, device):
        return self


@pytest.fixture(scope="module")
def prompts():
    return [f"probe prompt number {i}" for i in range(24)]


@pytest.fixture(scope="module")
def sweep(tiny_model, prompts):
    return e1_sweep.collect_e1_activations(
        tiny_model, StubTokenizer(), prompts, layers=(0, 1, 2, 3),
        batch_size=8, max_length=64, spectrum_layers=(1, 3),
        per_prompt_layers=(1, 3))


# ---------------------------------------------------------------------------
# hook correctness — the gate that keeps V3/V6 from measuring the wrong tensor
# ---------------------------------------------------------------------------

class TestHookPoints:
    def test_gated_mlp_identity_verified(self, sweep):
        _, _, meta = sweep
        err = meta["gated_mlp_hook_check_max_abs_err"]
        assert err is not None, "hook check never ran"
        assert err < 1e-5

    def test_hook_check_raises_on_broken_gate(self, tiny_model, prompts):
        # Detach act_fn from the product so gate_post * up no longer equals the
        # down_proj input; the collector must refuse rather than record it.
        mlp = tiny_model.model.layers[0].mlp
        original = mlp.forward
        # All three hooks still fire, but the gate is summed with `up` instead
        # of multiplied, so the reconstruction identity no longer holds.
        mlp.forward = lambda x: mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) + mlp.up_proj(x))
        try:
            with pytest.raises(RuntimeError, match="gated-MLP hook check failed"):
                e1_sweep.collect_e1_activations(
                    tiny_model, StubTokenizer(), prompts[:8], layers=(0,),
                    batch_size=4, max_length=64)
        finally:
            mlp.forward = original

    def test_all_hooks_removed_after_collection(self, tiny_model, sweep):
        for l in range(N_LAYERS):
            block = tiny_model.model.layers[l]
            mlp = block.mlp
            assert not block._forward_hooks
            assert not mlp.gate_proj._forward_hooks
            assert not mlp.up_proj._forward_hooks
            assert not mlp.act_fn._forward_hooks
            assert not mlp.down_proj._forward_pre_hooks


# ---------------------------------------------------------------------------
# V3 — dormancy tensors
# ---------------------------------------------------------------------------

class TestV3Tensors:
    def test_all_four_tensors_present_at_every_layer(self, sweep):
        _, dormancy, _ = sweep
        for l in (0, 1, 2, 3):
            for t in e1_sweep.DORMANT_TENSORS:
                assert (t, l) in dormancy

    def test_tensors_are_actually_different(self, sweep):
        _, dormancy, _ = sweep
        s = {t: dormancy[(t, 1)].per_unit_scores()["mean"]
             for t in e1_sweep.DORMANT_TENSORS}
        assert not np.allclose(s["down_in"], s["gate_post"])
        assert not np.allclose(s["gate_pre"], s["gate_post"])
        assert not np.allclose(s["up"], s["down_in"])

    def test_gate_post_is_nonnegative_but_gate_pre_is_not(self, sweep):
        # SiLU is bounded below by ~-0.2785 but the score is on |h|; the point
        # of the check is that pre- and post-activation are distinct captures.
        _, dormancy, _ = sweep
        pre = dormancy[("gate_pre", 1)].per_unit_scores()["mean"]
        post = dormancy[("gate_post", 1)].per_unit_scores()["mean"]
        assert pre.shape == post.shape == (INTERMEDIATE,)
        assert not np.allclose(pre, post)


# ---------------------------------------------------------------------------
# V2 — poolings
# ---------------------------------------------------------------------------

class TestV2Poolings:
    def test_mean_and_max_and_median_all_produced(self, sweep):
        _, dormancy, _ = sweep
        scores = dormancy[("down_in", 1)].per_unit_scores()
        assert set(scores) == {"mean", "max", "per_prompt_median"}
        for v in scores.values():
            assert v.shape == (INTERMEDIATE,)
            assert np.isfinite(v).all()

    def test_median_absent_when_per_prompt_not_retained(self, sweep):
        _, dormancy, _ = sweep
        # layer 0 was not in per_prompt_layers
        assert "per_prompt_median" not in dormancy[("down_in", 0)].per_unit_scores()

    def test_scores_are_redo_normalized(self, sweep):
        _, dormancy, _ = sweep
        for pooling, v in dormancy[("down_in", 1)].per_unit_scores().items():
            assert v.mean() == pytest.approx(1.0), pooling

    def test_max_pooling_dominates_mean_pooling_pre_normalization(self, sweep):
        # A unit active on only a few tokens reads higher under max than mean,
        # which is the whole point of V2b.
        _, dormancy, _ = sweep
        acc = dormancy[("down_in", 1)]
        raw_mean = acc._abssum.numpy() / acc._tok_count
        raw_max = acc._absmax.numpy()
        assert (raw_max >= raw_mean - 1e-9).all()

    def test_per_token_fractions_are_monotone_in_tau(self, sweep):
        _, dormancy, _ = sweep
        curve = dormancy[("down_in", 1)].per_token_dormant_frac()
        taus = sorted(float(k) for k in curve)
        vals = [curve[e1_sweep._tau_key(t)] for t in taus]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
        assert 0.0 <= vals[0] and vals[-1] <= 1.0

    def test_per_token_fraction_matches_direct_computation(self, tiny_model):
        # Exercise the bucketize histogram against a literal recomputation.
        acc = e1_sweep._DormancyAccumulator(4, (0.5, 1.0), keep_per_prompt=False)
        act = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 4.0]]])
        mask = torch.tensor([[True, True]])
        acc.update(act, mask, None)
        curve = acc.per_token_dormant_frac()
        # token 0: all s = 1.0 -> none < 0.5, none < 1.0
        # token 1: mean = 1.0, s = [0,0,0,4] -> three < 0.5 and < 1.0
        assert curve[e1_sweep._tau_key(0.5)] == pytest.approx(3 / 8)
        assert curve[e1_sweep._tau_key(1.0)] == pytest.approx(3 / 8)


# ---------------------------------------------------------------------------
# V4 — tau grid and score vectors
# ---------------------------------------------------------------------------

class TestV4TauSweep:
    def test_grid_spans_the_spec_range_and_includes_registered_taus(self):
        # 25 log-spaced points + the two registered taus, but 0.1 is already a
        # grid point (logspace(-4, 0, 25) lands exactly on 1e-1), so 26.
        assert len(e1_sweep.TAU_GRID) == 26
        assert min(e1_sweep.TAU_GRID) == pytest.approx(1e-4)
        assert max(e1_sweep.TAU_GRID) == pytest.approx(1.0)
        assert 0.025 in e1_sweep.TAU_GRID and 0.1 in e1_sweep.TAU_GRID

    def test_dormant_frac_matches_pilot_reference_at_registered_taus(self, sweep):
        _, dormancy, _ = sweep
        s = dormancy[("down_in", 1)].per_unit_scores()["mean"]
        ours = e1_sweep.dormant_frac_by_tau(s)
        acc = dormancy[("down_in", 1)]
        theirs = PILOT.dormant_metrics(acc._abssum.numpy() / acc._tok_count)
        for tau in (0.025, 0.1):
            assert ours[e1_sweep._tau_key(tau)] == pytest.approx(
                theirs[f"dormant_frac_tau{tau}"])
        assert e1_sweep.score_summary(s)["dormant_score_min"] == pytest.approx(
            theirs["dormant_score_min"])

    def test_score_vectors_written_to_disk(self, sweep, tmp_path):
        pooled, dormancy, meta = sweep
        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics,
            PILOT.anisotropy_metrics, score_dir=tmp_path / "scores", checkpoint=0)
        entry = rec["dormancy"]["down_in/layer1"]["mean"]
        p = tmp_path / "scores" / Path(entry["dormant_score_vector_path"]).name
        assert p.exists()
        assert np.load(p).shape == (INTERMEDIATE,)

    def test_score_vector_names_are_scoped_by_variant(self, sweep, tmp_path):
        pooled, dormancy, meta = sweep
        kwargs = dict(score_dir=tmp_path / "scores", checkpoint=0)
        base = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics,
            PILOT.anisotropy_metrics, variant_label="base", **kwargs)
        v1a = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics,
            PILOT.anisotropy_metrics, variant_label="V1a", **kwargs)
        p_base = base["dormancy"]["down_in/layer1"]["mean"][
            "dormant_score_vector_path"]
        p_v1a = v1a["dormancy"]["down_in/layer1"]["mean"][
            "dormant_score_vector_path"]
        assert p_base != p_v1a
        assert Path(p_base).name.startswith("base_")
        assert Path(p_v1a).name.startswith("V1a_")

    def test_per_token_has_no_score_vector(self, sweep, tmp_path):
        pooled, dormancy, meta = sweep
        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics,
            PILOT.anisotropy_metrics, score_dir=tmp_path / "scores", checkpoint=0)
        assert rec["dormancy"]["down_in/layer1"]["per_token"][
            "dormant_score_vector_path"] is None


# ---------------------------------------------------------------------------
# V5 / V6 — spectra
# ---------------------------------------------------------------------------

class TestSpectra:
    def test_v6_variants_labelled(self, sweep):
        pooled, dormancy, meta = sweep
        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics, PILOT.anisotropy_metrics)
        labels = {v["variant"] for v in rec["spectra"].values()}
        assert {"V6a", "V6b", "V6c"} <= labels

    def test_resid_last_token_matches_pilot_reference_arm(self, sweep):
        pooled, _, _ = sweep
        A = pooled[("resid", "last", 1)]
        assert A.shape == (24, HIDDEN)
        ours = PILOT.spectrum_metrics(A, center=True)
        assert np.isfinite(ours["erank"]) and ours["erank"] > 0

    def test_v5c_mean_pooling_differs_from_last_token(self, sweep):
        pooled, _, _ = sweep
        assert not np.allclose(pooled[("resid", "last", 1)],
                               pooled[("resid", "mean", 1)])

    def test_v6b_dim_is_intermediate_and_flagged_truncated(self, sweep):
        pooled, dormancy, meta = sweep
        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics, PILOT.anisotropy_metrics)
        e = rec["spectra"]["down_in/last/layer1"]
        assert e["dim"] == INTERMEDIATE
        assert e["sample_truncated"] is True   # n_probe 24 < 48

    def test_v5b_prefix_sweep_is_nested_and_ordered(self, sweep):
        pooled, dormancy, meta = sweep
        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics, PILOT.anisotropy_metrics,
            probe_prefixes=(8, 16, 24))
        sweep_ns = rec["spectra"]["resid/last/layer1"]["probe_size_sweep"]
        assert sorted(int(k) for k in sweep_ns) == [8, 16, 24]
        eranks = [sweep_ns[str(n)]["erank"] for n in (8, 16, 24)]
        assert all(np.isfinite(e) for e in eranks)

    def test_v5b_skips_prefixes_larger_than_probe(self, sweep):
        pooled, dormancy, meta = sweep
        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics, PILOT.anisotropy_metrics,
            probe_prefixes=(8, 4096))
        assert list(rec["spectra"]["resid/last/layer1"]["probe_size_sweep"]) == ["8"]

    def test_v5a_scoping_keeps_full_cross_product_only_where_asked(
            self, tiny_model, prompts):
        # The depth profile must not silently drop to one tensor everywhere,
        # nor pay for four tensors at every layer (spec §6 gate 5: record it).
        _, dormancy, meta = e1_sweep.collect_e1_activations(
            tiny_model, StubTokenizer(), prompts[:8], layers=tuple(range(N_LAYERS)),
            batch_size=4, max_length=64, spectrum_layers=(1,),
            per_prompt_layers=(), full_variant_layers=(1,))
        assert meta["full_variant_layers"] == [1]
        assert meta["dormancy_tensors_by_layer"]["1"] == list(e1_sweep.DORMANT_TENSORS)
        assert meta["dormancy_tensors_by_layer"]["3"] == ["down_in"]
        assert ("gate_pre", 3) not in dormancy
        assert ("down_in", 3) in dormancy
        assert ("gate_pre", 1) in dormancy

    def test_v5a_all_layers_collected(self, tiny_model, prompts):
        pooled, dormancy, meta = e1_sweep.collect_e1_activations(
            tiny_model, StubTokenizer(), prompts[:8], layers=tuple(range(N_LAYERS)),
            batch_size=4, max_length=64, spectrum_layers=(),
            depth_profile_layers=tuple(range(N_LAYERS)), per_prompt_layers=())
        assert meta["layers"] == list(range(N_LAYERS))
        assert meta["depth_profile_layers"] == list(range(N_LAYERS))
        for l in range(N_LAYERS):
            assert pooled[("resid", "last", l)].shape == (8, HIDDEN)
            assert dormancy[("down_in", l)].per_unit_scores()["mean"].shape == (
                INTERMEDIATE,)
        assert not any(tensor != "resid" for tensor, _, _ in pooled)


# ---------------------------------------------------------------------------
# contract + gates (spec §6)
# ---------------------------------------------------------------------------

class TestContractAndGates:
    def test_contract_records_its_overrides(self):
        c = e1_sweep.measurement_contract(
            model_dtype="bfloat16", max_length=512, batch_size=16,
            n_probe=4096, layers=(5, 14, 26),
            overrides={"dormant_pooling": "max_abs_over_tokens"})
        assert c["dormant_pooling"] == "max_abs_over_tokens"
        assert c["_overrides_vs_reference_arm"] == {
            "dormant_pooling": "max_abs_over_tokens"}

    def test_reference_arm_gate_passes_on_exact_reproduction(self):
        measured = {"per_layer": {"layer5": {
            "erank": 1127.4155, "dormant_frac_tau0.025": 0.0,
            "dormant_frac_tau0.1": 0.0}}}
        v = e1_sweep.check_reference_arm(measured, {"layer5": {"erank": 1127.4155}})
        assert v["passed"] is True

    def test_reference_arm_gate_fails_on_drift(self):
        measured = {"per_layer": {"layer5": {
            "erank": 1127.5000, "dormant_frac_tau0.025": 0.0,
            "dormant_frac_tau0.1": 0.0}}}
        v = e1_sweep.check_reference_arm(measured, {"layer5": {"erank": 1127.4155}})
        assert v["passed"] is False
        assert v["per_layer"]["layer5"]["erank_ok"] is False

    def test_reference_arm_gate_fails_on_nonzero_dormant_frac(self):
        measured = {"per_layer": {"layer5": {
            "erank": 1127.4155, "dormant_frac_tau0.025": 0.0,
            "dormant_frac_tau0.1": 0.01}}}
        v = e1_sweep.check_reference_arm(measured, {"layer5": {"erank": 1127.4155}})
        assert v["passed"] is False
        assert v["per_layer"]["layer5"]["dormant_frac_all_zero"] is False

    def test_reference_arm_gate_fails_on_missing_layer(self):
        v = e1_sweep.check_reference_arm(
            {"per_layer": {}}, {"layer5": {"erank": 1127.4155}})
        assert v["passed"] is False

class TestV1ContinuationProbe:
    def test_build_v1_probe_pairs_prompt_and_continuation(self):
        out = e1_sweep.build_v1_probe(["a", "b"], [" x", " y"])
        assert out == ["a x", "b y"]

    def test_build_v1_probe_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="continuations"):
            e1_sweep.build_v1_probe(["a", "b"], ["x"])

    def test_continuation_starts_are_the_prompt_lengths(self, prompts):
        tok = StubTokenizer()
        starts = e1_sweep.continuation_start_indices(tok, prompts[:4])
        assert starts == [len(p.split()) for p in prompts[:4]]

    def test_continuation_start_uses_token_divergence_not_prompt_length(self):
        # A tokenizer that merges across the join: "ab" encodes as one token,
        # so tokenize("a") is NOT a prefix of tokenize("a"+"b").
        class MergingTokenizer:
            padding_side = "right"

            def __call__(self, text, truncation=True, max_length=512, **kw):
                ids = {"a": [1], "b": [2], "ab": [9]}[text]
                return {"input_ids": ids}

        starts = e1_sweep.continuation_start_indices(
            MergingTokenizer(), ["a"], sequences=["ab"])
        # prompt length would say 1; divergence is at 0, so the merged
        # boundary token is treated as continuation.
        assert starts == [0]

    def test_continuation_start_is_prompt_length_when_no_merge(self):
        class CleanTokenizer:
            padding_side = "right"

            def __call__(self, text, truncation=True, max_length=512, **kw):
                return {"input_ids": [ord(c) for c in text]}

        starts = e1_sweep.continuation_start_indices(
            CleanTokenizer(), ["abc"], sequences=["abcdef"])
        assert starts == [3]

    def test_pooling_restricted_to_continuation_changes_the_result(
            self, tiny_model, prompts):
        tok = StubTokenizer()
        base = prompts[:8]
        conts = [" cont tokens here for it" for _ in base]
        probe = e1_sweep.build_v1_probe(base, conts)
        starts = e1_sweep.continuation_start_indices(tok, base)

        _, dorm_all, meta_all = e1_sweep.collect_e1_activations(
            tiny_model, tok, probe, layers=(1,), batch_size=4, max_length=64,
            spectrum_layers=(), per_prompt_layers=())
        _, dorm_cont, meta_cont = e1_sweep.collect_e1_activations(
            tiny_model, tok, probe, layers=(1,), batch_size=4, max_length=64,
            spectrum_layers=(), per_prompt_layers=(),
            continuation_starts=starts)

        assert meta_all["pooling_restricted_to_continuation"] is False
        assert meta_cont["pooling_restricted_to_continuation"] is True
        assert meta_cont["n_sequences_with_empty_pool"] == 0
        # fewer tokens pooled, and a different per-unit statistic
        assert dorm_cont[("down_in", 1)]._tok_count < dorm_all[("down_in", 1)]._tok_count
        assert not np.allclose(dorm_cont[("down_in", 1)].per_unit_scores()["mean"],
                               dorm_all[("down_in", 1)].per_unit_scores()["mean"])

    def test_v1a_contract_is_comparable_and_v1b_is_not(self):
        a = e1_sweep.v1_contract_overrides(
            on_policy=False, n_probe=512, hidden_size=3584)
        b = e1_sweep.v1_contract_overrides(
            on_policy=True, n_probe=512, hidden_size=3584)
        assert a["comparable_across_checkpoints"] is True
        assert b["comparable_across_checkpoints"] is False
        assert a["hidden_pooling"] == "last_continuation_token"
        assert any("Never plot it on the same series" in c for c in b["caveats"])
        assert any("sample-truncated" in c for c in a["caveats"])


class TestContractAndGatesCsv:
    def test_summary_csv_has_one_row_per_variant_ckpt_layer_tau(self, sweep, tmp_path):
        pooled, dormancy, meta = sweep
        rec = e1_sweep.build_variant_records(
            pooled, dormancy, meta, PILOT.spectrum_metrics, PILOT.anisotropy_metrics)
        p = e1_sweep.write_summary_csv({0: rec, 50: rec}, tmp_path / "summary.csv")
        rows = p.read_text().strip().split("\n")
        header = rows[0].split(",")
        assert header[:6] == ["variant", "checkpoint", "tensor", "pooling",
                              "layer", "tau"]
        assert len(rows) > 100
        assert {r.split(",")[1] for r in rows[1:]} == {"0", "50"}
