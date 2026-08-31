"""Unit tests for E4 detector calibration.

Deliberately runnable on a bare machine: everything that does not strictly
need torch is pure numpy, and the torch-dependent tests skip cleanly rather
than erroring. E1's suite had 26 tests that could only error without
`transformers` installed, which made a local green run impossible; this suite
is designed so the Mac mini can actually verify the code it is about to run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXP2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP2))

from src import e4_calibration as e4  # noqa: E402


# ---------------------------------------------------------------------------
# Arm W — weight-space dose
# ---------------------------------------------------------------------------

class TestLoraScaling:
    def test_plain_lora_scaling_is_alpha_over_r(self):
        assert e4.lora_scaling({"r": 16, "lora_alpha": 32}) == 2.0

    def test_rslora_scaling_is_alpha_over_sqrt_r(self):
        got = e4.lora_scaling({"r": 16, "lora_alpha": 32, "use_rslora": True})
        assert got == pytest.approx(32 / 4.0)

    def test_stage_a_adapter_config_gives_scaling_two(self):
        # WEIGHTS.md records r=16, alpha=32 for all three Stage-A adapters.
        assert e4.lora_scaling({"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}) == 2.0


class TestBaseKeyMapping:
    def test_peft_prefix_is_stripped(self):
        got = e4._base_key_for(
            "base_model.model.model.layers.3.mlp.up_proj.lora_A.weight")
        assert got == "model.layers.3.mlp.up_proj.weight"

    def test_lora_b_maps_to_the_same_base_weight_as_lora_a(self):
        a = e4._base_key_for("base_model.model.model.layers.7.self_attn.q_proj.lora_A.weight")
        b = e4._base_key_for("base_model.model.model.layers.7.self_attn.q_proj.lora_B.weight")
        assert a == b == "model.layers.7.self_attn.q_proj.weight"

    def test_non_lora_key_returns_none(self):
        assert e4._base_key_for("model.layers.0.mlp.up_proj.weight") is None


class TestRelativeDose:
    def test_aggregate_is_a_norm_ratio_not_a_mean_of_ratios(self):
        # One big module barely moved, one tiny module moved a lot. The mean of
        # per-module ratios would be dominated by the tiny module; the
        # aggregate must not be.
        delta = {"big": {"delta_fro": 1.0, "shape": [1000, 1000]},
                 "tiny": {"delta_fro": 1.0, "shape": [2, 2]}}
        base = {"big": {"base_fro": 1000.0, "shape": [1000, 1000]},
                "tiny": {"base_fro": 1.0, "shape": [2, 2]}}
        out = e4.relative_dose(delta, base)
        expected = np.sqrt(2.0) / np.sqrt(1000.0 ** 2 + 1.0)
        assert out["aggregate_relative_dose"] == pytest.approx(expected)
        assert out["mean_per_module_relative_dose"] == pytest.approx(0.5005)
        assert out["aggregate_relative_dose"] < out["mean_per_module_relative_dose"]

    def test_zero_delta_gives_zero_dose(self):
        delta = {"m": {"delta_fro": 0.0, "shape": [4, 4]}}
        base = {"m": {"base_fro": 3.0, "shape": [4, 4]}}
        assert e4.relative_dose(delta, base)["aggregate_relative_dose"] == 0.0

    def test_module_absent_from_base_is_reported_not_silently_dropped(self):
        delta = {"present": {"delta_fro": 1.0, "shape": [2, 2]},
                 "absent": {"delta_fro": 5.0, "shape": [2, 2]}}
        base = {"present": {"base_fro": 2.0, "shape": [2, 2]}}
        out = e4.relative_dose(delta, base)
        assert out["modules_missing_from_base"] == ["absent"]
        assert out["n_modules"] == 1
        # The missing module must not contribute to the aggregate either.
        assert out["aggregate_relative_dose"] == pytest.approx(0.5)


class TestLoraDeltaNorms:
    def test_reads_a_synthetic_adapter_and_matches_hand_computation(self, tmp_path):
        st = pytest.importorskip("safetensors.numpy")
        rng = np.random.default_rng(0)
        A = rng.normal(size=(4, 8)).astype(np.float32)   # r x in
        B = rng.normal(size=(6, 4)).astype(np.float32)   # out x r
        st.save_file({
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": A,
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": B,
        }, str(tmp_path / "adapter_model.safetensors"))
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"r": 4, "lora_alpha": 8, "peft_type": "LORA"}))

        out = e4.lora_delta_norms(tmp_path)
        key = "model.layers.0.mlp.up_proj.weight"
        want = np.linalg.norm(
            (B.astype(np.float64) @ A.astype(np.float64)) * 2.0)
        assert out[key]["delta_fro"] == pytest.approx(want)
        assert out[key]["shape"] == [6, 8]

    def test_zero_b_gives_exactly_zero_delta(self, tmp_path):
        """ckpt-0 has B=0 by LoRA initialization, so its dose must be exactly 0."""
        st = pytest.importorskip("safetensors.numpy")
        A = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
        B = np.zeros((6, 4), dtype=np.float32)
        st.save_file({
            "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": A,
            "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": B,
        }, str(tmp_path / "adapter_model.safetensors"))
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"r": 4, "lora_alpha": 8}))
        out = e4.lora_delta_norms(tmp_path)
        assert out["model.layers.0.mlp.up_proj.weight"]["delta_fro"] == 0.0


# ---------------------------------------------------------------------------
# Arm N — the perturbation ladder
# ---------------------------------------------------------------------------

def _tiny_linear_stack(dtype):
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = torch.nn.Linear(8, 16, bias=False)
            self.down_proj = torch.nn.Linear(16, 8, bias=False)
            self.untouched = torch.nn.Linear(8, 8, bias=False)

    m = Block().to(dtype)
    return m


class TestPerturbation:
    def test_achieved_dose_matches_the_request_in_float32(self):
        torch = pytest.importorskip("torch")
        m = _tiny_linear_stack(torch.float32)
        rec = e4.perturb_model_(m, 1e-2, target_modules=("up_proj", "down_proj"))
        assert rec["achieved_aggregate_dose"] == pytest.approx(1e-2, rel=1e-5)
        for row in rec["per_module"]:
            assert row["relative_dose"] == pytest.approx(1e-2, rel=1e-4)

    def test_only_target_modules_are_touched(self):
        torch = pytest.importorskip("torch")
        m = _tiny_linear_stack(torch.float32)
        before = m.untouched.weight.detach().clone()
        rec = e4.perturb_model_(m, 1e-1, target_modules=("up_proj", "down_proj"))
        assert rec["n_modules_perturbed"] == 2
        assert torch.equal(m.untouched.weight, before)

    def test_same_seed_and_dose_reproduce_the_same_weights(self):
        torch = pytest.importorskip("torch")
        a, b = _tiny_linear_stack(torch.float32), _tiny_linear_stack(torch.float32)
        b.load_state_dict(a.state_dict())
        e4.perturb_model_(a, 3e-3, seed=7, target_modules=("up_proj",))
        e4.perturb_model_(b, 3e-3, seed=7, target_modules=("up_proj",))
        assert torch.equal(a.up_proj.weight, b.up_proj.weight)

    def test_different_dose_gives_different_noise_draw(self):
        torch = pytest.importorskip("torch")
        a, b = _tiny_linear_stack(torch.float32), _tiny_linear_stack(torch.float32)
        b.load_state_dict(a.state_dict())
        e4.perturb_model_(a, 3e-3, seed=7, target_modules=("up_proj",))
        e4.perturb_model_(b, 1e-2, seed=7, target_modules=("up_proj",))
        assert not torch.equal(a.up_proj.weight, b.up_proj.weight)

    def test_achieved_dose_is_measured_after_the_dtype_cast(self):
        """A bf16 parameter cannot represent a 1e-6 relative nudge; the record
        must show the shortfall rather than echo the request back."""
        torch = pytest.importorskip("torch")
        m = _tiny_linear_stack(torch.bfloat16)
        rec = e4.perturb_model_(m, 1e-6, target_modules=("up_proj", "down_proj"))
        assert rec["requested_dose"] == 1e-6
        assert rec["achieved_aggregate_dose"] != 1e-6

    def test_zero_dose_leaves_weights_bit_identical(self):
        torch = pytest.importorskip("torch")
        m = _tiny_linear_stack(torch.float32)
        before = m.up_proj.weight.detach().clone()
        rec = e4.perturb_model_(m, 0.0, target_modules=("up_proj",))
        assert torch.equal(m.up_proj.weight, before)
        assert rec["achieved_aggregate_dose"] == pytest.approx(0.0)

    def test_record_carries_the_not_an_rlvr_update_caveat(self):
        torch = pytest.importorskip("torch")
        m = _tiny_linear_stack(torch.float32)
        rec = e4.perturb_model_(m, 1e-3, target_modules=("up_proj",))
        assert "not a model of an rlvr update" in rec["caveat"].lower()


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _record(eranks: dict, tensor="resid", pooling="last") -> dict:
    return {"spectra": {
        f"{tensor}/{pooling}/layer{l}": {
            "tensor": tensor, "pooling": pooling, "erank": v}
        for l, v in eranks.items()}}


class TestRulerTable:
    def test_relative_change_is_signed_percent(self):
        ref = {5: 100.0, 14: 200.0}
        other = {5: 101.0, 14: 198.0}
        rel = e4.relative_change(ref, other)
        assert rel[5] == pytest.approx(1.0)
        assert rel[14] == pytest.approx(-1.0)

    def test_max_abs_change_picks_the_largest_magnitude_not_the_largest_signed(self):
        arms = {"ref": _record({5: 100.0, 14: 100.0}),
                "x": _record({5: 101.0, 14: 95.0})}
        t = e4.ruler_table(arms, "ref")
        assert t["arms"]["x"]["max_abs_change_layer"] == 14
        assert t["arms"]["x"]["max_abs_change_pct"] == pytest.approx(5.0)
        assert t["arms"]["x"]["signed_change_at_max_pct"] == pytest.approx(-5.0)

    def test_reference_arm_is_excluded_from_its_own_table(self):
        arms = {"ref": _record({5: 100.0}), "x": _record({5: 110.0})}
        assert set(e4.ruler_table(arms, "ref")["arms"]) == {"x"}

    def test_missing_reference_arm_raises(self):
        with pytest.raises(KeyError):
            e4.ruler_table({"x": _record({5: 1.0})}, "ref")

    def test_reference_arm_without_spectra_raises(self):
        with pytest.raises(ValueError):
            e4.ruler_table({"ref": {"spectra": {}}, "x": _record({5: 1.0})}, "ref")

    def test_only_resid_last_enters_the_default_profile(self):
        rec = _record({5: 100.0})
        rec["spectra"]["down_in/last/layer5"] = {
            "tensor": "down_in", "pooling": "last", "erank": 999.0}
        assert e4.erank_by_layer(rec) == {5: 100.0}
        assert e4.erank_by_layer(rec, tensor="down_in") == {5: 999.0}


class TestPlatformReproduction:
    def test_reports_relative_drift_without_failing(self):
        out = e4.platform_reproduction_delta(
            {5: 1127.4155 * 1.001, 14: 1281.0450, 26: 1426.0597},
            {"layer5": 1127.4155, "layer14": 1281.0450, "layer26": 1426.0597})
        assert out["per_layer"]["layer5"]["rel_delta_pct"] == pytest.approx(0.1, abs=1e-6)
        assert out["max_abs_rel_delta_pct"] == pytest.approx(0.1, abs=1e-6)
        assert "not gated on" in out["interpretation"]

    def test_missing_layer_is_flagged(self):
        out = e4.platform_reproduction_delta({5: 1.0}, {"layer5": 1.0, "layer14": 2.0})
        assert "error" in out["per_layer"]["layer14"]


class TestDoseLadder:
    def test_ladder_is_strictly_increasing(self):
        assert list(e4.DOSE_LADDER) == sorted(e4.DOSE_LADDER)
        assert len(set(e4.DOSE_LADDER)) == len(e4.DOSE_LADDER)

    def test_target_modules_match_the_stage_a_adapter_config(self):
        # WEIGHTS.md: the Stage-A LoRA targeted exactly these seven.
        assert set(e4.TARGET_MODULES) == {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"}


class TestDormancyExtraction:
    def test_reads_the_registered_cell(self):
        rec = {"dormancy": {"down_in/layer5": {
            "mean": {"dormant_frac_by_tau": {"0.025": 0.0, "0.1": 0.5}}}}}
        assert e4.dormancy_by_layer(rec, tau="0.025") == {5: 0.0}
        assert e4.dormancy_by_layer(rec, tau="0.1") == {5: 0.5}

    def test_absent_pooling_yields_no_row(self):
        rec = {"dormancy": {"down_in/layer5": {
            "max": {"dormant_frac_by_tau": {"0.025": 0.3}}}}}
        assert e4.dormancy_by_layer(rec, pooling="mean") == {}


class TestFullParameterDose:
    """exp1.5 v3 trains every parameter, so its dose is a direct weight diff."""

    def _write(self, tmp_path, name, tensors):
        st = pytest.importorskip("safetensors.numpy")
        p = tmp_path / name
        st.save_file(tensors, str(p))
        return p

    def test_matches_hand_computed_frobenius_difference(self, tmp_path):
        pytest.importorskip("safetensors.numpy")
        rng = np.random.default_rng(3)
        W0 = rng.normal(size=(6, 8)).astype(np.float32)
        W = (W0 + 0.01 * rng.normal(size=(6, 8))).astype(np.float32)
        k = "model.layers.0.mlp.up_proj.weight"
        b = self._write(tmp_path, "base.safetensors", {k: W0})
        c = self._write(tmp_path, "ckpt.safetensors", {k: W})
        out = e4.full_delta_norms([c], [b])
        want = np.linalg.norm(W.astype(np.float64) - W0.astype(np.float64))
        assert out[k]["delta_fro"] == pytest.approx(want)

    def test_identical_weights_give_zero_dose(self, tmp_path):
        pytest.importorskip("safetensors.numpy")
        W = np.random.default_rng(4).normal(size=(4, 4)).astype(np.float32)
        k = "model.layers.0.self_attn.q_proj.weight"
        b = self._write(tmp_path, "base.safetensors", {k: W})
        c = self._write(tmp_path, "ckpt.safetensors", {k: W.copy()})
        assert e4.full_delta_norms([c], [b])[k]["delta_fro"] == 0.0

    def test_non_target_modules_are_excluded(self, tmp_path):
        pytest.importorskip("safetensors.numpy")
        rng = np.random.default_rng(5)
        keep = "model.layers.0.mlp.down_proj.weight"
        drop = "model.layers.0.input_layernorm.weight"
        t = {keep: rng.normal(size=(4, 4)).astype(np.float32),
             drop: rng.normal(size=(4,)).astype(np.float32)}
        b = self._write(tmp_path, "base.safetensors", t)
        c = self._write(tmp_path, "ckpt.safetensors", t)
        out = e4.full_delta_norms([c], [b])
        assert set(out) == {keep}

    def test_shape_mismatch_raises_instead_of_silently_skipping(self, tmp_path):
        pytest.importorskip("safetensors.numpy")
        k = "model.layers.0.mlp.up_proj.weight"
        b = self._write(tmp_path, "base.safetensors",
                        {k: np.zeros((4, 4), dtype=np.float32)})
        c = self._write(tmp_path, "ckpt.safetensors",
                        {k: np.zeros((4, 8), dtype=np.float32)})
        with pytest.raises(ValueError, match="shape"):
            e4.full_delta_norms([c], [b])

    def test_no_overlap_raises_rather_than_reporting_an_empty_dose(self, tmp_path):
        pytest.importorskip("safetensors.numpy")
        b = self._write(tmp_path, "base.safetensors",
                        {"model.layers.0.mlp.up_proj.weight":
                         np.zeros((4, 4), dtype=np.float32)})
        c = self._write(tmp_path, "ckpt.safetensors",
                        {"something.else.weight":
                         np.zeros((4, 4), dtype=np.float32)})
        with pytest.raises(ValueError, match="no target-module weights"):
            e4.full_delta_norms([c], [b])

    def test_output_feeds_relative_dose_unchanged(self, tmp_path):
        """The point of matching lora_delta_norms' shape."""
        pytest.importorskip("safetensors.numpy")
        k = "model.layers.0.mlp.up_proj.weight"
        W0 = np.full((4, 4), 1.0, dtype=np.float32)
        W = np.full((4, 4), 1.1, dtype=np.float32)
        b = self._write(tmp_path, "base.safetensors", {k: W0})
        c = self._write(tmp_path, "ckpt.safetensors", {k: W})
        delta = e4.full_delta_norms([c], [b])
        base = e4.base_weight_norms([b], wanted={k})
        out = e4.relative_dose(delta, base)
        assert out["aggregate_relative_dose"] == pytest.approx(0.1, rel=1e-5)
