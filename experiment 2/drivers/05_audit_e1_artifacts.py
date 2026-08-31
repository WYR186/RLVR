"""Strict local audit for E1 metric re-measurement artifacts.

The audit is intentionally independent of CUDA/model weights.  It validates the
reference gate, provenance, measurement contracts, score-vector integrity, and
variant/layer coverage after artifacts are copied back from Drive.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXP2 = Path(__file__).resolve().parent.parent
ROOT = EXP2.parent
DEFAULT_DIR = (
    ROOT / "eaaj-pilot" / "outputs"
    / "exp2_colab_guru_math7b_instruct_group8_e33527592dd9"
    / "measurements" / "e1_sweep"
)
STEPS = (0, 50, 100)
REF_LAYERS = (5, 14, 26)
ALL_LAYERS = tuple(range(28))
CONFIG_SHA12 = "e33527592dd9"
SPLITS_SHA16 = {
    "stage_a_train_ids": "a06fe6b80e7d40ca",
    "stage_b_train_ids": "df8623cf009e0690",
    "stage_b_eval_ids": "8ee975c7089dc72a",
    "probe_stage_a_topup_ids": "1e61252e7b54793e",
}
CONTRACT_FIELDS = {
    "model_eval", "model_dtype", "hidden_pooling", "dormant_pooling",
    "dormant_tensor", "dormant_score", "max_prompt_tokens",
    "activation_accumulator", "svd_dtype", "spectrum_centering",
    "n_probe", "layers", "batch_size", "taus",
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"missing artifact: {path}")
    return json.loads(path.read_text())


def audit_provenance(record: dict) -> None:
    p = record["provenance"]
    assert p["config_hash"] == CONFIG_SHA12
    assert p["splits_sha16"] == SPLITS_SHA16
    assert p["reference_arm_gate"] == "pass"


def audit_contract(record: dict, *, n_probe: int, layers: tuple[int, ...]) -> None:
    c = record["measurement_contract"]
    missing = CONTRACT_FIELDS - set(c)
    assert not missing, f"contract missing fields: {sorted(missing)}"
    assert c["model_eval"] is True
    assert c["activation_accumulator"] == "float32"
    assert c["svd_dtype"] == "float64"
    assert c["n_probe"] == n_probe
    assert tuple(c["layers"]) == layers
    assert len(c["taus"]) == 26
    assert 0.025 in c["taus"] and 0.1 in c["taus"]


def audit_reference(out: Path) -> None:
    expected = {
        0: {5: 1127.4155, 14: 1281.0450, 26: 1426.0597},
        50: {5: 1128.2271, 14: 1287.8799, 26: 1433.8730},
        100: {5: 1128.1812, 14: 1287.8093, 26: 1432.5480},
    }
    for step in STEPS:
        rec = load(out / f"reference_arm_ckpt{step}.json")
        gate = rec["gate"]
        assert gate["passed"] is True
        for layer, want in expected[step].items():
            layer_key = f"layer{layer}"
            got = rec["metrics"]["per_layer"][layer_key]["erank"]
            assert abs(got - want) < 1e-4, (step, layer, got, want)
            assert gate["per_layer"][layer_key]["erank_ok"] is True
            assert gate["per_layer"][layer_key]["dormant_frac_all_zero"] is True
            metrics = rec["metrics"]["per_layer"][layer_key]
            assert metrics["dormant_frac_tau0.025"] == 0.0
            assert metrics["dormant_frac_tau0.1"] == 0.0


def audit_record(out: Path, name: str, *, n_probe: int,
                 layers: tuple[int, ...], continuation: bool
                 ) -> tuple[set[Path], set[int], list[str]]:
    score_paths: set[Path] = set()
    spectrum_layers: set[int] = set()
    stale_vectors: list[str] = []
    for step in STEPS:
        rec = load(out / f"{name}_ckpt{step}.json")
        assert rec["checkpoint"] == step
        audit_provenance(rec)
        audit_contract(rec, n_probe=n_probe, layers=layers)
        meta = rec["meta"]
        assert meta["n_probe"] == n_probe
        assert meta["pooling_restricted_to_continuation"] is continuation
        assert meta["n_sequences_with_empty_pool"] == 0
        assert meta["gated_mlp_hook_check_max_abs_err"] == 0.0
        for key in rec["spectra"]:
            spectrum_layers.add(int(key.rsplit("layer", 1)[1]))
        for tensor_entry in rec["dormancy"].values():
            for pooling, values in tensor_entry.items():
                vector = values.get("dormant_score_vector_path")
                if vector is None:
                    assert pooling == "per_token"
                    continue
                path = out / vector
                assert path.is_file(), path
                arr = np.load(path)
                assert arr.shape == (18944,), (path, arr.shape)
                assert np.isfinite(arr).all(), path
                # The JSON summary and its vector must describe the same data.
                # This detects legacy cross-variant filename collisions.
                checks = {
                    "dormant_score_min": np.min(arr),
                    "dormant_score_p1": np.percentile(arr, 1),
                    "dormant_score_p5": np.percentile(arr, 5),
                    "dormant_score_median": np.median(arr),
                }
                if any(not np.isclose(values[k], got, rtol=0, atol=1e-10)
                       for k, got in checks.items()):
                    stale_vectors.append(
                        f"{name}/ckpt{step}/{tensor_entry}/{pooling}: {vector}")
                score_paths.add(path.resolve())
    return score_paths, spectrum_layers, stale_vectors


def audit_csv(path: Path) -> int:
    if not path.is_file():
        raise AssertionError(f"missing summary: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, path
    assert {int(r["checkpoint"]) for r in rows} == set(STEPS)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--require-v5a", action="store_true")
    args = ap.parse_args()
    out = args.artifact_dir.resolve()
    assert out.is_dir(), out

    audit_reference(out)
    base_vectors, base_spectrum_layers, base_stale = audit_record(
        out, "V2V3V4V5V6", n_probe=4096, layers=REF_LAYERS,
        continuation=False)
    v1_vectors, v1_spectrum_layers, v1_stale = audit_record(
        out, "V1a", n_probe=512, layers=REF_LAYERS, continuation=True)
    summary_rows = audit_csv(out / "summary.csv")
    v1_rows = audit_csv(out / "summary_v1a.csv")

    v5a_files = [out / f"V5a_ckpt{s}.json" for s in STEPS]
    v5a_present = all(p.is_file() for p in v5a_files)
    v5a_spectrum_layers: set[int] = set()
    if v5a_present:
        v5a_vectors, v5a_spectrum_layers, v5a_stale = audit_record(
            out, "V5a", n_probe=4096, layers=ALL_LAYERS,
            continuation=False)
        assert v5a_spectrum_layers == set(ALL_LAYERS), (
            "V5a residual spectrum does not cover all 28 layers",
            sorted(v5a_spectrum_layers),
        )
        assert not v5a_stale, f"V5a has stale score vectors: {v5a_stale[:3]}"
        assert v5a_vectors.isdisjoint(v1_vectors), (
            "V5a score paths collide with V1a score paths")
    elif args.require_v5a:
        raise AssertionError("V5a artifacts are required but absent")

    print("E1 ARTIFACT AUDIT PASS")
    print(f"  artifact_dir: {out}")
    print("  reference gate: 9/9 eranks within 1e-4; dormant_frac 0.0")
    print(f"  base spectrum layers: {sorted(base_spectrum_layers)}")
    print(f"  V1a spectrum layers: {sorted(v1_spectrum_layers)}")
    collisions = base_vectors & v1_vectors
    print(f"  score vectors referenced: base={len(base_vectors)}, V1a={len(v1_vectors)}")
    print(f"  base/V1a path collisions: {len(collisions)}")
    print(f"  stale JSON/vector pairs: base={len(base_stale)}, V1a={len(v1_stale)}")
    print(f"  summary rows: base={summary_rows}, V1a={v1_rows}")
    print(f"  V5a all-28-layer residual spectrum: {'PASS' if v5a_present else 'PENDING'}")
    if base_stale:
        status = ("V5a regenerated prompt-only vectors under variant-scoped names"
                  if v5a_present else
                  "V5a must regenerate them under variant-scoped names")
        print("  AUDIT FINDING: legacy V1a files overwrote base score vectors; " + status)


if __name__ == "__main__":
    main()
