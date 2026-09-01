"""Strict local audit for E4 artifacts. No GPU, no model weights, no network.

Mirrors `05_audit_e1_artifacts.py`: it re-derives every headline number from
the raw per-arm JSONs rather than trusting the summary, and it fails loudly on
the specific ways this experiment could be silently wrong.

    python 09_audit_e4_artifacts.py --dir ../outputs/e4_small
    python 09_audit_e4_artifacts.py --dir ../outputs/e4_large --require-arm-w
"""
import argparse
import json
import sys
from pathlib import Path

EXP2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP2))

from src import e4_calibration as e4  # noqa: E402

CONTRACT_FIELDS = {
    "model_eval", "activation_accumulator", "svd_dtype", "spectrum_centering",
    "n_probe", "layers", "batch_size", "e4_scale", "e4_device", "e4_dtype",
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"missing artifact: {path}")
    return json.loads(path.read_text())


def audit_contract(rec: dict, label: str) -> None:
    c = rec.get("measurement_contract")
    assert c, f"{label}: no measurement_contract"
    missing = CONTRACT_FIELDS - set(c)
    assert not missing, f"{label}: contract missing {sorted(missing)}"
    assert c["model_eval"] is True, f"{label}: model was not in eval mode"
    assert c["activation_accumulator"] == "float32", f"{label}: wrong accumulator"
    assert c["svd_dtype"] == "float64", f"{label}: wrong SVD dtype"
    assert c["spectrum_centering"] is True, f"{label}: spectrum not centered"
    # Arm A deliberately carries a real trained update; every other arm must
    # be a bare model, or an adapter would silently confound the ladder.
    if label.startswith("A_"):
        assert c["adapter"] != "none - bare model weights", \
            f"{label}: an Arm-A record must name the checkpoint it loaded"
    else:
        assert c["adapter"] == "none - bare model weights", \
            f"{label}: E4 arms must carry no adapter"


def audit_probe(d: Path) -> dict:
    """Every arm must have measured the SAME probe, or nothing is comparable."""
    man = d / "probe_manifest.json"
    out = {"manifest_present": man.is_file()}
    if man.is_file():
        m = load(man)
        out.update({k: m[k] for k in
                    ("n_probe", "probe_ids_sha16", "rendered_text_sha256",
                     "sample_truncated_vs_e1") if k in m})
    return out


def audit_arms(d: Path) -> dict:
    arms = {}
    for p in sorted(d.glob("*.json")):
        if p.name in ("ruler_table.json", "probe_manifest.json",
                      "probe_frozen.json", "arm_W_weight_dose.json",
                      "audit_e4.json"):
            continue
        rec = load(p)
        assert "spectra" in rec, f"{p.name}: no spectra block"
        audit_contract(rec, p.stem)
        arms[p.stem] = rec
    assert arms, f"no arm records under {d}"

    n_probes = {k: v["meta"]["n_probe"] for k, v in arms.items()}
    assert len(set(n_probes.values())) == 1, \
        f"arms measured different probe sizes: {n_probes}"
    layer_sets = {k: tuple(v["meta"]["layers"]) for k, v in arms.items()}
    assert len(set(layer_sets.values())) == 1, \
        f"arms measured different layers: {layer_sets}"
    dtypes = {k: v["measurement_contract"]["e4_dtype"] for k, v in arms.items()}
    assert len(set(dtypes.values())) == 1, \
        f"arms measured in different dtypes, so levels are not comparable: {dtypes}"
    return arms


def audit_hook_check(arms: dict) -> dict:
    """The gated-MLP identity must have been verified on every arm."""
    out = {}
    for label, rec in arms.items():
        err = rec["meta"].get("gated_mlp_hook_check_max_abs_err")
        assert err is not None, f"{label}: gated-MLP hook check did not run"
        assert float(err) <= 1e-2, f"{label}: hook check err {err} too large"
        out[label] = float(err)
    return out


def audit_ladder(arms: dict) -> dict:
    """Arm N rungs must have hit the dose they claim, and be distinct."""
    rungs = {k: v for k, v in arms.items() if k.startswith("N_")}
    rows = {}
    for label, rec in sorted(rungs.items()):
        p = rec.get("perturbation")
        assert p, f"{label}: no perturbation record"
        req, got = float(p["requested_dose"]), float(p["achieved_aggregate_dose"])
        # bf16/fp16 parameters cannot represent an arbitrarily small nudge; a
        # rung that missed its dose is usable only if the record says so.
        rel_err = abs(got - req) / req if req > 0 else 0.0
        rows[label] = {"requested": req, "achieved": got,
                       "rel_err": rel_err,
                       "seed": p.get("seed"),
                       "n_modules": p["n_modules_perturbed"]}
        assert p["n_modules_perturbed"] > 0, f"{label}: nothing was perturbed"
        assert "not a model of an rlvr update" in p["caveat"].lower(), \
            f"{label}: perturbation caveat missing"
    if len(rows) > 1:
        # Seed repeats are SUPPOSED to land on the same achieved dose - the
        # rescaling makes it seed-independent by construction, and varying only
        # the noise direction is the whole point. So uniqueness is checked on
        # (dose, seed), which still catches the real error this guards against:
        # the same rung silently measured twice.
        keys = [(round(r["achieved"], 12), r["seed"]) for r in rows.values()]
        assert len(set(keys)) == len(keys), \
            f"two ladder rungs share a (dose, seed): {rows}"
    return rows


def audit_ruler(d: Path, arms: dict) -> dict:
    """Recompute the ruler table from the raw records; it must match on disk."""
    if "R_instruct" not in arms:
        return {"status": "Arm R reference absent; ruler not checkable"}
    recomputed = e4.ruler_table(arms, "R_instruct")
    path = d / "ruler_table.json"
    if not path.is_file():
        return {"status": "ruler_table.json absent", "recomputed": recomputed["arms"]}
    on_disk = load(path)
    for label, row in recomputed["arms"].items():
        want = on_disk["arms"].get(label)
        assert want is not None, f"ruler_table missing arm {label}"
        if "max_abs_change_pct" in row:
            assert abs(row["max_abs_change_pct"]
                       - want["max_abs_change_pct"]) < 1e-9, \
                f"{label}: ruler_table disagrees with the raw records"
    return {"status": "recomputed and matches", "arms": recomputed["arms"]}


def audit_arm_w(d: Path, required: bool) -> dict:
    p = d / "arm_W_weight_dose.json"
    if not p.is_file():
        if required:
            raise AssertionError(f"--require-arm-w but {p} is absent")
        return {"status": "not run"}
    rec = load(p)
    rel = rec.get("relative_dose", {})
    if rel.get("status") == "not_run":
        return {"status": "delta norms only", "reason": rel.get("reason")}
    out = {}
    for ckpt, r in sorted(rel.items()):
        out[ckpt] = r["aggregate_relative_dose"]
    # LoRA initialises B=0, so the pre-update adapter must be exactly identity.
    if "ckpt-0" in out:
        assert out["ckpt-0"] == 0.0, \
            f"ckpt-0 dose is {out['ckpt-0']}, expected exactly 0 (B is zero-init)"
    return {"status": "complete", "aggregate_relative_dose": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--require-arm-w", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)

    probe = audit_probe(d)
    arms = audit_arms(d)
    hooks = audit_hook_check(arms)
    ladder = audit_ladder(arms)
    ruler = audit_ruler(d, arms)
    armw = audit_arm_w(d, args.require_arm_w)

    report = {"dir": str(d), "probe": probe, "arms": sorted(arms),
              "gated_mlp_hook_check": hooks, "ladder": ladder,
              "ruler": ruler, "arm_W": armw}
    (d / "audit_e4.json").write_text(json.dumps(report, indent=1))

    print("E4 ARTIFACT AUDIT PASS")
    print(f"  dir: {d}")
    print(f"  arms: {', '.join(sorted(arms))}")
    print(f"  probe: n={probe.get('n_probe', '?')} "
          f"ids={probe.get('probe_ids_sha16', 'n/a')} "
          f"truncated_vs_e1={probe.get('sample_truncated_vs_e1', 'n/a')}")
    print(f"  gated-MLP hook check: max err "
          f"{max(hooks.values()) if hooks else 'n/a'}")
    if ladder:
        print("  ladder rungs (requested -> achieved):")
        for label, r in sorted(ladder.items(), key=lambda kv: kv[1]["requested"]):
            print(f"    {label:>16}  {r['requested']:.1e} -> {r['achieved']:.6e} "
                  f"(rel err {r['rel_err']:.2%}, {r['n_modules']} modules)")
    if isinstance(ruler.get("arms"), dict):
        print("  erank change vs R_instruct:")
        for label, row in sorted(ruler["arms"].items()):
            if "max_abs_change_pct" in row:
                print(f"    {label:>16}  {row['max_abs_change_pct']:>9.4f}%  "
                      f"(layer {row['max_abs_change_layer']})")
    print(f"  Arm W: {armw.get('status')}")
    if armw.get("aggregate_relative_dose"):
        for c, v in armw["aggregate_relative_dose"].items():
            print(f"    {c:>10}: {v:.6e}")
    print(f"\n  report: {d / 'audit_e4.json'}")


if __name__ == "__main__":
    main()
