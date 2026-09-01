"""Regression tests for Arm A's Base reference and identity gate."""
import importlib.util
from pathlib import Path

import pytest


EXP2 = Path(__file__).resolve().parent.parent


def load_driver(filename, name):
    spec = importlib.util.spec_from_file_location(name, EXP2 / "drivers" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_driver("09_audit_e4_artifacts.py", "e4_audit")
report = load_driver("10_e4_report.py", "e4_report")


def record(erank, perturbation=None):
    out = {"spectra": {"resid/layer4": {
        "tensor": "resid", "pooling": "last", "erank": erank}}}
    if perturbation is not None:
        out["perturbation"] = perturbation
    return out


def test_arm_a_identity_is_checked_against_r_base():
    arms = {"R_base": record(100.0), "A_ckpt0": record(100.0),
            "A_ckpt500": record(101.0)}
    got = audit.audit_arm_a(arms)
    assert got["ckpt0_identity_passed"] is True
    assert got["arms"]["A_ckpt500"]["max_abs_change_pct"] == pytest.approx(1.0)


def test_arm_a_nonidentity_ckpt0_fails():
    arms = {"R_base": record(100.0), "A_ckpt0": record(100.01)}
    with pytest.raises(AssertionError, match="reproduce R_base exactly"):
        audit.audit_arm_a(arms)


def test_noise_keeps_instruct_reference_when_arm_a_uses_base():
    perturbation = {"requested_dose": 1e-3,
                    "achieved_aggregate_dose": 1e-3, "seed": 42}
    arms = {"R_base": record(100.0), "R_instruct": record(200.0),
            "N_dose_1e-3": record(220.0, perturbation),
            "A_ckpt500": record(110.0)}
    rows = {row["arm"]: row for row in report.ladder_rows(arms, "R_base")}
    assert rows["N_dose_1e-3"]["reference"] == "R_instruct"
    assert rows["N_dose_1e-3"]["max_abs_pct"] == pytest.approx(10.0)
    assert rows["A_ckpt500"]["reference"] == "R_base"
    assert rows["A_ckpt500"]["max_abs_pct"] == pytest.approx(10.0)
