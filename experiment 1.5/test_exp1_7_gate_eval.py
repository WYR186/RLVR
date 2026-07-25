"""Unit tests for the experiment-1.7 cross-trajectory gate."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import exp1_7_gate_eval as gate


class Exp17GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("EAAJ_TEST_TMP"))
        self.root = Path(self.tmp.name)
        self.configs = gate.DEFAULT_CONFIGS

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def complete_run(self, index: int, q_shift: float,
                     endpoint_drop: float | None) -> Path:
        run_dir = self.root / f"run{index}"
        self.write_json(run_dir / "phase1_complete.json", {"step": 500})
        self.write_json(run_dir / "ckpt-500" / "config.json", {})
        for checkpoint, erank in (
                (0, 100.0), (300, 100.0 * (1.0 + q_shift)),
                (400, 100.0 * (1.0 + q_shift)),
                (500, 100.0 * (1.0 + q_shift))):
            self.write_json(
                run_dir / "measurements" / f"metrics_ckpt{checkpoint}.json",
                {"per_layer": {"layer12": {"erank": erank}}})
        if endpoint_drop is not None:
            self.write_json(
                run_dir / "adaptation_seed42" / "ckpt-0" / "summary.json",
                {"delta_acc": endpoint_drop})
            self.write_json(
                run_dir / "adaptation_seed42" / "ckpt-500" / "summary.json",
                {"delta_acc": 0.0})
        return run_dir

    def test_probe_before_endpoint_cells_exist(self):
        runs = [self.complete_run(i, 0.09, None) for i in range(3)]
        payload = gate.evaluate(self.configs, runs)
        self.assertEqual(payload["verdict"], "PROBE")
        self.assertTrue(payload["q_pass"])

    def test_q_gate_unlocks_expansion(self):
        runs = [self.complete_run(i, q, 0.01)
                for i, q in enumerate((0.08, 0.09, 0.10))]
        payload = gate.evaluate(self.configs, runs)
        self.assertEqual(payload["verdict"], "EXPAND")
        self.assertTrue(payload["q_pass"])
        self.assertFalse(payload["endpoint_pass"])

    def test_outcome_gate_unlocks_expansion(self):
        runs = [self.complete_run(i, 0.01, drop)
                for i, drop in enumerate((0.04, 0.06, 0.07))]
        payload = gate.evaluate(self.configs, runs)
        self.assertEqual(payload["verdict"], "EXPAND")
        self.assertFalse(payload["q_pass"])
        self.assertTrue(payload["endpoint_pass"])

    def test_double_failure_stops(self):
        runs = [self.complete_run(i, 0.01, 0.01) for i in range(3)]
        payload = gate.evaluate(self.configs, runs)
        self.assertEqual(payload["verdict"], "STOP")
        self.assertFalse(payload["q_pass"])
        self.assertFalse(payload["endpoint_pass"])

    def test_two_safety_stops_override_other_evidence(self):
        runs = []
        for index in range(2):
            run_dir = self.root / f"run{index}"
            self.write_json(run_dir / "safety_stop.json", {"step": 55})
            runs.append(run_dir)
        runs.append(self.complete_run(2, 0.10, 0.10))
        payload = gate.evaluate(self.configs, runs)
        self.assertEqual(payload["verdict"], "STOP_COLLAPSE")


if __name__ == "__main__":
    unittest.main()
