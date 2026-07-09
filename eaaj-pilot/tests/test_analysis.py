import json
from pathlib import Path

from src.analysis import run_analysis


def test_phase4_analysis_writes_all_artifacts(tmp_path: Path):
    ckpts, layers = [0, 25, 50, 100, 200], [4, 12, 22]
    (tmp_path / "measurements").mkdir()
    (tmp_path / "adaptation").mkdir()
    for i, ckpt in enumerate(ckpts):
        per_layer = {}
        for layer in layers:
            per_layer[f"layer{layer}"] = {
                "erank": 10.0 - i, "erank_norm": (10.0 - i) / 100,
                "dormant_frac_tau0.025": i / 100,
                "dormant_frac_tau0.1": i / 50,
                "anisotropy_centered": i / 10,
            }
        (tmp_path / "measurements" / f"metrics_ckpt{ckpt}.json").write_text(
            json.dumps({"per_layer": per_layer}))
        out = tmp_path / "adaptation" / f"ckpt-{ckpt}"
        out.mkdir()
        (out / "summary.json").write_text(json.dumps({
            "acc_before": .1, "acc_after": .6 - i / 20,
            "delta_acc": .5 - i / 20}))
    (tmp_path / "gsm8k_eval.jsonl").write_text("".join(
        json.dumps({"step": c, "accuracy": .2 + i / 10}) + "\n"
        for i, c in enumerate(ckpts)))
    (tmp_path / "dashboard.jsonl").write_text("".join(
        json.dumps({"step": i + 1, "reward": i / 10}) + "\n" for i in range(10)))

    summary = run_analysis(tmp_path, {
        "stage_a": {"checkpoint_steps": ckpts},
        "measurement": {"layers": layers},
    })
    assert summary["n_checkpoints"] == 5
    for name in ("results_table.csv", "spearman_table.csv",
                 "fig_a_q_vs_updates.png", "fig_b_reward_vs_q.png",
                 "fig_c_scatter_q_vs_svamp.png", "analysis_summary.json"):
        assert (tmp_path / "analysis" / name).stat().st_size > 0
