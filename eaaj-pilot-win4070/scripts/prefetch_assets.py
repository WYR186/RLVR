#!/usr/bin/env python3
"""Prefetch the pinned model + datasets into the local HF cache.

Phase 1 loads the model with local_files_only=True, so the Windows box must
have everything cached before the first run. Ids and revisions come from
eaaj-pilot/pilot_config.json (single source of truth) — nothing here decides
anything scientific.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # eaaj-pilot-win4070/
PILOT = HERE.parent / "eaaj-pilot"
sys.path.insert(0, str(PILOT))


def main() -> None:
    cfg = json.loads((PILOT / "pilot_config.json").read_text())

    from huggingface_hub import snapshot_download
    print(f"model: {cfg['model_id']} @ {cfg['model_revision'][:12]}")
    snapshot_download(cfg["model_id"], revision=cfg["model_revision"])

    from datasets import load_dataset
    for repo, rev in cfg["dataset_revisions"].items():
        print(f"dataset: {repo} @ {rev[:12]}")
        if repo == "openai/gsm8k":
            load_dataset(repo, "main", revision=rev)
        else:
            load_dataset(repo, revision=rev)

    for name in ("gsm8k_splits.json", "svamp_splits.json", "probe_set_ids.json"):
        path = PILOT / "data" / name
        if not path.exists():
            raise FileNotFoundError(f"frozen split file missing from the repo: {path}")

    print("prefetch complete — phase 1 can now run offline")


if __name__ == "__main__":
    main()
