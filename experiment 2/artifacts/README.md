# E1 artifact archive

`e1_metric_remeasurement_2026-08-31.tar.gz` is the complete audited output of
the E1 metric re-measurement campaign. It contains the 12 checkpoint JSON
records, 366 score vectors, three summary CSV files, and the V5a execution log.

- Source run: `exp2_colab_guru_math7b_instruct_group8_e33527592dd9`
- Archive root: `e1_sweep/`
- Files: 382
- Uncompressed size: approximately 56 MB
- SHA-256: `9e5a8c99603fa2e0d250388ca9e484b82c34d734b5fac3f4a9c77e621a6558cb`

Extract from the repository root with:

```bash
tar -xzf "experiment 2/artifacts/e1_metric_remeasurement_2026-08-31.tar.gz" \
  -C eaaj-pilot/outputs/exp2_colab_guru_math7b_instruct_group8_e33527592dd9/measurements
```

Then verify the extracted artifacts with:

```bash
.venv/bin/python "experiment 2/drivers/05_audit_e1_artifacts.py" --require-v5a
```

The human-readable interpretation is in
`experiment 2/FINDING_E1_METRIC_REMEASUREMENT.md`; compute accounting and GPU
shutdown evidence are recorded in `eaaj-pilot/compute_log.md` and
`experiment 2/evidence/`.

## E4-small Windows RTX 4070 archive

`e4_small_win4070_2026-09-01.tar.gz` is the complete audited 0.5B E4 output,
including exp1.5 v3 full-parameter Arm W, both Arm-R records, all six Arm-N
rungs, the frozen probe and manifest, audit JSON, summary, figure, and the
preserved batch-16 OOM diagnostic record.

- Archive root: `e4_small/`
- Files: 16
- Uncompressed size: 3,329,503 bytes
- Archive size: 777,129 bytes
- SHA-256: `a172cc032ed5692290378441fb6903e218073d67461d6178655868e2c3bedd16`

Extract from the repository root with:

```bash
tar -xzf "experiment 2/artifacts/e4_small_win4070_2026-09-01.tar.gz" -C outputs
```

Then verify the extracted artifacts with:

```bash
python "experiment 2/drivers/09_audit_e4_artifacts.py" \
  --dir outputs/e4_small --require-arm-w
```

The interpretation and Windows execution notes are in
`experiment 2/FINDING_E4_SMALL_WIN4070.md` and `eaaj-pilot/compute_log.md`.
