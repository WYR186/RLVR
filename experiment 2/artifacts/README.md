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
