# Operational scripts

37 scripts. None are imported by the product; they are tools you run against a deployment.

## Evaluation and quality

| Script | Purpose |
|---|---|
| `eval_gate.py` | The repeat-run quality gate. `--repeat N` to measure flakiness, `--compare baseline.json` to fail on regression, `--floor metric=value` to enforce a bar. Reports a **rate**, never a single verdict. |
| `ragas_gate.py` | Certifies KBs against the RAGAS quality contract (`make ragas`). |
| `calibrate.py`, `calibrate_abstain.py`, `calibrate_faithfulness.py` | Sweep the abstain and faithfulness thresholds against a gold set instead of guessing them. |
| `score_doc_card.py`, `retrieval_metrics.py`, `answer_probe.py`, `shape_battery.py` | Narrower probes — behaviour checks, not quality metrics. |
| `benchmark.py`, `load_soak.py` | Throughput, federated scatter-gather scaling, and soak-drift detection. |

## Operations

| Script | Purpose |
|---|---|
| `wait_for_stores.py` | Readiness probe, run **inside** the compose network by `make up` (the stores publish no resolvable names on the host). |
| `backup.sh` / `restore.sh` | Snapshot and restore the stateful stores (`make backup`, `make restore BACKUP=…`). |
| `airgap-bundle.sh` / `airgap-install.sh` | Build and install a self-contained offline bundle. |
| `build_update_bundle.py`, `apply_update.py`, `verify_receipt.py` | Signed offline updates and trust receipts. |
| `build_model_sbom.py` | Model SBOM for the compliance pack. |
| `reindex_corpora.py`, `reenrich.py`, `reconcile_v2_index.py` | Re-index and repair after a chunking or embedder change. |
| `compliance_evidence.py`, `federation_qa.py`, `ocr_verify.py` | Evidence gathering for the governance surfaces. |
| `seed_demo_runtime.py` | Seeds fictional demo data (Acme personas) for a walkthrough. |
| `smoke_test.sh`, `install_prod.sh`, `bundle_prod.sh` | Deployment helpers. |

Scripts shell out to `docker` and `alembic` on `PATH` by design — hence the `S603`/`S607` ruff
exemption for this directory in `pyproject.toml`.
