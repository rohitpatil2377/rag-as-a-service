# Documentation index

Start with the root [`README.md`](../README.md) to run it, and [`CLAUDE.md`](../CLAUDE.md) for the
architecture map and the honest list of known limitations.

## Decisions

[`adr/`](adr/) — 26 Architecture Decision Records. **Read the relevant one before reversing
something that looks like an oversight**; it usually is not. Load-bearing ones: monolith-first
(0001), no RAG framework (0002), in-memory twins (0004), faithfulness routing (0005), retrieval
score semantics (0015), federation transport and its security (0017, 0018, 0020).

## Operating it

| Doc | Covers |
|---|---|
| [`PLATFORM_GUIDE.md`](PLATFORM_GUIDE.md) | The operator-facing tour, including the views worth drawing and the sizing table. |
| [`DEPLOY_PROD.md`](DEPLOY_PROD.md), [`deployment-runbook.md`](deployment-runbook.md) | Production deployment and the runbook. |
| [`production-sizing.md`](production-sizing.md) | Capacity numbers derived from a running system. |
| [`airgap-install.md`](airgap-install.md), [`air-gap-mode.md`](air-gap-mode.md), [`offline-update.md`](offline-update.md) | Running with no internet at all. |
| [`MODELS.md`](MODELS.md), [`model-sbom.md`](model-sbom.md) | Which models, served how, and their provenance. |
| [`Backup_DR.md`](Backup_DR.md) | Backup and disaster recovery. |
| [`deploy/`](deploy/) | Kubernetes ingress, observability and service-mesh cutover. |

## Behaviour and tuning

[`calibration.md`](calibration.md) (thresholds), [`golden-template.md`](golden-template.md) and
[`TEST_QUESTION_SET.md`](TEST_QUESTION_SET.md) (eval sets), [`OCR.md`](OCR.md),
[`indic-cross-lingual.md`](indic-cross-lingual.md), [`connector-*.md`](.) (REST, NoSQL, SharePoint).

## Governance and security

[`qa/`](qa/) — compliance evidence, observability, load tests, the RAGAS gate and known
pre-production behaviours. Plus [`trust-receipts.md`](trust-receipts.md),
[`residency-attestation.md`](residency-attestation.md), [`automated-erasure.md`](automated-erasure.md),
[`grant-token-spec.md`](grant-token-spec.md) and [`VAPT_Response.md`](VAPT_Response.md).

## Working notes

[`rag_v2/`](rag_v2/) is the discovery record behind the v2 pipeline — what was measured, what was
abandoned, and which acceptances were never scored. Read its [README](rag_v2/README.md) first: it
refers to corpora that are not part of this repository.
