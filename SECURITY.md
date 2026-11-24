# Security

## Reporting a vulnerability

Please open a [security advisory](https://github.com/rohitpatil2377/rag-as-a-service/security/advisories/new)
rather than a public issue. This is a personal project, so there is no SLA — expect best effort.

## What the platform enforces

Worth knowing before assessing it:

- **Tenant isolation by namespace.** One Qdrant collection, OpenSearch index, S3 prefix and
  Postgres schema per tenant, with `tenant_id` an explicit parameter on every store call.
  `tests/integration/test_tenant_isolation.py` and its adversarial sibling are the gates.
- **Grounded-or-abstain**, enforced at the type level — an uncited "grounded" answer cannot be
  constructed.
- **Egress allowlist.** `rag_common.sovereignty.assert_sovereign` fails closed: a host that is not
  allowlisted is refused, not called.
- **RBAC** over a closed capability set, with multi-level security (document classification vs
  member clearance) layered on top.
- **Envelope encryption** with a KMS/HSM seam, so revoking the key crypto-shreds a tenant's data.
- **Append-only audit chain** (SHA-256), with integrity verified by walking the chain.
- **Governed NL→SQL** — model-written SQL is parsed and refused unless it is a single SELECT over
  allowlisted tables, with a forbidden-function denylist and a hard LIMIT.

## Known limitations

These are documented rather than hidden; see the "Known limitations" section of
[`CLAUDE.md`](CLAUDE.md). The security-relevant ones:

- Faithfulness checks **entailment, not relevance** — a sentence can be entailed by a retrieved
  chunk that has nothing to do with the question.
- Query-time audit events are written under the org `tenant_id` while ingest events use the
  `kb_id`, so the quality dashboards under-report for multi-KB orgs.
- Heuristic graph extraction is noisy on dense documents.

## Dev defaults are not production defaults

`RAG_AUTH_MODE=dev_jwt`, the `RAG_DEV_JWT_SECRET` placeholder and the `minioadmin` store
credentials in `.env.example` and the compose files are **local development values**. Production
uses OIDC and injects secrets through the `SecretsProvider` (Vault/HSM). The in-memory adapters
are refused outright when `RAG_ENV=prod`.

No real credentials belong in this repository. If you find one, report it as above.
