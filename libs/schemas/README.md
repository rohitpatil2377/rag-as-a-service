# `rag_schemas`

The pydantic contracts every layer passes around. Deliberately small and dependency-free, so a
contract change is visible in one place.

| Schema | Invariant worth knowing |
|---|---|
| `DocumentEnvelope` | What ingestion accepts: bytes or an object reference, plus `content_hash` (SHA-256) used for delta and near-duplicate detection. |
| `Chunk` | A span of a document with its `tenant_id`, ACL, classification and metadata. Metadata set here becomes a retrieval filter later. |
| `RetrievalQuery` / `RetrievedSpan` / `RetrievalResult` | The retrieval contract, including `acl`, `clearance`, recency window and `chem_terms`. |
| `Citation` / `GroundedAnswer` | **The grounded-or-abstain invariant, enforced at the type level.** A `GroundedAnswer` with `status=GROUNDED` and zero citations fails validation, as does `status=ABSTAINED` carrying any. |
| `AuditEvent` | An append-only event in the SHA-256 hash chain. |

The `GroundedAnswer` validator is the reason the platform's central promise cannot be broken by a
careless caller: there is no code path that returns an uncited "grounded" answer, because such an
object cannot be constructed.
