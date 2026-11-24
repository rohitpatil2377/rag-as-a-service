"""Shared enums and primitive types for the data contracts."""

from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    """How a document was processed into text (drives the retrieval path later)."""

    TEXT = "text"  # born-digital / clean text
    OCR = "ocr"  # scanned -> OCR (Phase-0 default for images/scans)
    VISUAL = "visual"  # ColPali page-image embedding (Phase-2 stub)
    DATABASE = "database"  # structured row(s) from a unified KB's SQL/NoSQL source (docs+DB fusion)
    API = "api"  # JSON payload from a unified KB's connected REST API source
    SHAREPOINT = "sharepoint"  # snippet from a unified KB's connected SharePoint site


class SourceKind(StrEnum):
    """Origin of an ingested document (FR-A1..A3, A7)."""

    UPLOAD = "upload"
    OBJECT_STORE = "object_store"  # S3/MinIO (FR-A2)
    LAKEHOUSE = "lakehouse"  # Iceberg/Delta lineage pointer (FR-A1)
    RDBMS = "rdbms"
    REST_API = "rest_api"
    FILE_SHARE = "file_share"
    WEB = "web"


class ChunkStrategy(StrEnum):
    """Configurable chunking strategies (FR-C1). Phase-0 implements FIXED + CLAUSE."""

    FIXED = "fixed"
    CLAUSE = "clause"
    # RAG v2 Phase 5: chunks from the Phase-2 structure tree and emits the `ChunkLevel` grains.
    # Never selected by `RAG_CHUNK_STRATEGY` (whose pattern does not list it) — it needs a parsed
    # layout tree, so the flag-gated v2 ingest path selects it by HAVING one.
    LAYOUT_AWARE = "layout_aware"
    PARENT_CHILD = "parent_child"  # Phase-1
    CONTEXTUAL = "contextual"  # FR-C2, Phase-1


class ChunkLevel(StrEnum):
    """The GRAIN at which a v2 chunk is retrievable (RAG v2 Phase 5, "Levels emitted per section").

    Not every question answers at the same size. A narrow value lookup wants the one table ROW that
    holds the value; "what does 6.2 require" wants the whole SECTION; ordinary prose wants a body
    CHUNK. One fixed chunk size has to lose at least two of those three, which is the failure the
    upgrade plan opens with. The level is what lets retrieval ask for the grain the question needs.

    FOUR MEMBERS, ALL WITH A PRODUCER AS OF PHASE 4. The plan names five; ``doc`` is the one still
    DEFERRED and deliberately not reserved here, because an enum member retrieval can name but
    nothing can produce reads as a level that exists and is simply empty:

    * ``doc`` — one unit per document holding the profile card text (doc_type + purpose + facets).
      Phase 3 built the card, so the INPUT exists; the LEVEL still has no producer. The card
      reaches the index as the ``embed_text`` prefix on every other unit
      (``layout_chunking.embed_prefix``), which is where the plan puts it; a separate ``doc`` unit
      is an additional retrievable grain, and the argument Phase 5 made against the ``section``
      level applies to it unchanged until there is something in it that is not already indexed
      verbatim beside it.

    ``SECTION`` was built in Phase 5, measured and withdrawn, and Phase 4 restores it. Without a
    summary its text could only be the section's own concatenated blocks, which duplicated the
    ``CHUNK``/``ROW`` units beside it in every reachable case; the member was kept against the
    rule above precisely because Phase 4 was going to supply the summary, and it now does. The
    unit's text is the summary, which is not a substring of anything else in the index — see
    ``rag_core.layout_chunking.LayoutAwareChunker._section_level``.

    ``PROPOSITION`` is Phase 4's own level: standalone requirement/fact statements extracted from
    a section, each carrying a validated verbatim quote (``rag_core.section_enrich``). A
    document gets them only on the v2 ingest path with section enrichment on; a consumer must not
    assume any document has them.
    """

    SECTION = "section"
    CHUNK = "chunk"
    ROW = "row"
    PROPOSITION = "proposition"


class RetrievalMode(StrEnum):
    """Query routing mode (FR-D1/D3/D4/D5). Phase-0 implements HYBRID only."""

    HYBRID = "hybrid"  # dense + BM25 + RRF (Phase-0)
    DENSE = "dense"
    SPARSE = "sparse"
    VISUAL = "visual"  # Phase-2
    GRAPH = "graph"  # Phase-2


class AnswerStatus(StrEnum):
    """Outcome of the grounded-or-abstain gate (FR-E2) + faithfulness routing (FR-E4)."""

    GROUNDED = "grounded"
    ABSTAINED = "abstained"
    # Faithfulness-borderline: answer is cited but not fully supported -> HITL review,
    # NOT returned to the end user as final (FR-E4 three-way routing).
    FLAGGED_FOR_REVIEW = "flagged_for_review"


class AuditAction(StrEnum):
    """Auditable actions across the platform (FR-G4)."""

    INGEST = "ingest"
    INDEX = "index"
    # GraphRAG entity/edge extraction completed for a document (runs after INDEX). A distinct action
    # so the ingestion DAG can prove the graph stage ran, not merely infer it from indexing.
    GRAPH = "graph"
    QUERY = "query"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    ABSTAIN = "abstain"
    # Faithfulness verification verdict (FR-E4) — distinct, independently queryable action
    # carrying the score + unsupported_claims for every GROUNDED/FLAGGED/ABSTAINED outcome.
    FAITHFULNESS = "faithfulness"
    # Evidence that FAILED verification and was dropped from a composite answer. Distinct from
    # FAITHFULNESS, which records that verification happened and with what verdict: this records
    # what then became of the evidence. Without it a batch could fail grounding, be excluded from
    # the answer and leave no trace of its DISPOSITION anywhere — the faithfulness event says the
    # score was low, never that the material was withheld. Emitted even when the review could not
    # be persisted, because that is exactly the case the audit trail must not lose.
    FAITHFULNESS_BATCH_EXCLUDED = "faithfulness_batch_excluded"
    # Guardrail block (FR-G1/G3) — prompt-injection (input) or toxicity (output).
    GUARDRAIL = "guardrail"
    # A document's data-classification was auto-raised (or the ingest blocked) from detected PII.
    CLASSIFICATION = "classification"
    # HITL gate actions (FR-F6)
    HITL_VIEW = "hitl_view"
    HITL_ACCEPT = "hitl_accept"
    HITL_MODIFY = "hitl_modify"
    HITL_FLAG = "hitl_flag"
    HITL_OVERRIDE = "hitl_override"
    # End-user feedback on an answer (FR-I2) — thumbs up/down + optional comment,
    # trace-correlated so it feeds quality dashboards and golden sets.
    FEEDBACK = "feedback"
    # DPDP right-to-erasure / retention purge (FR-H8) — a tamper-evident compliance
    # certificate: what was erased, by whom, when. Distinct from routine deletion.
    ERASURE = "erasure"
    # Continuous evaluation (FR-G9) — a scheduled/on-demand quality run (golden set +
    # sampled live traffic) and a distinct breach ALERT when metrics fall below floor.
    EVAL_RUN = "eval_run"
    EVAL_ALERT = "eval_alert"
    # Envelope-encryption key rotation (FR-H9 BYOK) — a tenant DEK was rotated and stored
    # secrets re-encrypted; recorded so the key lifecycle is itself auditable.
    KEY_ROTATION = "key_rotation"
    # Governed NL→SQL query (FR-D10) — the NL question, the model's SQL, the validated SQL,
    # and the row count, so every database read is attributable and reviewable.
    SQL_QUERY = "sql_query"
    # Interpretive reading of a stored IMAGE (FR-B6). The one path that is deliberately NOT
    # grounded-or-abstain, and it also reads RAW PIXELS -- bypassing the PII masking applied to
    # extracted text, so an identifier masked in the index can still be described aloud. Both are
    # reasons every call is recorded with its question and its interpret flag.
    IMAGE_ANALYSIS = "image_analysis"
    # Agentic-workflow run (FR-F) — the ordered decision trace (which steps ran, which branch
    # fired) so an orchestrated multi-step run is fully explainable and reproducible.
    WORKFLOW_RUN = "workflow_run"
    # Egress-host approval workflow — a user requested an external host, an owner/admin approved
    # or rejected it. Recorded so every widening of the sovereignty allowlist is attributable.
    EGRESS = "egress"
    # Deletion of a document or a whole knowledge base — records who removed what and when, so
    # data removal is attributable on the tamper-evident trail (distinct from a DPDP ERASURE).
    DELETE = "delete"
    # Maker-checker (dual-control) approval — a sensitive action was requested, then approved or
    # rejected by a DIFFERENT admin (four-eyes). Records both halves for segregation-of-duties.
    APPROVAL = "approval"
    # Federated RAG federation (cross-tenant P2P RAG). A publisher department issues/revokes a
    # cryptographic grant authorising another department to query specific topics; a federated
    # query records which peers were contacted and which grants authorised the cross-dept access.
    GRANT_ISSUE = "grant_issue"
    GRANT_REVOKE = "grant_revoke"
    FEDERATED_QUERY = "federated_query"
    # A federated query a peer REFUSED (grant rejected/revoked at use). Its own action so the
    # SOC/SIEM can alert on denials without parsing every FEDERATED_QUERY payload.
    FEDERATION_DENIED = "federation_denied"
    # A federation anomaly worth a look (e.g. one caller denied by many peers = grant probing).
    FEDERATION_ALERT = "federation_alert"
    # Federated EVALUATION (FR-G9 mesh edge) — a per-grant quality-attribution run, and a
    # distinct QUALITY ALERT naming a peer whose content DEGRADES the joint answer.
    FEDERATED_EVAL = "federated_eval"
    FEDERATION_QUALITY_ALERT = "federation_quality_alert"
    # Billing (FR-H4+) — a per-tenant per-period invoice was computed (immutable statement on
    # the chain) and, distinctly, a budget-exceeded alert against the tenant plan.
    BILLING = "billing"
    # Trust receipt (H1) — a signed, portable provenance proof for one grounded answer, recorded
    # on the chain so the answer's authorisation + grounding + model are independently verifiable.
    RECEIPT = "receipt"
    # Unified KB — a connected structured source (NoSQL / REST / SharePoint; SQL also records its
    # own SQL_QUERY) contributed evidence to a fused answer. One row per source per query, so a
    # cross-source answer is attributable to exactly which sources it drew on.
    SOURCE_QUERY = "source_query"
    # Agent orchestrator (Agentic RAG + MCP) — one dynamic plan-and-execute agent run: the plan,
    # the critic verdict, which steps ran, and whether it paused for human approval of a critical
    # tool call. The per-step critical-action approvals are separate dual-control APPROVAL events.
    AGENT_RUN = "agent_run"
    # Per-KB custom named role defined or assigned by a KB admin (extends the RBAC spine). Records
    # who defined which roles / assigned which members, so every clearance elevation is attributable.
    ROLE_ASSIGN = "role_assign"
    # Per-KB model-connection change governance — a KB MANAGER requested a model change (add/update/
    # default/delete/embedder-apply), and an org owner/admin approved or rejected it. Recorded so
    # every change to a KB's model wiring by a non-admin is attributable through an approval.
    MODEL_REQUEST = "model_request"
