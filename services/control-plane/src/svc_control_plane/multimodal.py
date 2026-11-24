"""Multimodal (vision) RAG — additive, flag-gated (``RAG_MULTIMODAL_ENABLED``, default OFF).

Two capabilities, both sovereign and default-safe:

1. **Caption at ingest** — when an image document is ingested, a sovereign vision LLM
   describes it and the caption is ADDED to the image's OCR text by the ingestion OCR check
   (``svc_doc_intelligence.ocr_check``, via ``wiring.describe_image_for_ingest``). A text query
   then retrieves the image by what it says AND what it shows. The raw image bytes already sit
   in the object store (the normal ingestion path put them there).

2. **See at answer time** — :func:`describe_image` fetches an image's raw bytes and asks the
   vision LLM a question about it. This is the genuinely-new capability: reading a diagram or
   photo the OCR text can't convey.

Composes with Federated RAG federation unchanged: an image chunk crosses a node boundary as
its caption text only (a `RetrievedSpan`), never raw bytes — the same rule as any document.
"""

from __future__ import annotations

from svc_doc_intelligence.ocr_check import trim_degenerate_tail

from rag_clients.factory import (
    get_catalog,
    get_object_store,
    get_vision_embedder,
    get_vision_llm_gateway,
    get_vision_vector_store,
)
from rag_common import get_logger, get_settings
from rag_common.tenancy import raw_object_key
from rag_core.vision import VisionLLMGateway
from rag_schemas import DocumentEnvelope

_log = get_logger(__name__)

# This caption is INDEXED and can be cited as grounded evidence, so it must stay strictly
# observational. It may say what is visibly present and how the page is organised; it may NOT
# explain what any of it means. That line is the whole reason a grounded summary of an image is
# trustworthy -- the moment a caption asserts "these are neutralisation reactions", the model's own
# knowledge has entered the corpus wearing a citation, and nothing downstream can tell it apart
# from the document. Interpretation lives in _INTERPRET_PROMPT, which is labelled ungrounded.
#
# Structure, though, IS visible, and asking for it is what makes an image summarisable at all. The
# old prompt asked only for "what it shows ... and the key facts", and on a page of 60 hand-drawn
# formulas that yielded a transcription: asked to summarise, the pipeline could only read the
# formulas back, because a list of formulas was the only thing in the index. Naming the KINDS of
# content and their rough counts gives retrieval something to summarise FROM.
_CAPTION_PROMPT = (
    "Describe this image for search and retrieval.\n"
    "1. Open with one sentence saying what the image IS as a whole.\n"
    "2. Then describe what it contains, GROUPING like things together and naming each group: "
    "visible text and labels, formulas or notation, diagrams and figures, tables, objects, "
    "and any structure or layout a reader would navigate by.\n"
    "3. Give approximate counts for repeated items rather than transcribing every one.\n"
    "Describe only what is visibly there. Do not explain, interpret, or add background "
    "knowledge -- another step does that."
)


async def _vision_gateway(tenant_id: str | None) -> VisionLLMGateway:
    """The KB's vision gateway when we know the tenant, else the process-wide one."""
    if tenant_id:
        from svc_control_plane import wiring  # noqa: PLC0415 — avoid import cycle at module load

        gw: VisionLLMGateway = await wiring.resolve_tenant_vision_gateway(tenant_id)
        return gw
    return get_vision_llm_gateway()


def is_image(doc: DocumentEnvelope) -> bool:
    return doc.mime.startswith("image/")


async def caption_image(image: bytes, *, mime: str = "image/jpeg", tenant_id: str | None = None) -> str:
    """A sovereign vision LLM's searchable description of an image.

    ``tenant_id`` lets the KB's own console-configured model answer — without it the vision path
    reads RAG_VISION_LLM_* alone and 401s on a deployment whose credential lives in a model
    connection."""
    gateway = await _vision_gateway(tenant_id)
    return await gateway.complete_with_images(_CAPTION_PROMPT, [image])


# Two prompts, and the difference between them is the whole grounding contract.
#
# STRICT is the platform's normal rule: answer from what is there, say so when it does not cover
# the question. Everything else in this codebase works this way.
#
# INTERPRET deliberately suspends that rule. Some documents cannot be USEFUL under it: a page of
# chemical formulas, a circuit diagram, a musical score carry meaning that is not written on them.
# Asked "what does this reaction do?", a strict reader can only reply that the image does not say
# -- which is true, and useless. Interpreting it requires domain knowledge the document does not
# contain, i.e. exactly what grounded-or-abstain exists to forbid.
#
# So the exception is explicit, narrow and labelled rather than a loosening of the main pipeline:
# a separate call, a separate audit action, and a response that states it is not grounded. The
# model is still told to separate what it READ from what it KNOWS, so a reader can see which half
# to trust.
_STRICT_PROMPT = (
    "Answer the question using ONLY what is visible in the image. "
    "If the image does not show it, say so.\n\nQuestion: {q}"
)
_INTERPRET_PROMPT = (
    # The QUESTION leads. An earlier version opened with "STEP 1 - transcribe everything", which
    # a general chat model tolerated and a document-OCR model obeyed literally: Chandra returned
    # a faithful transcription of the whole page and never reached the question. Transcribing is
    # also redundant here -- the OCR text is already indexed -- and it spends the model's
    # attention on copying instead of reasoning. Ask what is wanted; require evidence for it.
    "You are examining a document image for a subject-matter reader. Answer the QUESTION below.\n"
    "- Ground every claim in what is actually drawn or written, and point to the specific marks "
    "you are reading. Do not transcribe the whole image.\n"
    "- Where answering needs subject knowledge the image does not state, you MAY use your own "
    "knowledge, but label that part clearly as interpretation so the reader can tell it apart.\n"
    "- If the question asks about several things, organise the answer under short headings.\n"
    "- Say plainly when something is illegible or genuinely absent rather than guessing.\n\n"
    "QUESTION: {q}"
)


async def describe_image(kb_id: str, doc_id: str, question: str, *, interpret: bool = False) -> str:
    """Answer a question about a stored image by letting the vision LLM SEE it.

    Fetches the image's raw bytes from the object store (the normal ingest path stored them
    at the standard raw key) and asks the sovereign vision model. Distinct from the text RAG
    answer: this reads pixels, not the caption.

    ``interpret=False`` (default) keeps the platform's grounding rule -- only what is visible.
    ``interpret=True`` permits the model's own domain knowledge and is therefore NOT a grounded
    answer; the caller is responsible for labelling it as such. See the prompts above.
    """
    if not get_settings().multimodal_enabled:
        raise RuntimeError("multimodal is not enabled (RAG_MULTIMODAL_ENABLED)")
    raw = await get_object_store().get(kb_id, raw_object_key(doc_id))
    gateway = await _vision_gateway(kb_id)  # the KB's own model, same as the caption path
    template = _INTERPRET_PROMPT if interpret else _STRICT_PROMPT
    answer = await gateway.complete_with_images(template.format(q=question.strip()), [raw])
    # The same degeneracy guard the OCR path uses, for the same reason and the same model class.
    # This is the THIRD place the failure has turned up -- OCR transcription, caption, and now a
    # live question -- which is the argument for guarding wherever a vision model's free text is
    # read, rather than at whichever call site last embarrassed itself.
    answer, looped = trim_degenerate_tail(answer)
    if looped:
        _log.warning("image_analysis_degenerate_output", doc_id=doc_id, kept_chars=len(answer))
    return answer


def _cross_modal_on() -> bool:
    settings = get_settings()
    return settings.multimodal_enabled and settings.vision_recall_enabled


async def index_image_vectors(tenant_id: str, raw: bytes, chunk_ids: list[str]) -> None:
    """Index an image's CLIP vector into the cross-modal store under each of its caption chunk
    ids, so a text query can retrieve the image by VISUAL similarity — not only its caption.

    Additive + default-OFF: a no-op unless multimodal AND vision_recall are both enabled. All of
    an image's caption chunks describe the ONE image, so the same visual vector is attached to
    each; any chunk id the visual leg surfaces then hydrates to a real chunk via the catalog.
    """
    if not _cross_modal_on() or not chunk_ids:
        return
    embedder = get_vision_embedder()
    store = get_vision_vector_store()
    chunks = list((await get_catalog().get_chunks(tenant_id, chunk_ids)).values())
    if not chunks:
        return
    [vec] = await embedder.embed_image([raw])
    await store.ensure_collection(tenant_id, embedder.dim)
    await store.upsert(tenant_id, chunks, [vec] * len(chunks))
    _log.info("image_vectors_indexed", tenant_id=tenant_id, chunks=len(chunks), dim=embedder.dim)


async def purge_image_vectors(tenant_id: str, chunk_ids: list[str]) -> None:
    """Drop image vectors for these chunk ids from the cross-modal store (document delete /
    replace / re-index swap). Safe on non-image ids — only image chunk ids exist there and
    ``delete_chunks`` is a no-op for unknown ids. No-op unless cross-modal is enabled."""
    if not _cross_modal_on() or not chunk_ids:
        return
    await get_vision_vector_store().delete_chunks(tenant_id, chunk_ids)


async def purge_image_tenant(tenant_id: str) -> None:
    """Drop the whole cross-modal collection for a KB (KB purge). No-op unless enabled."""
    if not _cross_modal_on():
        return
    await get_vision_vector_store().delete_tenant(tenant_id)
