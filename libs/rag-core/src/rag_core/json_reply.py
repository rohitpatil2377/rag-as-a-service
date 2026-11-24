"""Isolating the JSON object out of a model reply — shared by every "one small JSON call" adapter.

Lifted verbatim out of :mod:`rag_core.query_plan`, which had it as a private helper, when
:mod:`rag_core.doc_profile` became its second caller. A second copy was the alternative, and
two copies of "where does the JSON start" drift: a reply the planner tolerates and the profiler
rejects (or the reverse) would be a difference nobody chose and no test would show.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def scan_json_object(reply: str) -> tuple[str | None, str]:
    """The first balanced ``{...}`` object in a reply, AND why there isn't one.

    WHY THE REASON IS RETURNED RATHER THAN LEFT TO THE CALLER TO GUESS. "No JSON object" has two
    causes that need different fixes and look identical from outside:

    * ``absent`` — there is no ``{`` at all. The model answered in prose. That is an
      instruction-following failure, and the fix is the prompt or another attempt.
    * ``unclosed`` — a ``{`` opens and the reply ENDS while still inside it. The model was
      answering correctly and ran out of room. The fix is output budget or a smaller input, and
      another attempt only re-rolls the reply's length.

    Measured cost of conflating them (2026-11-02, `corrigendum-5.pdf`): three substantive
    sections lost their Phase-4 enrichment, the error string said "no JSON object in the reply",
    and that was committed as "the model returned prose" — from the string, without reading a
    reply. Every one was a valid object truncated mid-structure at the output cap.

    ``unclosed`` IS A STRUCTURAL FACT, not an inference about tokens: the scan says the object
    never closed. It is not proof of token exhaustion, which only the provider's own
    ``finish_reason`` can give, and :class:`~rag_core.protocols.LLMGateway` returns a bare
    string so that signal is not available here. A caller that gains it should prefer it.

    The scan does not interpret strings, so a ``}`` inside a string value can close the object
    early. That is pre-existing behaviour and is kept deliberately: one scan, so the reason can
    never disagree with the extraction."""
    text = _FENCE_RE.sub("", reply.strip())
    start = text.find("{")
    if start == -1:
        return None, "absent"
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], ""
    return None, "unclosed"


def extract_json_object(reply: str) -> str | None:
    """Best-effort isolation of the first balanced ``{...}`` object in a model reply.

    The reason-free form of :func:`scan_json_object`, kept because most callers only need the
    blob."""
    return scan_json_object(reply)[0]


def extract_json_array(reply: str) -> str | None:
    """Best-effort isolation of the first balanced ``[...]`` array in a model reply.

    The companion to :func:`extract_json_object`, for the one prompt in this codebase whose
    schema is a LIST rather than an object (the Phase-9 map extraction, which returns zero or
    more items per section). Written as its own scan rather than by generalising the object
    scanner over a bracket pair, because the two must not be interchangeable: a caller expecting
    an object and handed an array — or the reverse — should get None and retry, not a value of
    the wrong shape that fails later in the parse with a worse message.

    Brackets inside a STRING are not tracked, exactly as in :func:`extract_json_object`: the
    scan is balanced-delimiter only. A quoted ``"]"`` inside the reply therefore ends the scan
    early and ``json.loads`` rejects the fragment, which is the safe direction — the caller
    reports an unparseable reply and retries rather than silently taking a prefix.

    A TRUNCATED ARRAY IS SALVAGED to its last COMPLETE element, which an object scan cannot do
    and is the reason this is not simply the object scanner over a different bracket pair. A
    list reply is the one shape where running out of budget mid-way still leaves most of the
    answer intact: the model emitted N whole items and was cut off inside the (N+1)th, and
    discarding all N because the last one is half-written throws away work that is complete and
    checkable. MEASURED on the real AEGCL corrigendum — 18 of 38 Phase-9 map calls returned a
    reply cut off mid-object at the token budget, and every one of them carried usable items.

    The salvage is deliberately conservative: it closes the array after the last element whose
    braces balanced, so a half-written object is dropped whole rather than repaired, and a reply
    with no complete element at all still returns None."""
    text = _FENCE_RE.sub("", reply.strip())
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    last_complete = -1  # end of the last top-level object, for the truncation salvage below
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1:  # back out to the array level: one whole element just closed
                last_complete = i
    return f"{text[start : last_complete + 1]}]" if last_complete > start else None
