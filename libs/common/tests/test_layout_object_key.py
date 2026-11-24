"""layout_object_key: the layout overlay's key follows raw_object_key's rule, not a parallel one.

A crawl/web doc id is a URL, whose ``:`` and ``//`` are illegal object names. ``raw_object_key``
already encodes those to a deterministic safe key; if the layout key re-implemented that rule it
would diverge on exactly the ids it exists for, and a crawl document's overlay would be written
or purged under a key nothing else refers to.
"""

from __future__ import annotations

from rag_common.tenancy import layout_object_key, raw_object_key

# The id shapes this platform actually produces: an upload, a path-ish upload, a crawl URL, a
# crawl URL that differs only in its path, and one with nothing object-safe in it at all.
_DOC_IDS = (
    "brief",
    "Acme_Night_Shift.pdf",
    "reports/q1.pdf",
    "http://localhost:8899/latest/",
    "https://docs.example.com/a",
    "https://docs.example.com/b",
    "://:://",
)


def test_an_object_safe_doc_id_passes_through_under_the_layout_prefix() -> None:
    assert layout_object_key("brief") == "layout/brief"
    assert layout_object_key("Acme_Night_Shift.pdf") == "layout/Acme_Night_Shift.pdf"
    assert layout_object_key("reports/q1.pdf") == "layout/reports/q1.pdf"  # single-slash path is fine


def test_a_crawl_url_doc_id_becomes_a_legal_object_name() -> None:
    key = layout_object_key("http://localhost:8899/latest/")
    assert key.startswith("layout/")
    rest = key.removeprefix("layout/")
    assert ":" not in rest and "//" not in rest
    assert layout_object_key("http://localhost:8899/latest/") == key  # deterministic


def test_the_safe_key_encoding_is_the_same_helper_as_the_raw_key() -> None:
    """The contract that keeps a crawl document's artifacts in lockstep: swap the prefix and the
    two keys are identical. A second implementation of the slug+digest rule would pass the two
    tests above and still fail this one."""
    for doc_id in _DOC_IDS:
        assert layout_object_key(doc_id) == "layout/" + raw_object_key(doc_id).removeprefix("raw/")


def test_the_safe_key_rule_itself_is_pinned_to_a_literal() -> None:
    """The equivalence test above would still pass if the slug/digest rule CHANGED — raw and
    layout keys would move in lockstep and orphan every object already stored under the old key,
    with no test objecting. So the rule is pinned to literals: slug = non-[A-Za-z0-9._-] runs
    collapsed to "_" and trimmed, capped at 80 chars, then "-" + the first 12 hex of
    sha1(doc_id). Changing it must fail here and be a deliberate migration decision."""
    assert raw_object_key("http://localhost:8899/latest/") == "raw/http_localhost_8899_latest-271dd83fd21e"
    assert layout_object_key("http://localhost:8899/latest/") == "layout/http_localhost_8899_latest-271dd83fd21e"
    assert raw_object_key("https://docs.example.com/a") == "raw/https_docs.example.com_a-b41719a8bad6"
    assert layout_object_key("https://docs.example.com/a") == "layout/https_docs.example.com_a-b41719a8bad6"


def test_two_different_urls_do_not_collide() -> None:
    a = layout_object_key("https://docs.example.com/a")
    b = layout_object_key("https://docs.example.com/b")
    assert a != b


def test_a_layout_key_is_never_a_raw_key() -> None:
    """Separate prefixes, so an operator listing ``raw/`` never finds overlay JSON in it and the
    overlay can never be served as a document's raw bytes."""
    for doc_id in _DOC_IDS:
        assert layout_object_key(doc_id) != raw_object_key(doc_id)
