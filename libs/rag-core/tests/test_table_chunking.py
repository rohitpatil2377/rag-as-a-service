"""A table must survive chunking as something a reader can still interpret (FR-C1).

Measured on a real pharmacy SOP: the font table and the department-code table were both split at
the character cap, and the second fragment began mid-table with no header row --

    chunk A: "5.6.1 FONTS: | Content of SOP | Size of Font | | --- | --- | | Header |..."
    chunk B: "| 12 Bold in Upper Case | | Write up of SOP | 12 in Sentence Case |..."

Retrieval then behaved in a way that is worse than a clean failure. Asked to LIST the department
codes, the platform answered correctly, header and all. Asked the narrow question -- "what is the
department code for Quality Control?" -- it ABSTAINED, because the fragment it retrieved held a
bare "| Q |" that supports no such statement. The abstain is correct for what the model was
shown; the chunking is what was wrong. Narrow lookups are the common case for SOP search.

What these pin is the behaviour that HOLDS today: a table survives chunking whole, every row
reaches the index, and a narrow lookup finds its row and its header in one chunk.

The real defect behind the abstain above turned out NOT to be here -- the chunk was intact and
retrieval scored it 0.87. It was the faithfulness premise, which cut the table mid-way and handed
the model a header with no answer under it (see test_nli_faithfulness). A table CAN still be split
when it exceeds the cap, and repeating the header on each fragment remains worth doing; it needs a
change to how the chunker re-accumulates pieces, and a golden set to show it helps rather than
trades one failure for another.
"""

from __future__ import annotations

from datetime import UTC, datetime

from rag_core.chunking import ClauseChunker
from rag_schemas import DocumentEnvelope
from rag_schemas.common import SourceKind

_HEADER = "| Department | Department Code |"
_SEP = "| --- | --- |"
_ROWS = [
    "| Mfg. & Packing | B |",
    "| Capsules | C |",
    "| Liquid Orals | L |",
    "| Packing | P |",
    "| Quality Control | Q |",
    "| Tablets | T |",
    "| Engineering | E |",
    "| Stores | S |",
    "| Human Resources | H |",
    "| Quality Assurance | A |",
]


def _doc() -> DocumentEnvelope:
    return DocumentEnvelope(
        tenant_id="t",
        doc_id="d",
        source_id="s",
        source_kind=SourceKind.UPLOAD,
        mime="text/plain",
        ingested_at=datetime(2026, 9, 18, tzinfo=UTC),
    )


def _table(rows: list[str]) -> str:
    return "\n".join([_HEADER, _SEP, *rows])


def _chunks(text: str, max_chars: int) -> list[str]:
    return [c.text for c in ClauseChunker(max_chars=max_chars).chunk(_doc(), text)]


def test_a_table_that_fits_is_never_split() -> None:
    text = f"5.7 Department codes are as follows:\n{_table(_ROWS[:4])}\nEnd of section."
    holding = [c for c in _chunks(text, 900) if "Quality" in c or "Capsules" in c]
    assert holding, "the table vanished"
    for c in holding:
        assert _HEADER in c, "a fragment lost the header row"


def test_every_row_survives_the_split() -> None:
    text = f"Codes:\n{_table(_ROWS)}\n"
    joined = " ".join(_chunks(text, 260))
    for row in _ROWS:
        cell = row.split("|")[1].strip()
        assert cell in joined, f"row lost entirely: {cell}"


def test_a_narrow_lookup_has_its_row_and_header_in_one_chunk() -> None:
    """The failure this exists to prevent: 'Quality Control' and 'Q' must appear in the SAME
    chunk as the header, or the faithfulness gate cannot support an answer from it."""
    text = f"Codes:\n{_table(_ROWS)}\n"
    holding = [c for c in _chunks(text, 260) if "Quality Control" in c]
    assert holding, "no chunk holds the row at all"
    assert any(_HEADER in c and "| Q |" in c for c in holding), "the row is separated from its header"


def test_prose_around_a_table_is_unaffected() -> None:
    text = (
        "1.0 OBJECTIVE. To lay down the procedure for numbering. "
        "2.0 SCOPE. This applies to all SOPs written at the site.\n"
        f"{_table(_ROWS[:3])}\n"
        "6.0 REFERENCES. None applicable to this procedure."
    )
    joined = " ".join(_chunks(text, 400))
    assert "1.0 OBJECTIVE" in joined and "6.0 REFERENCES" in joined


def test_a_pipe_in_prose_is_not_a_table() -> None:
    """A stray pipe must not turn ordinary prose into a table region and change its chunking."""
    text = "The value is A | B in the legacy notation. " * 12
    chunks = _chunks(text, 300)
    assert len(chunks) > 1, "prose should still chunk normally"
    assert all("---" not in c for c in chunks)
