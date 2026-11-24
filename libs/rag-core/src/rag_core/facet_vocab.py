"""A document's facet vocabulary, pinned to a revision so a re-ingest cannot re-roll it.

WHY A FACET ID IS NOT DESCRIPTIVE METADATA. It is load-bearing twice over:

  * it is the JOIN KEY of Phase 8's facet union (``select_sections`` step 4 unions in every
    section whose ``facet_ids`` intersect the router's facets), and
  * de-underscored it IS the cross-encoder probe text (``pipeline._section_probes`` sends
    ``facet.replace("_", " ")``), which decides the score stage.

So two labels that mean the same thing are not the same thing to this system.
``performance_and_scalability`` and ``performance_and_scalability_metrics`` join nothing to each
other and score differently.

WHAT WAS MEASURED (2026-11-02, `corrigendum-5.pdf`, three ingests of identical bytes). Phase 3
invents this vocabulary per call, so the three runs produced three 10-facet vocabularies sharing
ONE id, with pairwise Jaccard 0.053-0.250. Downstream, six of the eight sections that lost their
score admission had been carried by a single probe, ``hardware and infrastructure specifications``,
which two of the three runs never re-invented. `replace_document_profile`'s own docstring had
already noticed the ids churn -- "the ids a second run produces are NOT the ids the first
produced" -- and chose to replace the rows wholesale. That keeps one document's taxonomy
self-consistent and still re-rolls selection on every re-ingest. This pins it instead.

THE REVISION IS THE BYTES PLUS THE SCHEMA. ``content_hash`` because a vocabulary describes a
document and must not outlive the text it described; ``profile_version`` because a changed profile
SCHEMA can change what a facet means even when the bytes are identical. Same discipline as
``EnrichmentManifest.content_hash``.

CHANGED BYTES DO NOT INHERIT. A new revision takes the new profile call's taxonomy, because
carrying facets forward onto an amended document would describe text that may no longer be there.
What :func:`reconcile` provides for that case is a REPORT of what changed, so the change is
visible rather than silent.

NO FUZZY MATCHING, EVER, AT RUNTIME. :meth:`FacetVocabulary.resolve` accepts an exact id or an
APPROVED alias and nothing else. Similarity-based joining would replace a nondeterministic model
label with a nondeterministic similarity threshold -- a lateral move that is harder to audit,
because a wrong join is then a float away rather than a row away. Unmapped proposals are reported
as CANDIDATES for a controlled update and are never adopted on their own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Bumped when the MEANING of a stored vocabulary changes, so a reader can refuse one it cannot
#: interpret rather than misread it. Not bumped for a new optional field.
VOCAB_VERSION = 1


def document_revision(content_hash: str, profile_version: int) -> str:
    """The revision a vocabulary is pinned to: ``<content_hash>/p<profile_version>``.

    Empty when there is no content hash. A vocabulary that cannot say which bytes it describes is
    not evidence about the document in front of you, and :meth:`FacetVocabulary.matches` refuses
    an empty revision for that reason."""
    return f"{content_hash}/p{profile_version}" if content_hash else ""


@dataclass(frozen=True, slots=True)
class CanonicalFacet:
    """One approved facet: a stable id, the canonical label, and the aliases approved for it.

    ``facet_id`` is the identity and the probe text (de-underscored). ``name`` and ``description``
    are the human-readable canonical forms -- they are stored and shown, and deliberately not used
    for joining or probing, so a cosmetic relabel cannot move selection.

    ``aliases`` are ids a PREVIOUS or a proposing profile call used for this same facet, approved
    explicitly. They are matched exactly, never fuzzily."""

    facet_id: str
    name: str = ""
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class FacetVocabulary:
    """The approved vocabulary for one document revision."""

    revision: str = ""
    facets: tuple[CanonicalFacet, ...] = field(default_factory=tuple)
    vocab_version: int = VOCAB_VERSION

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(f.facet_id for f in self.facets)

    def matches(self, revision: str) -> bool:
        """True when this vocabulary describes the revision the caller holds."""
        return bool(self.revision) and self.revision == revision

    def resolve(self, proposed_id: str) -> str | None:
        """The canonical id for a proposed one: itself if canonical, else an APPROVED alias's
        owner, else None. Exact matching only, case- and separator-normalised.

        Normalisation is deliberately narrow -- case and the ``-``/space-to-underscore spelling of
        the SAME token sequence. ``Performance_And_Scalability`` and ``performance and scalability``
        are the same id written differently; ``performance_and_scalability_metrics`` is a different
        id and stays unmapped until someone approves it as an alias."""
        want = _norm(proposed_id)
        if not want:
            return None
        for facet in self.facets:
            if _norm(facet.facet_id) == want:
                return facet.facet_id
        for facet in self.facets:
            if any(_norm(a) == want for a in facet.aliases):
                return facet.facet_id
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_version": self.vocab_version,
            "revision": self.revision,
            "facets": [
                {"facet_id": f.facet_id, "name": f.name, "description": f.description, "aliases": list(f.aliases)}
                for f in self.facets
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> FacetVocabulary | None:
        """A stored vocabulary, or None when there isn't a readable one.

        None for absent, for a non-mapping, for a version this code does not understand, and for
        one with no facets -- NOT an empty vocabulary, which would read as "this document has no
        facets" and silence Phase 8's facet union for it."""
        if not isinstance(data, Mapping):
            return None
        if data.get("vocab_version") != VOCAB_VERSION:
            return None
        raw = data.get("facets")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
            return None
        facets: list[CanonicalFacet] = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            fid = str(item.get("facet_id") or "").strip()
            if not fid:
                continue
            aliases = item.get("aliases")
            facets.append(
                CanonicalFacet(
                    facet_id=fid,
                    name=str(item.get("name") or ""),
                    description=str(item.get("description") or ""),
                    aliases=tuple(str(a) for a in aliases) if isinstance(aliases, (list, tuple)) else (),
                )
            )
        if not facets:
            return None
        return cls(revision=str(data.get("revision") or ""), facets=tuple(facets))


@dataclass(frozen=True, slots=True)
class VocabularyDecision:
    """What was done with a proposed taxonomy, and why — the record that makes this auditable.

    ``reused`` False with a non-empty ``stored_revision`` is the interesting case: a vocabulary
    existed and described OTHER bytes, so the new proposal was adopted rather than carried
    forward."""

    #: The vocabulary to use for this ingest. Never empty-facetted unless the proposal was.
    vocabulary: FacetVocabulary
    reused: bool = False
    stored_revision: str = ""
    #: Proposed ids that resolved onto a canonical id (proposed -> canonical), alias hits included.
    mapped: Mapping[str, str] = field(default_factory=dict)
    #: Proposed ids with no canonical id and no approved alias. CANDIDATES for a controlled
    #: vocabulary update -- reported, never adopted. Phase 4 cannot assign them, which is the
    #: point: an unapproved label must not reach a join key or a probe.
    unmapped: tuple[str, ...] = field(default_factory=tuple)
    #: Canonical ids the proposal did not mention. Not an error -- Phase 4 decides which sections
    #: a label describes, and a facet no section carries simply selects nothing.
    unproposed: tuple[str, ...] = field(default_factory=tuple)


def reconcile(
    stored: FacetVocabulary | None, proposed: Sequence[CanonicalFacet], *, revision: str
) -> VocabularyDecision:
    """Decide this ingest's vocabulary from what is stored and what Phase 3 just proposed.

    Three cases, and only one of them reuses:

      * the stored vocabulary describes THIS revision -> REUSE it, and map the proposal onto it so
        the mapping (and anything unmapped) is reported;
      * a stored vocabulary describes other bytes, or there is none -> ADOPT the proposal as this
        revision's vocabulary. An amended document gets the taxonomy of the amended text;
      * nothing stored and nothing proposed -> an empty vocabulary, which is what a document whose
        profile call failed has. The caller already treats a profile-less document as having no
        facet vocabulary, so this changes nothing for it.
    """
    if stored is not None and stored.matches(revision):
        mapped: dict[str, str] = {}
        unmapped: list[str] = []
        for facet in proposed:
            canonical = stored.resolve(facet.facet_id)
            if canonical is None:
                unmapped.append(facet.facet_id)
            else:
                mapped[facet.facet_id] = canonical
        return VocabularyDecision(
            vocabulary=stored,
            reused=True,
            stored_revision=stored.revision,
            mapped=mapped,
            unmapped=tuple(unmapped),
            unproposed=tuple(f for f in stored.ids if f not in set(mapped.values())),
        )
    return VocabularyDecision(
        vocabulary=FacetVocabulary(revision=revision, facets=tuple(proposed)),
        reused=False,
        stored_revision=stored.revision if stored is not None else "",
    )


def _norm(value: str) -> str:
    """Lowercase, and the same token sequence spelled with spaces, hyphens or underscores."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")
