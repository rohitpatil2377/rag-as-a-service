"""Lexical anchoring — is a sentence drawn from ONE retrieved passage? Pure logic, no I/O.

NLI answers "does this sentence conflict with the sources?". A sentence recalled from the
model's own memory conflicts with nothing, so it passes: live, a true-but-unrelated sentence
scored 1.0 citing a source that never mentioned it. The anchor is the complementary check — a
sentence genuinely drawn from the sources shares their vocabulary, one recalled does not.

The first anchor counted a sentence's content words found in its best passage. Four defects,
each fixed here:

1. **Every word weighed the same.** "Acme", "policy" and "employees" occur in every passage of an
   HR corpus, so a fabricated sentence built from them anchored as well as one quoting "Rs.200".
   Words are now weighted by inverse document frequency OVER THE RETRIEVED EVIDENCE: a word in
   every passage says little about which passage a sentence came from; a word in none of them is
   the strongest evidence it came from none. Computed from the spans in hand — no corpus lookup,
   no extra round trip, and it answers the question actually being asked (anchored in THIS
   evidence), not how rare a word is in the knowledge base at large.
2. **Inflections did not match.** "approves" never met "approved"; measured live on the
   compound-question gate, that alone turned a supported answer into an abstention. Shared-stem
   matching, the same rule that gate uses, with no stemmer dependency.
3. **It was blind outside ASCII.** Words were ``[A-Za-z0-9]+``, so a Hindi or Tamil sentence had no
   words at all and anchored at 1.0 — the check was silently off for every Indian-language answer.
   Words are now Unicode, and compared only within the scripts the evidence contains: a Hindi
   answer drawn from English sources shares no words with them by construction, and punishing
   that would reject every correct cross-lingual answer. Where no script is shared the anchor
   cannot judge, says so (1.0), and the multilingual NLI check carries the sentence alone.
4. **Longer passages won.** A long span holds more distinct words, so it is likelier to contain a
   sentence's words by chance. The normalization is pivoted on the evidence's own average length
   and only ever PENALISES a longer-than-average span — it never rewards a short one, because
   measured live a short, dense table-of-contents chunk is exactly what a symmetric normalization
   picks. Strength ``b`` is operator-tunable; 0 turns it off.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Collection, Sequence
from os.path import commonprefix

# Letters and digits in ANY script; underscore is a word character to `\w` but not to a reader.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
# Latin tokens this short are function words or noise ("of", "to", "an"); other scripts pack a
# content word into fewer code points, so they keep two.
_MIN_LATIN_LEN = 3
_MIN_OTHER_LEN = 2
# Characters of shared prefix before two surface forms count as the same word. Matches
# rag_clients.nli_faithfulness._MIN_SHARED_STEM so both lexical checks agree on what a word is.
DEFAULT_MIN_SHARED_STEM = 5


def script_of(token: str) -> str:
    """The Unicode script of a token's first letter ("LATIN", "DEVANAGARI", ...), "DIGIT" for numbers."""
    for ch in token:
        if ch.isdigit():
            continue
        name = unicodedata.name(ch, "")
        return name.split(" ", 1)[0] if name else "UNKNOWN"
    return "DIGIT"


def content_terms(text: str, *, stopwords: Collection[str] = ()) -> list[str]:
    """Casefolded content words, in order, in any script."""
    out: list[str] = []
    # Fold notation for COMPARISON only. Ingestion now PRESERVES subscripts and superscripts in
    # stored text (H₂SO₄ stays H₂SO₄, and 4FeO₄²⁻ keeps its charge), but a generator answering
    # about it writes "H2SO4" as often as not, and a reader types it that way too. Without this
    # fold "so₄" and "so4" are different tokens, the sentence looks unanchored in every span, and
    # a correct answer about the formula scores 0.0 and abstains. The displayed text is untouched.
    for raw in _WORD.findall(unicodedata.normalize("NFKC", text)):
        tok = raw.casefold()
        if tok in stopwords:
            continue
        minimum = _MIN_LATIN_LEN if script_of(tok) in ("LATIN", "DIGIT") else _MIN_OTHER_LEN
        if len(tok) >= minimum:
            out.append(tok)
    return out


# A shorter term that IS the whole stem of the longer one ("atom"/"atoms", "bond"/"bonded") is
# the same word even below ``min_stem`` — a four-letter noun can never reach a five-character
# shared prefix, so the plural of every short word read as a different word. Kept in step with
# rag_clients.nli_faithfulness._MIN_WHOLE_STEM so both lexical checks agree on what a word is.
MIN_WHOLE_STEM = 4


def same_term(left: str, right: str, *, min_stem: int = DEFAULT_MIN_SHARED_STEM) -> bool:
    """Equal, or sharing a long enough stem that only an ending separates them."""
    if left == right:
        return True
    shared = len(commonprefix([left, right]))
    if shared >= min_stem:
        return True
    return shared >= MIN_WHOLE_STEM and shared == min(len(left), len(right))


def _matches(term: str, vocabulary: set[str], min_stem: int) -> bool:
    return term in vocabulary or any(same_term(term, v, min_stem=min_stem) for v in vocabulary)


def idf_weights(
    terms: Collection[str], evidence: Sequence[set[str]], *, min_stem: int = DEFAULT_MIN_SHARED_STEM
) -> dict[str, float]:
    """Smoothed inverse document frequency of each term across the evidence passages.

    ``log((N + 1) / (df + 1)) + 1``: always >= 1, so a word present in every passage still counts
    for something, and a word present in none weighs the most. With a single passage every present
    word weighs 1.0 and every absent one ~1.69 — the check degrades to the unweighted one, plus a
    penalty for words the evidence does not contain, rather than to nonsense."""
    n = len(evidence)
    return {
        term: math.log((n + 1) / (sum(1 for vocab in evidence if _matches(term, vocab, min_stem)) + 1)) + 1.0
        for term in set(terms)
    }


def anchor_score(
    sentence: str,
    passages: Sequence[str],
    *,
    stopwords: Collection[str] = (),
    length_norm: float = 0.0,
    min_stem: int = DEFAULT_MIN_SHARED_STEM,
) -> float:
    """How strongly ``sentence`` is anchored in its best single passage, in [0, 1].

    IDF-weighted share of the sentence's words that one passage contains, divided by that
    passage's length penalty. 1.0 when there is nothing the check can judge: no content words, no
    evidence, or no script in common with the evidence (see module docstring, point 3)."""
    tokenised = [content_terms(p, stopwords=stopwords) for p in passages]
    evidence = [set(t) for t in tokenised if t]
    if not evidence:
        return 1.0
    evidence_scripts = {script_of(t) for vocab in evidence for t in vocab} - {"DIGIT"}
    want = {t for t in content_terms(sentence, stopwords=stopwords) if script_of(t) in evidence_scripts or t.isdigit()}
    if not {t for t in want if not t.isdigit()}:
        return 1.0
    weights = idf_weights(want, evidence, min_stem=min_stem)
    total = sum(weights.values())
    lengths = [len(t) for t in tokenised if t]
    average = sum(lengths) / len(lengths)
    best = 0.0
    for vocab, length in zip(evidence, lengths, strict=True):
        covered = sum(w for term, w in weights.items() if _matches(term, vocab, min_stem))
        penalty = max(1.0, 1.0 - length_norm + length_norm * (length / average)) if length_norm > 0 else 1.0
        best = max(best, (covered / total) / penalty)
    return best
