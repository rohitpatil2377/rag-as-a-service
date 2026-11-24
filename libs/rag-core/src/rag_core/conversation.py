"""Conversational follow-up condensation (FR-I1) — pure logic, no framework.

A multi-turn chat turn like *"what about for premium tenants?"* is meaningless to a
retriever on its own — the subject lives in the previous turns. Before retrieval, a
follow-up is **condensed** into a standalone query that carries the full intent, so the
existing grounded-or-abstain pipeline sees a complete question.

Two backends behind one seam (mirrors faithfulness / router / transform):

* :func:`heuristic_condense` — deterministic, zero-latency. Detects whether a turn is a
  follow-up (dangling pronoun, follow-up opener, or very short) and, if so, folds in the
  most recent user turn. Good enough to lift retrieval recall on the common cases and the
  only sane choice for the CPU dev tier.
* the LLM backend (in the control plane, using the tenant's own gateway) rewrites the turn
  properly — orchestrated where the gateway lives; this module supplies its prompt.

The condenser never invents facts: it only rewrites the *query*, and the answer that comes
back is still grounded-or-abstain against real sources.
"""

from __future__ import annotations

import re

_MAX_FOLLOWUP_WORDS = 5  # a very short turn is almost always a follow-up

# --- small talk -------------------------------------------------------------------
# A greeting or pleasantry is NOT a knowledge question. Without this guard the condenser
# treats "hi" as a short follow-up, folds in the previous turn ("What does the document say
# about CAG — hi") and the platform answers a question the user never asked. Detected
# deterministically (no model) and answered conversationally, without retrieval.
# Vocabulary of pleasantry words. A turn is small talk only when EVERY word is drawn from
# this set — so "hello good morning" and "thanks a lot" match, while "good leave policy"
# (which carries a real subject) does not.
_SMALLTALK_WORDS = frozenset(
    {
        # greetings
        "hi",
        "hii",
        "hiii",
        "hiiii",
        "hey",
        "heya",
        "hello",
        "helo",
        "hellow",
        "yo",
        "namaste",
        "namaskar",
        "hola",
        "greetings",
        # time-of-day
        "good",
        "morning",
        "afternoon",
        "evening",
        "day",
        "night",
        "gm",
        "gn",
        # gratitude / acknowledgement
        "thanks",
        "thank",
        "thankyou",
        "thx",
        "ty",
        "cheers",
        "welcome",
        "ok",
        "okay",
        "k",
        "kk",
        "cool",
        "nice",
        "great",
        "awesome",
        "perfect",
        "got",
        "it",
        "sure",
        # sign-offs
        "bye",
        "goodbye",
        "see",
        "ya",
        "later",
        "regards",
        # polite filler / address
        "there",
        "team",
        "all",
        "sir",
        "madam",
        "again",
        "buddy",
        "friend",
        "dear",
        "a",
        "lot",
        "much",
        "you",
        "u",
        "how",
        "are",
        "is",
        "your",
        "my",
        "im",
        "i",
        "am",
        "fine",
        "please",
        "very",
        "so",
        "and",
        "for",
        "that",
        "this",
        # probing the assistant
        "test",
        "testing",
        "ping",
        "hmm",
        "hmmm",
    }
)
_SMALLTALK_MAX_WORDS = 6
_PUNCT_RE = re.compile(r"[^\w\s]+")


def is_smalltalk(text: str) -> bool:
    """True for greetings/pleasantries that carry no question — answer these conversationally
    instead of retrieving. Conservative by construction: a turn qualifies only when it is
    short AND every single word is a pleasantry, so any real subject word ("leave", "policy")
    or a question mark disqualifies it."""
    if "?" in text:
        return False
    words = _PUNCT_RE.sub(" ", text.strip().lower()).split()
    if not words or len(words) > _SMALLTALK_MAX_WORDS:
        return False
    return all(w in _SMALLTALK_WORDS for w in words)


# Openers that signal the turn depends on prior context.
_FOLLOWUP_OPENERS = (
    "what about",
    "how about",
    "and ",
    "what else",
    "tell me more",
    "why",
    "why not",
    "and why",
    "what if",
    "who",
    "when",
    "where",
    "which one",
    "the same",
    "does it",
    "do they",
    "can it",
    "is it",
    "are they",
)

# Standalone pronouns/determiners that, appearing early, point at an earlier subject.
_DANGLING_REFERENCES = frozenset(
    {"it", "its", "it's", "they", "them", "their", "that", "those", "these", "this", "he", "she", "his", "her"}
)


# Words that describe how an answer should LOOK, never what it should be about. A turn built
# only from these is asking for the previous answer in a different shape.
#
# Reported live: after a table of ten SOPs, "not in table formmat but in bullet point" abstained.
# Standing alone that is correct -- it names no subject, and retrieval scored it 0.0013. It was
# never meant to stand alone. Folding the earlier question back in gives retrieval its subject and
# the model re-answers in the requested shape.
#
# All-or-nothing by construction: EVERY significant word must appear here, so "bullet points for
# the leave policy" keeps its own subject and is answered as the new question it is. Misspellings
# people actually type are included -- "formmat" is what prompted this.
_FORMATTING_WORDS = frozenset(
    {
        "table",
        "tables",
        "tabular",
        "grid",
        "bullet",
        "bullets",
        "point",
        "points",
        "pointwise",
        "bulleted",
        "list",
        "listed",
        "numbered",
        "number",
        "order",
        "ordered",
        "steps",
        "format",
        "formmat",
        "formatting",
        "form",
        "style",
        "shape",
        "layout",
        "paragraph",
        "paragraphs",
        "prose",
        "text",
        "sentence",
        "sentences",
        "short",
        "shorter",
        "brief",
        "briefly",
        "concise",
        "concisely",
        "long",
        "longer",
        "detail",
        "detailed",
        "expand",
        "expanded",
        "instead",
        "rather",
        "make",
        "give",
        "show",
        "put",
        "rewrite",
        "redo",
        "again",
        "markdown",
        "plain",
        "simple",
        "simpler",
        # Function and politeness words that survive _significant() but name no subject. Without
        # them "not in table formmat but in bullet point" fails the all-or-nothing test on "not"
        # and "but" alone -- which is exactly the turn that prompted this rule.
        "not",
        "but",
        "and",
        "with",
        "without",
        "into",
        "just",
        "only",
        "also",
        "please",
        "kindly",
        "want",
        "need",
        "prefer",
        "would",
        "could",
        "can",
        "previous",
        "above",
        "same",
        "answer",
        "output",
        "response",
    }
)


def is_formatting_turn(question: str) -> bool:
    """True when the turn asks only for a different PRESENTATION of the previous answer."""
    words = _significant(question)
    return bool(words) and words <= _FORMATTING_WORDS


def is_follow_up(question: str) -> bool:
    """True when ``question`` cannot stand on its own — it references earlier context."""
    q = question.strip().lower()
    if not q:
        return False
    words = q.split()
    # A dangling reference at the start ("what about it", "do they …") depends on prior context.
    if words[0].strip("?.,") in _DANGLING_REFERENCES:
        return True
    if any(q.startswith(op) for op in _FOLLOWUP_OPENERS):
        return True
    # A presentation-only turn ("in bullet points", "not in table format") refers to the previous
    # ANSWER; it carries no subject of its own, so it can only be understood against it.
    if is_formatting_turn(q):
        return True
    # Short AND without its own subject (≤1 significant content word) → a follow-up like
    # "the amount?" or "why?". A short question that DOES carry a subject ("torch bearer
    # eligibility") stands on its own and must NOT be folded into the previous turn.
    return len(words) <= _MAX_FOLLOWUP_WORDS and len(_significant(q)) <= 1


_STOPWORDS = frozenset(
    {"the", "a", "an", "and", "or", "of", "for", "to", "is", "are", "what", "which", "one", "about", "how", "why"}
)
_MIN_SIGNIFICANT_LEN = 2
_MIN_OWN_SUBJECT_WORDS = 2  # a follow-up with this many of its own topic words is a new topic


def _significant(text: str) -> set[str]:
    words = (w.strip("?.,").lower() for w in text.split())
    return {w for w in words if len(w) > _MIN_SIGNIFICANT_LEN and w not in _STOPWORDS}


def heuristic_condense(history_user_turns: list[str], question: str) -> str:
    """Fold the most recent user turn into a follow-up so retrieval sees the full intent.

    ``history_user_turns`` is prior *user* questions, oldest→newest. Returns ``question``
    unchanged when there is no history or the turn already stands on its own."""
    if not history_user_turns or not is_follow_up(question):
        return question.strip()
    # The last REAL question, not merely the last turn. A presentation-only turn carries no
    # subject, so folding one into another ("in bullet points" -> "make it shorter") produces a
    # pair that still has nothing to retrieve -- measured live as an abstain on the second one.
    prior = [t for t in history_user_turns if not is_formatting_turn(t)]
    if not prior:
        return question.strip()
    previous = prior[-1].strip().rstrip("?.")
    q = question.strip()
    q_words = _significant(q)
    prev_words = _significant(previous)
    # Don't fold if the follow-up already restates the previous subject (its meaningful
    # words are all present in the previous turn) — that would just duplicate it.
    if previous.lower() in q.lower() or (q_words and q_words <= prev_words):
        return q
    # Topic switch: the follow-up carries its own subject (≥2 significant words) that shares
    # nothing with the previous turn → keep it standalone, never merge two unrelated topics.
    #
    # A presentation-only turn is exempt: it shares nothing with the previous turn BY DEFINITION
    # (it names a shape, not a subject), so the topic-switch test reads it as a new topic and
    # hands retrieval a question with nothing to retrieve.
    if len(q_words) >= _MIN_OWN_SUBJECT_WORDS and not (q_words & prev_words) and not is_formatting_turn(q):
        return q
    return f"{previous} — {q}"


def build_condense_prompt(history_user_turns: list[str], question: str) -> str:
    """Prompt for the LLM condensation backend: rewrite the latest turn as standalone.

    Kept deliberately tight and grounded — the model rewrites the QUERY only; it must not
    answer, and it must not add facts that are not in the conversation."""
    recent = "\n".join(f"- {t}" for t in history_user_turns[-6:])
    return (
        "Rewrite the FOLLOW-UP question as a single, self-contained question that includes "
        "the subject from the earlier questions, so it can be understood on its own. "
        "Do NOT answer it. Do NOT add any information not present in the questions. "
        "Return only the rewritten question.\n\n"
        f"Earlier questions:\n{recent}\n\n"
        f"Follow-up: {question}\n\n"
        "Self-contained question:"
    )


def default_title(question: str, *, limit: int = 60) -> str:
    """A readable conversation title from its first question."""
    t = " ".join(question.strip().split())
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"
