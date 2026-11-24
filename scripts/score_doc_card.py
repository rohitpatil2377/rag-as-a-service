#!/usr/bin/env python3
"""Score a Phase-3 profile card and Phase-4 section enrichment against a gold card.

WHY THIS IS A SEPARATE SCRIPT from ``eval_gate.py``. That gate scores ANSWERS: it asks a live KB
a question and grades the reply. Phases 3 and 4 produce neither a question nor an answer — they
produce an artifact about the document, built once at ingest. Their acceptance is stated in the
plan as prose about that artifact, which is why it sat unscored while every other phase had a
number: there was nothing to run it with.

Both inputs are what the pipeline already emits, so this scores the real thing:

  --card      the Phase-3 profile, as ``DocProfile.as_json()`` writes it
              (``documents.profile_json``)
  --sections  the Phase-4 enriched sections, a JSON list of objects carrying at least
              ``text`` and whichever of ``summary``/``key_values``/``amends``/``propositions``
              the enricher produced

QUOTE VALIDATION IS THE POINT OF THE PHASE-4 HALF. The plan's §4 rule is "discard any item whose
quote is not a substring (after whitespace normalisation) of the section text", so the share of
quotes that validate measures directly whether the enricher is grounded or inventing. Normalised
the same way the pipeline normalises, because a quote that differs only in whitespace is the
extractor's formatting, not a fabrication.

    scripts/score_doc_card.py --card profile.json --sections sections.json \
        --gold eval/gold/aegcl_doc_card.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

_WS = re.compile(r"\s+")
_EXIT_FAIL = 1
_MAX_OFFENDERS_SHOWN = 5  # enough to see the pattern, not a dump of every bad quote


def _norm(text: str) -> str:
    """NFKC-folded, whitespace-collapsed, case-insensitive — the comparison the plan implies."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().lower()


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _quotes_of(section: dict[str, Any]) -> list[str]:
    """Every verbatim quote the enricher attached to this section, from all three item kinds."""
    out: list[str] = []
    for kv in section.get("key_values") or []:
        if isinstance(kv, dict) and kv.get("quote"):
            out.append(str(kv["quote"]))
    amends = section.get("amends")
    if isinstance(amends, dict) and amends.get("quote"):
        out.append(str(amends["quote"]))
    for prop in section.get("propositions") or []:
        if isinstance(prop, dict) and prop.get("quote"):
            out.append(str(prop["quote"]))
    return out


def _score_profile(card: dict[str, Any], gold: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(passes, failures) for the Phase-3 half."""
    ok: list[str] = []
    bad: list[str] = []
    want_type = gold["doc_type"]
    got_type = str(card.get("doc_type", ""))
    (ok if _norm(want_type) in _norm(got_type) else bad).append(f"doc_type: wanted ~{want_type!r}, got {got_type!r}")

    rel_spec = gold["requires_relation"]
    relations = card.get("relations") or []
    typed = [r for r in relations if _norm(str(r.get("type", ""))) == _norm(rel_spec["type"])]
    hit = [r for r in typed if any(_contains(str(r.get("target_ref", "")), t) for t in rel_spec["target_any_of"])]
    (ok if hit else bad).append(
        f"relation {rel_spec['type']!r} -> one of {rel_spec['target_any_of']}: "
        f"{len(typed)} of that type, {len(hit)} on target"
    )

    quoted = [r for r in relations if str(r.get("evidence_quote", "")).strip()]
    rate = len(quoted) / len(relations) if relations else 1.0
    (ok if rate >= gold["min_relation_quote_rate"] else bad).append(
        f"relation evidence_quote rate: {rate:.2f} (need {gold['min_relation_quote_rate']:.2f})"
    )

    # `DocProfile.as_json()` writes the facet list under `facet_taxonomy`; `facets` is the
    # in-memory attribute name. Accept either, so this scores a stored card and a hand-built one
    # the same way rather than silently seeing zero facets.
    raw_facets = card.get("facet_taxonomy") or card.get("facets") or []
    facet_ids = [_norm(str(f.get("facet_id", "") if isinstance(f, dict) else f)) for f in raw_facets]
    covered, missing = [], []
    for theme in gold["facet_themes"]:
        if any(any(_norm(alt) in fid for fid in facet_ids) for alt in theme["any_of"]):
            covered.append(theme["name"])
        else:
            missing.append(theme["name"])
    frac = len(covered) / len(gold["facet_themes"])
    (ok if frac >= gold["min_theme_coverage"] else bad).append(
        f"facet theme coverage: {len(covered)}/{len(gold['facet_themes'])} = {frac:.2f} "
        f"(need {gold['min_theme_coverage']:.2f}); missing {missing or 'none'}"
    )
    return ok, bad


def _score_sections(sections: list[dict[str, Any]], gold: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(passes, failures) for the Phase-4 half."""
    ok: list[str] = []
    bad: list[str] = []

    for case in gold.get("amends_cases") or []:
        found = [s for s in sections if _contains(str(s.get("text", "")), case["locate_by"])]
        if not found:
            bad.append(f"amends {case['locate_by']!r}: no section contains that text")
            continue
        got = next((s.get("amends") for s in found if isinstance(s.get("amends"), dict)), None)
        if not got:
            bad.append(f"amends {case['locate_by']!r}: section found, but no amends captured")
            continue
        old_ok = _contains(str(got.get("old") or ""), case["old"])
        new_ok = _contains(str(got.get("new") or ""), case["new"])
        (ok if old_ok and new_ok else bad).append(
            f"amends {case['locate_by']!r}: old={got.get('old')!r} (want ~{case['old']!r}, "
            f"{'ok' if old_ok else 'MISS'}), new={got.get('new')!r} (want ~{case['new']!r}, "
            f"{'ok' if new_ok else 'MISS'})"
        )

    for case in gold.get("proposition_cases") or []:
        found = [s for s in sections if _contains(str(s.get("text", "")), case["locate_by"])]
        if not found:
            bad.append(f"propositions {case['locate_by']!r}: no section contains that text")
            continue
        blob = " ".join(
            str(p.get("text", "")) for s in found for p in (s.get("propositions") or []) if isinstance(p, dict)
        )
        missing = [m for m in case["must_mention"] if not _contains(blob, m)]
        (ok if not missing else bad).append(
            f"propositions {case['locate_by']!r}: must mention {case['must_mention']}, missing {missing or 'none'}"
        )

    total = valid = 0
    offenders: list[str] = []
    for s in sections:
        body = str(s.get("text", ""))
        for q in _quotes_of(s):
            total += 1
            if _contains(body, q):
                valid += 1
            elif len(offenders) < _MAX_OFFENDERS_SHOWN:
                offenders.append(q[:70])
    rate = valid / total if total else 1.0
    (ok if rate >= gold["min_quote_validation_rate"] else bad).append(
        f"quote validation: {valid}/{total} = {rate:.3f} "
        f"(need {gold['min_quote_validation_rate']:.2f}){'; e.g. ' + str(offenders) if offenders else ''}"
    )
    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=Path, required=True, help="Phase-3 profile JSON")
    ap.add_argument("--sections", type=Path, required=True, help="Phase-4 enriched sections JSON")
    ap.add_argument("--gold", type=Path, required=True, help="gold card YAML")
    args = ap.parse_args()

    import yaml

    gold = yaml.safe_load(args.gold.read_text())
    card = json.loads(args.card.read_text())
    sections = json.loads(args.sections.read_text())
    if isinstance(sections, dict):
        sections = sections.get("sections", [])

    print(f"\n  gold   : {args.gold.name}  [doc {gold.get('doc_id')}]")
    p_ok, p_bad = _score_profile(card, gold["profile"])
    s_ok, s_bad = _score_sections(sections, gold["sections"])

    print("\n  PHASE 3 — document profile")
    for line in p_ok:
        print(f"    PASS  {line}")
    for line in p_bad:
        print(f"    FAIL  {line}")
    print("\n  PHASE 4 — section enrichment")
    for line in s_ok:
        print(f"    PASS  {line}")
    for line in s_bad:
        print(f"    FAIL  {line}")

    failed = len(p_bad) + len(s_bad)
    total = failed + len(p_ok) + len(s_ok)
    print(f"\n  {total - failed}/{total} assertions pass\n")
    return _EXIT_FAIL if failed else 0


if __name__ == "__main__":
    sys.exit(main())
