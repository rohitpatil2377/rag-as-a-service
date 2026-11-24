"""Answer verification: claim split, value check, enumeration check (RAG v2 Phase 10).

THIS FILE CARRIES THE PHASE'S ACCEPTANCE, and unlike Phases 3 and 4 it can. The plan's Phase-10
acceptance is "on a seeded error set (answers with injected wrong numbers / dropped list items),
the verifier catches >= 90%" — a claim about what the VERIFIER does to a fault somebody injected,
not a claim about what a model extracts from a document nobody has supplied. A seeded error set is
therefore legitimate evidence rather than a stand-in: the faults are the input, and whether they
are caught is a property of the code under test. See ``test_the_seeded_error_set_acceptance``.

AND IT CARRIES THE COMPANION MEASUREMENT THE PLAN DOES NOT ASK FOR, which matters just as much.
A verifier is adopted or abandoned on its FALSE POSITIVE rate, not its catch rate:
``rag_core.numeracy``'s own docstring says it — "a guard that fires on correct answers gets
turned off". So the same fixtures are run in their faithful form and the flag rate on those is
asserted at zero. A catch rate quoted without it would be half a measurement.

The fixture text is composed here and is representative rather than real — tender, SOP and
technical-specification phrasing of the kind this corpus holds. That is fine for THIS acceptance
in a way it was not for Phases 3 and 4: nothing here depends on a model reading a real document.
"""

from __future__ import annotations

import json

from rag_core.verification import (
    VERIFY_VERSION,
    Claim,
    ClaimVerdict,
    VerificationReport,
    enumeration_items,
    extract_values,
    split_claims,
    unverified_footnote,
    verify_claims,
    without_unverified,
)

# --- the seeded error set -------------------------------------------------------------------------
#
# Each case is (kind, quote, faithful_claim, corrupted_claim). The faithful claim must pass and the
# corrupted one must be caught. Kinds mirror the plan's own list: a wrong figure, a swapped unit, a
# swapped standard, a shifted date, a dropped list item, and a figure with no source at all.
_SEEDED: tuple[tuple[str, str, str, str], ...] = (
    (
        "number",
        "The bidder shall have an average annual turnover of Rs. 19.44 Crores.",
        "The required average annual turnover is Rs. 19.44 Crores [1].",
        "The required average annual turnover is Rs. 35 Crores [1].",
    ),
    (
        "unit",
        "Every bidder shall furnish an EMD of Rs. 5 lakh with the technical bid.",
        "The EMD is Rs. 5 lakh [1].",
        "The EMD is Rs. 5 Crores [1].",
    ),
    (
        "standard",
        "The controller shall comply with IEC 62443-4-2 for component security.",
        "The controller must comply with IEC 62443-4-2 [1].",
        "The controller must comply with IEC 62443-3-3 [1].",
    ),
    (
        "standard",
        "The vendor shall be certified to ISO 27001 at the time of bidding.",
        "Certification to ISO 27001 is required [1].",
        "Certification to ISO 27002 is required [1].",
    ),
    (
        "standard",
        "Emission limits shall be as per CISPR 32 Class A.",
        "Emission limits follow CISPR 32 [1].",
        "Emission limits follow CISPR 22 [1].",
    ),
    (
        "percentage",
        "An uptime of 99.5% shall be maintained over each quarter.",
        "The uptime commitment is 99.5% [1].",
        "The uptime commitment is 99.9% [1].",
    ),
    (
        "date",
        "Bids close on 24-09-2026 at 15:00 hours.",
        "Bids close on 24 September 2026 [1].",
        "Bids close on 24-10-2026 [1].",
    ),
    (
        "date",
        "The corrigendum is effective from 01-04-2027.",
        "It takes effect on 1 April 2027 [1].",
        "It takes effect on 01-04-2026 [1].",
    ),
    (
        "dropped_item",
        "The vendor shall supply HBOM and SBOM for every delivered component.",
        "The vendor must supply HBOM and SBOM [1].",
        "The vendor must supply SBOM [1].",
    ),
    (
        "dropped_item",
        "The scope covers the DAM, the NMS and the ITSM modules.",
        "The scope covers the DAM, the NMS and the ITSM modules [1].",
        "The scope covers the DAM and the NMS modules [1].",
    ),
    (
        "dropped_item",
        "Supply shall include video wall, workstation, switches and firewalls.",
        "Supply includes video wall, workstation, switches and firewalls [1].",
        "Supply includes video wall, workstation and switches [1].",
    ),
    (
        "dropped_item",
        "Each rack shall carry the NG-SIEM and the HCI-1 appliance.",
        "Each rack carries the NG-SIEM and the HCI-1 appliance [1].",
        "Each rack carries the NG-SIEM appliance [1].",
    ),
    (
        "quantity",
        "The transformer shall be rated at 33 kV on the primary side.",
        "The primary rating is 33 kV [1].",
        "The primary rating is 11 kV [1].",
    ),
    (
        "quantity",
        "Each workstation shall be supplied with 32 GB of RAM.",
        "Each workstation has 32 GB of RAM [1].",
        "Each workstation has 16 GB of RAM [1].",
    ),
    (
        "quantity",
        "The core uplink shall be 10 Gbps.",
        "The core uplink is 10 Gbps [1].",
        "The core uplink is 40 Gbps [1].",
    ),
    (
        "quantity",
        "Usable storage capacity shall not be less than 100 TB.",
        "Usable capacity is at least 100 TB [1].",
        "Usable capacity is at least 200 TB [1].",
    ),
    (
        "quantity",
        "The display shall provide a brightness of 700 nits.",
        "Brightness is 700 nits [1].",
        "Brightness is 500 nits [1].",
    ),
    (
        "quantity",
        "The NG-SIEM shall be sized for 5000 EPS sustained.",
        "It is sized for 5000 EPS [1].",
        "It is sized for 15000 EPS [1].",
    ),
    (
        "quantity",
        "Equipment shall operate between 0 °C and 50 °C.",
        "The operating range is 0 °C to 50 °C [1].",
        "The operating range is 0 °C to 70 °C [1].",
    ),
    (
        "quantity",
        "Payment shall be released within 30 days of acceptance.",
        "Payment follows within 30 days [1].",
        "Payment follows within 60 days [1].",
    ),
    (
        "unit",
        "The penalty shall not exceed Rs. 35 lakh in aggregate.",
        "The penalty cap is Rs. 35 lakh [1].",
        "The penalty cap is Rs. 35 Crores [1].",
    ),
    (
        "number",
        "The contrast ratio shall be at least 4000:1.",
        "The minimum contrast ratio is 4000:1 [1].",
        "The minimum contrast ratio is 5000:1 [1].",
    ),
    (
        "fabricated",
        "The warranty shall be comprehensive and on-site for the full term.",
        "The warranty is comprehensive and on-site [1].",
        "The warranty runs for 5 years [1].",
    ),
    (
        "fabricated",
        "The equipment shall meet the technical specification in Annexure B.",
        "The equipment meets the technical specification in Annexure B [1].",
        "The equipment meets IEC 61850 [1].",
    ),
)

# --- faithful answers that must NOT be flagged -----------------------------------------------------
#
# Every one of these is a legitimate restatement. A hit on any of them is a false positive, and a
# false positive is what gets a guard switched off — so this list is as much the acceptance as the
# one above. Each case names the restatement habit it is defending.
_FAITHFUL: tuple[tuple[str, str, str], ...] = (
    (
        "thousands separator",
        "The fee payable is Rs. 50,000 per annum.",
        "The annual fee is Rs. 50000 [1].",
    ),
    (
        "trailing decimal zero",
        "The engagement runs for 24.0 months from the effective date.",
        "The engagement runs for 24 months [1].",
    ),
    (
        "plural unit",
        "A turnover of Rs. 19.44 Crore is required.",
        "A turnover of Rs. 19.44 Crores is required [1].",
    ),
    (
        "percent spelled out",
        "Availability shall be 99.5 percent.",
        "Availability shall be 99.5% [1].",
    ),
    (
        "date written in words",
        "The deadline is 24-09-2026.",
        "The deadline is 24th September 2026 [1].",
    ),
    (
        "date written in digits",
        "The deadline is 24 September 2026.",
        "The deadline is 24/09/2026 [1].",
    ),
    (
        "less specific standard",
        "The controller shall comply with IEC 62443-4-2.",
        "The controller must comply with IEC 62443 [1].",
    ),
    (
        "prose joined by and, only one half restated",
        "The bidder shall submit the EMD and the technical bid shall be sealed.",
        "The bidder shall submit the EMD [1].",
    ),
    (
        "a quote's figures the claim simply does not restate",
        "The deposit is refundable within 30 days of award.",
        "The deposit is refundable [1].",
    ),
    (
        "paraphrase keeping every figure",
        "An uptime of 99.5% shall be maintained over each quarter.",
        "Quarterly uptime must reach 99.5% [1].",
    ),
    (
        "markdown bullet with inline emphasis",
        "The core uplink shall be 10 Gbps.",
        "- The **core uplink** is 10 Gbps [1].",
    ),
    (
        "an ordered-list marker is layout, not a claimed figure",
        "The vendor shall supply the HBOM.",
        "1. The vendor shall supply the HBOM [1].",
    ),
    (
        "a sentence-initial capital is not an acronym list",
        "Payment shall be released within 30 days of acceptance.",
        "Payment is released within 30 days [1].",
    ),
    (
        "full list restated in a different order",
        "The scope covers the DAM, the NMS and the ITSM modules.",
        "The scope covers the ITSM, the DAM and the NMS modules [1].",
    ),
)

_ACCEPTANCE_FLOOR = 0.90


def _caught(quote: str, claim: str) -> bool:
    return not verify_claims(claim, [quote]).ok


# --- the plan's acceptance -------------------------------------------------------------------------


def test_the_seeded_error_set_acceptance() -> None:
    """The plan's Phase-10 acceptance: >= 90% of injected faults caught.

    Reported per kind as well as overall, because an aggregate that hides a kind catching nothing
    would pass the threshold while leaving a whole class of error invisible."""
    missed: list[tuple[str, str]] = []
    by_kind: dict[str, list[bool]] = {}
    for kind, quote, _ok, bad in _SEEDED:
        hit = _caught(quote, bad)
        by_kind.setdefault(kind, []).append(hit)
        if not hit:
            missed.append((kind, bad))

    rate = (len(_SEEDED) - len(missed)) / len(_SEEDED)
    assert rate >= _ACCEPTANCE_FLOOR, f"catch rate {rate:.0%} < {_ACCEPTANCE_FLOOR:.0%}; missed {missed}"
    # No kind may be entirely blind, whatever the aggregate says.
    for kind, hits in by_kind.items():
        assert any(hits), f"every injected {kind} fault was missed"


def test_no_faithful_answer_is_flagged() -> None:
    """The companion measurement. A guard that fires on correct answers gets turned off."""
    flagged = [(habit, claim) for habit, quote, claim in _FAITHFUL if _caught(quote, claim)]
    assert not flagged, f"false positives on faithful restatements: {flagged}"


def test_the_faithful_form_of_every_seeded_case_passes() -> None:
    """The same fixtures the catch rate is measured on, restated correctly. This is what makes the
    catch rate mean "it caught the injected fault" rather than "it flags everything"."""
    flagged = [(kind, ok) for kind, quote, ok, _bad in _SEEDED if _caught(quote, ok)]
    assert not flagged, f"false positives on the faithful form of seeded cases: {flagged}"


# --- claim splitting -------------------------------------------------------------------------------


def test_a_bullet_list_is_one_claim_per_bullet() -> None:
    answer = "- The EMD is Rs. 5 lakh [1].\n- The turnover floor is Rs. 19.44 Crores [2]."
    claims = split_claims(answer)
    assert [c.citations for c in claims] == [(1,), (2,)]
    assert claims[0].text == "The EMD is Rs. 5 lakh"


def test_an_abbreviation_is_not_a_sentence_boundary() -> None:
    """The bug that made the first cut of this module useless. This corpus writes money as
    "Rs. 19.44 Crores", so splitting after "Rs." tore every monetary claim in half and left the
    figure in a fragment short enough to be dropped — after which the verifier checked a subject
    with no value in it and passed everything."""
    claims = split_claims("The required turnover is Rs. 19.44 Crores [1].")
    assert len(claims) == 1
    assert "19.44" in claims[0].text


def test_citation_markers_are_recorded_then_stripped() -> None:
    """A marker is written by the citation assembler, not asserted by the model, so leaving it in
    would make every value check see a stray number."""
    claim = split_claims("The uplink is 10 Gbps [2, 3].")[0]
    assert claim.citations == (2, 3)
    assert "[" not in claim.text and "2" not in claim.text.replace("10", "")


def test_a_fragment_with_no_claim_in_it_is_dropped() -> None:
    for answer in ("[1]", "।", "Here:", "- "):
        assert split_claims(answer) == ()


def test_an_ordered_list_marker_is_not_a_claimed_figure() -> None:
    assert "1." not in split_claims("1. The uplink is 10 Gbps [1].")[0].text


# --- values ----------------------------------------------------------------------------------------


def test_a_date_canonicalises_across_formats_and_conventions() -> None:
    """Phase 3's ``_date`` refused to guess between day-first and month-first for good reason, so
    a date reduces to its year plus its two other components SORTED — identical under either
    reading, every separator and every month spelling."""
    forms = ("24-09-2026", "24/09/2026", "09/24/2026", "24 September 2026", "September 24, 2026", "24th Sep 2026")
    assert len({next(iter(v for v in extract_values(f) if v.startswith("date:"))) for f in forms}) == 1


def test_a_standards_designation_is_one_identifier_not_three_figures() -> None:
    values = extract_values("per IEC 62443-4-2")
    assert "std:iec 62443-4-2" in values
    assert "num:4" not in values and "num:2" not in values


def test_a_lowercase_word_cannot_look_like_a_standards_body() -> None:
    """ "when 50" must not read as "EN 50". Standards bodies are matched case-SENSITIVELY, which is
    the only thing that makes the two-letter ones usable at all."""
    assert not any(v.startswith("std:") for v in extract_values("the result when 50 bidders apply"))


def test_a_quantity_carries_its_unit_so_a_unit_swap_is_caught() -> None:
    assert "qty:35 lakh" in extract_values("Rs. 35 lakh")
    assert "qty:35 crore" in extract_values("Rs. 35 Crores")


# --- enumerations ----------------------------------------------------------------------------------


def test_the_plans_own_worked_example() -> None:
    assert enumeration_items("The vendor shall supply HBOM and SBOM for every component.") == ("hbom", "sbom")


def test_prose_joined_by_and_is_not_a_list() -> None:
    assert enumeration_items("the bidder shall submit the EMD and the technical bid shall be sealed") == ()


def test_an_article_prefixed_acronym_list_is_read_through_the_fallback() -> None:
    assert enumeration_items("Submit the HBOM, the SBOM and the compliance matrix.") == ("hbom", "sbom")


def test_a_claim_mentioning_none_of_a_quotes_items_reports_no_omission() -> None:
    """It is about something else in the quote, not a partial answer to the list."""
    report = verify_claims(
        "The components must be delivered to site [1].",
        ["The vendor shall supply HBOM and SBOM for every component."],
    )
    assert report.ok


# --- the premise is the CITED quote, not the union -------------------------------------------------


def test_a_figure_from_another_citation_is_a_mis_attribution() -> None:
    """The point of the whole phase. ``numeracy.unsupported_numbers`` asks "does this figure appear
    in ANY retrieved span" and so cannot see this: the figure is real, and the claim hangs it off a
    span that does not state it."""
    quotes = ["an EMD of Rs. 5 lakh", "a turnover of Rs. 19.44 Crores"]
    assert verify_claims("The EMD is Rs. 19.44 Crores [1].", quotes).failures
    assert verify_claims("The EMD is Rs. 5 lakh [1].", quotes).ok


def test_an_uncited_claim_falls_back_to_the_union_and_says_so() -> None:
    """A lead-in or a connective legitimately carries no marker; punishing the generator's
    formatting is not the job."""
    report = verify_claims("The EMD is Rs. 5 lakh.", ["an EMD of Rs. 5 lakh", "a turnover of Rs. 19.44 Crores"])
    assert report.ok
    assert all(v.uncited for v in report.verdicts)


def test_a_marker_pointing_past_the_citations_falls_back_rather_than_failing() -> None:
    """An out-of-range marker is a generator formatting fault, not a false claim."""
    assert verify_claims("The EMD is Rs. 5 lakh [9].", ["an EMD of Rs. 5 lakh"]).ok


def test_an_answer_with_no_citations_is_not_checked() -> None:
    """An abstain has none, and "every claim failed" would be a false reading of "there was
    nothing to check against"."""
    report = verify_claims("I could not find that in the documents.", [])
    assert report.checked == 0 and report.ok


# --- the actions -----------------------------------------------------------------------------------


def test_the_footnote_names_the_claims_that_did_not_check_out() -> None:
    report = verify_claims("The EMD is Rs. 5 Crores [1].", ["an EMD of Rs. 5 lakh"])
    footnote = unverified_footnote(report)
    assert "Could not be verified" in footnote and "Rs. 5 Crores" in footnote


def test_a_clean_answer_gets_no_footnote() -> None:
    assert unverified_footnote(verify_claims("The EMD is Rs. 5 lakh [1].", ["an EMD of Rs. 5 lakh"])) == ""


def test_dropping_removes_the_failing_claim_and_still_footnotes_it() -> None:
    """The plan's own wording. A silent deletion would leave the reader with a shorter answer and
    no way to know something was removed."""
    answer = "The EMD is Rs. 5 lakh [1].\nThe turnover floor is Rs. 35 Crores [2]."
    quotes = ["an EMD of Rs. 5 lakh", "a turnover of Rs. 19.44 Crores"]
    report = verify_claims(answer, quotes)
    edited = without_unverified(answer, report)

    assert "Rs. 5 lakh" in edited
    assert "The turnover floor is Rs. 35 Crores [2]." not in edited.split("Could not be verified")[0]
    assert "Could not be verified" in edited


def test_an_answer_whose_every_claim_fails_is_not_erased() -> None:
    """A verifier that disagrees with everything is far likelier to be wrong about the answer than
    the answer is to be entirely fabricated — and an empty GROUNDED answer is a contract
    violation, not a conservative outcome."""
    answer = "The EMD is Rs. 5 Crores [1]."
    report = verify_claims(answer, ["an EMD of Rs. 5 lakh"])
    edited = without_unverified(answer, report)
    assert edited.startswith(answer)
    assert "Could not be verified" in edited


# --- the stored report -----------------------------------------------------------------------------


def test_the_report_json_stores_the_failures_in_full_and_the_passes_as_a_count() -> None:
    """``query_log.verification`` exists to answer "what did this answer get wrong"; recording
    every passing claim verbatim would store the whole answer a second time."""
    answer = "The EMD is Rs. 5 lakh [1].\nThe turnover is Rs. 35 Crores [1]."
    stored = json.loads(verify_claims(answer, ["an EMD of Rs. 5 lakh"]).as_json())

    assert stored["verify_version"] == VERIFY_VERSION
    assert stored["checked"] == 2 and stored["failed"] == 1
    assert len(stored["failures"]) == 1
    assert stored["failures"][0]["reasons"] == ["value_not_in_quote"]


def test_a_verdict_reports_stable_reason_slugs() -> None:
    assert ClaimVerdict(claim="c", ordinal=0, ok=False, missing_values=("num:1",)).reasons == ("value_not_in_quote",)
    assert ClaimVerdict(claim="c", ordinal=0, ok=False, omitted_items=("hbom",)).reasons == ("enumeration_omitted",)
    assert ClaimVerdict(claim="c", ordinal=0, ok=True).reasons == ()


def test_an_empty_report_is_clean() -> None:
    assert VerificationReport().ok and VerificationReport().checked == 0


def test_a_claim_is_a_frozen_record() -> None:
    claim = Claim(text="t", ordinal=0, citations=(1,))
    assert claim.text == "t" and claim.citations == (1,)
