"""PII engine: checksum-aware detection + redact/mask/hash strategies (FR-B6/G2)."""

from __future__ import annotations

import pytest
from svc_doc_intelligence.pii import (
    DEFAULT_ENTITIES,
    RegexPIIMasker,
    _luhn_ok,
    _verhoeff_ok,
    build_pii_masker,
    parse_entities,
)

_VALID_AADHAAR = "234567890125"  # Verhoeff-valid
_VALID_CARD = "4111 1111 1111 1111"  # Luhn-valid (Visa test)


def test_checksums() -> None:
    assert _verhoeff_ok(_VALID_AADHAAR) is True
    assert _verhoeff_ok("123456789012") is False
    assert _luhn_ok(_VALID_CARD) is True
    assert _luhn_ok("4111 1111 1111 1112") is False


def test_redact_default() -> None:
    m = RegexPIIMasker()
    out = m.mask(f"mail a@b.com aadhaar {_VALID_AADHAAR} pan ABCDE1234F phone +919812345678")
    assert "[REDACTED:EMAIL]" in out.text
    assert "[REDACTED:AADHAAR]" in out.text
    assert "[REDACTED:PAN]" in out.text
    assert "[REDACTED:PHONE]" in out.text
    assert out.total == 4


def test_checksum_avoids_false_positives() -> None:
    """A 12-digit invoice number is NOT an Aadhaar; a non-Luhn card is left alone."""
    out = RegexPIIMasker().mask("invoice 123456789012 ref 1234567890123456")
    assert "123456789012" in out.text
    assert "AADHAAR" not in out.counts
    assert "CREDIT_CARD" not in out.counts


def test_phone_masks_grouped_formats() -> None:
    """Grouped/separated Indian mobiles leaked in cleartext before the regex tolerated an
    internal separator (QA ISS-012). Every common form must now be masked."""
    for raw in ("+91-98200-10005", "98200-10005", "98200 10005", "9820010005", "+91 9820010005"):
        out = RegexPIIMasker().mask(f"call us on {raw} today")
        assert "PHONE" in out.counts, raw
        assert raw not in out.text, raw
    # ...but a long id / section ref is not a phone number.
    kept = RegexPIIMasker().mask("order 100200300400 section 12345")
    assert "PHONE" not in kept.counts


def test_mask_strategy_keeps_last_four() -> None:
    out = RegexPIIMasker(strategy="mask").mask("pan ABCDE1234F")
    assert "234F" in out.text and "ABCDE" not in out.text


def test_hash_strategy_is_deterministic() -> None:
    m = RegexPIIMasker(strategy="hash", hash_salt="s")
    out = m.mask("pan ABCDE1234F seen again ABCDE1234F")
    token = out.text.split("[PAN:")[1].split("]")[0]
    assert out.text.count(f"[PAN:{token}]") == 2  # same value → same token
    # different salt → different token
    other = RegexPIIMasker(strategy="hash", hash_salt="t").mask("pan ABCDE1234F")
    assert token not in other.text


def test_entity_subset_and_parse() -> None:
    out = RegexPIIMasker(entities=("EMAIL",)).mask("a@b.com and pan ABCDE1234F")
    assert out.counts == {"EMAIL": 1}  # PAN not in the chosen set
    assert parse_entities("email, phone, bogus") == ("EMAIL", "PHONE")
    assert parse_entities("") is None  # empty → default set
    assert set(DEFAULT_ENTITIES) >= {"EMAIL", "AADHAAR", "PAN", "CREDIT_CARD"}


def test_build_pii_masker_regex_default() -> None:
    masker = build_pii_masker("regex", strategy="redact")
    assert isinstance(masker, RegexPIIMasker)
    assert masker.mask("a@b.com").total == 1


# --- Banking entity pack (context-anchored) ---------------------------------------------


def test_banking_ids_masked_but_keyword_kept() -> None:
    m = RegexPIIMasker()
    out = m.mask("A/c No: 123456789012, MICR 400002045, CVV 573, CIF No: CUST00123456")
    # Only the sensitive VALUE is masked; the "A/c No:" / "CVV" keywords stay for readability.
    assert "A/c No: [REDACTED:BANK_ACCOUNT]" in out.text
    assert "MICR [REDACTED:MICR]" in out.text
    assert "CVV [REDACTED:CVV]" in out.text
    assert "CIF No: [REDACTED:CUSTOMER_ID]" in out.text
    assert out.counts == {"BANK_ACCOUNT": 1, "MICR": 1, "CVV": 1, "CUSTOMER_ID": 1}


def test_banking_no_false_positive_on_policy_amounts() -> None:
    """The killer requirement: a bare long number in policy prose must NOT be masked as an
    account number — only a number ANCHORED to an account/CVV/etc. keyword is."""
    out = RegexPIIMasker().mask(
        "Reimbursement up to 500000 per section 123456789012. The allowance is Rs 25,000 for 2024."
    )
    assert "500000" in out.text and "123456789012" in out.text
    assert "BANK_ACCOUNT" not in out.counts
    assert out.total == 0


def test_bank_account_mask_strategy_keeps_last_four() -> None:
    out = RegexPIIMasker(strategy="mask").mask("Account Number 123456789012")
    assert "9012" in out.text and "12345678" not in out.text


def test_cvv_is_fully_masked() -> None:
    out = RegexPIIMasker(strategy="mask").mask("cvv: 573")
    assert "573" not in out.text  # keep_last=0 → nothing revealed for a CVV


# --- a clause number is not an IP address ---------------------------------------------------------
#
# FOUND IN PRODUCTION DATA, by tracing why an amendment never reached an answer. The AEGCL
# corrigendum says "BID DATA SHEET (BDS), ITB 1.3.9.1 item 5"; it was stored as
# "ITB [REDACTED:IP] item 5". `1.3.9.1` is a syntactically valid IPv4 address, so the IP detector
# claimed it and the clause identity was destroyed AT INGEST -- after which no downstream stage
# could recover it. Retrieval could not match the clause, the map could not extract it, the answer
# could not cite it, and the supersession metric scored the obligation as unreported. Four-part
# numbering is ordinary in tender and standards documents, so this damaged exactly the corpora the
# platform exists for, silently.
#
# REDACTION REMAINS THE DEFAULT. These tests assert the suppression is narrow: it needs an explicit
# contractual cue immediately before the number. A bare dotted quad is still masked, because
# failing to redact a real address is the worse error and the digits alone cannot settle it.


@pytest.mark.parametrize(
    "text",
    [
        "BID DATA SHEET (BDS), ITB 1.3.9.1 item 5",
        "clause 1.3.9.1",
        "Clause 1.3.9.1 applies",
        "Section 1.3.9.1",
        "ITB 2.2.1",
        "as per Sl no: D.3 of Appendix to ITB-1 1.3.9.1",
        "see paragraph 1.3.9.1 of the BDS",
        "Annexure 2.10.4.1",
        "Table 1.2.3.4 of the schedule",
    ],
)
def test_a_contractual_identifier_survives_ingestion_unchanged(text: str) -> None:
    """THE INVARIANT IS THE STORED TEXT, not whether a regex matched. What broke downstream was
    the chunk that got written, so that is what is asserted: byte-identical, no counts, no spans."""
    result = RegexPIIMasker().mask(text)

    assert result.text == text
    assert "IP" not in result.counts
    assert result.spans == []


@pytest.mark.parametrize("text", ["version 2.0.1.3", "Version 2.0.1.3", "rev 1.2.3.4", "release 1.2.3.4"])
def test_a_version_identifier_is_preserved_by_its_own_cue_group(text: str) -> None:
    """TESTED SEPARATELY BECAUSE ITS JUSTIFICATION IS SEPARATE. A version string is not a
    contractual clause, and an earlier draft of this fix put `version` into the clause-cue list
    unannounced -- which is how an allowlist stops meaning what its name says. It is suppressed
    for the same cost (a redacted version destroys which release a requirement refers to; the
    corpus cites "CERT-In Technical Guidelines ... Version 2.0") under its own named group."""
    result = RegexPIIMasker().mask(text)

    assert result.text == text
    assert "IP" not in result.counts


@pytest.mark.parametrize(
    "text",
    [
        "v 10.0.0.1",
        "item 1.2.3.4",
        "format 1.2.3.4",
    ],
)
def test_a_cue_nobody_demonstrated_does_not_suppress(text: str) -> None:
    """The allowlist holds only cues with a case behind them. A bare `v` is one letter and would
    suppress far more than it should; `item` and `format` were in an earlier draft of this fix
    with no example requiring them. Redaction is the default, so an unjustified cue is removed
    rather than left in because it is harmless-looking."""
    assert "[REDACTED:IP]" in RegexPIIMasker().mask(text).text


@pytest.mark.parametrize(
    "text",
    [
        "Server IP: 192.168.1.1",
        "10.0.0.1",
        "connect to 10.0.0.1 on port 443",
        "the gateway is 192.168.0.254",
        "Host 192.0.2.11 is unreachable",
        "ip address 8.8.8.8",
    ],
)
def test_a_real_address_is_still_redacted(text: str) -> None:
    """The half that must not regress. A bare quad with no contractual cue stays masked."""
    result = RegexPIIMasker().mask(text)

    assert "[REDACTED:IP]" in result.text
    assert result.counts.get("IP") == 1


def test_a_clause_cue_does_not_shield_an_address_elsewhere_in_the_sentence() -> None:
    """The suppression is anchored to the word immediately before the number, deliberately. A
    wider window would let any sentence mentioning "section" hide a real address -- and a document
    that discusses both clauses and network configuration is the normal case, not an edge one."""
    result = RegexPIIMasker().mask("Section 1.3.9.1 requires the server at 192.168.1.1 to be hardened.")

    assert "Section 1.3.9.1" in result.text, "the clause survives"
    assert "192.168.1.1" not in result.text, "the address does not"
    assert result.counts.get("IP") == 1
