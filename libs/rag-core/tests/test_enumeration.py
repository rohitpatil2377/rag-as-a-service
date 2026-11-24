from rag_core.enumeration import enumerated_items


def test_ten_sop_ids_are_detected_in_order() -> None:
    q = (
        "give a detailed summary for all the 10 SOPs from PBSOP001, PBSOP002, PBSOP003, PBSOP004, "
        "PBSOP005, PBSOP006, PBSOP007, PBSOP008, PBSOP009, and PBSOP010"
    )
    assert enumerated_items(q) == [f"PBSOP{n:03d}" for n in range(1, 11)]


def test_hyphenated_and_mixed_case_ids_count_once() -> None:
    assert enumerated_items("compare xqa-001, XQA-001, SOP-2026-11 and ISO9001") == [
        "xqa-001",
        "SOP-2026-11",
        "ISO9001",
    ]


def test_two_ids_is_a_comparison_not_an_enumeration() -> None:
    assert enumerated_items("what differs between PBSOP001 and PBSOP002?") == []


def test_years_versions_and_ordinals_are_not_items() -> None:
    assert enumerated_items("what changed in 2023, 2024 and 2025 under v2.1, v2.2 and the 3rd revision?") == []


def test_plain_prose_has_no_items() -> None:
    assert enumerated_items("What are the contents of the header of an SOP?") == []
