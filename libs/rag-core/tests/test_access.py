"""Access rules: need-to-know ACL (FR-A6) + hierarchical clearance (banking pack)."""

from __future__ import annotations

from rag_core.access import is_cleared, is_entitled
from rag_schemas.classification import ClassificationLevel, classification_label, parse_classification


def test_is_entitled_public_and_groups() -> None:
    assert is_entitled([], []) is True  # empty span acl = tenant-public
    assert is_entitled(["finance"], []) is False  # restricted span, caller has no groups
    assert is_entitled(["finance"], ["hr"]) is False
    assert is_entitled(["finance"], ["finance", "hr"]) is True


def test_is_cleared_is_hierarchical() -> None:
    conf = int(ClassificationLevel.CONFIDENTIAL)  # 2
    # A Confidential-cleared caller sees Public/Internal/Confidential, not Restricted.
    assert is_cleared(int(ClassificationLevel.PUBLIC), conf) is True
    assert is_cleared(int(ClassificationLevel.INTERNAL), conf) is True
    assert is_cleared(int(ClassificationLevel.CONFIDENTIAL), conf) is True
    assert is_cleared(int(ClassificationLevel.RESTRICTED), conf) is False
    # Default clearance 0 sees only Public — least privilege.
    assert is_cleared(0, 0) is True
    assert is_cleared(1, 0) is False


def test_parse_and_label() -> None:
    assert parse_classification("confidential") == 2
    assert parse_classification("RESTRICTED") == 3
    assert parse_classification(2) == 2
    assert parse_classification("bogus") == 0  # unknown → Public (safe default)
    assert parse_classification(None) == 0
    assert parse_classification(99) == 0  # out of range → Public
    assert classification_label(3) == "restricted"
    assert classification_label(0) == "public"
    assert classification_label(99) == "public"
