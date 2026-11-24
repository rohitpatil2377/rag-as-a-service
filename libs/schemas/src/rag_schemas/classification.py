"""Hierarchical data classification (banking pack).

Four ordered sensitivity levels. A caller's *clearance* is one of these levels; they may see
a span only if the span's classification is at or below their clearance (Confidential clearance
sees Confidential + Internal + Public, never Restricted). This is the classic BFSI model and
complements the need-to-know ACL groups (FR-A6) — a span must pass BOTH checks.

Default is PUBLIC(0) everywhere, so existing content and callers are unchanged until a document
is explicitly classified higher and a caller is given a higher clearance.
"""

from __future__ import annotations

from enum import IntEnum


class ClassificationLevel(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


_MAX_LEVEL = max(int(level) for level in ClassificationLevel)
_BY_NAME = {level.name.lower(): int(level) for level in ClassificationLevel}


def parse_classification(value: str | int | None) -> int:
    """Coerce a label ('confidential') or int (2) to a valid level; unknown/None → PUBLIC."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value if 0 <= value <= _MAX_LEVEL else 0
    return _BY_NAME.get(value.strip().lower(), 0)


def classification_label(level: int) -> str:
    """The human label for a level (out-of-range → 'public')."""
    try:
        return ClassificationLevel(level).name.lower()
    except ValueError:
        return "public"
