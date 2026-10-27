"""Session ids end up in file names (cold tier) and in SQL keys, so they are restricted everywhere."""

from __future__ import annotations

import re

SESSION_ID_PATTERN = "[A-Za-z0-9_.-]{1,64}"
_SESSION_ID = re.compile(SESSION_ID_PATTERN)


def is_valid_session_id(session_id: object) -> bool:
    """1-64 characters of letters, digits, `_`, `.`, `-`; `.`, `..` and `index` (the cold tier's index file) are not allowed."""
    return (
        isinstance(session_id, str)
        and _SESSION_ID.fullmatch(session_id) is not None
        and session_id not in (".", "..", "index")
    )


def validate_session_id(session_id: str) -> str:
    if not is_valid_session_id(session_id):
        raise ValueError("session_id must be 1-64 characters: letters, digits, _ . - (and not '.', '..' or 'index')")
    return session_id
