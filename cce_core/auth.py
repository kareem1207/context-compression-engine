"""
API keys that are limited to some sessions (REST API and proxy).

  CCE_API_KEY=secret                      one key that may use every session (as before)
  CCE_API_KEYS="alice-key=user-1,user-2;team-key=team-*;admin-key=*"
                                          key=session patterns, keys separated by ';', patterns by ','
                                          a pattern is a session id, a prefix ending in '*', or '*' for all

A key that is valid but not allowed a session gets 403 for it, whether or not the session exists, and does not
see other sessions in listings. Calls that touch no session (/health, /v1/compress, /v1/models) work with any
valid key. Keys are compared in constant time. Keys cannot contain '=', ';' or ','.
"""

from __future__ import annotations

import hmac
import os


def parse_keys(spec: str | None) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for part in (spec or "").split(";"):
        part = part.strip()
        if not part:
            continue
        key, sep, patterns = part.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"CCE_API_KEYS entry {part[:12]!r}... must look like key=pattern,pattern")
        keys[key] = [p.strip() for p in patterns.split(",") if p.strip()]
    return keys


def configured_keys() -> dict[str, list[str]]:
    """Every configured key with its session patterns (empty when no key is configured: no authentication)."""
    keys = parse_keys(os.getenv("CCE_API_KEYS"))
    single = os.getenv("CCE_API_KEY")
    if single:
        keys.setdefault(single, ["*"])
    return keys


def scope_for(given: str | None, keys: dict[str, list[str]]) -> list[str] | None:
    """The session patterns of the key that was given, or None if it matches no key."""
    if not given:
        return None
    found = None
    for key, patterns in keys.items():          # no early exit: every key is compared
        if hmac.compare_digest(given.encode(), key.encode()):
            found = patterns
    return found


def allows(patterns: list[str], session_id: str) -> bool:
    for p in patterns:
        if p == "*" or p == session_id or (p.endswith("*") and session_id.startswith(p[:-1])):
            return True
    return False
