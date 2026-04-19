"""
CCE Segmenter
Converts a raw conversation (list of message dicts) into a list of Turn
objects — the atomic unit everything else operates on.

Input format (OpenAI-style, works with llama.cpp / Gemma):
    [
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]

Also accepts a plain string — split on newlines, alternating user/assistant.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from cce_core.ingestion import tokenizer


Role = Literal["user", "assistant", "system", "tool"]


@dataclass
class Turn:
    """Single conversation turn — the atomic unit of CCE."""
    turn_id: str
    role: Role
    content: str
    token_count: int
    timestamp: datetime
    index: int                        # position in the conversation (0-based)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "role": self.role,
            "content": self.content,
            "token_count": self.token_count,
            "timestamp": self.timestamp.isoformat(),
            "index": self.index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            turn_id=d["turn_id"],
            role=d["role"],
            content=d["content"],
            token_count=d.get("token_count", tokenizer.count(d["content"])),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            index=d["index"],
            metadata=d.get("metadata", {}),
        )

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        return f"Turn(index={self.index}, role={self.role!r}, tokens={self.token_count}, content={preview!r})"


_SPEAKER_RE = re.compile(r"^(user|human|assistant|ai|system)\s*:\s*", re.IGNORECASE)


def segment(
    messages: list[dict] | str,
    session_id: str | None = None,
) -> list[Turn]:
    """
    Convert raw messages into a list of Turn objects.

    Args:
        messages: Either a list of {"role", "content"} dicts, or a plain
                  string where each line is a turn (role auto-detected).
        session_id: Optional — attached to each turn's metadata.

    Returns:
        Ordered list of Turn objects, index 0 = oldest.
    """
    if isinstance(messages, str):
        messages = _parse_plain_text(messages)

    turns: list[Turn] = []
    for i, msg in enumerate(messages):
        role = _normalize_role(msg.get("role", "user"))
        content = (msg.get("content") or "").strip()

        if not content:
            continue  # skip empty turns silently

        turn = Turn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            token_count=tokenizer.count(content),
            timestamp=datetime.now(timezone.utc),
            index=i,
            metadata={"session_id": session_id} if session_id else {},
        )
        turns.append(turn)

    # Re-index after filtering empty turns
    for i, t in enumerate(turns):
        t.index = i

    return turns


def segment_incremental(
    new_message: dict,
    existing_turns: list[Turn],
    session_id: str | None = None,
) -> Turn:
    """
    Add a single new message to an existing list of turns.
    Used in stateful mode — called after every new message arrives.
    Mutates existing_turns in place and returns the new Turn.
    """
    role = _normalize_role(new_message.get("role", "user"))
    content = (new_message.get("content") or "").strip()

    turn = Turn(
        turn_id=str(uuid.uuid4()),
        role=role,
        content=content,
        token_count=tokenizer.count(content),
        timestamp=datetime.now(timezone.utc),
        index=len(existing_turns),
        metadata={"session_id": session_id} if session_id else {},
    )
    existing_turns.append(turn)
    return turn


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_role(raw: str) -> Role:
    r = raw.lower().strip()
    if r in ("user", "human"):
        return "user"
    if r in ("assistant", "ai", "bot", "model"):
        return "assistant"
    if r == "system":
        return "system"
    if r == "tool":
        return "tool"
    return "user"  # safe default


def _parse_plain_text(text: str) -> list[dict]:
    """
    Parse a plain-text conversation string into message dicts.
    Handles "User: ..." / "Assistant: ..." prefixes or falls back to
    alternating user/assistant assignment.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    messages = []
    default_roles = ["user", "assistant"]

    for i, line in enumerate(lines):
        m = _SPEAKER_RE.match(line)
        if m:
            role = _normalize_role(m.group(1))
            content = line[m.end():].strip()
        else:
            role = default_roles[i % 2]
            content = line
        messages.append({"role": role, "content": content})

    return messages