"""
CCE MCP Schema
Pydantic input/output models for all MCP tool parameters.
"""

from __future__ import annotations

from typing import Annotated, Optional
from pydantic import AfterValidator, BaseModel, Field, ConfigDict

from cce_core.ids import validate_session_id

# Session ids become file names in the cold tier, so they are checked like the REST API checks them.
SessionId = Annotated[str, AfterValidator(validate_session_id)]


class IngestTurnInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(
        ...,
        description="Unique session identifier (e.g. 'user-123-chat')",
        min_length=1, max_length=64,
    )
    role: str = Field(
        ...,
        description="Speaker role: 'user' | 'assistant' | 'system'",
        pattern=r"^(user|assistant|system|tool)$",
    )
    content: str = Field(
        ...,
        description="Message content text",
        min_length=1,
    )


class IngestHistoryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(
        ...,
        description="Unique session identifier",
        min_length=1, max_length=64,
    )
    messages: list[dict] = Field(
        ...,
        description="List of {'role': str, 'content': str} message dicts",
        min_length=1,
    )


class RetrieveContextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(
        ...,
        description="Session to retrieve context from",
        min_length=1, max_length=64,
    )
    query: str = Field(
        ...,
        description="The current user query — used to find relevant past context",
        min_length=1,
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of memory nodes to retrieve (default: config value)",
        ge=1, le=20,
    )
    fmt: str = Field(
        default="messages",
        description="Output format: 'messages' (list of dicts), 'string', or 'dict'",
        pattern=r"^(messages|string|dict)$",
    )
    include_micro: bool = Field(
        default=False,
        description="Include per-turn micro summaries in past context blocks",
    )
    since: Optional[str] = Field(default=None, description="ISO 8601 moment: only notes and recent turns from then on")
    until: Optional[str] = Field(default=None, description="ISO 8601 moment: only notes and recent turns up to then")
    last_turns: Optional[int] = Field(default=None, ge=1, description="only the last N turns of the session and the notes reaching into them")
    explain: bool = Field(
        default=False,
        description="Also return why the context looks as it does: chosen notes and quotes, scores, token budget split, what was dropped",
    )


class SummarizeSessionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(
        ...,
        description="Session to summarize",
        min_length=1, max_length=64,
    )


class CloseSessionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(
        ...,
        description="Session to close",
        min_length=1, max_length=64,
    )
    checkpoint: bool = Field(
        default=True,
        description="If True, writes macro summary to cold tier before closing",
    )


class SessionStatsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(
        ...,
        description="Session to get stats for",
        min_length=1, max_length=64,
    )


class StatelessCompressInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    messages: list[dict] = Field(
        ...,
        description="Full conversation history as list of {'role', 'content'} dicts",
        min_length=1,
    )
    query: str = Field(
        ...,
        description="Current user query — used to retrieve most relevant past context",
        min_length=1,
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of memory nodes to retrieve",
        ge=1, le=20,
    )
    fmt: str = Field(
        default="messages",
        description="Output format: 'messages', 'string', or 'dict'",
        pattern=r"^(messages|string|dict)$",
    )


class PinMemoryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(..., description="Session", min_length=1, max_length=64)
    action: str = Field(..., description="add | list | edit | remove", pattern=r"^(add|list|edit|remove)$")
    text: Optional[str] = Field(default=None, max_length=4000, description="text to pin (add) or the new text (edit)")
    turn: Optional[int] = Field(default=None, ge=0, description="index of a stored turn to pin (add)")
    pin_id: Optional[str] = Field(default=None, description="the pin to edit or remove")


class ForgetMemoryInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(..., description="Session", min_length=1, max_length=64)
    node_id: Optional[str] = Field(default=None, description="forget this note and the text it covered")
    start: Optional[int] = Field(default=None, ge=0, description="first turn index to forget")
    end: Optional[int] = Field(default=None, ge=0, description="last turn index to forget")
    confirm: bool = Field(default=False, description="must be true: forgetting cannot be undone")


class ExportSessionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: SessionId = Field(..., description="Session to export", min_length=1, max_length=64)
