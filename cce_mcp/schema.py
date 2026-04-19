"""
CCE MCP Schema
Pydantic input/output models for all MCP tool parameters.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class IngestTurnInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description="Unique session identifier (e.g. 'user-123-chat')",
        min_length=1, max_length=200,
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

    session_id: str = Field(
        ...,
        description="Unique session identifier",
        min_length=1, max_length=200,
    )
    messages: list[dict] = Field(
        ...,
        description="List of {'role': str, 'content': str} message dicts",
        min_length=1,
    )


class RetrieveContextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description="Session to retrieve context from",
        min_length=1, max_length=200,
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


class SummarizeSessionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description="Session to summarize",
        min_length=1, max_length=200,
    )


class CloseSessionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description="Session to close",
        min_length=1, max_length=200,
    )
    checkpoint: bool = Field(
        default=True,
        description="If True, writes macro summary to cold tier before closing",
    )


class SessionStatsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    session_id: str = Field(
        ...,
        description="Session to get stats for",
        min_length=1, max_length=200,
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