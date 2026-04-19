"""
CCE Context Builder
Takes retrieval results + hot tier turns and assembles the final
compressed context payload that gets injected into the LLM prompt.

The payload has three sections, ordered for LLM readability:

  [1] PAST CONTEXT (warm tier — compressed)
      Retrieved memory nodes, formatted as titled topic blocks.
      Most relevant first (already ranked by Retriever).

  [2] RECENT CONTEXT (hot tier — verbatim)
      Last N turns exactly as spoken. Always included in full.

  [3] SYSTEM NOTE (optional)
      Brief note telling the LLM how to interpret the above.

The builder respects a token budget (context_max_tokens from config).
If the budget is tight, it trims past context first, never recent context.

Output formats:
  "messages"  — list of {"role", "content"} dicts (OpenAI / llama.cpp style)
  "string"    — single formatted string (for models that take raw text)
  "dict"      — structured dict with all sections for inspection/debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cce_core.config import CCEConfig, DEFAULT_CONFIG
from cce_core.ingestion import tokenizer
from cce_core.ingestion.segmenter import Turn
from cce_core.retrieval.retriever import RetrievalResult


OutputFormat = Literal["messages", "string", "dict"]


_SYSTEM_NOTE = (
    "The following conversation context has been compressed by the "
    "Context Compression Engine (CCE). 'Past context' contains summarized "
    "topic blocks from earlier in the conversation. 'Recent context' contains "
    "the verbatim last few messages. Use both to answer accurately."
)

_PAST_HEADER = "=== PAST CONTEXT (compressed) ==="
_RECENT_HEADER = "=== RECENT CONTEXT ==="
_SEPARATOR = "---"


@dataclass
class ContextPayload:
    """
    The assembled context payload — ready to inject into an LLM call.

    Attributes:
        past_blocks: Formatted text blocks from retrieved memory nodes.
        recent_turns: Verbatim hot tier turns.
        system_note: Explanation for the LLM.
        token_count: Total estimated tokens in the payload.
        retrieved_node_ids: IDs of nodes included (for tracing).
        was_truncated: True if past context was trimmed to fit budget.
    """
    past_blocks: list[str]
    recent_turns: list[Turn]
    system_note: str
    token_count: int
    retrieved_node_ids: list[str] = field(default_factory=list)
    was_truncated: bool = False

    def to_messages(self) -> list[dict]:
        """
        Export as OpenAI-style message list.
        Structure:
          system: system_note + past context
          user/assistant: recent turns (verbatim)
        """
        messages: list[dict] = []

        # System message: note + compressed past
        system_parts = [self.system_note]
        if self.past_blocks:
            system_parts.append("")
            system_parts.append(_PAST_HEADER)
            system_parts.extend(self.past_blocks)

        messages.append({
            "role": "system",
            "content": "\n".join(system_parts),
        })

        # Recent turns verbatim
        for turn in self.recent_turns:
            messages.append({"role": turn.role, "content": turn.content})

        return messages

    def to_string(self) -> str:
        """Export as a single formatted string."""
        parts: list[str] = [self.system_note, ""]

        if self.past_blocks:
            parts.append(_PAST_HEADER)
            parts.extend(self.past_blocks)
            parts.append("")

        if self.recent_turns:
            parts.append(_RECENT_HEADER)
            for turn in self.recent_turns:
                parts.append(f"{turn.role.upper()}: {turn.content}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Export as structured dict for debugging/inspection."""
        return {
            "system_note": self.system_note,
            "past_blocks": self.past_blocks,
            "recent_turns": [
                {"role": t.role, "content": t.content} for t in self.recent_turns
            ],
            "token_count": self.token_count,
            "retrieved_node_ids": self.retrieved_node_ids,
            "was_truncated": self.was_truncated,
            "past_block_count": len(self.past_blocks),
            "recent_turn_count": len(self.recent_turns),
        }

    def export(self, fmt: OutputFormat = "messages"):
        if fmt == "messages":
            return self.to_messages()
        if fmt == "string":
            return self.to_string()
        return self.to_dict()

    def __repr__(self):
        return (
            f"ContextPayload(past_blocks={len(self.past_blocks)}, "
            f"recent_turns={len(self.recent_turns)}, "
            f"tokens={self.token_count}, truncated={self.was_truncated})"
        )


class ContextBuilder:
    """
    Assembles the final LLM context payload from retrieval results + hot tier.

    Usage:
        builder = ContextBuilder(config)
        payload = builder.build(
            results=retriever.retrieve(store, query),
            hot_turns=store.get_hot_turns(),
        )
        messages = payload.to_messages()
    """

    def __init__(self, config: CCEConfig = DEFAULT_CONFIG):
        self.config = config

    def build(
        self,
        results: list[RetrievalResult],
        hot_turns: list[Turn],
        system_note: str = _SYSTEM_NOTE,
        max_tokens: int | None = None,
        include_micro: bool = False,
    ) -> ContextPayload:
        """
        Build the context payload.

        Args:
            results: Ranked retrieval results from the Retriever.
            hot_turns: Verbatim recent turns from the hot tier.
            system_note: System message prefix for the LLM.
            max_tokens: Token budget. Defaults to config.context_max_tokens.
            include_micro: If True, include per-turn micro summaries under
                           each topic block (more detail, more tokens).

        Returns:
            ContextPayload ready to export.
        """
        budget = max_tokens or self.config.context_max_tokens

        # Reserve tokens for: system note + recent turns + overhead
        system_tokens = tokenizer.count(system_note)
        recent_tokens = sum(tokenizer.count(t.content) + 4 for t in hot_turns)
        overhead = 50  # headers, separators
        past_budget = budget - system_tokens - recent_tokens - overhead

        # Build past blocks within budget
        past_blocks, included_ids, was_truncated = self._build_past_blocks(
            results, past_budget, include_micro
        )

        total_tokens = system_tokens + recent_tokens + overhead + sum(
            tokenizer.count(b) for b in past_blocks
        )

        return ContextPayload(
            past_blocks=past_blocks,
            recent_turns=hot_turns,
            system_note=system_note,
            token_count=total_tokens,
            retrieved_node_ids=included_ids,
            was_truncated=was_truncated,
        )

    def build_stateless(
        self,
        results: list[RetrievalResult],
        all_turns: list[Turn],
        max_tokens: int | None = None,
    ) -> ContextPayload:
        """
        Stateless mode: no separate hot tier. Split all_turns into
        recent (last hot_tier_max_turns) and past (compress the rest).
        """
        n_hot = self.config.hot_tier_max_turns
        hot_turns = all_turns[-n_hot:] if len(all_turns) > n_hot else all_turns
        return self.build(results, hot_turns, max_tokens=max_tokens)

    def build_empty(self, hot_turns: list[Turn]) -> ContextPayload:
        """
        Build a payload with no past context — just recent turns.
        Used when the warm tier is empty (brand new session).
        """
        recent_tokens = sum(tokenizer.count(t.content) + 4 for t in hot_turns)
        return ContextPayload(
            past_blocks=[],
            recent_turns=hot_turns,
            system_note=_SYSTEM_NOTE,
            token_count=recent_tokens,
            retrieved_node_ids=[],
            was_truncated=False,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_past_blocks(
        self,
        results: list[RetrievalResult],
        budget: int,
        include_micro: bool,
    ) -> tuple[list[str], list[str], bool]:
        """
        Greedily pack retrieval results into text blocks within the token budget.
        Returns (blocks, included_node_ids, was_truncated).
        """
        blocks: list[str] = []
        included_ids: list[str] = []
        used_tokens = 0
        was_truncated = False

        for result in results:
            block = self._format_block(result, include_micro)
            block_tokens = tokenizer.count(block)

            if used_tokens + block_tokens > budget:
                # Try truncating the block to fit remaining budget
                remaining = budget - used_tokens
                if remaining > 30:   # worth including a partial block
                    block = self._truncate_block(result, remaining)
                    blocks.append(block)
                    included_ids.append(result.node.node_id)
                was_truncated = True
                break

            blocks.append(block)
            included_ids.append(result.node.node_id)
            used_tokens += block_tokens

        return blocks, included_ids, was_truncated

    def _format_block(
        self,
        result: RetrievalResult,
        include_micro: bool,
    ) -> str:
        """
        Format a single retrieval result as a titled text block.

        Output:
            [Topic: <label>] (turns <start>-<end>)
            <meso summary>
            - <micro 1>       ← only if include_micro=True
            - <micro 2>
            ---
        """
        node = result.node
        header = f"[Topic: {node.topic_label}] (turns {node.turn_start}-{node.turn_end})"
        lines = [header, node.meso_summary]

        if include_micro and node.micro_summaries:
            for micro in node.micro_summaries:
                lines.append(f"  - {micro}")

        lines.append(_SEPARATOR)
        return "\n".join(lines)

    def _truncate_block(self, result: RetrievalResult, max_tokens: int) -> str:
        """Produce a shortened version of a block to fit remaining budget."""
        node = result.node
        header = f"[Topic: {node.topic_label}] (turns {node.turn_start}-{node.turn_end})"
        truncated_summary = tokenizer.truncate_to_tokens(
            node.meso_summary, max_tokens - tokenizer.count(header) - 5
        )
        return f"{header}\n{truncated_summary}\n{_SEPARATOR}"