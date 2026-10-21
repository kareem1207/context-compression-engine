"""
CCE Context Builder
Takes retrieval results + hot tier turns and assembles the final
compressed context payload that gets injected into the LLM prompt.

The payload has three sections, ordered for LLM readability:

  [1] PAST CONTEXT (warm tier — compressed)
      Retrieved memory nodes, formatted as titled topic blocks.
      Most relevant first (already ranked by Retriever).

  [1b] EARLIER MESSAGES (live sessions — verbatim)
      Turns that left the hot tier but are not in a topic note yet. Included
      as they were said, so nothing is missing between past and recent context.

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

from dataclasses import dataclass, field, replace
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
_PINNED_HEADER = "=== PINNED (always included) ==="
_PENDING_HEADER = "=== EARLIER MESSAGES (not summarised yet) ==="
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
    pending_turns: list[Turn] = field(default_factory=list)
    explain: dict = field(default_factory=dict)   # why this payload looks as it does (see ContextBuilder.build)
    pinned: list[str] = field(default_factory=list)   # texts pinned by the user, sent every time

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
        if self.pinned:
            system_parts.append("")
            system_parts.append(_PINNED_HEADER)
            system_parts.extend(f"- {p}" for p in self.pinned)
        if self.past_blocks:
            system_parts.append("")
            system_parts.append(_PAST_HEADER)
            system_parts.extend(self.past_blocks)
        if self.pending_turns:
            system_parts.append("")
            system_parts.append(_PENDING_HEADER)
            system_parts.extend(f"{t.role.upper()}: {t.content}" for t in self.pending_turns)

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

        if self.pinned:
            parts.append(_PINNED_HEADER)
            parts.extend(f"- {p}" for p in self.pinned)
            parts.append("")

        if self.past_blocks:
            parts.append(_PAST_HEADER)
            parts.extend(self.past_blocks)
            parts.append("")

        if self.pending_turns:
            parts.append(_PENDING_HEADER)
            parts.extend(f"{t.role.upper()}: {t.content}" for t in self.pending_turns)
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
            "pending_turns": [
                {"role": t.role, "content": t.content} for t in self.pending_turns
            ],
            "recent_turns": [
                {"role": t.role, "content": t.content} for t in self.recent_turns
            ],
            "token_count": self.token_count,
            "retrieved_node_ids": self.retrieved_node_ids,
            "was_truncated": self.was_truncated,
            "explain": self.explain,
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
        pending_turns: list[Turn] | None = None,
        pending_rank_fn=None,
        pinned: list[str] | None = None,
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
            pending_turns: Turns not yet compressed into a note (live sessions), oldest first.
                           Included verbatim; if they do not fit, `pending_rank_fn` (turns ->
                           turns, best match first) picks which, otherwise the newest are kept.

        Returns:
            ContextPayload ready to export.
        """
        budget = max_tokens or self.config.context_max_tokens
        budget -= max(16, budget // 32)   # counting the whole text at once can differ a little from counting its parts

        # Reserve tokens for: system note + recent turns + overhead
        system_tokens = tokenizer.count(system_note)
        overhead = 80  # headers, separators, line breaks between blocks
        # The cap is hard: if the recent turns alone do not fit, the oldest are shortened or dropped.
        hot_turns, recent_cut = self._fit_recent(hot_turns, budget - system_tokens - overhead)
        recent_tokens = sum(tokenizer.count(t.content) + 4 for t in hot_turns)
        past_budget = budget - system_tokens - recent_tokens - overhead

        # Pinned text is sent every time, before anything else competes for the budget (at most a quarter of it).
        pins: list[str] = []
        pin_tokens = 0
        for text in pinned or []:
            cost = tokenizer.count(text) + 4
            if pin_tokens + cost > max(past_budget // 4, 0):
                break
            pins.append(text)
            pin_tokens += cost
        if pins:
            pin_tokens += 12                                    # header
        past_budget -= pin_tokens

        # Pending turns come first: they are the only copy of that part of the conversation.
        pending, pending_tokens, pending_cut = self._pack_pending(
            pending_turns or [], past_budget if not results else int(past_budget * 0.6), pending_rank_fn
        )
        past_budget -= pending_tokens

        if results and all(r.overview_lines for r in results):
            self._share_overview_budget(results, past_budget)

        # Build past blocks within budget
        past_blocks, included_ids, was_truncated = self._build_past_blocks(
            results, past_budget, include_micro
        )

        total_tokens = system_tokens + recent_tokens + overhead + pending_tokens + pin_tokens + sum(
            tokenizer.count(b) for b in past_blocks
        )

        past_tokens = [tokenizer.count(b) for b in past_blocks]
        block_tokens = dict(zip(included_ids, past_tokens))
        explain = {
            "budget": {
                "max_tokens": max_tokens or self.config.context_max_tokens,
                "usable": budget,
                "system_note": system_tokens,
                "overhead": overhead,
                "recent_turns": recent_tokens,
                "pending_turns": pending_tokens,
                "pinned": pin_tokens,
                "past_notes": sum(past_tokens),
                "total": total_tokens,
            },
            "recent": {"turns": len(hot_turns), "shortened_or_dropped": recent_cut},
            "pending": {"turns_included": len(pending), "turns_available": len(pending_turns or []), "cut": pending_cut},
            "notes": [
                {
                    "node_id": r.node.node_id,
                    "topic": r.node.topic_label,
                    "turns": [r.node.turn_start, r.node.turn_end],
                    "included": r.node.node_id in block_tokens,
                    "tokens": block_tokens.get(r.node.node_id, 0),
                    "scores": {
                        "dense": round(r.semantic_score, 4),
                        "recency": round(r.recency_score, 4),
                        "bm25": round(r.keyword_score, 4),
                        "fused": round(r.composite_score, 5),
                    },
                    "quotes": [q[:200] for q in (r.evidence or ([r.best_turn] if r.best_turn else []))],
                    "overview_lines": r.overview_lines,
                }
                for r in results
            ],
            "dropped_for_budget": [r.node.node_id for r in results if r.node.node_id not in block_tokens],
        }

        return ContextPayload(
            past_blocks=past_blocks,
            recent_turns=hot_turns,
            system_note=system_note,
            token_count=total_tokens,
            retrieved_node_ids=included_ids,
            was_truncated=was_truncated or pending_cut or recent_cut,
            pending_turns=pending,
            explain=explain,
            pinned=pins,
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

    def build_empty(
        self,
        hot_turns: list[Turn],
        pending_turns: list[Turn] | None = None,
        pending_rank_fn=None,
        pinned: list[str] | None = None,
    ) -> ContextPayload:
        """
        Build a payload with no past context — just recent turns (and pending ones, if any).
        Used when the warm tier is empty (brand new session).
        """
        return self.build([], hot_turns, pending_turns=pending_turns, pending_rank_fn=pending_rank_fn, pinned=pinned)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _share_overview_budget(self, results: list[RetrievalResult], budget: int) -> None:
        """Every topic of an overview gets a share of the budget: fewer and shorter lines when there are many."""
        def cost() -> int:
            return sum(tokenizer.count(self._format_block(r, False)) + 2 for r in results)

        for lines_each in (None, 1):
            if lines_each:
                for r in results:
                    r.overview_lines = r.overview_lines[:lines_each]
            if cost() <= budget:
                return
        for limit in (20, 14, 10, 7):
            for r in results:
                r.overview_lines = [tokenizer.truncate_to_tokens(l, limit) for l in r.overview_lines]
            if cost() <= budget:
                return

    def _fit_recent(self, turns: list[Turn], budget: int) -> tuple[list[Turn], bool]:
        """Keep the newest turns that fit in `budget` tokens; shorten the first one that does not."""
        kept: list[Turn] = []
        left = budget
        cut = False
        for t in reversed(turns):
            cost = tokenizer.count(t.content) + 4
            if cost <= left:
                kept.append(t)
                left -= cost
                continue
            cut = True
            if left > 24:
                kept.append(replace(t, content=tokenizer.truncate_to_tokens(t.content, left - 4)))
            break
        kept.reverse()
        return kept, cut

    def _pack_pending(
        self,
        pending: list[Turn],
        budget: int,
        rank_fn,
    ) -> tuple[list[Turn], int, bool]:
        """Fit pending turns into `budget` tokens. Returns (turns in order, tokens used, was_cut)."""
        if not pending or budget <= 0:
            return [], 0, bool(pending)
        cost = {t.index: tokenizer.count(f"{t.role.upper()}: {t.content}") + 4 for t in pending}
        if sum(cost.values()) + 12 <= budget:
            return list(pending), sum(cost.values()) + 12, False
        order = rank_fn(list(pending)) if rank_fn else list(reversed(pending))
        picked: list[Turn] = []
        used = 12   # header
        for t in order:
            if used + cost[t.index] <= budget:
                picked.append(t)
                used += cost[t.index]
        picked.sort(key=lambda t: t.index)
        return picked, (used if picked else 0), True

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
            block_tokens = tokenizer.count(block) + 2   # + the line break that joins it to the next

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
        if result.overview_lines:      # overview question: a few representative sentences, not the summary
            return "\n".join([header, *(f"- {l}" for l in result.overview_lines), _SEPARATOR])
        lines = [header, node.meso_summary]

        # Quote the messages that match the question best when the summary left them out.
        quotes = list(result.evidence) or ([result.best_turn] if result.best_turn else [])
        seen = set()
        for q in quotes:
            q = (q or "").strip()
            if q and q not in node.meso_summary and q not in seen:
                seen.add(q)
                lines.append(f"Relevant: {tokenizer.truncate_to_tokens(q, 80)}")

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