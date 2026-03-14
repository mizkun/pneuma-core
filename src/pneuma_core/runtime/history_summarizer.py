"""HistorySummarizer: compresses overflow conversation history via LLM summary."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pneuma_core.llm.adapter import LLMRequest

if TYPE_CHECKING:
    from pneuma_core.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM_PROMPT = """\
あなたは会話履歴を要約するアシスタントです。
以下の会話履歴を簡潔に要約してください。重要な情報（名前、話題、決定事項、感情の変化など）を漏らさず、\
短くまとめてください。要約は日本語で書いてください。"""

_SUMMARIZE_WITH_PREVIOUS_PROMPT = """\
あなたは会話履歴を要約するアシスタントです。
以下に「これまでの要約」と「新しい会話履歴」があります。
これらを統合して、1つの簡潔な要約を作成してください。重要な情報（名前、話題、決定事項、感情の変化など）を漏らさず、\
短くまとめてください。要約は日本語で書いてください。

【これまでの要約】
{previous_summary}"""


class HistorySummarizer:
    """Summarizes overflow conversation history using LLM.

    When conversation history exceeds the limit, the overflow messages
    are summarized and the summary is prepended as a system message.
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self._llm = llm
        self._current_summary: str | None = None

    @property
    def current_summary(self) -> str | None:
        """Return the current accumulated summary, or None if no summary exists."""
        return self._current_summary

    def should_summarize(self, history: list[dict], limit: int) -> bool:
        """Check if history exceeds the limit and needs summarization."""
        return len(history) > limit

    async def summarize(
        self, history: list[dict], limit: int
    ) -> str | None:
        """Summarize overflow messages (those beyond the limit).

        Args:
            history: Full conversation history.
            limit: Maximum number of recent messages to keep.

        Returns:
            Summary text, or None if summarization failed.
        """
        if not self.should_summarize(history, limit):
            return self._current_summary

        overflow = history[: len(history) - limit]
        overflow_text = self._format_messages(overflow)

        try:
            if self._current_summary is not None:
                system_prompt = _SUMMARIZE_WITH_PREVIOUS_PROMPT.format(
                    previous_summary=self._current_summary
                )
            else:
                system_prompt = _SUMMARIZE_SYSTEM_PROMPT

            request = LLMRequest(
                system_prompt=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"以下の会話を要約してください:\n\n{overflow_text}",
                    }
                ],
                model="claude-haiku-4-5-20251001",
                temperature=0.3,
                max_tokens=512,
            )
            response = await self._llm.generate(request)
            self._current_summary = response.content
            return self._current_summary
        except Exception:
            logger.warning("History summarization failed, returning None")
            return None

    async def trim_with_summary(
        self, history: list[dict], limit: int
    ) -> list[dict]:
        """Trim history with summarization.

        If history exceeds limit, summarize overflow and return
        [summary_message] + recent messages. If summarization fails,
        fall back to simple truncation.

        Args:
            history: Full conversation history.
            limit: Maximum number of recent messages to keep.

        Returns:
            Trimmed history with optional summary at the head.
        """
        if not self.should_summarize(history, limit):
            return history

        summary = await self.summarize(history, limit)
        recent = history[-limit:]

        if summary is not None:
            summary_message = {
                "role": "system",
                "content": f"[要約] {summary}",
            }
            return [summary_message] + recent

        # Fallback: simple truncation
        return recent

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """Format messages for the summarization prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
