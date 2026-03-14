"""ProactiveEngine: decides when and how a character should proactively reach out.

This is NOT the World Engine ThinkCycle. This is a simpler system that:
1. Periodically evaluates if the character has something to say
2. Considers time of day, recent activity, pending items
3. Generates a natural message if reaching out is appropriate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProactiveConfig:
    """Configuration for proactive reach-out behaviour."""

    check_interval_seconds: float = 3600.0  # Check every hour
    model: str = "claude-haiku-4-5-20251001"
    quiet_hours_start: int = 23  # Don't reach out after 11pm
    quiet_hours_end: int = 7  # Don't reach out before 7am
    max_reach_outs_per_day: int = 3  # Don't spam


@dataclass
class ProactiveResult:
    """Result of a proactive evaluation."""

    should_send: bool
    message: str | None = None
    reason: str | None = None  # Why reaching out (for logging)


class ProactiveEngine:
    """Decides when and how a character should proactively reach out to the user.

    This is a lightweight system designed for a single character's proactive
    behaviour (e.g., morning greetings, TODO reminders, check-ins).
    """

    def __init__(
        self,
        character_id: str,
        llm: LLMAdapter,
        config: ProactiveConfig | None = None,
    ) -> None:
        self._character_id = character_id
        self._llm = llm
        self._config = config or ProactiveConfig()
        self._reach_out_count_today: int = 0
        self._last_reach_out_date: date | None = None
        self._last_check_time: datetime | None = None

    # ------------------------------------------------------------------
    # Public checks
    # ------------------------------------------------------------------

    def should_check(self, now: datetime) -> bool:
        """Whether enough time has passed to check for proactive actions."""
        if self._last_check_time is None:
            return True
        elapsed = (now - self._last_check_time).total_seconds()
        return elapsed >= self._config.check_interval_seconds

    def is_quiet_hours(self, now: datetime) -> bool:
        """Whether it's currently quiet hours (don't disturb).

        Handles the case where quiet hours wrap past midnight
        (e.g., 23:00 - 07:00).
        """
        hour = now.hour
        if self._config.quiet_hours_start > self._config.quiet_hours_end:
            # Wraps midnight: e.g., 23-7
            return hour >= self._config.quiet_hours_start or hour < self._config.quiet_hours_end
        return self._config.quiet_hours_start <= hour < self._config.quiet_hours_end

    def can_reach_out(self, now: datetime) -> bool:
        """Whether reaching out is allowed (not quiet hours, not over daily limit)."""
        if self.is_quiet_hours(now):
            return False
        # Reset counter on new day
        today = now.date()
        if self._last_reach_out_date != today:
            self._reach_out_count_today = 0
            self._last_reach_out_date = today
        return self._reach_out_count_today < self._config.max_reach_outs_per_day

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        now: datetime,
        context: str = "",
    ) -> ProactiveResult:
        """Evaluate whether to reach out and generate a message if so.

        Args:
            now: Current time (should be JST for proper quiet hours).
            context: Additional context (e.g., pending TODOs, last conversation summary).

        Returns:
            ProactiveResult with should_send and optional message.
        """
        self._last_check_time = now

        if not self.can_reach_out(now):
            return ProactiveResult(should_send=False)

        # Ask LLM if there's something worth saying
        result = await self._generate(now, context)

        if result.should_send:
            self._reach_out_count_today += 1

        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _generate(self, now: datetime, context: str) -> ProactiveResult:
        """Ask the LLM whether the character should reach out."""
        time_str = now.strftime("%Y-%m-%d %H:%M")

        context_block = ""
        if context:
            context_block = f"\n追加コンテキスト:\n{context}\n"

        system_prompt = (
            f"あなたは「{self._character_id}」です。ユーザーに自分から話しかけるかどうかを判断してください。\n"
            f"\n"
            f"現在時刻: {time_str}\n"
            f"{context_block}\n"
            f"以下の場合に話しかけてください:\n"
            f"- 朝の挨拶の時間帯（7-9時）\n"
            f"- 未完了のTODOがある場合のリマインド\n"
            f"- 前回の会話から時間が経っている場合の気遣い\n"
            f"- 特に話したいことがある場合\n"
            f"\n"
            f"以下のJSON形式で応答してください:\n"
            f'{{\n'
            f'  "should_send": true/false,\n'
            f'  "message": "送信するメッセージ（should_send=trueの場合）",\n'
            f'  "reason": "判断理由（ログ用）"\n'
            f'}}\n'
            f"\n"
            f"話しかけない場合は should_send: false で応答してください。\n"
            f"無理に話しかける必要はありません。"
        )

        try:
            response = await self._llm.generate(
                LLMRequest(
                    system_prompt=system_prompt,
                    messages=[],
                    model=self._config.model,
                )
            )
            return self._parse_response(response.content)
        except Exception:
            logger.warning("ProactiveEngine: LLM generation failed")
            return ProactiveResult(should_send=False, reason="LLM error")

    @staticmethod
    def _parse_response(raw: str) -> ProactiveResult:
        """Parse the JSON response from the LLM."""
        # Strip markdown code fences if present (Haiku tendency)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("ProactiveEngine: failed to parse LLM response as JSON")
            return ProactiveResult(should_send=False, reason="parse error")

        should_send = bool(data.get("should_send", False))
        message = data.get("message") if should_send else None
        reason = data.get("reason")

        return ProactiveResult(
            should_send=should_send,
            message=message,
            reason=reason,
        )
