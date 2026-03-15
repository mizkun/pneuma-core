"""Tests for ProactiveEngine — decides when a character should proactively reach out."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest, LLMResponse
from pneuma_core.runtime.proactive import ProactiveConfig, ProactiveEngine, ProactiveResult

JST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jst(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Convenience for creating JST datetimes."""
    return datetime(year, month, day, hour, minute, tzinfo=JST)


def _make_llm_response(should_send: bool, message: str | None = None, reason: str | None = None) -> LLMResponse:
    """Create an LLMResponse that mimics the JSON output expected from the proactive prompt."""
    payload = {"should_send": should_send}
    if message is not None:
        payload["message"] = message
    if reason is not None:
        payload["reason"] = reason
    return LLMResponse(
        content=json.dumps(payload, ensure_ascii=False),
        model="claude-haiku-4-5-20251001",
        usage={"input_tokens": 100, "output_tokens": 50},
    )


def _build_engine(
    llm: LLMAdapter | None = None,
    config: ProactiveConfig | None = None,
) -> ProactiveEngine:
    mock_llm = llm or AsyncMock(spec=LLMAdapter)
    return ProactiveEngine(
        character_id="mira",
        llm=mock_llm,
        config=config,
    )


# ===========================================================================
# TestProactiveConfig
# ===========================================================================

class TestProactiveConfig:
    """Default and custom config values."""

    def test_defaults(self) -> None:
        cfg = ProactiveConfig()
        assert cfg.check_interval_seconds == 3600.0
        assert cfg.model == "claude-haiku-4-5-20251001"
        assert cfg.quiet_hours_start == 23
        assert cfg.quiet_hours_end == 7
        assert cfg.max_reach_outs_per_day == 3

    def test_custom_values(self) -> None:
        cfg = ProactiveConfig(
            check_interval_seconds=1800.0,
            model="claude-sonnet-4-20250514",
            quiet_hours_start=22,
            quiet_hours_end=8,
            max_reach_outs_per_day=5,
        )
        assert cfg.check_interval_seconds == 1800.0
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.quiet_hours_start == 22
        assert cfg.quiet_hours_end == 8
        assert cfg.max_reach_outs_per_day == 5


# ===========================================================================
# TestQuietHours
# ===========================================================================

class TestQuietHours:
    """Quiet hours boundary tests (default: 23:00 - 07:00 JST)."""

    def test_is_quiet_hours_late_night(self) -> None:
        engine = _build_engine()
        now = _jst(2026, 3, 2, 23, 30)
        assert engine.is_quiet_hours(now) is True

    def test_is_quiet_hours_early_morning(self) -> None:
        engine = _build_engine()
        now = _jst(2026, 3, 2, 5, 0)
        assert engine.is_quiet_hours(now) is True

    def test_is_not_quiet_hours_daytime(self) -> None:
        engine = _build_engine()
        now = _jst(2026, 3, 2, 12, 0)
        assert engine.is_quiet_hours(now) is False

    def test_is_not_quiet_hours_evening(self) -> None:
        engine = _build_engine()
        now = _jst(2026, 3, 2, 20, 0)
        assert engine.is_quiet_hours(now) is False

    def test_boundary_start(self) -> None:
        """23:00 exactly is quiet hours."""
        engine = _build_engine()
        now = _jst(2026, 3, 2, 23, 0)
        assert engine.is_quiet_hours(now) is True

    def test_boundary_end(self) -> None:
        """07:00 exactly is NOT quiet hours (end is exclusive)."""
        engine = _build_engine()
        now = _jst(2026, 3, 2, 7, 0)
        assert engine.is_quiet_hours(now) is False


# ===========================================================================
# TestCanReachOut
# ===========================================================================

class TestCanReachOut:
    """Whether reaching out is allowed."""

    def test_can_reach_out_normal(self) -> None:
        """Daytime, under daily limit -> True."""
        engine = _build_engine()
        now = _jst(2026, 3, 2, 12, 0)
        assert engine.can_reach_out(now) is True

    def test_cannot_reach_out_quiet_hours(self) -> None:
        """During quiet hours -> False."""
        engine = _build_engine()
        now = _jst(2026, 3, 2, 23, 30)
        assert engine.can_reach_out(now) is False

    def test_cannot_reach_out_daily_limit(self) -> None:
        """Reached max_reach_outs_per_day -> False."""
        engine = _build_engine(config=ProactiveConfig(max_reach_outs_per_day=2))
        now = _jst(2026, 3, 2, 12, 0)
        # Simulate 2 reach-outs already done today
        engine._reach_out_count_today = 2
        engine._last_reach_out_date = now.date()
        assert engine.can_reach_out(now) is False

    def test_daily_limit_resets_on_new_day(self) -> None:
        """New day resets counter -> True."""
        engine = _build_engine(config=ProactiveConfig(max_reach_outs_per_day=2))
        yesterday = _jst(2026, 3, 1, 12, 0)
        engine._reach_out_count_today = 2
        engine._last_reach_out_date = yesterday.date()

        today = _jst(2026, 3, 2, 12, 0)
        assert engine.can_reach_out(today) is True
        # Counter should have been reset
        assert engine._reach_out_count_today == 0


# ===========================================================================
# TestShouldCheck
# ===========================================================================

class TestShouldCheck:
    """Whether enough time has passed to check for proactive actions."""

    def test_first_check_always_true(self) -> None:
        """No previous check -> True."""
        engine = _build_engine()
        now = _jst(2026, 3, 2, 12, 0)
        assert engine.should_check(now) is True

    def test_check_interval_respected(self) -> None:
        """Too soon -> False."""
        engine = _build_engine(config=ProactiveConfig(check_interval_seconds=3600.0))
        t1 = _jst(2026, 3, 2, 12, 0)
        engine._last_check_time = t1

        t2 = t1 + timedelta(minutes=30)  # Only 30 min later
        assert engine.should_check(t2) is False

    def test_check_after_interval(self) -> None:
        """Enough time passed -> True."""
        engine = _build_engine(config=ProactiveConfig(check_interval_seconds=3600.0))
        t1 = _jst(2026, 3, 2, 12, 0)
        engine._last_check_time = t1

        t2 = t1 + timedelta(hours=1, minutes=1)  # 61 min later
        assert engine.should_check(t2) is True


# ===========================================================================
# TestEvaluate
# ===========================================================================

class TestEvaluate:
    """Async evaluate method tests."""

    @pytest.mark.asyncio
    async def test_evaluate_sends_message(self) -> None:
        """LLM says should_send=true -> message returned."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        mock_llm.generate.return_value = _make_llm_response(
            should_send=True,
            message="おはよう！今日もいい天気だね。",
            reason="朝の挨拶の時間帯",
        )
        engine = _build_engine(llm=mock_llm)
        now = _jst(2026, 3, 2, 8, 0)

        result = await engine.evaluate(now)

        assert result.should_send is True
        assert result.message == "おはよう！今日もいい天気だね。"
        assert result.reason == "朝の挨拶の時間帯"

    @pytest.mark.asyncio
    async def test_evaluate_quiet_hours_skips_llm(self) -> None:
        """Quiet hours -> no LLM call, should_send=False."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        engine = _build_engine(llm=mock_llm)
        now = _jst(2026, 3, 2, 23, 30)

        result = await engine.evaluate(now)

        assert result.should_send is False
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_increments_counter(self) -> None:
        """Successful reach_out increments count."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        mock_llm.generate.return_value = _make_llm_response(
            should_send=True,
            message="元気？",
            reason="気遣い",
        )
        engine = _build_engine(llm=mock_llm)
        now = _jst(2026, 3, 2, 12, 0)

        assert engine._reach_out_count_today == 0

        await engine.evaluate(now)
        assert engine._reach_out_count_today == 1

        # Second call
        now2 = now + timedelta(minutes=5)
        await engine.evaluate(now2)
        assert engine._reach_out_count_today == 2

    @pytest.mark.asyncio
    async def test_evaluate_no_send_does_not_increment(self) -> None:
        """LLM says should_send=false -> counter unchanged."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        mock_llm.generate.return_value = _make_llm_response(should_send=False, reason="特に用はない")
        engine = _build_engine(llm=mock_llm)
        now = _jst(2026, 3, 2, 12, 0)

        result = await engine.evaluate(now)
        assert result.should_send is False
        assert engine._reach_out_count_today == 0

    @pytest.mark.asyncio
    async def test_evaluate_uses_configured_model(self) -> None:
        """LLM request uses the model from config."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        mock_llm.generate.return_value = _make_llm_response(should_send=False)
        config = ProactiveConfig(model="claude-haiku-4-5-20251001")
        engine = _build_engine(llm=mock_llm, config=config)
        now = _jst(2026, 3, 2, 12, 0)

        await engine.evaluate(now)

        # Verify the LLM was called with the right model
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        request: LLMRequest = call_args[0][0]
        assert request.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_evaluate_updates_last_check_time(self) -> None:
        """evaluate() should update _last_check_time regardless of result."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        mock_llm.generate.return_value = _make_llm_response(should_send=False)
        engine = _build_engine(llm=mock_llm)
        now = _jst(2026, 3, 2, 12, 0)

        assert engine._last_check_time is None
        await engine.evaluate(now)
        assert engine._last_check_time == now

    @pytest.mark.asyncio
    async def test_evaluate_passes_context_to_llm(self) -> None:
        """Context string is included in the LLM prompt."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        mock_llm.generate.return_value = _make_llm_response(should_send=False)
        engine = _build_engine(llm=mock_llm)
        now = _jst(2026, 3, 2, 12, 0)

        await engine.evaluate(now, context="未完了TODO: レポート提出")

        mock_llm.generate.assert_called_once()
        request: LLMRequest = mock_llm.generate.call_args[0][0]
        assert "未完了TODO: レポート提出" in request.system_prompt

    @pytest.mark.asyncio
    async def test_evaluate_quiet_hours_still_updates_check_time(self) -> None:
        """Even during quiet hours, _last_check_time is updated."""
        mock_llm = AsyncMock(spec=LLMAdapter)
        engine = _build_engine(llm=mock_llm)
        now = _jst(2026, 3, 2, 2, 0)  # 2am = quiet

        await engine.evaluate(now)
        assert engine._last_check_time == now


# ===========================================================================
# TestProactiveResult
# ===========================================================================

class TestProactiveResult:
    """ProactiveResult data class."""

    def test_default_values(self) -> None:
        r = ProactiveResult(should_send=False)
        assert r.should_send is False
        assert r.message is None
        assert r.reason is None

    def test_with_all_fields(self) -> None:
        r = ProactiveResult(
            should_send=True,
            message="おはよう！",
            reason="朝の挨拶",
        )
        assert r.should_send is True
        assert r.message == "おはよう！"
        assert r.reason == "朝の挨拶"
