"""Tests for RetryableLLMCall (Issue #12 / AC-2).

invariant: llm-response-validated-strict — 失敗時は N 回まで自動再試行し、
それでも失敗ならログ + 明示的フォールバック値を返す（exponential backoff）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from pneuma_core.llm.retry import RetryableLLMCall
from pneuma_core.llm.validation import StructuredResponseError


class TestRetryableLLMCall:
    @pytest.mark.asyncio
    async def test_returns_first_success(self) -> None:
        """1 回目で成功したらそのまま値を返す."""
        call = AsyncMock(return_value={"ok": True})
        runner = RetryableLLMCall(max_retries=2, initial_delay=0.0, backoff=2.0)

        result = await runner.run(call)

        assert result == {"ok": True}
        assert call.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_until_success(self) -> None:
        """途中で成功したら以降は呼ばない."""
        call = AsyncMock(side_effect=[
            StructuredResponseError("fail 1"),
            {"ok": True},
        ])
        runner = RetryableLLMCall(max_retries=2, initial_delay=0.0, backoff=2.0)

        result = await runner.run(call)

        assert result == {"ok": True}
        assert call.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_raises(self) -> None:
        """N 回全部失敗で最後の例外を再送出（max_retries=2 → 試行 3 回）."""
        err = StructuredResponseError("always fail")
        call = AsyncMock(side_effect=err)
        runner = RetryableLLMCall(max_retries=2, initial_delay=0.0, backoff=2.0)

        with pytest.raises(StructuredResponseError):
            await runner.run(call)

        # max_retries=2 means 1 initial + 2 retries = 3 attempts
        assert call.call_count == 3

    @pytest.mark.asyncio
    async def test_default_max_retries_is_two(self) -> None:
        """デフォルト max_retries = 2."""
        runner = RetryableLLMCall()
        assert runner.max_retries == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        """exponential backoff: initial_delay * backoff^attempt の間隔で sleep."""
        err = StructuredResponseError("fail")
        call = AsyncMock(side_effect=err)
        runner = RetryableLLMCall(max_retries=3, initial_delay=0.5, backoff=2.0)

        with patch("pneuma_core.llm.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(StructuredResponseError):
                await runner.run(call)

        # max_retries=3 → 3 retries → 3 sleep calls
        # delays: 0.5, 1.0, 2.0
        delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert delays == [0.5, 1.0, 2.0]

    @pytest.mark.asyncio
    async def test_no_sleep_when_initial_delay_zero(self) -> None:
        """initial_delay=0 でも sleep(0) は呼ぶ（互換のため）."""
        err = StructuredResponseError("fail")
        call = AsyncMock(side_effect=err)
        runner = RetryableLLMCall(max_retries=1, initial_delay=0.0, backoff=2.0)

        with patch("pneuma_core.llm.retry.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(StructuredResponseError):
                await runner.run(call)

        # 1 retry → 1 sleep call (with delay=0.0)
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args_list[0].args[0] == 0.0

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates(self) -> None:
        """retryable_exceptions に含まれない例外はリトライせず即時再送出."""
        err = ValueError("not retryable")
        call = AsyncMock(side_effect=err)
        runner = RetryableLLMCall(
            max_retries=2,
            initial_delay=0.0,
            retryable_exceptions=(StructuredResponseError,),
        )

        with pytest.raises(ValueError):
            await runner.run(call)

        assert call.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_zero_means_one_attempt(self) -> None:
        """max_retries=0 だと再試行なし（1 回呼ぶだけ）."""
        err = StructuredResponseError("fail")
        call = AsyncMock(side_effect=err)
        runner = RetryableLLMCall(max_retries=0, initial_delay=0.0)

        with pytest.raises(StructuredResponseError):
            await runner.run(call)

        assert call.call_count == 1
