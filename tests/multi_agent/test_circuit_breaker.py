"""Tests for CircuitBreaker (Issue #6 / Phase 0 C).

AC: 1 セッション当たりの LLM 呼び出し回数・TTS リクエスト数・elapsed time
に上限がある（spec invariant: turn-rate-limit）。
"""

from __future__ import annotations

import time

import pytest

from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


class TestCircuitBreakerLimits:
    """AC-1: 各上限を超えるとセッションは tripped 状態に遷移する."""

    def test_turn_limit(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(max_turns=3))
        for _ in range(3):
            assert cb.allow() is True
            cb.record_turn()
        # 4 ターン目で拒否
        assert cb.allow() is False
        assert cb.state == CircuitState.TRIPPED
        assert "turn" in cb.trip_reason

    def test_token_limit(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(max_tokens=100))
        assert cb.allow() is True
        cb.record_tokens(50)
        assert cb.allow() is True
        cb.record_tokens(60)
        assert cb.allow() is False
        assert "token" in cb.trip_reason

    def test_elapsed_limit(self) -> None:
        cb = CircuitBreaker(
            CircuitBreakerConfig(max_elapsed_seconds=0.05)
        )
        cb.start()
        assert cb.allow() is True
        time.sleep(0.1)
        assert cb.allow() is False
        assert "elapsed" in cb.trip_reason

    def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.state == CircuitState.CLOSED

    def test_remaining_turns(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(max_turns=5))
        assert cb.remaining_turns() == 5
        cb.record_turn()
        cb.record_turn()
        assert cb.remaining_turns() == 3


class TestCircuitBreakerDefaults:
    """Default config is generous enough for a single session."""

    def test_default_allows_many_turns(self) -> None:
        cb = CircuitBreaker()
        # default should easily allow 50 turns
        for _ in range(50):
            assert cb.allow() is True
            cb.record_turn()
