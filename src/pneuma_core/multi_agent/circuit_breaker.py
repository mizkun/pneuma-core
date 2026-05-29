"""CircuitBreaker: コスト暴走防止のセーフティ.

invariant: turn-rate-limit — 1 セッション当たりの LLM 呼び出し回数・
TTS リクエスト数・経過時間に上限がある。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    """サーキットブレーカーの状態."""

    CLOSED = "closed"      # 通常運転（リクエスト通す）
    TRIPPED = "tripped"    # 上限超え（拒否）


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """サーキットブレーカーの上限値.

    Defaults are chosen so that a typical 15-min Phase 0 C2 text run
    can complete (~50-100 turns).
    """

    max_turns: int = 200
    max_tokens: int = 200_000
    max_elapsed_seconds: float = 30 * 60.0  # 30 分


class CircuitBreaker:
    """ターン / トークン / 経過時間で session を強制終了するセーフティ.

    Usage:
        cb = CircuitBreaker()
        cb.start()
        while cb.allow():
            ...
            cb.record_turn()
            cb.record_tokens(usage.input_tokens + usage.output_tokens)
        if cb.state == CircuitState.TRIPPED:
            print(f"Circuit tripped: {cb.trip_reason}")
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._turn_count = 0
        self._token_count = 0
        self._started_at: float | None = None
        self._trip_reason: str = ""

    def start(self) -> None:
        """セッション開始時刻を記録."""
        self._started_at = time.monotonic()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def max_turns(self) -> int:
        """1 セッションのターン上限（構造的オチの終盤判定などに使う）."""
        return self._config.max_turns

    def remaining_turns(self) -> int:
        return max(0, self._config.max_turns - self._turn_count)

    def allow(self) -> bool:
        """次のターンを許可するか判定。許可しない場合は state を TRIPPED に."""
        if self._state == CircuitState.TRIPPED:
            return False

        if self._turn_count >= self._config.max_turns:
            self._trip(
                f"turn limit reached: {self._turn_count} >= "
                f"{self._config.max_turns}"
            )
            return False

        if self._token_count >= self._config.max_tokens:
            self._trip(
                f"token limit reached: {self._token_count} >= "
                f"{self._config.max_tokens}"
            )
            return False

        if (
            self._started_at is not None
            and (time.monotonic() - self._started_at)
            >= self._config.max_elapsed_seconds
        ):
            elapsed = time.monotonic() - self._started_at
            self._trip(
                f"elapsed time limit reached: {elapsed:.1f}s >= "
                f"{self._config.max_elapsed_seconds:.1f}s"
            )
            return False

        return True

    def record_turn(self) -> None:
        self._turn_count += 1

    def record_tokens(self, tokens: int) -> None:
        self._token_count += max(0, int(tokens))

    def _trip(self, reason: str) -> None:
        self._state = CircuitState.TRIPPED
        self._trip_reason = reason
