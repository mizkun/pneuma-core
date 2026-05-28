"""RetryableLLMCall — exponential-backoff retry wrapper for LLM calls.

invariant: llm-response-validated-strict — 失敗時は N 回まで自動再試行し、
それでも失敗ならログ + 明示的フォールバック値を返す（exponential backoff）。

Issue #12.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pneuma_core.llm.validation import StructuredResponseError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableLLMCall:
    """任意の async LLM 呼び出しを最大 N 回リトライするラッパ.

    デフォルト挙動:
        - max_retries=2: 初回 + 2 回リトライ = 計 3 試行
        - exponential backoff: delay = initial_delay * backoff^attempt
        - retryable_exceptions に含まれる例外のみリトライ対象
    """

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff: float = 2.0,
        retryable_exceptions: tuple[type[BaseException], ...] = (
            StructuredResponseError,
        ),
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if initial_delay < 0:
            raise ValueError("initial_delay must be >= 0")
        if backoff <= 0:
            raise ValueError("backoff must be > 0")
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff = backoff
        self.retryable_exceptions = retryable_exceptions

    async def run(self, call: Callable[[], Awaitable[T]]) -> T:
        """call を実行。retryable_exceptions が出たら最大 N 回まで再試行する.

        Args:
            call: 引数なしの async callable（クロージャで引数を束ねること）.

        Returns:
            call が最終的に返した値.

        Raises:
            最後の試行で発生した例外（再送出）.
        """
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await call()
            except self.retryable_exceptions as e:
                last_exc = e
                if attempt >= self.max_retries:
                    logger.warning(
                        "LLM call exhausted retries (%d attempts): %s",
                        attempt + 1,
                        e,
                    )
                    raise
                delay = self.initial_delay * (self.backoff ** attempt)
                logger.info(
                    "LLM call attempt %d/%d failed (%s), retrying in %.2fs",
                    attempt + 1,
                    self.max_retries + 1,
                    type(e).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

        # 理論的には到達しないが安全策
        assert last_exc is not None
        raise last_exc
