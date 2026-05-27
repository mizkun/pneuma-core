"""Tests for text_runner CLI (Issue #7 / Phase 0 C2)."""

from __future__ import annotations

import pytest

from pneuma_core.cli.text_runner import run


@pytest.mark.asyncio
async def test_text_runner_completes_within_turn_limit(capsys) -> None:
    """AC: --turn-limit 5 --use-mock-llm でエラーなく完走する."""
    rc = await run(
        duration_minutes=1.0,
        context="部室",
        turn_limit=5,
        use_mock=True,
        loop_delay_seconds=0.0,
    )
    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out
    # 全 3 キャラの名前が出る
    assert "なでしこ" in out
    assert "千明" in out
    assert "あおい" in out
    # PAD 表示の形式が出る
    assert "P=" in out and "A=" in out and "D=" in out
    # circuit breaker trip メッセージ
    assert "circuit breaker tripped" in out or "Done:" in out


@pytest.mark.asyncio
async def test_text_runner_short_session(capsys) -> None:
    """AC: 2 ターンでも session end まで含めて完走."""
    rc = await run(
        duration_minutes=1.0,
        context="テスト",
        turn_limit=2,
        use_mock=True,
        loop_delay_seconds=0.0,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Session End" in out
    assert "Final Snapshot" in out
