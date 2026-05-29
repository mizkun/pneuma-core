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
        save_log=False,
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
        save_log=False,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Session End" in out
    assert "Final Snapshot" in out


@pytest.mark.asyncio
async def test_text_runner_saves_transcript_log(tmp_path) -> None:
    """AC (Issue #7): save_log=True で log_dir にトランスクリプトが Markdown 保存される."""
    rc = await run(
        duration_minutes=1.0,
        context="ログ保存テスト",
        turn_limit=3,
        use_mock=True,
        loop_delay_seconds=0.0,
        save_log=True,
        log_dir=str(tmp_path),
    )
    assert rc == 0

    logs = list(tmp_path.glob("text-runner-trial-*.md"))
    assert len(logs) == 1, f"expected 1 transcript log, found {len(logs)}"

    content = logs[0].read_text(encoding="utf-8")
    # ヘッダとキャストが残っている
    assert "Pneuma multi-agent text runner" in content
    assert "なでしこ" in content
    assert "Session End" in content
    assert "Final Snapshot" in content


@pytest.mark.asyncio
async def test_text_runner_no_save_log_writes_nothing(tmp_path) -> None:
    """save_log=False のとき log_dir にファイルが作られない."""
    rc = await run(
        duration_minutes=1.0,
        context="保存しないテスト",
        turn_limit=2,
        use_mock=True,
        loop_delay_seconds=0.0,
        save_log=False,
        log_dir=str(tmp_path),
    )
    assert rc == 0
    assert list(tmp_path.glob("*.md")) == []
