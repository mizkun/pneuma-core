"""Tests for SessionDriver (web_server backend, Issue #7 / Phase 0 C2)."""

from __future__ import annotations

import json

import pytest

from pneuma_core.cli.web_server import SessionDriver
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter


def test_bootstrap_snapshot_has_three_characters() -> None:
    """AC: 起動直後（session 開始前）に snapshot が 3 キャラ分返る."""
    driver = SessionDriver(
        llm=MockLLMAdapter(seed=1),
        llm_label="mock-llm",
        context="test",
        use_mock=True,
    )
    snap = driver.latest_snapshot()
    assert len(snap["characters"]) == 3
    names = {c["name"] for c in snap["characters"]}
    assert names == {"なでしこ", "千明", "あおい"}
    assert snap["llm_label"] == "mock-llm"


def test_snapshot_json_serializable() -> None:
    """AC: snapshot は JSON シリアライズ可能（SSE で流せる）."""
    driver = SessionDriver(
        llm=MockLLMAdapter(seed=1),
        llm_label="mock-llm",
        context="test",
        use_mock=True,
    )
    snap = driver.latest_snapshot()
    payload = json.dumps(snap, ensure_ascii=False)
    assert "なでしこ" in payload


def test_subscribe_receives_initial_snapshot() -> None:
    """AC: subscribe() すると最初に snapshot イベントが配信される."""
    driver = SessionDriver(
        llm=MockLLMAdapter(seed=1),
        llm_label="mock-llm",
        context="test",
        use_mock=True,
    )
    q = driver.subscribe()
    msg = q.get_nowait()
    assert msg["type"] == "snapshot"
    assert "characters" in msg["data"]
    driver.unsubscribe(q)


@pytest.mark.asyncio
async def test_run_one_session_broadcasts_events() -> None:
    """AC: _run_one_session() がイベントを subscriber に届ける."""
    driver = SessionDriver(
        llm=MockLLMAdapter(seed=1),
        llm_label="mock-llm",
        context="test",
        use_mock=True,
        turn_limit_per_session=3,
        inter_turn_seconds=0.0,
        inter_session_seconds=0.1,
    )
    q = driver.subscribe()
    # consume initial snapshot
    q.get_nowait()
    await driver._run_one_session(1)
    # session_start + 3 turns + 3 snapshots + session_end + 1 snapshot
    types = []
    while not q.empty():
        types.append(q.get_nowait()["type"])
    assert "session_start" in types
    assert types.count("turn") == 3
    assert "session_end" in types
