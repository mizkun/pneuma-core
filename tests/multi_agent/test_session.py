"""Tests for MultiAgentSession (Issue #6 / Phase 0 C).

invariants 統合確認:
- thinking-on-every-turn: 各ターンで全キャラの emotion が更新される
- observation-turn-state-update: 他キャラの発話が自分の history に積まれる
- utterance-action-pair: speech / action のペアが出力される
- multi-agent-session-end: N 体ぶんの分析が走る
"""

from __future__ import annotations

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession
from pneuma_core.multi_agent.session_end import (
    MultiAgentSessionEndPipeline,
)


def _make_char(
    char_id: str, name: str, extraversion: float = 0.5
) -> Character:
    return Character(
        id=char_id,
        name=name,
        personality=Personality(
            openness=0.6, conscientiousness=0.5, extraversion=extraversion,
            agreeableness=0.7, neuroticism=0.4,
        ),
        values=Values(
            self_transcendence=0.5, self_enhancement=0.5,
            openness_to_change=0.5, conservation=0.5,
        ),
        profile="テストキャラ",
        speaking_style="普通",
    )


@pytest.mark.asyncio
async def test_run_single_turn_produces_utterance() -> None:
    """AC: 1 ターン回すと 1 つの Utterance が記録される."""
    chars = [
        _make_char("nade", "なでしこ", 0.95),
        _make_char("chia", "千明", 0.85),
        _make_char("aoi", "あおい", 0.5),
    ]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=42),
        shared_context="部室で雑談",
    )
    result = await session.run_turn()
    assert result is not None
    assert result.turn_index == 1
    # 最初の発話者は外向性最大 → なでしこ
    assert result.speaker.id == "nade"
    assert len(conv.history) == 1
    assert conv.history[0].speech  # 非空


@pytest.mark.asyncio
async def test_thinking_on_every_turn_updates_all_emotions() -> None:
    """AC (thinking-on-every-turn): 1 ターンで全キャラの emotion が更新される."""
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
    )
    result = await session.run_turn()
    assert result is not None
    # 全キャラの emotion が emotion_changes に含まれる
    assert set(result.emotion_changes.keys()) == {"a", "b"}


@pytest.mark.asyncio
async def test_observation_turn_state_update() -> None:
    """AC (observation-turn-state-update): 自分のターンでなくても発話が積まれる."""
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.4)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
    )
    await session.run_turn()
    # b 視点では a の発話が user として見える
    view_b = conv.history_as_messages(viewer_id="b")
    assert len(view_b) == 1
    assert view_b[0]["role"] == "user"


@pytest.mark.asyncio
async def test_run_multiple_turns_rotates_speakers() -> None:
    """AC: 複数ターンで複数のキャラが発話する（FloorController 連携）.

    extraversion を近い値にして、recent_penalty で順番が回ることを確認。
    """
    chars = [
        _make_char("a", "なでしこ", 0.7),
        _make_char("b", "千明", 0.65),
        _make_char("c", "あおい", 0.6),
    ]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
    )
    for _ in range(9):
        result = await session.run_turn()
        assert result is not None
    speakers_seen = {u.speaker_id for u in conv.history}
    assert speakers_seen == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_circuit_breaker_stops_session() -> None:
    """AC (turn-rate-limit): circuit breaker が tripped したら None を返す."""
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    cb = CircuitBreaker(CircuitBreakerConfig(max_turns=2))
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
        circuit_breaker=cb,
    )
    r1 = await session.run_turn()
    r2 = await session.run_turn()
    r3 = await session.run_turn()  # tripped
    assert r1 is not None
    assert r2 is not None
    assert r3 is None
    assert cb.state.value == "tripped"


@pytest.mark.asyncio
async def test_snapshot_serializable() -> None:
    """AC: snapshot() は Web UI に流せる dict を返す."""
    import json as json_mod
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
    )
    await session.run_turn()
    snap = session.snapshot()
    # JSON シリアライズできる
    js = json_mod.dumps(snap, ensure_ascii=False)
    assert "characters" in snap
    assert len(snap["characters"]) == 2
    assert "history_tail" in snap
    assert isinstance(js, str)


@pytest.mark.asyncio
async def test_session_end_pipeline_runs_for_all_characters() -> None:
    """AC (multi-agent-session-end): N 体ぶんの分析結果が返る."""
    chars = [
        _make_char("a", "なでしこ", 0.9),
        _make_char("b", "千明", 0.7),
        _make_char("c", "あおい", 0.5),
    ]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
    )
    for _ in range(6):
        await session.run_turn()

    pipeline = MultiAgentSessionEndPipeline(llm=MockLLMAdapter(seed=2))
    result = await pipeline.run(session)
    assert len(result.per_character) == 3
    # mock LLM が空でない episodic を返すので、合計 > 0
    assert result.total_episodic >= 3
