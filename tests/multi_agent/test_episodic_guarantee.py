"""Regression tests for instrumentation reliability — Issue #23 バグ2.

SessionEnd で発話したキャラの episodic が空にならない保証
(invariant: speaker-episodic-guaranteed).

AC-2: 発話数>0 のキャラは SessionEnd で episodic >= 1 件。
"""

from __future__ import annotations

import json

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession
from pneuma_core.multi_agent.session_end import (
    MultiAgentSessionEndPipeline,
)


def _make_char(char_id: str, name: str, extraversion: float = 0.5) -> Character:
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


class _EmptyEpisodicMock(MockLLMAdapter):
    """session_end 分析で episodic を必ず空で返す（取りこぼし再現）."""

    def _gen_session_end(self) -> str:  # type: ignore[override]
        return json.dumps(
            {
                "episodic_memories": [],
                "semantic_updates": [],
                "relationship_changes": [],
            },
            ensure_ascii=False,
        )

    async def structured_completion(self, **kwargs):  # type: ignore[override]
        if kwargs.get("tool_name") == "record_session_memories":
            return json.loads(self._gen_session_end())
        return await super().structured_completion(**kwargs)


@pytest.mark.asyncio
async def test_speaker_gets_episodic_even_when_llm_returns_empty() -> None:
    """AC-2: LLM 分析が空 episodic を返しても、発話したキャラは episodic >= 1."""
    chars = [
        _make_char("a", "なでしこ", 0.95),
        _make_char("b", "千明", 0.7),
        _make_char("c", "あおい", 0.5),
    ]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
    )
    for _ in range(8):
        await session.run_turn()

    # 何人かは必ず喋っている
    speakers = {u.speaker_id for u in conv.history}
    assert speakers

    pipeline = MultiAgentSessionEndPipeline(llm=_EmptyEpisodicMock(seed=2))
    result = await pipeline.run(session)

    # 発話数>0 の各キャラは episodic >= 1（フォールバック保証）
    by_id = {r.character_id: r for r in result.per_character}
    for ch in chars:
        st = session.character_states[ch.id]
        if st.speak_count > 0:
            assert len(st.episodic) >= 1, (
                f"{ch.name}: 発話 {st.speak_count} 回なのに episodic 空"
            )
            assert len(by_id[ch.id].episodic) >= 1


@pytest.mark.asyncio
async def test_non_speaker_episodic_not_forced() -> None:
    """発話していないキャラには episodic 保証を強制しない（過剰生成しない）.

    1 ターンだけ回して、喋っていないキャラの episodic は空のままでよい。
    """
    chars = [
        _make_char("a", "なでしこ", 0.95),
        _make_char("b", "千明", 0.3),
    ]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
    )
    await session.run_turn()  # 1 ターン → なでしこのみ発話

    non_speakers = [
        ch for ch in chars if session.character_states[ch.id].speak_count == 0
    ]
    assert non_speakers  # 少なくとも 1 人は未発話

    pipeline = MultiAgentSessionEndPipeline(llm=_EmptyEpisodicMock(seed=2))
    await pipeline.run(session)

    for ch in non_speakers:
        st = session.character_states[ch.id]
        # 未発話キャラには空 episodic を許容（保証は発話キャラのみ）
        assert len(st.episodic) == 0
