"""Regression tests for instrumentation reliability — Issue #23 バグ1.

PAD/emotion とテキスト(thought)の整合 (invariant: pad-text-consistency).

AC-1: 生成順序が固定され、emotion_label と thought の感情方向が正反対に
      ならない（再生成 or 補正）。MockLLM でテスト可能な形で検証する。
"""

from __future__ import annotations

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.emotion_text_consistency import (
    label_emotion_direction,
    resolve_emotion_text_consistency,
    thought_emotion_direction,
)
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession


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


# ─────────────── direction helpers (unit) ───────────────


def test_label_direction_positive() -> None:
    assert label_emotion_direction("happy") > 0
    assert label_emotion_direction("teasing") > 0


def test_label_direction_negative() -> None:
    assert label_emotion_direction("sad_lite") < 0
    assert label_emotion_direction("embarrassed") < 0


def test_label_direction_neutral() -> None:
    assert label_emotion_direction("neutral") == 0
    assert label_emotion_direction("surprised") == 0


def test_thought_direction_positive() -> None:
    assert thought_emotion_direction("やったー、嬉しい！楽しいな") > 0


def test_thought_direction_negative() -> None:
    assert thought_emotion_direction("最悪だ……悲しいし悔しい") < 0


def test_thought_direction_neutral_or_unknown() -> None:
    # 感情語が無い内心は方向不明（0）扱い
    assert thought_emotion_direction("次の話題どうしよう") == 0
    assert thought_emotion_direction("") == 0


# ─────────────── resolve_emotion_text_consistency (unit) ───────────────


def test_resolve_keeps_label_when_consistent() -> None:
    """整合しているときは label を変えない（happy thought + happy label）."""
    label, changed = resolve_emotion_text_consistency(
        emotion_label="happy", thought="嬉しいな、楽しい！"
    )
    assert label == "happy"
    assert changed is False


def test_resolve_corrects_label_when_contradictory() -> None:
    """正反対のときは label を thought 側（負方向）へ補正する.

    AC-1: emotion=happy(正) なのに thought が「悲しい・最悪」(負) のときは
    矛盾なので label を負方向のラベルに合わせる。
    """
    label, changed = resolve_emotion_text_consistency(
        emotion_label="happy", thought="最悪だ……すごく悲しい"
    )
    assert changed is True
    assert label_emotion_direction(label) < 0


def test_resolve_corrects_negative_label_with_positive_thought() -> None:
    """逆方向: emotion=sad_lite(負) なのに thought が自信満々(正)."""
    label, changed = resolve_emotion_text_consistency(
        emotion_label="sad_lite", thought="やったー！絶対いける、楽しい"
    )
    assert changed is True
    assert label_emotion_direction(label) > 0


def test_resolve_noop_when_thought_neutral() -> None:
    """thought が感情中立なら label をいじらない（過補正しない）."""
    label, changed = resolve_emotion_text_consistency(
        emotion_label="happy", thought="次の話題どうしよう"
    )
    assert label == "happy"
    assert changed is False


# ─────────────── Session 統合 ───────────────


@pytest.mark.asyncio
async def test_session_utterance_label_consistent_with_thought() -> None:
    """AC-1 (pad-text-consistency): セッションが出す Utterance の emotion_label と
    thought の感情方向が正反対にならない.

    MockLLM が「label=happy だが thought が強い負」という矛盾ケースを必ず作る
    ように細工し、Session の整合ロジックが補正することを検証する。
    """

    class _ContradictoryMock(MockLLMAdapter):
        """発話の thought を必ず強い負方向にして矛盾を仕込む."""

        def _gen_chat(self, request):  # type: ignore[override]
            import json
            return json.dumps(
                {
                    "speech": "そうだね",
                    "thought": "最悪だ……すごく悲しいし悔しい",
                    "action": "うつむく",
                },
                ensure_ascii=False,
            )

        def _gen_emotion(self, request):  # type: ignore[override]
            # PAD/label は正方向（happy）を返して矛盾を作る
            import json
            return json.dumps(
                {
                    "pleasure": 0.6,
                    "arousal": 0.6,
                    "dominance": 0.2,
                    "emotion_label": "happy",
                    "situation": "雑談中",
                },
                ensure_ascii=False,
            )

        async def structured_completion(self, **kwargs):  # type: ignore[override]
            if kwargs.get("tool_name") == "report_emotion":
                import json
                from pneuma_core.llm.adapter import LLMRequest
                return json.loads(self._gen_emotion(
                    LLMRequest(system_prompt="", messages=[])
                ))
            return await super().structured_completion(**kwargs)

    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=_ContradictoryMock(seed=1),
    )
    result = await session.run_turn()
    assert result is not None
    u = result.utterance
    # thought は負方向。emotion_label が正(happy/teasing)のままなら矛盾 → NG。
    t_dir = thought_emotion_direction(u.thought)
    l_dir = label_emotion_direction(u.emotion_label)
    assert t_dir < 0  # 前提: thought は負
    # 整合: 正反対(積が負)になっていないこと
    assert not (t_dir * l_dir < 0), (
        f"emotion_label={u.emotion_label!r} が thought={u.thought!r} と正反対"
    )


@pytest.mark.asyncio
async def test_structured_ending_does_not_force_overwrite_emotion() -> None:
    """AC-1: 構造的オチ(終盤収束)が感情を強制上書きして矛盾を生まない.

    structured_ending ON + 終盤でも、emotion_label と thought の方向が
    正反対にならないこと（締めだから happy 固定、のような上書きをしない）。
    """
    from pneuma_core.multi_agent.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
    )
    from pneuma_core.multi_agent.fun_config import FunEngineConfig

    class _NegThoughtMock(MockLLMAdapter):
        def _gen_chat(self, request):  # type: ignore[override]
            import json
            return json.dumps(
                {
                    "speech": "もう終わりか",
                    "thought": "つまらない、最悪な締めだ……悲しい",
                    "action": "ため息",
                },
                ensure_ascii=False,
            )

    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    cb = CircuitBreaker(CircuitBreakerConfig(max_turns=4))
    session = MultiAgentSession(
        conversation=conv,
        llm=_NegThoughtMock(seed=1),
        circuit_breaker=cb,
        fun_config=FunEngineConfig(
            enable_intent=True, enable_structured_ending=True
        ),
    )
    last = None
    for _ in range(3):
        last = await session.run_turn()
    assert last is not None
    u = last.utterance
    t_dir = thought_emotion_direction(u.thought)
    l_dir = label_emotion_direction(u.emotion_label)
    assert not (t_dir * l_dir < 0)


class _ContradictoryStateMock(MockLLMAdapter):
    """発話の thought は強い負、推定 emotion は正(happy) という矛盾を仕込む.

    観測される計器（state.emotion / snapshot）まで整合補正が届くかを検証する
    ための共通 mock。
    """

    def _gen_chat(self, request):  # type: ignore[override]
        import json
        return json.dumps(
            {
                "speech": "そうだね",
                "thought": "最悪だ……すごく悲しいし悔しい、怖い",
                "action": "うつむく",
            },
            ensure_ascii=False,
        )

    def _gen_emotion(self, request):  # type: ignore[override]
        import json
        return json.dumps(
            {
                "pleasure": 0.6,
                "arousal": 0.6,
                "dominance": 0.2,
                "emotion_label": "happy",
                "situation": "雑談中",
            },
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_run_turn_corrects_speaker_state_emotion() -> None:
    """🔴 致命: 整合補正が観測される計器 state.emotion に届く.

    AC-1 (pad-text-consistency): 再推定後に確定する speaker の state.emotion
    .emotion_label が、その thought（負方向）と正反対(happy)のままにならない。
    CLI 主表示 / Web バッジは state.emotion を読むため、ここが補正されないと
    視聴者に矛盾が見える。
    """
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=_ContradictoryStateMock(seed=1),
    )
    result = await session.run_turn()
    assert result is not None

    speaker_state = session.character_states[result.speaker.id]
    t_dir = thought_emotion_direction(speaker_state.last_thought)
    l_dir = label_emotion_direction(speaker_state.emotion.emotion_label)
    assert t_dir < 0  # 前提: thought は負方向
    # 観測される計器(state.emotion)が thought と正反対になっていないこと
    assert not (t_dir * l_dir < 0), (
        f"state.emotion.emotion_label={speaker_state.emotion.emotion_label!r} が "
        f"thought={speaker_state.last_thought!r} と正反対"
    )


@pytest.mark.asyncio
async def test_snapshot_emotion_matches_history_tail() -> None:
    """🟠 重要: snapshot の characters[].emotion と history_tail[] が一致.

    AC-1 (pad-text-consistency): 最新発話者の snapshot characters[].emotion
    .emotion_label（state.emotion 由来）と、対応する history_tail[].emotion_label
    （Utterance 由来）が食い違わない（同じ補正後ラベル）。
    """
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=_ContradictoryStateMock(seed=1),
    )
    result = await session.run_turn()
    assert result is not None

    snap = session.snapshot()
    # 最新発話者を history_tail の末尾から取る
    last_hist = snap["history_tail"][-1]
    speaker_id = last_hist["speaker_id"]
    char_emo = next(
        c["emotion"]["emotion_label"]
        for c in snap["characters"]
        if c["id"] == speaker_id
    )
    assert char_emo == last_hist["emotion_label"], (
        f"snapshot characters[].emotion={char_emo!r} != "
        f"history_tail[].emotion_label={last_hist['emotion_label']!r}"
    )


@pytest.mark.asyncio
async def test_run_turn_utterance_matches_state_emotion() -> None:
    """🔴 致命: Utterance.emotion_label と state.emotion.emotion_label が一致.

    AC-1 (pad-text-consistency): 補正後の最終ラベルが Utterance と state の
    両方で同一であること（CLI/Web/snapshot/Utterance がすべて同じ値）。
    """
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.5)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=_ContradictoryStateMock(seed=1),
    )
    result = await session.run_turn()
    assert result is not None

    speaker_state = session.character_states[result.speaker.id]
    assert (
        result.utterance.emotion_label == speaker_state.emotion.emotion_label
    ), (
        f"Utterance.emotion_label={result.utterance.emotion_label!r} != "
        f"state.emotion.emotion_label={speaker_state.emotion.emotion_label!r}"
    )
