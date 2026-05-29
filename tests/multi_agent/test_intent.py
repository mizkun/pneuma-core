"""Tests for the 思惑エンジン (intent engine) — Issue #20.

AC-Test Binding:
- AC-1 (intent-driven-utterance): 各キャラが状況に応じた短期ゴール（思惑）を持ち、
  発話プロンプトに注入される。
- AC-2 (intent-emergent-not-scripted): 短期ゴールは長期欲求 + 状況から生成され、
  展開は台本化されない（LLM 由来 / 決定論的でない fallback でも可）。
- AC-3 (speech-thought-separation): speech（表）と thought（裏の本音）が分離生成される。
- AC-4: snapshot に各キャラの思惑（surface/hidden）と直近 thought が出る。
- AC-5: FloorController が intent_relevance を加味する。
"""

from __future__ import annotations

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.floor_controller import (
    FloorController,
    Utterance,
)
from pneuma_core.multi_agent.intent import Intent, IntentGenerator
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession


def _make_char(
    char_id: str,
    name: str,
    extraversion: float = 0.5,
    self_enhancement: float = 0.5,
) -> Character:
    return Character(
        id=char_id,
        name=name,
        personality=Personality(
            openness=0.6, conscientiousness=0.5, extraversion=extraversion,
            agreeableness=0.7, neuroticism=0.4,
        ),
        values=Values(
            self_transcendence=0.5, self_enhancement=self_enhancement,
            openness_to_change=0.5, conservation=0.5,
        ),
        profile="テストキャラ",
        speaking_style="普通",
    )


# ─────────────────────────── Intent dataclass ───────────────────────────


def test_intent_holds_surface_hidden_and_intensity() -> None:
    """AC-3: Intent は表のゴール・裏の本音・強さを構造化して持つ."""
    intent = Intent(
        character_id="c1",
        surface_goal="お化け屋敷を盛り上げたい",
        hidden_goal="本当はなでしこに負けたくないだけ",
        intensity=0.8,
    )
    assert intent.surface_goal == "お化け屋敷を盛り上げたい"
    assert intent.hidden_goal == "本当はなでしこに負けたくないだけ"
    assert intent.intensity == 0.8


def test_intent_hidden_goal_is_optional() -> None:
    """AC-3: 裏の本音は無くてもよい（表のゴールだけのケース）."""
    intent = Intent(character_id="c1", surface_goal="楽しく話したい", intensity=0.5)
    assert intent.hidden_goal is None


def test_intent_intensity_out_of_range_raises() -> None:
    """intensity は 0.0〜1.0 の範囲."""
    with pytest.raises(ValueError):
        Intent(character_id="c1", surface_goal="x", intensity=1.5)
    with pytest.raises(ValueError):
        Intent(character_id="c1", surface_goal="x", intensity=-0.1)


# ─────────────────────────── IntentGenerator ───────────────────────────


@pytest.mark.asyncio
async def test_intent_generator_produces_structured_intent() -> None:
    """AC-1 / AC-2: 長期欲求 + 状況から構造化された Intent を生成する（MockLLM）."""
    char = _make_char("chia", "千明", self_enhancement=0.9)
    gen = IntentGenerator(llm=MockLLMAdapter(seed=1))
    intent = await gen.generate(
        character=char,
        shared_context="文化祭の出し物を決める",
    )
    assert isinstance(intent, Intent)
    assert intent.character_id == "chia"
    # surface_goal は必ず非空（思惑が無い = 駆動力が無いので不可）
    assert intent.surface_goal
    # intensity は 0〜1 の範囲に収まる
    assert 0.0 <= intent.intensity <= 1.0


@pytest.mark.asyncio
async def test_intent_generator_uses_tool_use_path_for_structured_adapter() -> None:
    """AC-2: structured_completion 対応 adapter なら Tool Use 経路を通る.

    MockLLMAdapter.supports_structured_completion=True なので、
    report_intent tool 経由で schema を満たす dict が返ることを確認する。
    """
    captured: dict = {}

    class _CapturingMock(MockLLMAdapter):
        async def structured_completion(self, **kwargs):  # type: ignore[override]
            captured.update(kwargs)
            return await super().structured_completion(**kwargs)

    char = _make_char("nade", "なでしこ")
    gen = IntentGenerator(llm=_CapturingMock(seed=2))
    await gen.generate(character=char, shared_context="夏祭りに行く相談")
    assert captured.get("tool_name") == "report_intent"
    schema = captured.get("schema", {})
    props = schema.get("properties", {})
    assert "surface_goal" in props
    assert "hidden_goal" in props
    assert "intensity" in props


@pytest.mark.asyncio
async def test_intent_generator_falls_back_without_crashing() -> None:
    """AC-2: LLM が壊れても degraded fallback の Intent を返す（クラッシュしない）."""

    class _BrokenLLM:
        # structured_completion を持たない → generate fallback 経路
        async def generate(self, request):  # noqa: ANN001
            raise RuntimeError("boom")

    char = _make_char("aoi", "あおい")
    gen = IntentGenerator(llm=_BrokenLLM())
    intent = await gen.generate(character=char, shared_context="部室で雑談")
    assert isinstance(intent, Intent)
    assert intent.character_id == "aoi"
    # fallback でも surface_goal は非空（キャラの長期欲求から導出）
    assert intent.surface_goal


# ─────────────────────── Session 統合（思惑注入） ───────────────────────


@pytest.mark.asyncio
async def test_session_generates_intents_for_all_characters() -> None:
    """AC-1: セッションは各キャラの思惑を生成して保持する."""
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.6)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=1),
        shared_context="文化祭の準備",
    )
    await session.ensure_intents()
    assert set(session.intents.keys()) == {"a", "b"}
    for intent in session.intents.values():
        assert isinstance(intent, Intent)
        assert intent.surface_goal


@pytest.mark.asyncio
async def test_run_turn_injects_intent_into_prompt() -> None:
    """AC-1: 発話生成プロンプトに話者の surface_goal（思惑）が注入される."""
    captured_prompts: list[str] = []

    class _PromptCapturingMock(MockLLMAdapter):
        async def generate(self, request):  # type: ignore[override]
            captured_prompts.append(request.system_prompt)
            return await super().generate(request)

    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.6)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=_PromptCapturingMock(seed=3),
        shared_context="文化祭の準備",
    )
    result = await session.run_turn()
    assert result is not None
    speaker_intent = session.intents[result.speaker.id]
    # 発話生成の system_prompt（最初の generate 呼び出し）に surface_goal が入っている
    speech_prompt = captured_prompts[0]
    assert speaker_intent.surface_goal in speech_prompt


@pytest.mark.asyncio
async def test_run_turn_records_thought_separately_from_speech() -> None:
    """AC-3: 各ターンで speech（表）と thought（裏の本音）が分離記録される."""
    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.6)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=4),
    )
    result = await session.run_turn()
    assert result is not None
    # Utterance は thought フィールドを持つ（speech とは別フィールド）
    assert hasattr(result.utterance, "thought")
    # 直近 thought が話者の runtime state に保持される
    state = session.character_states[result.speaker.id]
    assert hasattr(state, "last_thought")


@pytest.mark.asyncio
async def test_snapshot_includes_intent_and_recent_thought() -> None:
    """AC-4: snapshot に各キャラの思惑（surface/hidden）と直近 thought が含まれる."""
    import json as json_mod

    chars = [_make_char("a", "なでしこ", 0.9), _make_char("b", "千明", 0.6)]
    conv = Conversation(participants=chars)
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=5),
        shared_context="文化祭の準備",
    )
    await session.run_turn()
    snap = session.snapshot()
    # JSON シリアライズ可能
    json_mod.dumps(snap, ensure_ascii=False)
    for ch in snap["characters"]:
        assert "intent" in ch
        assert "surface_goal" in ch["intent"]
        assert "hidden_goal" in ch["intent"]
        assert "intensity" in ch["intent"]
        assert "recent_thought" in ch
    # history_tail の各発話にも thought が入る
    for u in snap["history_tail"]:
        assert "thought" in u


# ─────────────────── FloorController の intent_relevance ───────────────────


def test_floor_controller_adds_intent_relevance() -> None:
    """AC-5: 直前発話が自分の思惑に関わるキャラのスコアが上がる.

    同条件の 2 キャラで、直前発話に片方の思惑キーワードが含まれるとき、
    そのキャラのスコアの方が高くなる。
    """
    cat = _make_char("cat", "千明", extraversion=0.6)
    dog = _make_char("dog", "あおい", extraversion=0.6)
    intents = {
        "cat": Intent(character_id="cat", surface_goal="お化け屋敷をやりたい", intensity=0.9),
        "dog": Intent(character_id="dog", surface_goal="カフェをやりたい", intensity=0.9),
    }
    fc = FloorController()
    # 「お化け屋敷」に言及する直前発話 → cat の思惑に関連
    history = [
        Utterance(
            speaker_id="other",
            speaker_name="なでしこ",
            speech="お化け屋敷ってちょっと怖そう",
        )
    ]
    score_cat = fc.score(cat, history, intents=intents)
    score_dog = fc.score(dog, history, intents=intents)
    assert score_cat > score_dog


def test_floor_controller_intent_is_optional_backward_compatible() -> None:
    """AC-5: intents を渡さなくても従来通り動作する（後方互換）."""
    ch = _make_char("c", "千明", extraversion=0.7)
    history = [Utterance(speaker_id="other", speaker_name="X", speech="やあ")]
    fc = FloorController()
    # intents 無しでもエラーにならず数値を返す
    s = fc.score(ch, history)
    assert isinstance(s, float)


def test_floor_controller_next_speaker_accepts_intents() -> None:
    """AC-5: next_speaker は intents を加味して発話者を選べる."""
    chars = [
        _make_char("a", "なでしこ", extraversion=0.6),
        _make_char("b", "千明", extraversion=0.6),
    ]
    intents = {
        "a": Intent(character_id="a", surface_goal="花火の話をしたい", intensity=0.9),
        "b": Intent(character_id="b", surface_goal="ご飯の話をしたい", intensity=0.9),
    }
    history = [
        Utterance(speaker_id="x", speaker_name="X", speech="今度の花火大会どうする？")
    ]
    fc = FloorController()
    speaker = fc.next_speaker(history, chars, intents=intents)
    # 花火に関連する思惑を持つ a が選ばれる
    assert speaker.id == "a"
