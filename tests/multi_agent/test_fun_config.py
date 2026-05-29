"""Tests for the 面白さエンジン (fun engine) toggles — Issue #22.

設計の土台: .vibe/references/fun-engine-design.md
核心思想: 台本化しない。各トグルは「傾向・性質をプロンプトに与える」だけで、
具体的な出力（セリフ/展開/結末）は LLM の自律に委ねる。

AC-Test Binding:
- AC-1 (fun-engine-toggleable): FunEngineConfig で 6 要素を独立に ON/OFF できる。
- AC-2: 全 OFF（enable_intent のみ true）で既存挙動が不変。
- AC-3 (quirk): enable_quirk ON でキャラのクセが発話プロンプトに注入される。
- AC-4 (terse): enable_terse ON で発話 max_tokens が下がる + 短く指示が入る。
- AC-5 (lateral): enable_lateral_thinking ON で連想・アナロジーが促される。
- AC-6 (emotion): enable_emotion_dynamics ON で感情が baseline に引き戻され温度差が出る。
- AC-7 (ending): enable_structured_ending ON で終盤に収束（決着）指示が入る。
"""

from __future__ import annotations

import dataclasses

import pytest

from pneuma_core.llm.adapter import LLMRequest, LLMResponse
from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.fun_config import FunEngineConfig
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession


def _make_char(
    char_id: str,
    name: str,
    *,
    extraversion: float = 0.5,
    openness: float = 0.6,
    neuroticism: float = 0.4,
    agreeableness: float = 0.7,
    quirk: str = "",
) -> Character:
    return Character(
        id=char_id,
        name=name,
        personality=Personality(
            openness=openness,
            conscientiousness=0.5,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
        ),
        values=Values(
            self_transcendence=0.5, self_enhancement=0.5,
            openness_to_change=0.5, conservation=0.5,
        ),
        profile="テストキャラ",
        speaking_style="普通",
        quirk=quirk,
    )


class RecordingMockLLM(MockLLMAdapter):
    """発話 (chat) リクエストの LLMRequest を記録する MockLLMAdapter.

    トグルが発話プロンプト / max_tokens に効いているかをテストで観測するため、
    最後に generate() に渡された chat 用 LLMRequest を保持する。
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.last_chat_request: LLMRequest | None = None
        self.chat_requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # chat（発話生成）プロンプトだけ記録する。感情分析等は無視。
        if "というキャラクターです" in request.system_prompt:
            self.last_chat_request = request
            self.chat_requests.append(request)
        return await super().generate(request)


def _session(
    chars: list[Character],
    fun_config: FunEngineConfig | None = None,
    *,
    llm: MockLLMAdapter | None = None,
    shared_context: str = "部室で雑談",
    circuit_breaker: CircuitBreaker | None = None,
) -> MultiAgentSession:
    conv = Conversation(participants=chars)
    kwargs: dict = {
        "conversation": conv,
        "llm": llm or MockLLMAdapter(seed=42),
        "shared_context": shared_context,
    }
    if fun_config is not None:
        kwargs["fun_config"] = fun_config
    if circuit_breaker is not None:
        kwargs["circuit_breaker"] = circuit_breaker
    return MultiAgentSession(**kwargs)


# ───────────────────────── AC-1: 6 flags toggleable ─────────────────────────


def test_fun_config_has_six_independent_flags() -> None:
    """AC-1: FunEngineConfig は 6 つの独立フラグを持つ."""
    fields = {f.name for f in dataclasses.fields(FunEngineConfig)}
    assert fields == {
        "enable_intent",
        "enable_quirk",
        "enable_terse",
        "enable_lateral_thinking",
        "enable_emotion_dynamics",
        "enable_structured_ending",
    }


def test_fun_config_default_is_intent_only() -> None:
    """AC-1/AC-2: デフォルトは enable_intent のみ True、他は False（挙動不変）."""
    cfg = FunEngineConfig()
    assert cfg.enable_intent is True
    assert cfg.enable_quirk is False
    assert cfg.enable_terse is False
    assert cfg.enable_lateral_thinking is False
    assert cfg.enable_emotion_dynamics is False
    assert cfg.enable_structured_ending is False


def test_fun_config_flags_set_independently() -> None:
    """AC-1: 各フラグは独立に設定できる."""
    cfg = FunEngineConfig(
        enable_intent=False,
        enable_quirk=True,
        enable_terse=False,
        enable_lateral_thinking=True,
        enable_emotion_dynamics=False,
        enable_structured_ending=True,
    )
    assert cfg.enable_intent is False
    assert cfg.enable_quirk is True
    assert cfg.enable_lateral_thinking is True
    assert cfg.enable_structured_ending is True
    assert cfg.enable_terse is False
    assert cfg.enable_emotion_dynamics is False


# ───────────────────── AC-2: default leaves behavior unchanged ─────────────────


@pytest.mark.asyncio
async def test_default_config_runs_turn_like_before() -> None:
    """AC-2: デフォルト config でも従来どおり 1 ターンで Utterance が出る."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9),
             _make_char("b", "千明", extraversion=0.5)]
    session = _session(chars, FunEngineConfig())
    result = await session.run_turn()
    assert result is not None
    assert result.utterance.speech


@pytest.mark.asyncio
async def test_default_intent_enabled_injects_intent() -> None:
    """AC-2: デフォルト（enable_intent True）では思惑が生成される（#20 挙動不変）."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9)]
    session = _session(chars, FunEngineConfig())
    await session.ensure_intents()
    assert "a" in session.intents
    assert session.intents["a"].surface_goal


@pytest.mark.asyncio
async def test_intent_disabled_skips_intent_generation() -> None:
    """AC-1: enable_intent False で思惑生成がスキップされる（思惑非注入）."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9)]
    session = _session(chars, FunEngineConfig(enable_intent=False))
    await session.ensure_intents()
    assert session.intents == {}
    # 発話プロンプトにも思惑セクションが入らない
    llm = RecordingMockLLM(seed=1)
    session2 = _session(chars, FunEngineConfig(enable_intent=False), llm=llm)
    await session2.run_turn()
    assert llm.last_chat_request is not None
    assert "表のゴール" not in llm.last_chat_request.system_prompt


# ───────────────────────────── AC-3: quirk ─────────────────────────────


@pytest.mark.asyncio
async def test_quirk_off_does_not_inject_quirk() -> None:
    """AC-3: enable_quirk False のとき quirk は発話プロンプトに入らない."""
    chars = [_make_char("a", "千明", extraversion=0.9,
                        quirk="何でも勝負・競争に変換して考える")]
    llm = RecordingMockLLM(seed=1)
    session = _session(chars, FunEngineConfig(), llm=llm)
    await session.run_turn()
    assert llm.last_chat_request is not None
    assert "勝負・競争に変換" not in llm.last_chat_request.system_prompt


@pytest.mark.asyncio
async def test_quirk_on_injects_quirk_into_prompt() -> None:
    """AC-3: enable_quirk True で quirk が発話プロンプトに注入され、
    『クセに引き寄せる』『優等生回答を避ける』が促される（出力は委ねる）."""
    chars = [_make_char("a", "千明", extraversion=0.9,
                        quirk="何でも勝負・競争に変換して考える")]
    llm = RecordingMockLLM(seed=1)
    session = _session(chars, FunEngineConfig(enable_quirk=True), llm=llm)
    await session.run_turn()
    assert llm.last_chat_request is not None
    sp = llm.last_chat_request.system_prompt
    assert "何でも勝負・競争に変換して考える" in sp
    # 優等生的な深掘りを避ける促し（傾向の付与）
    assert "クセ" in sp


@pytest.mark.asyncio
async def test_quirk_on_without_quirk_text_is_noop() -> None:
    """AC-3: quirk 未設定キャラは enable_quirk ON でも壊れない."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9, quirk="")]
    session = _session(chars, FunEngineConfig(enable_quirk=True))
    result = await session.run_turn()
    assert result is not None


# ───────────────────────────── AC-4: terse ─────────────────────────────


@pytest.mark.asyncio
async def test_terse_off_keeps_default_max_tokens() -> None:
    """AC-4: enable_terse False のとき従来の max_tokens が使われる."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9)]
    llm = RecordingMockLLM(seed=1)
    session = _session(chars, FunEngineConfig(), llm=llm)
    await session.run_turn()
    assert llm.last_chat_request is not None
    default_tokens = llm.last_chat_request.max_tokens

    llm2 = RecordingMockLLM(seed=1)
    session2 = _session(chars, FunEngineConfig(enable_terse=True), llm=llm2)
    await session2.run_turn()
    assert llm2.last_chat_request is not None
    assert llm2.last_chat_request.max_tokens < default_tokens


@pytest.mark.asyncio
async def test_terse_on_injects_short_instruction() -> None:
    """AC-4: enable_terse True で『短く・大喜利的に』がプロンプトに入る."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9)]
    llm = RecordingMockLLM(seed=1)
    session = _session(chars, FunEngineConfig(enable_terse=True), llm=llm)
    await session.run_turn()
    assert llm.last_chat_request is not None
    assert "短く" in llm.last_chat_request.system_prompt
    assert "大喜利" in llm.last_chat_request.system_prompt


# ──────────────────────── AC-5: lateral thinking ────────────────────────


@pytest.mark.asyncio
async def test_lateral_off_does_not_promote_analogy() -> None:
    """AC-5: enable_lateral_thinking False のとき連想プロンプトは入らない."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9, openness=0.9)]
    llm = RecordingMockLLM(seed=1)
    session = _session(chars, FunEngineConfig(), llm=llm)
    await session.run_turn()
    assert llm.last_chat_request is not None
    assert "アナロジー" not in llm.last_chat_request.system_prompt


@pytest.mark.asyncio
async def test_lateral_on_promotes_analogy_for_high_openness() -> None:
    """AC-5: openness が高いキャラに連想・アナロジーが促される（中身は自律）."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9, openness=0.9)]
    llm = RecordingMockLLM(seed=1)
    session = _session(
        chars, FunEngineConfig(enable_lateral_thinking=True), llm=llm
    )
    await session.run_turn()
    assert llm.last_chat_request is not None
    sp = llm.last_chat_request.system_prompt
    assert "アナロジー" in sp or "連想" in sp


# ──────────────────────── AC-6: emotion dynamics ────────────────────────


@pytest.mark.asyncio
async def test_emotion_dynamics_pulls_toward_personality_baseline() -> None:
    """AC-6: enable_emotion_dynamics ON で、推定 PAD が各キャラの性格 baseline
    側に引き戻される（全員同方向への高止まりを解消）.

    同じ mock seed で OFF/ON を比較し、ON のほうが baseline に近いことを確認。
    性格が極端に異なる 2 体で、ON のときに PAD の差（温度差）が縮まないこと
    （= baseline 由来の個体差が残る）を確認する。
    """
    from pneuma_core.emotion.baseline import personality_to_pad_baseline

    # neuroticism 高 / 低の 2 体（baseline が大きく異なる）
    hot = _make_char("hot", "千明", extraversion=0.95, neuroticism=0.8,
                     agreeableness=0.3)
    cool = _make_char("cool", "あおい", extraversion=0.3, neuroticism=0.1,
                      agreeableness=0.9)

    # OFF
    session_off = _session(
        [hot, cool], FunEngineConfig(), llm=MockLLMAdapter(seed=7)
    )
    await session_off.run_turn()
    off_hot = session_off.character_states["hot"].emotion
    off_cool = session_off.character_states["cool"].emotion

    # ON
    session_on = _session(
        [hot, cool], FunEngineConfig(enable_emotion_dynamics=True),
        llm=MockLLMAdapter(seed=7),
    )
    await session_on.run_turn()
    on_hot = session_on.character_states["hot"].emotion
    on_cool = session_on.character_states["cool"].emotion

    base_hot = personality_to_pad_baseline(hot.personality)
    base_cool = personality_to_pad_baseline(cool.personality)

    # ON は各キャラの baseline pleasure に OFF より近い（引き戻し）
    assert abs(on_hot.pleasure - base_hot[0]) <= abs(off_hot.pleasure - base_hot[0])
    assert abs(on_cool.pleasure - base_cool[0]) <= abs(off_cool.pleasure - base_cool[0])

    # ON では 2 体の pleasure 差が baseline の差に近づく（温度差が生まれる）
    on_spread = abs(on_hot.pleasure - on_cool.pleasure)
    base_spread = abs(base_hot[0] - base_cool[0])
    # baseline 差が出ている前提（性格を極端にしている）
    assert base_spread > 0.05
    # ON の温度差は baseline 差の一定割合以上を保持している
    assert on_spread >= base_spread * 0.3


# ─────────────────────── AC-7: structured ending ───────────────────────


@pytest.mark.asyncio
async def test_structured_ending_off_no_closure_prompt() -> None:
    """AC-7: enable_structured_ending False では終盤でも収束指示が入らない."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9)]
    llm = RecordingMockLLM(seed=1)
    cb = CircuitBreaker(CircuitBreakerConfig(max_turns=2))
    session = _session(chars, FunEngineConfig(), llm=llm, circuit_breaker=cb)
    await session.run_turn()
    assert llm.last_chat_request is not None
    assert "決着" not in llm.last_chat_request.system_prompt
    assert "結論" not in llm.last_chat_request.system_prompt


@pytest.mark.asyncio
async def test_structured_ending_injects_closure_only_in_endgame() -> None:
    """AC-7: enable_structured_ending ON で、残りターンが少ない『終盤』にのみ
    『そろそろ結論・決着に』という外圧が注入される（結末そのものは台本化しない）.

    序盤（残りターン多い）では収束指示は入らない。
    """
    chars = [_make_char("a", "なでしこ", extraversion=0.9)]
    llm = RecordingMockLLM(seed=1)
    # max_turns=4: 序盤(turn1,2)は通常、終盤(turn3,4)で収束指示
    cb = CircuitBreaker(CircuitBreakerConfig(max_turns=4))
    session = _session(
        chars, FunEngineConfig(enable_structured_ending=True),
        llm=llm, circuit_breaker=cb,
    )
    # turn 1（序盤）: 収束指示なし
    await session.run_turn()
    early_sp = llm.last_chat_request.system_prompt
    assert "決着" not in early_sp and "結論" not in early_sp

    # 終盤まで進める
    await session.run_turn()  # turn 2
    await session.run_turn()  # turn 3（終盤）
    late_sp = llm.last_chat_request.system_prompt
    assert "決着" in late_sp or "結論" in late_sp


@pytest.mark.asyncio
async def test_structured_ending_requires_circuit_breaker_turn_budget() -> None:
    """AC-7: 終盤判定は CircuitBreaker の残りターン数で行う（メカニカル）."""
    chars = [_make_char("a", "なでしこ", extraversion=0.9)]
    llm = RecordingMockLLM(seed=1)
    cb = CircuitBreaker(CircuitBreakerConfig(max_turns=2))
    session = _session(
        chars, FunEngineConfig(enable_structured_ending=True),
        llm=llm, circuit_breaker=cb,
    )
    # max_turns=2: turn1 ですでに終盤（残り 1）→ 収束指示が入る
    await session.run_turn()
    assert "決着" in llm.last_chat_request.system_prompt or \
        "結論" in llm.last_chat_request.system_prompt


# ─────────────────────── all-on smoke (台本化しない確認) ───────────────────────


@pytest.mark.asyncio
async def test_all_flags_on_runs_without_error() -> None:
    """AC-1: 全フラグ ON でもセッションが回る（各トグルが共存できる）."""
    chars = [
        _make_char("a", "なでしこ", extraversion=0.9, openness=0.9,
                   quirk="何でも食べ物に結びつける"),
        _make_char("b", "千明", extraversion=0.85, quirk="何でも勝負に変換"),
        _make_char("c", "あおい", extraversion=0.5, quirk="達観して人生訓で返す"),
    ]
    cfg = FunEngineConfig(
        enable_intent=True,
        enable_quirk=True,
        enable_terse=True,
        enable_lateral_thinking=True,
        enable_emotion_dynamics=True,
        enable_structured_ending=True,
    )
    cb = CircuitBreaker(CircuitBreakerConfig(max_turns=6))
    session = _session(chars, cfg, llm=MockLLMAdapter(seed=3), circuit_breaker=cb)
    for _ in range(6):
        result = await session.run_turn()
        if result is None:
            break
    assert len(session.conversation.history) >= 1
