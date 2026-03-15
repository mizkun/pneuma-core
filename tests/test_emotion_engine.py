"""Tests for EmotionEngine (Issue #28)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from pneuma_core.emotion import (
    exponential_decay,
    pad_to_emotion_label,
    personality_to_pad_baseline,
)
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.personality import Personality
from pneuma_core.runtime.emotion_engine import EmotionConfig, EmotionEngine, EmotionResult


def _make_personality(
    openness: float = 0.7,
    conscientiousness: float = 0.6,
    extraversion: float = 0.8,
    agreeableness: float = 0.7,
    neuroticism: float = 0.3,
) -> Personality:
    return Personality(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
    )


def _make_llm_response(
    pleasure: float = 0.6,
    arousal: float = 0.4,
    dominance: float = 0.2,
    emotion_label: str = "喜び",
    situation: str = "楽しい会話をしている",
) -> str:
    return json.dumps(
        {
            "pleasure": pleasure,
            "arousal": arousal,
            "dominance": dominance,
            "emotion_label": emotion_label,
            "situation": situation,
        },
        ensure_ascii=False,
    )


# --- EmotionConfig ---


class TestEmotionConfig:
    """EmotionConfig の検証."""

    def test_default_values(self) -> None:
        config = EmotionConfig()
        assert config.recent_messages_limit == 10
        assert config.decay_half_life == 3600.0

    def test_custom_values(self) -> None:
        config = EmotionConfig(
            recent_messages_limit=5,
            decay_half_life=1800.0,
        )
        assert config.recent_messages_limit == 5
        assert config.decay_half_life == 1800.0


# --- LLM 感情推定 ---


class TestEstimate:
    """LLM による感情推定の検証."""

    @pytest.mark.asyncio
    async def test_estimate_returns_emotional_state(self) -> None:
        """LLM から PAD 感情値を推定できる."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(
                content=_make_llm_response(
                    pleasure=0.6,
                    arousal=0.4,
                    dominance=0.2,
                    emotion_label="喜び",
                    situation="楽しい会話",
                )
            )
        )

        engine = EmotionEngine(llm=mock_llm)
        personality = _make_personality()

        result = await engine.estimate(
            personality=personality,
            messages=[
                {"role": "user", "content": "こんにちは！"},
                {"role": "assistant", "content": "やあ！"},
            ],
        )

        assert isinstance(result, EmotionalState)
        assert result.pleasure == pytest.approx(0.6)
        assert result.arousal == pytest.approx(0.4)
        assert result.dominance == pytest.approx(0.2)
        assert result.emotion_label == "喜び"
        assert result.situation == "楽しい会話"

    @pytest.mark.asyncio
    async def test_estimate_truncates_messages(self) -> None:
        """recent_messages_limit に従ってメッセージを切り詰める."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        config = EmotionConfig(recent_messages_limit=3)
        engine = EmotionEngine(llm=mock_llm, config=config)

        messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        await engine.estimate(personality=_make_personality(), messages=messages)

        # LLM に渡されたメッセージが3件に制限されている
        call_args = mock_llm.generate.call_args[0][0]
        assert len(call_args.messages) == 3

    @pytest.mark.asyncio
    async def test_estimate_malformed_response_returns_neutral(self) -> None:
        """不正な LLM レスポンスの場合、中立的な感情を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content="not valid json")
        )

        engine = EmotionEngine(llm=mock_llm)

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        assert isinstance(result, EmotionalState)
        assert result.pleasure == pytest.approx(0.0)
        assert result.arousal == pytest.approx(0.0)
        assert result.dominance == pytest.approx(0.0)
        assert result.emotion_label == "中立"

    @pytest.mark.asyncio
    async def test_estimate_llm_exception_returns_neutral(self) -> None:
        """LLM が例外を投げた場合、中立的な感情を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("API error"))

        engine = EmotionEngine(llm=mock_llm)

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        assert isinstance(result, EmotionalState)
        assert result.emotion_label == "中立"

    @pytest.mark.asyncio
    async def test_estimate_out_of_range_clamped(self) -> None:
        """PAD 値が範囲外の場合、クランプされる."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(
                content=_make_llm_response(pleasure=1.5, arousal=-1.5, dominance=0.0)
            )
        )

        engine = EmotionEngine(llm=mock_llm)

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.pleasure == pytest.approx(1.0)
        assert result.arousal == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_estimate_missing_fields_returns_neutral(self) -> None:
        """必須フィールドが欠落した場合、中立的な感情を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(
                content=json.dumps({"pleasure": 0.5})
            )
        )

        engine = EmotionEngine(llm=mock_llm)

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result.emotion_label == "中立"

    @pytest.mark.asyncio
    async def test_estimate_sanitizes_long_strings(self) -> None:
        """emotion_label と situation が長すぎる場合に切り詰められる."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(
                content=_make_llm_response(
                    emotion_label="あ" * 100,
                    situation="い" * 300,
                )
            )
        )

        engine = EmotionEngine(llm=mock_llm)

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        assert len(result.emotion_label) <= 50
        assert len(result.situation) <= 200

    @pytest.mark.asyncio
    async def test_estimate_strips_newlines(self) -> None:
        """emotion_label と situation の改行が除去される."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(
                content=_make_llm_response(
                    emotion_label="喜び\n注入",
                    situation="楽しい\r\n会話",
                )
            )
        )

        engine = EmotionEngine(llm=mock_llm)

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "\n" not in result.emotion_label
        assert "\r" not in result.situation
        assert result.emotion_label == "喜び 注入"
        assert result.situation == "楽しい  会話"

    @pytest.mark.asyncio
    async def test_estimate_includes_personality_in_prompt(self) -> None:
        """LLM に渡される system_prompt に personality (Big Five) 情報が含まれる."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        engine = EmotionEngine(llm=mock_llm)
        personality = _make_personality(
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.9,
            agreeableness=0.4,
            neuroticism=0.2,
        )

        await engine.estimate(
            personality=personality,
            messages=[{"role": "user", "content": "こんにちは"}],
        )

        # LLM に渡された system_prompt を取得
        call_args = mock_llm.generate.call_args[0][0]
        system_prompt = call_args.system_prompt

        # Big Five の各値がプロンプトに含まれていること
        assert "0.8" in system_prompt  # openness
        assert "0.6" in system_prompt  # conscientiousness
        assert "0.9" in system_prompt  # extraversion
        assert "0.4" in system_prompt  # agreeableness
        assert "0.2" in system_prompt  # neuroticism

    @pytest.mark.asyncio
    async def test_estimate_prompt_contains_trait_names(self) -> None:
        """system_prompt に Big Five の特性名が含まれる."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        engine = EmotionEngine(llm=mock_llm)

        await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        call_args = mock_llm.generate.call_args[0][0]
        system_prompt = call_args.system_prompt

        # 特性名（日本語または英語）がプロンプトに含まれていること
        assert "openness" in system_prompt.lower() or "開放性" in system_prompt
        assert "conscientiousness" in system_prompt.lower() or "誠実性" in system_prompt
        assert "extraversion" in system_prompt.lower() or "外向性" in system_prompt
        assert "agreeableness" in system_prompt.lower() or "協調性" in system_prompt
        assert "neuroticism" in system_prompt.lower() or "神経症傾向" in system_prompt

    @pytest.mark.asyncio
    async def test_estimate_different_personalities_produce_different_prompts(self) -> None:
        """異なる personality が渡された場合、異なるプロンプトが生成される."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        engine = EmotionEngine(llm=mock_llm)

        # 1つ目の personality で呼び出し
        await engine.estimate(
            personality=_make_personality(openness=0.9, neuroticism=0.1),
            messages=[{"role": "user", "content": "hi"}],
        )
        prompt_1 = mock_llm.generate.call_args[0][0].system_prompt

        # 2つ目の personality で呼び出し
        await engine.estimate(
            personality=_make_personality(openness=0.2, neuroticism=0.8),
            messages=[{"role": "user", "content": "hi"}],
        )
        prompt_2 = mock_llm.generate.call_args[0][0].system_prompt

        # 異なる personality なのでプロンプトも異なる
        assert prompt_1 != prompt_2


# --- ベースライン回帰 ---


class TestDecayToBaseline:
    """指数減衰によるベースライン回帰の検証."""

    def test_decay_towards_baseline(self) -> None:
        """感情がベースラインに向かって減衰する."""
        engine = EmotionEngine(llm=AsyncMock())
        personality = _make_personality()
        state = EmotionalState(
            pleasure=0.8, arousal=0.6, dominance=0.4,
            emotion_label="喜び", situation="test",
        )

        decayed = engine.decay_towards_baseline(
            state=state,
            personality=personality,
            elapsed_seconds=3600.0,  # 1 hour = half_life
        )

        # 1半減期後: 差分が半分になる
        baseline = personality_to_pad_baseline(personality)
        original_diff_p = 0.8 - baseline[0]
        decayed_diff_p = decayed.pleasure - baseline[0]
        assert abs(decayed_diff_p) < abs(original_diff_p)

    def test_no_decay_at_zero_elapsed(self) -> None:
        """経過0秒では減衰しない."""
        engine = EmotionEngine(llm=AsyncMock())
        state = EmotionalState(
            pleasure=0.8, arousal=0.6, dominance=0.4,
            emotion_label="喜び", situation="test",
        )

        decayed = engine.decay_towards_baseline(
            state=state,
            personality=_make_personality(),
            elapsed_seconds=0.0,
        )

        assert decayed.pleasure == pytest.approx(0.8)
        assert decayed.arousal == pytest.approx(0.6)
        assert decayed.dominance == pytest.approx(0.4)

    def test_large_elapsed_near_baseline(self) -> None:
        """非常に長い時間が経過するとベースラインに近づく."""
        engine = EmotionEngine(llm=AsyncMock())
        personality = _make_personality()
        state = EmotionalState(
            pleasure=0.9, arousal=-0.8, dominance=0.7,
            emotion_label="怒り", situation="test",
        )

        decayed = engine.decay_towards_baseline(
            state=state,
            personality=personality,
            elapsed_seconds=100000.0,  # ~28 hours
        )

        baseline = personality_to_pad_baseline(personality)
        assert decayed.pleasure == pytest.approx(baseline[0], abs=0.05)
        assert decayed.arousal == pytest.approx(baseline[1], abs=0.05)
        assert decayed.dominance == pytest.approx(baseline[2], abs=0.05)

    def test_decay_updates_emotion_label(self) -> None:
        """減衰後の感情ラベルが PAD 値に基づいて更新される."""
        engine = EmotionEngine(llm=AsyncMock())
        state = EmotionalState(
            pleasure=0.8, arousal=0.6, dominance=0.4,
            emotion_label="喜び（高揚）", situation="test",
        )

        decayed = engine.decay_towards_baseline(
            state=state,
            personality=_make_personality(),
            elapsed_seconds=3600.0,
        )

        # ラベルが PAD 値から再計算される
        expected_label = pad_to_emotion_label(
            decayed.pleasure, decayed.arousal, decayed.dominance
        )
        assert decayed.emotion_label == expected_label


# --- Issue #56: モデル選択パラメータ ---


class TestModelSelection:
    """EmotionEngine のモデル選択パラメータの検証 (#56)."""

    @pytest.mark.asyncio
    async def test_model_parameter_accepted(self) -> None:
        """EmotionEngine が model パラメータを受け取れる."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        engine = EmotionEngine(llm=mock_llm, model="claude-haiku-4-20250514")

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(result, EmotionalState)

    @pytest.mark.asyncio
    async def test_model_set_in_llm_request(self) -> None:
        """model パラメータが LLMRequest.model に設定される."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        engine = EmotionEngine(llm=mock_llm, model="claude-haiku-4-20250514")

        await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        call_args = mock_llm.generate.call_args[0][0]
        assert call_args.model == "claude-haiku-4-20250514"

    @pytest.mark.asyncio
    async def test_model_none_by_default(self) -> None:
        """model を指定しない場合は LLMRequest.model が None."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        engine = EmotionEngine(llm=mock_llm)

        await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "hi"}],
        )

        call_args = mock_llm.generate.call_args[0][0]
        assert call_args.model is None

    @pytest.mark.asyncio
    async def test_model_with_config(self) -> None:
        """model と config を同時に指定できる."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )

        config = EmotionConfig(recent_messages_limit=5)
        engine = EmotionEngine(
            llm=mock_llm,
            config=config,
            model="claude-haiku-4-20250514",
        )

        messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        await engine.estimate(personality=_make_personality(), messages=messages)

        call_args = mock_llm.generate.call_args[0][0]
        # model が設定される
        assert call_args.model == "claude-haiku-4-20250514"
        # config の recent_messages_limit も効く
        assert len(call_args.messages) == 5


# --- Issue #131: 毎ターン直接推定 ---


class TestEmotionConfigSimplified:
    """EmotionConfig の簡素化された設定の検証 (#131)."""

    def test_no_safety_net_interval(self) -> None:
        """safety_net_interval が廃止されている."""
        config = EmotionConfig()
        assert not hasattr(config, "safety_net_interval")


class TestEvaluate:
    """EmotionEngine.evaluate() の毎ターン直接推定検証 (#131)."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_emotion_result(self) -> None:
        """evaluate が EmotionResult を返す."""

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )
        engine = EmotionEngine(llm=mock_llm)

        current_state = EmotionalState(
            pleasure=0.0, arousal=0.0, dominance=0.0,
            emotion_label="中立", situation="",
        )

        result = await engine.evaluate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "嬉しい！"}],
            turn_count=1,
            current_state=current_state,
        )

        assert isinstance(result, EmotionResult)
        assert isinstance(result.state, EmotionalState)
        assert result.trigger_type == "triggered"

    @pytest.mark.asyncio
    async def test_evaluate_always_calls_llm(self) -> None:
        """毎ターン必ず LLM が1回呼ばれる（Tier 0/1 によるスキップなし）."""

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )
        engine = EmotionEngine(llm=mock_llm)

        current_state = EmotionalState(
            pleasure=0.3, arousal=0.1, dominance=0.0,
            emotion_label="安らぎ", situation="穏やかな会話中",
        )

        # 感情キーワードがないメッセージでも LLM が呼ばれる
        result = await engine.evaluate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "明日の天気は？"}],
            turn_count=1,
            current_state=current_state,
        )

        assert result.trigger_type == "triggered"
        assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_returns_estimated_state(self) -> None:
        """evaluate が LLM 推定結果の EmotionalState を返す."""

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(
                pleasure=0.7, arousal=0.5, dominance=0.2,
                emotion_label="喜び", situation="嬉しいことがあった",
            ))
        )
        engine = EmotionEngine(llm=mock_llm)

        current_state = EmotionalState(
            pleasure=0.0, arousal=0.0, dominance=0.0,
            emotion_label="中立", situation="",
        )

        result = await engine.evaluate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "めっちゃ嬉しい！最高！"}],
            turn_count=1,
            current_state=current_state,
        )

        assert result.trigger_type == "triggered"
        assert result.state.emotion_label == "喜び"
        assert result.state.pleasure == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_evaluate_only_one_llm_call_per_turn(self) -> None:
        """毎ターン Haiku 1回だけ呼ばれる（Tier 1 ゲートなし）."""

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )
        engine = EmotionEngine(llm=mock_llm)

        current_state = EmotionalState(
            pleasure=0.0, arousal=0.0, dominance=0.0,
            emotion_label="中立", situation="",
        )

        await engine.evaluate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "嬉しい！"}],
            turn_count=1,
            current_state=current_state,
        )

        # Tier 1 ゲートがないので LLM は1回だけ呼ばれる
        assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_uses_model_parameter(self) -> None:
        """evaluate で model パラメータが LLMRequest に設定される."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )
        engine = EmotionEngine(llm=mock_llm, model="claude-haiku-4-20250514")

        current_state = EmotionalState(
            pleasure=0.0, arousal=0.0, dominance=0.0,
            emotion_label="中立", situation="",
        )

        await engine.evaluate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "こんにちは"}],
            turn_count=1,
            current_state=current_state,
        )

        call_args = mock_llm.generate.call_args[0][0]
        assert call_args.model == "claude-haiku-4-20250514"

    @pytest.mark.asyncio
    async def test_evaluate_llm_failure_returns_neutral(self) -> None:
        """LLM が失敗した場合は中立状態を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("API error"))
        engine = EmotionEngine(llm=mock_llm)

        current_state = EmotionalState(
            pleasure=0.3, arousal=0.1, dominance=0.0,
            emotion_label="安らぎ", situation="穏やか",
        )

        result = await engine.evaluate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "嬉しい！"}],
            turn_count=1,
            current_state=current_state,
        )

        assert result.trigger_type == "triggered"
        # estimate() が中立を返す
        assert result.state.emotion_label == "中立"

    @pytest.mark.asyncio
    async def test_evaluate_no_emotion_trigger_dependency(self) -> None:
        """EmotionEngine が EmotionTrigger に依存しない."""
        # EmotionEngine のインスタンスに _trigger 属性がないこと
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )
        engine = EmotionEngine(llm=mock_llm)
        assert not hasattr(engine, "_trigger")

    @pytest.mark.asyncio
    async def test_evaluate_consistent_across_turn_counts(self) -> None:
        """どの turn_count でも同じように毎ターン推定される（safety net 不要）."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )
        engine = EmotionEngine(llm=mock_llm)

        current_state = EmotionalState(
            pleasure=0.0, arousal=0.0, dominance=0.0,
            emotion_label="中立", situation="",
        )

        for turn in [1, 7, 15, 30, 100]:
            mock_llm.generate.reset_mock()
            result = await engine.evaluate(
                personality=_make_personality(),
                messages=[{"role": "user", "content": "普通の会話です"}],
                turn_count=turn,
                current_state=current_state,
            )
            assert result.trigger_type == "triggered"
            assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_existing_estimate_still_works(self) -> None:
        """既存の estimate() メソッドは後方互換性を維持."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response())
        )
        engine = EmotionEngine(llm=mock_llm)

        result = await engine.estimate(
            personality=_make_personality(),
            messages=[{"role": "user", "content": "こんにちは"}],
        )
        assert isinstance(result, EmotionalState)
        assert result.pleasure == pytest.approx(0.6)
