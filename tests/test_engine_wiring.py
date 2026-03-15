"""Tests for engine wiring: emotion decay (Issue #125).

These tests verify that emotion decay is properly wired
into RuntimeEngine.process_message().
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.message import MessageInput
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.engine import RuntimeEngine


NOW = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)


def _make_character() -> Character:
    return Character(
        id="test-001",
        name="テスト",
        personality=Personality(
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.7,
            agreeableness=0.9,
            neuroticism=0.3,
        ),
        values=Values(
            self_transcendence=0.8,
            self_enhancement=0.3,
            openness_to_change=0.7,
            conservation=0.4,
        ),
        profile="テスト用キャラ",
        speaking_style="普通の口調",
    )


def _make_emotional_state(
    pleasure: float = 0.6,
    arousal: float = 0.4,
    dominance: float = 0.2,
) -> EmotionalState:
    return EmotionalState(
        pleasure=pleasure,
        arousal=arousal,
        dominance=dominance,
        emotion_label="喜び",
        situation="テスト中",
    )


def _make_message_input(content: str = "こんにちは！") -> MessageInput:
    return MessageInput(
        content=content,
        sender_id="user-1",
        sender_name="ユーザー",
        sender_type="human",
    )


def _make_llm_response(content: str = "やあ！元気？") -> MagicMock:
    response = MagicMock()
    response.content = content
    response.model = "test-model"
    response.usage = {}
    return response


def _make_emotion_response(
    pleasure: float = 0.5,
    arousal: float = 0.3,
    dominance: float = 0.1,
) -> MagicMock:
    response = MagicMock()
    response.content = json.dumps(
        {
            "pleasure": pleasure,
            "arousal": arousal,
            "dominance": dominance,
            "emotion_label": "喜び",
            "situation": "楽しい会話",
        },
        ensure_ascii=False,
    )
    response.model = "test-model"
    response.usage = {}
    return response


def _setup_mocks():
    """Create standard mock dependencies for RuntimeEngine."""
    storage = AsyncMock()
    storage.get_character = AsyncMock(return_value=_make_character())
    storage.get_emotional_state = AsyncMock(return_value=_make_emotional_state())
    storage.get_goals = AsyncMock(return_value=GoalTree())
    storage.save_emotional_state = AsyncMock()
    storage.save_change = AsyncMock()
    llm = AsyncMock()
    llm.generate = AsyncMock(
        side_effect=[
            _make_llm_response(),
            _make_emotion_response(),
        ]
    )

    embedding_service = AsyncMock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 10)
    embedding_service.embed_batch = AsyncMock(return_value=[[0.1] * 10])

    memory_store = AsyncMock()
    memory_store.get_episodic_by_character = AsyncMock(return_value=[])
    memory_store.get_semantic_by_character = AsyncMock(return_value=[])
    memory_store.add_episodic = AsyncMock()
    memory_store.add_semantic = AsyncMock()
    memory_store.find_similar_episodic = AsyncMock(return_value=[])

    return storage, llm, embedding_service, memory_store


# =============================================================================
# Test 1: Emotion Decay
# =============================================================================


class TestEmotionDecay:
    """engine.py が process_message 中で感情減衰を適用することを検証."""

    @pytest.mark.asyncio
    async def test_emotion_decays_towards_baseline_after_time_passes(self) -> None:
        """前回の感情更新から時間が経過している場合、感情がベースラインに向かって減衰する."""
        storage, llm, embedding_service, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
        )

        # Set a past last_emotion_update to trigger decay
        engine._last_emotion_update = NOW - timedelta(minutes=30)

        with patch.object(
            engine._emotion_engine,
            "decay_towards_baseline",
            wraps=engine._emotion_engine.decay_towards_baseline,
        ) as mock_decay, patch(
            "pneuma_core.runtime.engine.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            await engine.process_message(_make_message_input())

            # decay_towards_baseline should have been called
            mock_decay.assert_called_once()
            call_args = mock_decay.call_args
            # Verify elapsed_seconds is approximately 30 minutes (1800s)
            elapsed = call_args.kwargs.get("elapsed_seconds") or call_args[0][2]
            assert elapsed > 0, "elapsed_seconds should be positive"

    @pytest.mark.asyncio
    async def test_no_decay_on_first_message(self) -> None:
        """初回メッセージでは _last_emotion_update が None のため減衰が適用されない."""
        storage, llm, embedding_service, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
        )

        with patch.object(
            engine._emotion_engine,
            "decay_towards_baseline",
        ) as mock_decay, patch(
            "pneuma_core.runtime.engine.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            await engine.process_message(_make_message_input())

            # decay should NOT be called when there's no previous timestamp
            mock_decay.assert_not_called()

    @pytest.mark.asyncio
    async def test_last_emotion_update_is_set_after_process_message(self) -> None:
        """process_message 実行後、_last_emotion_update が現在時刻に更新される."""
        storage, llm, embedding_service, memory_store = _setup_mocks()

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
        )

        assert engine._last_emotion_update is None

        with patch("pneuma_core.runtime.engine.datetime") as mock_dt:
            mock_dt.now.return_value = NOW
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            await engine.process_message(_make_message_input())

        assert engine._last_emotion_update == NOW
