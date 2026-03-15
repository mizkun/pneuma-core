"""Tests for conversation history summarization (Issue #118).

Tests for:
- Default history_limit changed from 100 to 30
- Summarization of older messages when history exceeds limit
- Summary included in conversation context
- Context maintained across summarization boundary
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.message import MessageInput
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.engine import RuntimeEngine


def _make_character() -> Character:
    return Character(
        id="aine-001",
        name="アイネ",
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
        profile="好奇心旺盛な AI",
        speaking_style="丁寧だけど親しみやすい口調",
    )


def _make_emotional_state() -> EmotionalState:
    return EmotionalState(
        pleasure=0.3,
        arousal=0.1,
        dominance=0.0,
        emotion_label="安らぎ",
        situation="穏やかな会話中",
    )


def _make_message_input(
    content: str = "こんにちは！",
    sender_id: str = "user-1",
    sender_name: str = "ユーザー",
    sender_type: str = "human",
) -> MessageInput:
    return MessageInput(
        content=content,
        sender_id=sender_id,
        sender_name=sender_name,
        sender_type=sender_type,
    )


def _make_llm_response(content: str = "やあ！元気？") -> AsyncMock:
    response = AsyncMock()
    response.content = content
    response.model = "test-model"
    response.usage = {}
    return response


def _make_summary_response(summary: str = "ユーザーとアイネが挨拶を交わした。") -> AsyncMock:
    response = AsyncMock()
    response.content = summary
    response.model = "claude-haiku-4-5-20251001"
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
    llm.generate = AsyncMock(return_value=_make_llm_response())

    embedding_service = AsyncMock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 10)

    memory_store = AsyncMock()
    memory_store.get_episodic_by_character = AsyncMock(return_value=[])
    memory_store.get_semantic_by_character = AsyncMock(return_value=[])

    return storage, llm, embedding_service, memory_store


class TestDefaultHistoryLimit:
    """デフォルトの history_limit が 30 に変更されていることのテスト."""

    def test_default_history_limit_is_30(self) -> None:
        """デフォルトの history_limit は 30."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
        )
        assert engine._history_limit == 30

    def test_custom_history_limit(self) -> None:
        """history_limit はカスタマイズ可能."""
        storage, llm, embedding, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=50,
        )
        assert engine._history_limit == 50


class TestHistorySummarization:
    """会話履歴の要約圧縮テスト."""

    @pytest.mark.asyncio
    async def test_summary_generated_when_history_exceeds_limit(self) -> None:
        """履歴が上限を超えたとき、古いメッセージが要約される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        call_count = 0
        summary_text = "[これまでの会話の要約]\n- ユーザーとアイネが挨拶を交わした"

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            # Summarization calls use haiku model
            if request.model == "claude-haiku-4-5-20251001":
                return _make_summary_response(summary_text)
            return _make_llm_response(f"応答{call_count}")

        llm.generate = AsyncMock(side_effect=mock_generate)

        # Use small limit to trigger summarization quickly
        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,  # Small limit: 3 turns = 6 messages will trigger
        )

        # Send 4 messages (each creates user + assistant = 2 entries = 8 total)
        for i in range(4):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # After exceeding limit, summarization should have been triggered
        assert engine._conversation_summary is not None

    @pytest.mark.asyncio
    async def test_summary_included_in_history_as_system_message(self) -> None:
        """要約は履歴の先頭にシステムメッセージとして含まれる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        summary_text = "[これまでの会話の要約]\n- 挨拶を交わした"

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                return _make_summary_response(summary_text)
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        # Send enough messages to trigger summarization
        for i in range(5):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # Check that the messages sent to LLM include summary as first message
        last_main_call = None
        for c in llm.generate.call_args_list:
            req = c[0][0]
            if req.model != "claude-haiku-4-5-20251001":
                last_main_call = c

        assert last_main_call is not None
        request = last_main_call[0][0]
        # First message should be the summary
        assert request.messages[0]["role"] == "system"
        assert "[これまでの会話の要約]" in request.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_history_stays_within_limit_after_summarization(self) -> None:
        """要約後も履歴はリミット内に収まる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                return _make_summary_response("要約テスト")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=10,
        )

        # Send many messages
        for i in range(20):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # History should never exceed limit
        assert len(engine._history) <= 10

    @pytest.mark.asyncio
    async def test_summary_uses_haiku_model(self) -> None:
        """要約生成には Haiku モデルが使われる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        haiku_calls = []

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                haiku_calls.append(request)
                return _make_summary_response("要約")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        for i in range(5):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # Haiku should have been called for summarization
        assert len(haiku_calls) > 0
        for req in haiku_calls:
            assert req.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_summary_accumulates_across_multiple_compressions(self) -> None:
        """複数回の圧縮で要約が蓄積される（前の要約 + 新しいメッセージの要約）."""
        storage, llm, embedding, memory_store = _setup_mocks()

        summarization_count = 0

        async def mock_generate(request):
            nonlocal summarization_count
            if request.model == "claude-haiku-4-5-20251001":
                summarization_count += 1
                return _make_summary_response(f"要約回{summarization_count}")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        # Send many messages to trigger multiple summarizations
        for i in range(15):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # Multiple summarizations should have occurred
        assert summarization_count >= 2

    @pytest.mark.asyncio
    async def test_summarization_failure_does_not_break_response(self) -> None:
        """要約生成が失敗しても応答は正常に返る."""
        storage, llm, embedding, memory_store = _setup_mocks()

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                raise RuntimeError("Summarization API error")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        # Should not raise even when summarization fails
        for i in range(5):
            result = await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )
            assert result.content != ""

    @pytest.mark.asyncio
    async def test_no_summarization_when_within_limit(self) -> None:
        """履歴がリミット内なら要約は生成されない."""
        storage, llm, embedding, memory_store = _setup_mocks()

        haiku_calls = []

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                haiku_calls.append(request)
                return _make_summary_response("要約")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=30,  # Large enough for 3 turns
        )

        # Send only a few messages (within limit)
        for i in range(3):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # No haiku calls for summarization should have been made
        assert len(haiku_calls) == 0
        assert engine._conversation_summary is None

    @pytest.mark.asyncio
    async def test_summary_prompt_includes_old_messages(self) -> None:
        """要約プロンプトには削除される古いメッセージが含まれる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        summarization_requests = []

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                summarization_requests.append(request)
                return _make_summary_response("要約")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        for i in range(5):
            await engine.process_message(
                _make_message_input(content=f"特定のメッセージ{i}")
            )

        # Summarization prompt should reference the old messages
        assert len(summarization_requests) > 0
        first_summary_req = summarization_requests[0]
        # The system prompt or messages should contain the old conversation content
        prompt_text = first_summary_req.system_prompt
        assert "メッセージ" in prompt_text or any(
            "メッセージ" in m.get("content", "") for m in first_summary_req.messages
        )

    @pytest.mark.asyncio
    async def test_summary_includes_previous_summary_context(self) -> None:
        """2回目以降の要約では、前回の要約も含まれる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        summarization_requests = []
        summary_count = 0

        async def mock_generate(request):
            nonlocal summary_count
            if request.model == "claude-haiku-4-5-20251001":
                summary_count += 1
                summarization_requests.append(request)
                return _make_summary_response(f"要約{summary_count}回目")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        # Send enough messages to trigger 2+ summarizations
        for i in range(15):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # After second summarization, prompt should include previous summary
        if len(summarization_requests) >= 2:
            second_req = summarization_requests[1]
            prompt_text = second_req.system_prompt
            # Previous summary should be referenced
            assert "要約" in prompt_text or any(
                "要約" in m.get("content", "") for m in second_req.messages
            )


class TestHistorySummarizationPipeline:
    """要約の入力→処理→保持→再利用のパイプラインテスト."""

    @pytest.mark.asyncio
    async def test_full_pipeline_summary_persists_across_turns(self) -> None:
        """要約が生成された後のターンでも要約が維持される."""
        storage, llm, embedding, memory_store = _setup_mocks()

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                return _make_summary_response("仕事の疲れについて話し、休息を提案した")
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        # Fill history to trigger summarization
        for i in range(5):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # Summary should be set
        assert engine._conversation_summary is not None
        summary_after_compression = engine._conversation_summary

        # Send one more message
        await engine.process_message(
            _make_message_input(content="新しいメッセージ")
        )

        # Summary should still be present (may have been updated)
        assert engine._conversation_summary is not None

    @pytest.mark.asyncio
    async def test_pipeline_messages_sent_to_llm_include_summary(self) -> None:
        """LLM に送信されるメッセージリストに要約が含まれる."""
        storage, llm, embedding, memory_store = _setup_mocks()

        summary_text = "[これまでの会話の要約]\n- 天気の話をした\n- 週末の予定を聞いた"
        main_call_messages = []

        async def mock_generate(request):
            if request.model == "claude-haiku-4-5-20251001":
                return _make_summary_response(summary_text)
            main_call_messages.append(request.messages[:])
            return _make_llm_response("応答")

        llm.generate = AsyncMock(side_effect=mock_generate)

        engine = RuntimeEngine(
            character_id="aine-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding,
            memory_store=memory_store,
            history_limit=6,
        )

        # Trigger summarization
        for i in range(5):
            await engine.process_message(
                _make_message_input(content=f"メッセージ{i}")
            )

        # After summarization, LLM calls should include summary in messages
        # Find a main call that was made after summarization
        found_summary_in_messages = False
        for msgs in main_call_messages:
            for m in msgs:
                if m.get("role") == "system" and "これまでの会話の要約" in m.get("content", ""):
                    found_summary_in_messages = True
                    break

        assert found_summary_in_messages, (
            "要約がLLMに送信されるメッセージリストに含まれていない"
        )
