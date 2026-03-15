"""Tests for Middleware Protocol and RuntimeEngine middleware integration (#145)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.message import MessageInput, MessageOutput
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.runtime.middleware import Middleware, PipelineContext


NOW = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)


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


def _make_emotional_state() -> EmotionalState:
    return EmotionalState(
        pleasure=0.3,
        arousal=0.1,
        dominance=0.0,
        emotion_label="安らぎ",
        situation="テスト中",
    )


def _make_message_input(content: str = "こんにちは！") -> MessageInput:
    return MessageInput(
        content=content,
        sender_id="user-1",
        sender_name="ユーザー",
        sender_type="human",
    )


def _make_llm_response(content: str = '{"speech": "やあ！", "thought": "テスト", "action": null}') -> MagicMock:
    response = MagicMock()
    response.content = content
    response.model = "test-model"
    response.usage = {}
    return response


def _make_emotion_response() -> MagicMock:
    response = MagicMock()
    response.content = json.dumps({
        "pleasure": 0.5,
        "arousal": 0.3,
        "dominance": 0.1,
        "emotion_label": "喜び",
        "situation": "楽しい会話",
    }, ensure_ascii=False)
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

    memory_store = AsyncMock()
    memory_store.get_episodic_by_character = AsyncMock(return_value=[])
    memory_store.get_semantic_by_character = AsyncMock(return_value=[])

    return storage, llm, embedding_service, memory_store


# =============================================================================
# Test 1: Middleware Protocol exists and has correct interface
# =============================================================================


class TestMiddlewareProtocol:
    """Middleware Protocol が正しいインターフェースを持つことを検証."""

    def test_middleware_protocol_has_pre_process(self) -> None:
        """Middleware には pre_process メソッドがある."""
        assert hasattr(Middleware, "pre_process")

    def test_middleware_protocol_has_post_process(self) -> None:
        """Middleware には post_process メソッドがある."""
        assert hasattr(Middleware, "post_process")


# =============================================================================
# Test 2: PipelineContext has required fields
# =============================================================================


class TestPipelineContext:
    """PipelineContext が必要なフィールドを持つことを検証."""

    def test_pipeline_context_has_character(self) -> None:
        """PipelineContext には character フィールドがある."""
        ctx = PipelineContext(
            character=_make_character(),
            emotion=_make_emotional_state(),
            goals=GoalTree(),
            memories=[],
            system_prompt="test",
            history=[],
            turn_count=1,
            metadata={},
        )
        assert ctx.character.name == "テスト"

    def test_pipeline_context_has_metadata(self) -> None:
        """PipelineContext には metadata dict がある（ミドルウェア間のデータ共有用）."""
        ctx = PipelineContext(
            character=_make_character(),
            emotion=_make_emotional_state(),
            goals=GoalTree(),
            memories=[],
            system_prompt="test",
            history=[],
            turn_count=1,
            metadata={"key": "value"},
        )
        assert ctx.metadata["key"] == "value"


# =============================================================================
# Test 3: RuntimeEngine accepts middlewares parameter
# =============================================================================


class TestEngineMiddlewareIntegration:
    """RuntimeEngine がミドルウェアを受け入れて実行することを検証."""

    def test_engine_accepts_middlewares_parameter(self) -> None:
        """RuntimeEngine のコンストラクタが middlewares パラメータを受け入れる."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[],
        )
        assert engine is not None

    @pytest.mark.asyncio
    async def test_middleware_pre_process_is_called(self) -> None:
        """process_message 中にミドルウェアの pre_process が呼ばれる."""
        storage, llm, embedding_service, memory_store = _setup_mocks()

        class TrackingMiddleware:
            def __init__(self):
                self.pre_called = False
                self.post_called = False

            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                self.pre_called = True
                return msg

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                self.post_called = True
                return output

        mw = TrackingMiddleware()
        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[mw],
        )

        await engine.process_message(_make_message_input())
        assert mw.pre_called, "pre_process should have been called"

    @pytest.mark.asyncio
    async def test_middleware_post_process_is_called(self) -> None:
        """process_message 中にミドルウェアの post_process が呼ばれる."""
        storage, llm, embedding_service, memory_store = _setup_mocks()

        class TrackingMiddleware:
            def __init__(self):
                self.post_called = False

            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                return msg

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                self.post_called = True
                return output

        mw = TrackingMiddleware()
        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[mw],
        )

        await engine.process_message(_make_message_input())
        assert mw.post_called, "post_process should have been called"

    @pytest.mark.asyncio
    async def test_middleware_can_modify_input(self) -> None:
        """pre_process でメッセージ内容を変更できる."""
        storage, llm, embedding_service, memory_store = _setup_mocks()

        class ModifyingMiddleware:
            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                return dc_replace(msg, content=msg.content + " [modified]")

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                return output

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[ModifyingMiddleware()],
        )

        await engine.process_message(_make_message_input("hello"))
        # Verify the modified content was sent to LLM
        call_args = llm.generate.call_args_list[0]
        messages = call_args[0][0].messages if call_args[0] else call_args[1].get("messages", [])
        # The last user message in history should contain "[modified]"
        user_messages = [m for m in messages if m["role"] == "user"]
        assert any("[modified]" in m["content"] for m in user_messages)

    @pytest.mark.asyncio
    async def test_multiple_middlewares_run_in_order(self) -> None:
        """複数のミドルウェアが登録順に実行される."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        execution_order: list[str] = []

        class FirstMiddleware:
            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                execution_order.append("first_pre")
                return msg

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                execution_order.append("first_post")
                return output

        class SecondMiddleware:
            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                execution_order.append("second_pre")
                return msg

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                execution_order.append("second_post")
                return output

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[FirstMiddleware(), SecondMiddleware()],
        )

        await engine.process_message(_make_message_input())
        assert execution_order == ["first_pre", "second_pre", "second_post", "first_post"]

    @pytest.mark.asyncio
    async def test_pipeline_context_has_correct_data(self) -> None:
        """ミドルウェアに渡される PipelineContext が正しいデータを持つ."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        captured_context: PipelineContext | None = None

        class CapturingMiddleware:
            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                nonlocal captured_context
                captured_context = context
                return msg

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                return output

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[CapturingMiddleware()],
        )

        await engine.process_message(_make_message_input())

        assert captured_context is not None
        assert captured_context.character.name == "テスト"
        assert captured_context.emotion is not None
        assert isinstance(captured_context.goals, GoalTree)
        assert isinstance(captured_context.memories, list)
        assert isinstance(captured_context.metadata, dict)


# =============================================================================
# Test 4: RuntimeEngine no longer accepts old optional parameters
# =============================================================================


class TestEngineSlimConstructor:
    """RuntimeEngine が旧 optional パラメータを受け付けないことを検証."""

    def test_engine_rejects_conversation_logger_parameter(self) -> None:
        """conversation_logger パラメータは受け付けない."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        with pytest.raises(TypeError):
            RuntimeEngine(
                character_id="test-001",
                storage=storage,
                llm=llm,
                embedding_service=embedding_service,
                memory_store=memory_store,
                conversation_logger=MagicMock(),  # type: ignore[call-arg]
            )

    def test_engine_rejects_diary_writer_parameter(self) -> None:
        """diary_writer パラメータは受け付けない."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        with pytest.raises(TypeError):
            RuntimeEngine(
                character_id="test-001",
                storage=storage,
                llm=llm,
                embedding_service=embedding_service,
                memory_store=memory_store,
                diary_writer=MagicMock(),  # type: ignore[call-arg]
            )

    def test_engine_rejects_diary_coaching_parameter(self) -> None:
        """diary_coaching パラメータは受け付けない."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        with pytest.raises(TypeError):
            RuntimeEngine(
                character_id="test-001",
                storage=storage,
                llm=llm,
                embedding_service=embedding_service,
                memory_store=memory_store,
                diary_coaching=MagicMock(),  # type: ignore[call-arg]
            )

    def test_engine_rejects_task_assistant_parameter(self) -> None:
        """task_assistant パラメータは受け付けない."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        with pytest.raises(TypeError):
            RuntimeEngine(
                character_id="test-001",
                storage=storage,
                llm=llm,
                embedding_service=embedding_service,
                memory_store=memory_store,
                task_assistant=MagicMock(),  # type: ignore[call-arg]
            )

    def test_engine_rejects_prompt_dumper_parameter(self) -> None:
        """prompt_dumper パラメータは受け付けない."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        with pytest.raises(TypeError):
            RuntimeEngine(
                character_id="test-001",
                storage=storage,
                llm=llm,
                embedding_service=embedding_service,
                memory_store=memory_store,
                prompt_dumper=MagicMock(),  # type: ignore[call-arg]
            )


# =============================================================================
# Test 5: Middleware can replicate old functionality
# =============================================================================


class TestMiddlewareFunctionality:
    """ミドルウェアで旧機能を再現できることを検証."""

    @pytest.mark.asyncio
    async def test_logging_middleware_pattern(self) -> None:
        """post_process でログ記録パターンが動作する."""
        storage, llm, embedding_service, memory_store = _setup_mocks()
        logged_outputs: list[MessageOutput] = []

        class LoggingMiddleware:
            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                return msg

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                logged_outputs.append(output)
                return output

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[LoggingMiddleware()],
        )

        await engine.process_message(_make_message_input())
        assert len(logged_outputs) == 1
        assert logged_outputs[0].content is not None

    @pytest.mark.asyncio
    async def test_content_injection_middleware_pattern(self) -> None:
        """pre_process でコンテンツ注入パターン（タスク、コーチング）が動作する."""
        storage, llm, embedding_service, memory_store = _setup_mocks()

        class ContentInjectionMiddleware:
            """Simulates TaskAssistant/DiaryCoaching by appending context to message."""

            async def pre_process(self, msg: MessageInput, context: PipelineContext) -> MessageInput:
                return dc_replace(msg, content=msg.content + "\n[タスク: 買い物に行く]")

            async def post_process(self, msg: MessageInput, output: MessageOutput, context: PipelineContext) -> MessageOutput:
                return output

        engine = RuntimeEngine(
            character_id="test-001",
            storage=storage,
            llm=llm,
            embedding_service=embedding_service,
            memory_store=memory_store,
            middlewares=[ContentInjectionMiddleware()],
        )

        await engine.process_message(_make_message_input("おはよう"))
        # The injected content should reach the LLM
        call_args = llm.generate.call_args_list[0]
        request = call_args[0][0] if call_args[0] else call_args[1]
        user_messages = [m for m in request.messages if m["role"] == "user"]
        assert any("[タスク: 買い物に行く]" in m["content"] for m in user_messages)
