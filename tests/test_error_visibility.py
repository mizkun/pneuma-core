"""Tests for error visibility via SystemMessage (Issue #119)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pneuma_core.models.message import MessageInput, MessageOutput, SystemMessage


class TestSystemMessageModel:
    """SystemMessage データクラスの検証."""

    def test_create_warning(self) -> None:
        """warning タイプの SystemMessage を作成できる."""
        sm = SystemMessage(
            type="warning",
            message="Memory search failed",
            component="memory_search",
        )
        assert sm.type == "warning"
        assert sm.message == "Memory search failed"
        assert sm.component == "memory_search"

    def test_create_error(self) -> None:
        """error タイプの SystemMessage を作成できる."""
        sm = SystemMessage(
            type="error",
            message="LLM generation failed",
            component="llm",
        )
        assert sm.type == "error"
        assert sm.message == "LLM generation failed"
        assert sm.component == "llm"

    def test_create_info(self) -> None:
        """info タイプの SystemMessage を作成できる."""
        sm = SystemMessage(
            type="info",
            message="Using cached prompt",
            component="prompt_cache",
        )
        assert sm.type == "info"
        assert sm.message == "Using cached prompt"
        assert sm.component == "prompt_cache"

    def test_frozen_dataclass(self) -> None:
        """SystemMessage は frozen（不変）である."""
        sm = SystemMessage(
            type="warning",
            message="test",
            component="test",
        )
        with pytest.raises(AttributeError):
            sm.type = "error"  # type: ignore[misc]

    def test_equality(self) -> None:
        """同じ値の SystemMessage は等しい."""
        sm1 = SystemMessage(type="warning", message="test", component="comp")
        sm2 = SystemMessage(type="warning", message="test", component="comp")
        assert sm1 == sm2

    def test_different_values_not_equal(self) -> None:
        """異なる値の SystemMessage は等しくない."""
        sm1 = SystemMessage(type="warning", message="test", component="comp")
        sm2 = SystemMessage(type="error", message="test", component="comp")
        assert sm1 != sm2


class TestMessageOutputSystemMessages:
    """MessageOutput の system_messages フィールドの検証."""

    def test_default_empty_list(self) -> None:
        """system_messages はデフォルトで空のリスト."""
        from pneuma_core.models.emotion import EmotionalState

        output = MessageOutput(
            content="hello",
            emotion=EmotionalState(
                pleasure=0.0,
                arousal=0.0,
                dominance=0.0,
                emotion_label="neutral",
                situation="test",
            ),
        )
        assert output.system_messages == []
        assert isinstance(output.system_messages, list)

    def test_with_system_messages(self) -> None:
        """system_messages にメッセージを含められる."""
        from pneuma_core.models.emotion import EmotionalState

        msgs = [
            SystemMessage(
                type="warning",
                message="Memory search failed",
                component="memory_search",
            ),
            SystemMessage(
                type="warning",
                message="Todo context build failed",
                component="todo",
            ),
        ]
        output = MessageOutput(
            content="hello",
            emotion=EmotionalState(
                pleasure=0.0,
                arousal=0.0,
                dominance=0.0,
                emotion_label="neutral",
                situation="test",
            ),
            system_messages=msgs,
        )
        assert len(output.system_messages) == 2
        assert output.system_messages[0].type == "warning"
        assert output.system_messages[0].component == "memory_search"
        assert output.system_messages[1].component == "todo"

    def test_system_messages_separate_from_content(self) -> None:
        """system_messages はキャラクター応答（content）とは独立している."""
        from pneuma_core.models.emotion import EmotionalState

        msgs = [
            SystemMessage(
                type="error",
                message="LLM failed",
                component="llm",
            ),
        ]
        output = MessageOutput(
            content="fallback response",
            emotion=EmotionalState(
                pleasure=0.0,
                arousal=0.0,
                dominance=0.0,
                emotion_label="neutral",
                situation="test",
            ),
            system_messages=msgs,
        )
        # content はキャラクター応答
        assert output.content == "fallback response"
        # system_messages はシステムメッセージ
        assert output.system_messages[0].message == "LLM failed"
        # 明確に分離されている
        assert "LLM failed" not in output.content

    def test_no_mutation_across_instances(self) -> None:
        """各 MessageOutput の system_messages は独立している（default_factory）."""
        from pneuma_core.models.emotion import EmotionalState

        emotion = EmotionalState(
            pleasure=0.0,
            arousal=0.0,
            dominance=0.0,
            emotion_label="neutral",
            situation="test",
        )
        output1 = MessageOutput(content="a", emotion=emotion)
        output2 = MessageOutput(content="b", emotion=emotion)
        # frozen なので直接 append はできないが、リストが同一オブジェクトでないことを確認
        assert output1.system_messages is not output2.system_messages
