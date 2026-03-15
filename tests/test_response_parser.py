"""Tests for response_parser (Issue #86).

Structured output: speech / thought / action separation.
"""

from __future__ import annotations

import json

import pytest

from pneuma_core.models.message import StructuredResponse
from pneuma_core.runtime.response_parser import parse_structured_response


class TestParseStructuredResponse:
    """parse_structured_response のユニットテスト."""

    def test_valid_json_all_fields(self) -> None:
        """有効な JSON で全フィールド (speech/thought/action) がパースされる."""
        raw = json.dumps({
            "speech": "こんにちは",
            "thought": "嬉しそうだな",
            "action": "微笑む",
        }, ensure_ascii=False)

        result = parse_structured_response(raw)

        assert isinstance(result, StructuredResponse)
        assert result.speech == "こんにちは"
        assert result.thought == "嬉しそうだな"
        assert result.action == "微笑む"

    def test_valid_json_speech_only(self) -> None:
        """speech のみの JSON がパースされる."""
        raw = json.dumps({
            "speech": "ふふ、頑張ってるね",
        }, ensure_ascii=False)

        result = parse_structured_response(raw)

        assert result.speech == "ふふ、頑張ってるね"
        assert result.thought is None
        assert result.action is None

    def test_speech_null_with_action(self) -> None:
        """speech=null でも action がパースされる."""
        raw = json.dumps({
            "speech": None,
            "thought": "考え中…",
            "action": "首をかしげる",
        }, ensure_ascii=False)

        result = parse_structured_response(raw)

        assert result.speech is None
        assert result.thought == "考え中…"
        assert result.action == "首をかしげる"

    def test_plain_text_fallback(self) -> None:
        """JSON でないプレーンテキストは speech にフォールバックされる."""
        raw = "ふふ、元気だよ！"

        result = parse_structured_response(raw)

        assert result.speech == "ふふ、元気だよ！"
        assert result.thought is None
        assert result.action is None

    def test_markdown_code_block_wrapping(self) -> None:
        """マークダウンコードブロックで囲まれた JSON がパースされる."""
        inner = json.dumps({
            "speech": "了解だよ",
            "thought": "なるほど",
            "action": "うなずく",
        }, ensure_ascii=False)
        raw = f"```json\n{inner}\n```"

        result = parse_structured_response(raw)

        assert result.speech == "了解だよ"
        assert result.thought == "なるほど"
        assert result.action == "うなずく"

    def test_invalid_json_fallback(self) -> None:
        """不正な JSON はプレーンテキストとしてフォールバックされる."""
        raw = '{"speech": "incomplete'

        result = parse_structured_response(raw)

        assert result.speech == '{"speech": "incomplete'
        assert result.thought is None
        assert result.action is None

    def test_json_without_structured_keys_fallback(self) -> None:
        """構造化キーがない JSON はフォールバックされる."""
        raw = json.dumps({"message": "hello", "status": "ok"})

        result = parse_structured_response(raw)

        assert result.speech == raw
        assert result.thought is None
        assert result.action is None

    def test_empty_string_fallback(self) -> None:
        """空文字列はフォールバックされる."""
        raw = ""

        result = parse_structured_response(raw)

        assert result.speech == ""
        assert result.thought is None
        assert result.action is None

    def test_non_dict_json_fallback(self) -> None:
        """配列の JSON はフォールバックされる."""
        raw = json.dumps(["hello", "world"])

        result = parse_structured_response(raw)

        assert result.speech == raw
        assert result.thought is None
        assert result.action is None

    def test_thought_action_optional_missing_is_none(self) -> None:
        """thought/action が JSON に存在しない場合は None."""
        raw = json.dumps({
            "speech": "おはよう",
        }, ensure_ascii=False)

        result = parse_structured_response(raw)

        assert result.speech == "おはよう"
        assert result.thought is None
        assert result.action is None


class TestStructuredResponse:
    """StructuredResponse データモデルの検証."""

    def test_creation_with_all_fields(self) -> None:
        """全フィールド指定で作成できる."""
        sr = StructuredResponse(
            speech="hello",
            thought="thinking",
            action="smiling",
        )
        assert sr.speech == "hello"
        assert sr.thought == "thinking"
        assert sr.action == "smiling"

    def test_defaults(self) -> None:
        """デフォルトで全フィールド None."""
        sr = StructuredResponse()
        assert sr.speech is None
        assert sr.thought is None
        assert sr.action is None

    def test_frozen(self) -> None:
        """frozen=True で変更不可."""
        sr = StructuredResponse(speech="test")
        with pytest.raises(AttributeError):
            sr.speech = "changed"  # type: ignore[misc]
