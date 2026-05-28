"""Tests for strip_code_fences (Issue #12 / AC-3).

invariant: llm-response-pre-strip — Tool Use を使えないレガシーパス（テスト・
モック・後方互換）でも、マークダウンコードブロック (```json ... ```) や前後の
説明文は前処理で除去してから JSON parse する。
"""

from __future__ import annotations

import json

import pytest

from pneuma_core.runtime.response_parser import strip_code_fences


class TestStripCodeFences:
    def test_pure_json_unchanged(self) -> None:
        """マークダウンも周辺テキストもない JSON は素通り."""
        raw = '{"a": 1, "b": 2}'
        assert strip_code_fences(raw) == raw

    def test_strips_json_fenced_block(self) -> None:
        """```json ... ``` を剥がす."""
        raw = '```json\n{"a": 1}\n```'
        out = strip_code_fences(raw)
        assert json.loads(out) == {"a": 1}

    def test_strips_plain_fenced_block(self) -> None:
        """``` ... ``` （言語指定なし）も剥がす."""
        raw = '```\n{"a": 1}\n```'
        out = strip_code_fences(raw)
        assert json.loads(out) == {"a": 1}

    def test_strips_description_text_around_json(self) -> None:
        """JSON 前後の説明文を除去して中身の {...} を取り出す."""
        raw = (
            "Here is the emotion analysis:\n"
            '{"pleasure": 0.5, "arousal": 0.3}\n'
            "Hope that helps!"
        )
        out = strip_code_fences(raw)
        assert json.loads(out) == {"pleasure": 0.5, "arousal": 0.3}

    def test_strips_fence_and_description(self) -> None:
        """説明文 + コードフェンス両方を剥がす."""
        raw = (
            "キャラクターは喜んでいます。\n"
            "```json\n"
            '{"pleasure": 0.7, "arousal": 0.5}\n'
            "```\n"
            "以上です。"
        )
        out = strip_code_fences(raw)
        assert json.loads(out) == {"pleasure": 0.7, "arousal": 0.5}

    def test_nested_braces_preserved(self) -> None:
        """ネスト中括弧は壊れない."""
        raw = '```json\n{"outer": {"inner": 42}}\n```'
        out = strip_code_fences(raw)
        assert json.loads(out) == {"outer": {"inner": 42}}

    def test_empty_string_returns_empty(self) -> None:
        """空文字列はそのまま空文字列."""
        assert strip_code_fences("") == ""

    def test_only_whitespace_returns_empty(self) -> None:
        """空白のみは空文字列扱い."""
        assert strip_code_fences("   \n  \n").strip() == ""

    def test_no_json_in_text_returns_input(self) -> None:
        """JSON っぽい部分がないテキストはそのまま返す（呼び出し側で fallback）."""
        raw = "This is just plain text without any braces"
        assert strip_code_fences(raw) == raw

    def test_only_open_brace_returns_input(self) -> None:
        """閉じカッコがないと切り出さず原文を返す."""
        raw = "Some text { with no close"
        out = strip_code_fences(raw)
        # 切り出せないなら原文の少なくとも一部が含まれること
        assert "{" in out

    def test_multiline_json_payload(self) -> None:
        """複数行 JSON も正しく抽出."""
        raw = (
            "```json\n"
            "{\n"
            '  "pleasure": 0.7,\n'
            '  "arousal": 0.5,\n'
            '  "emotion_label": "happy"\n'
            "}\n"
            "```\n"
            "\nキャラクターは楽しんでいます。"
        )
        out = strip_code_fences(raw)
        parsed = json.loads(out)
        assert parsed["pleasure"] == pytest.approx(0.7)
        assert parsed["emotion_label"] == "happy"

    def test_handles_full_real_world_failure_case(self) -> None:
        """朝の動作確認で実際に発生したパターン."""
        raw = (
            "```json\n"
            "{\n"
            '  "pleasure": 0.7,\n'
            '  "arousal": 0.5,\n'
            '  "dominance": 0.3,\n'
            '  "emotion_label": "happy",\n'
            '  "situation": "楽しい会話"\n'
            "}\n"
            "```\n"
            "\nキャラクターは元気いっぱい！"
        )
        out = strip_code_fences(raw)
        parsed = json.loads(out)
        assert parsed["pleasure"] == pytest.approx(0.7)
        assert parsed["emotion_label"] == "happy"
