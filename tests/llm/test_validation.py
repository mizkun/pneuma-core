"""Tests for ResponseValidator (Issue #12 / AC-2).

invariant: llm-response-validated-strict — 全構造化 LLM レスポンスは Pydantic
モデルで strict parse される。失敗時は StructuredResponseError。
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from pneuma_core.llm.validation import ResponseValidator, StructuredResponseError


class _EmotionPayload(BaseModel):
    """Pydantic モデル — テスト用."""

    pleasure: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=-1.0, le=1.0)
    dominance: float = Field(ge=-1.0, le=1.0)
    emotion_label: str
    situation: str


class TestResponseValidator:
    def test_valid_payload_returns_model(self) -> None:
        """整合した dict は Pydantic モデルに parse される."""
        validator = ResponseValidator(_EmotionPayload)
        result = validator.validate({
            "pleasure": 0.5,
            "arousal": 0.3,
            "dominance": 0.1,
            "emotion_label": "happy",
            "situation": "chat",
        })

        assert isinstance(result, _EmotionPayload)
        assert result.pleasure == pytest.approx(0.5)
        assert result.emotion_label == "happy"

    def test_missing_required_field_raises(self) -> None:
        """必須フィールド欠落で StructuredResponseError."""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate({
                "pleasure": 0.5,
                "arousal": 0.3,
                # dominance missing
                "emotion_label": "happy",
                "situation": "chat",
            })

    def test_type_mismatch_raises(self) -> None:
        """型違反で StructuredResponseError."""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate({
                "pleasure": "high",  # not a number
                "arousal": 0.3,
                "dominance": 0.0,
                "emotion_label": "happy",
                "situation": "chat",
            })

    def test_out_of_range_raises(self) -> None:
        """Field 制約 (ge/le) 違反で StructuredResponseError."""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate({
                "pleasure": 5.0,  # > 1.0
                "arousal": 0.3,
                "dominance": 0.0,
                "emotion_label": "happy",
                "situation": "chat",
            })

    def test_non_dict_raises(self) -> None:
        """dict 以外を渡したら StructuredResponseError."""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate("not a dict")  # type: ignore[arg-type]

    def test_empty_dict_raises(self) -> None:
        """空 dict は欠落として StructuredResponseError."""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate({})

    def test_error_message_includes_field_info(self) -> None:
        """エラーメッセージにフィールド情報が含まれる."""
        validator = ResponseValidator(_EmotionPayload)
        try:
            validator.validate({
                "pleasure": 0.5,
                "arousal": 0.3,
                "dominance": 0.0,
                "emotion_label": "happy",
                # situation missing
            })
        except StructuredResponseError as e:
            assert "situation" in str(e).lower() or "missing" in str(e).lower()
            return
        raise AssertionError("StructuredResponseError was not raised")

    def test_strict_mode_rejects_string_to_float_coercion(self) -> None:
        """invariant: llm-response-validated-strict — strict=True で
        文字列→数値の暗黙 coercion が拒絶される。lax mode だと "0.5" は 0.5 にコース
        されて pass してしまうが、strict mode では型違反として reject されるべき。"""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate({
                "pleasure": "0.5",  # 文字列、lax だと coerce される
                "arousal": 0.3,
                "dominance": 0.0,
                "emotion_label": "happy",
                "situation": "chat",
            })

    def test_strict_mode_rejects_bool_to_float_coercion(self) -> None:
        """invariant: llm-response-validated-strict — strict=True で
        bool→float の暗黙 coercion も拒絶される。"""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate({
                "pleasure": True,  # bool、lax だと 1.0 にコース
                "arousal": 0.3,
                "dominance": 0.0,
                "emotion_label": "happy",
                "situation": "chat",
            })

    def test_strict_mode_rejects_int_string_to_float(self) -> None:
        """invariant: llm-response-validated-strict — strict=True で
        整数文字列→数値の暗黙 coercion も拒絶される。"""
        validator = ResponseValidator(_EmotionPayload)
        with pytest.raises(StructuredResponseError):
            validator.validate({
                "pleasure": 0.5,
                "arousal": "1",  # 整数文字列、lax だと 1.0 にコース
                "dominance": 0.0,
                "emotion_label": "happy",
                "situation": "chat",
            })
