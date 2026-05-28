"""ResponseValidator — strict Pydantic validation for structured LLM responses.

invariant: llm-response-validated-strict — 全構造化 LLM レスポンスは Pydantic
モデルで strict parse される。失敗時は StructuredResponseError。

Issue #12.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredResponseError(Exception):
    """構造化 LLM レスポンスの境界違反.

    Cases:
        - Tool Use レスポンスが取得できなかった
        - Pydantic strict parse に失敗した
        - 渡された値が dict でなかった
    """


class ResponseValidator(Generic[T]):
    """Pydantic モデルで構造化 LLM レスポンスを strict parse する.

    Example:
        validator = ResponseValidator(MyModel)
        parsed: MyModel = validator.validate(some_dict)
    """

    def __init__(self, model_cls: type[T]) -> None:
        self._model_cls = model_cls

    def validate(self, data: object) -> T:
        """data を model_cls に strict mode で parse する.

        Pydantic v2 のデフォルトは lax mode で型 coercion を行う（"0.5" → 0.5）。
        invariant: llm-response-validated-strict に従い、strict=True で
        暗黙の型 coercion を禁止する。

        Args:
            data: LLM が返した値（dict を期待）.

        Returns:
            検証済みの Pydantic モデルインスタンス.

        Raises:
            StructuredResponseError: dict でない / 必須欠落 / 型違反 / 制約違反 /
                文字列→数値などの暗黙 coercion.
        """
        if not isinstance(data, dict):
            raise StructuredResponseError(
                f"expected dict for {self._model_cls.__name__}, got {type(data).__name__}"
            )

        try:
            return self._model_cls.model_validate(data, strict=True)
        except ValidationError as e:
            raise StructuredResponseError(
                f"validation failed for {self._model_cls.__name__}: {e}"
            ) from e
