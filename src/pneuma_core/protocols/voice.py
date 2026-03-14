"""Voice Protocols: TTSAdapter and STTService."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSAdapter(Protocol):
    """テキストを音声バイナリに変換するアダプターの抽象インターフェース."""

    async def synthesize(
        self, text: str, *, emotion_tags: list[str] | None = None
    ) -> bytes:
        """テキストを音声バイナリに変換する."""
        ...


@runtime_checkable
class STTService(Protocol):
    """音声バイナリをテキストに変換するサービスの抽象インターフェース."""

    async def transcribe(self, audio: bytes, *, language: str = "ja") -> str:
        """音声バイナリをテキストに変換する."""
        ...
