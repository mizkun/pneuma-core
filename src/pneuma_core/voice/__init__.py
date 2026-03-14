"""Voice module: TTS / STT protocol definitions."""

from pneuma_core.voice.stt import STTService
from pneuma_core.voice.tts import TTSAdapter

__all__ = [
    "STTService",
    "TTSAdapter",
]
