"""PAD 3D emotion model: EmotionalState, Mood, EmotionLabel."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EmotionLabel(str, Enum):
    """AITuber 用 6 種の感情ラベル (Issue #9 / Phase 0 E).

    `angry` は採用しない（女子高生雑談での過剰演技回避）。

    TTS / 立ち絵切り替え / ダッシュボード描画の単一ソース・オブ・トゥルース。
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    TEASING = "teasing"
    SURPRISED = "surprised"
    EMBARRASSED = "embarrassed"
    SAD_LITE = "sad_lite"


def _validate_pad(value: float, name: str) -> float:
    if not (-1.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between -1.0 and 1.0, got {value}")
    return value


@dataclass(frozen=True)
class EmotionalState:
    """PAD 3次元感情状態.

    pleasure: -1.0〜1.0（不快〜快）
    arousal: -1.0〜1.0（沈静〜興奮）
    dominance: -1.0〜1.0（服従〜支配）
    emotion_label: 離散ラベル（表示用）
    situation: 現在の状況（1文）
    """

    pleasure: float
    arousal: float
    dominance: float
    emotion_label: str
    situation: str

    def __post_init__(self) -> None:
        _validate_pad(self.pleasure, "pleasure")
        _validate_pad(self.arousal, "arousal")
        _validate_pad(self.dominance, "dominance")


@dataclass(frozen=True)
class Mood:
    """中期感情（ムード）: 感情の移動平均.

    PAD と同じ -1.0〜1.0 の範囲。
    """

    pleasure: float
    arousal: float
    dominance: float

    def __post_init__(self) -> None:
        _validate_pad(self.pleasure, "pleasure")
        _validate_pad(self.arousal, "arousal")
        _validate_pad(self.dominance, "dominance")

    def update(self, state: EmotionalState, alpha: float = 0.3) -> Mood:
        """感情を反映した新しい Mood を返す（指数移動平均）.

        new_value = (1 - alpha) * current + alpha * emotion
        """
        return Mood(
            pleasure=(1 - alpha) * self.pleasure + alpha * state.pleasure,
            arousal=(1 - alpha) * self.arousal + alpha * state.arousal,
            dominance=(1 - alpha) * self.dominance + alpha * state.dominance,
        )
