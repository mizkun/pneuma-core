"""Tests for EmotionLabel enum and pad_to_label mapping (Issue #9)."""

from __future__ import annotations

import pytest

from pneuma_core.models.emotion import EmotionalState, EmotionLabel
from pneuma_core.runtime.emotion_engine import pad_to_label


class TestEmotionLabel:
    """AC-1: EmotionLabel enum has 6 fixed values (no angry)."""

    def test_has_six_values(self) -> None:
        values = {e.value for e in EmotionLabel}
        assert values == {
            "neutral", "happy", "teasing",
            "surprised", "embarrassed", "sad_lite",
        }

    def test_no_angry(self) -> None:
        assert "angry" not in {e.value for e in EmotionLabel}

    def test_string_value(self) -> None:
        assert EmotionLabel.NEUTRAL.value == "neutral"
        assert EmotionLabel.HAPPY.value == "happy"
        assert EmotionLabel.TEASING.value == "teasing"
        assert EmotionLabel.SURPRISED.value == "surprised"
        assert EmotionLabel.EMBARRASSED.value == "embarrassed"
        assert EmotionLabel.SAD_LITE.value == "sad_lite"


class TestPadToLabel:
    """AC-2/3: pad_to_label maps PAD state to one of 6 labels."""

    @staticmethod
    def _state(p: float, a: float, d: float) -> EmotionalState:
        return EmotionalState(
            pleasure=p, arousal=a, dominance=d,
            emotion_label="neutral", situation="",
        )

    # surprised: A > 0.7 (highest priority)
    def test_high_arousal_is_surprised(self) -> None:
        # high P, high A → surprised wins over happy
        assert pad_to_label(self._state(0.5, 0.9, 0.0)) == EmotionLabel.SURPRISED

    def test_high_arousal_negative_pleasure_is_surprised(self) -> None:
        assert pad_to_label(self._state(-0.5, 0.9, 0.0)) == EmotionLabel.SURPRISED

    def test_high_arousal_boundary(self) -> None:
        # exactly 0.7 not yet surprised
        assert pad_to_label(self._state(0.5, 0.7, 0.0)) == EmotionLabel.HAPPY
        # 0.71 → surprised
        assert pad_to_label(self._state(0.5, 0.71, 0.0)) == EmotionLabel.SURPRISED

    # happy: P > 0.3, A > 0.3 (and not surprised)
    def test_happy_high_p_high_a(self) -> None:
        assert pad_to_label(self._state(0.5, 0.5, 0.0)) == EmotionLabel.HAPPY

    def test_happy_boundary(self) -> None:
        # P=0.31, A=0.31 → happy
        assert pad_to_label(self._state(0.31, 0.31, 0.0)) == EmotionLabel.HAPPY

    # teasing: P > 0.3, A in [-0.1, 0.3]
    def test_teasing_positive_p_low_a(self) -> None:
        assert pad_to_label(self._state(0.5, 0.1, 0.0)) == EmotionLabel.TEASING

    def test_teasing_boundary_a(self) -> None:
        # A = 0.3 → teasing (since happy needs A > 0.3)
        assert pad_to_label(self._state(0.5, 0.3, 0.0)) == EmotionLabel.TEASING
        # A = -0.1 → teasing
        assert pad_to_label(self._state(0.5, -0.1, 0.0)) == EmotionLabel.TEASING

    # embarrassed: 0 < P < 0.3, D > 0.2
    def test_embarrassed_low_positive_p_high_d(self) -> None:
        assert pad_to_label(self._state(0.2, 0.0, 0.5)) == EmotionLabel.EMBARRASSED

    def test_embarrassed_d_boundary(self) -> None:
        # D = 0.2 not yet embarrassed → neutral
        assert pad_to_label(self._state(0.2, 0.0, 0.2)) == EmotionLabel.NEUTRAL
        # D = 0.21 → embarrassed
        assert pad_to_label(self._state(0.2, 0.0, 0.21)) == EmotionLabel.EMBARRASSED

    # sad_lite: P < -0.1
    def test_sad_lite(self) -> None:
        assert pad_to_label(self._state(-0.5, 0.0, 0.0)) == EmotionLabel.SAD_LITE

    def test_sad_lite_boundary(self) -> None:
        assert pad_to_label(self._state(-0.1, 0.0, 0.0)) == EmotionLabel.NEUTRAL
        assert pad_to_label(self._state(-0.11, 0.0, 0.0)) == EmotionLabel.SAD_LITE

    # neutral fallback
    def test_neutral(self) -> None:
        assert pad_to_label(self._state(0.0, 0.0, 0.0)) == EmotionLabel.NEUTRAL

    def test_neutral_low_positive_p_low_d(self) -> None:
        # low positive P but low D → neutral
        assert pad_to_label(self._state(0.1, 0.0, 0.0)) == EmotionLabel.NEUTRAL

    # priority chain: surprised > happy > teasing > embarrassed > sad_lite > neutral
    def test_surprised_beats_happy(self) -> None:
        # P > 0.3, A > 0.3, A > 0.7 → surprised
        assert pad_to_label(self._state(0.5, 0.8, 0.0)) == EmotionLabel.SURPRISED

    def test_sad_lite_beats_embarrassed(self) -> None:
        # P < -0.1 always wins over embarrassed (P>0 path)
        assert pad_to_label(self._state(-0.2, 0.0, 0.5)) == EmotionLabel.SAD_LITE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
