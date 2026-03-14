"""Emotion engine: PAD 3D emotional state estimation."""

from pneuma_core.emotion.baseline import personality_to_pad_baseline
from pneuma_core.emotion.decay import exponential_decay
from pneuma_core.emotion.pad_mapping import pad_to_emotion_label

__all__ = ["exponential_decay", "pad_to_emotion_label", "personality_to_pad_baseline"]
