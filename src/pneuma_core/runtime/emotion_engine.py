"""EmotionEngine: LLM-based PAD emotion estimation and lifecycle management."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pneuma_core.emotion.baseline import personality_to_pad_baseline
from pneuma_core.emotion.decay import exponential_decay
from pneuma_core.emotion.pad_mapping import pad_to_emotion_label
from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.models.emotion import EmotionalState, EmotionLabel
from pneuma_core.models.personality import Personality


def pad_to_label(state: EmotionalState) -> EmotionLabel:
    """PAD 値から AITuber 6 種感情ラベルにマップ (Issue #9).

    優先順位（上から先に判定）:
        1. surprised : A > 0.7（極端に高い覚醒）
        2. sad_lite  : P < -0.1（不快）
        3. happy     : P > 0.3 かつ A > 0.3
        4. teasing   : P > 0.3 かつ -0.1 <= A <= 0.3
        5. embarrassed : 0 < P < 0.3 かつ D > 0.2
        6. neutral   : 上記以外
    """
    p, a, d = state.pleasure, state.arousal, state.dominance

    # 1. surprised (highest arousal wins)
    if a > 0.7:
        return EmotionLabel.SURPRISED

    # 2. sad_lite (negative pleasure)
    if p < -0.1:
        return EmotionLabel.SAD_LITE

    # 3. happy (high positive pleasure + active)
    if p > 0.3 and a > 0.3:
        return EmotionLabel.HAPPY

    # 4. teasing (positive pleasure + calm)
    if p > 0.3 and -0.1 <= a <= 0.3:
        return EmotionLabel.TEASING

    # 5. embarrassed (low positive pleasure + high dominance)
    if 0 < p < 0.3 and d > 0.2:
        return EmotionLabel.EMBARRASSED

    return EmotionLabel.NEUTRAL

logger = logging.getLogger(__name__)

_MD_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)

_SYSTEM_PROMPT_BASE = """\
あなたは会話の感情分析エキスパートです。
以下の会話履歴を読み、キャラクターの現在の感情状態を PAD モデルで推定してください。

## キャラクターの性格特性（Big Five）
- 開放性 (Openness): {openness}
- 誠実性 (Conscientiousness): {conscientiousness}
- 外向性 (Extraversion): {extraversion}
- 協調性 (Agreeableness): {agreeableness}
- 神経症傾向 (Neuroticism): {neuroticism}

キャラクターの性格特性を考慮して、感情状態を推定してください。

以下の JSON 形式で回答してください:
{{
  "pleasure": <-1.0〜1.0>,
  "arousal": <-1.0〜1.0>,
  "dominance": <-1.0〜1.0>,
  "emotion_label": "<感情ラベル>",
  "situation": "<現在の状況を1文で>"
}}
"""


def _build_system_prompt(personality: Personality) -> str:
    """Build system prompt with personality information."""
    return _SYSTEM_PROMPT_BASE.format(
        openness=personality.openness,
        conscientiousness=personality.conscientiousness,
        extraversion=personality.extraversion,
        agreeableness=personality.agreeableness,
        neuroticism=personality.neuroticism,
    )

NEUTRAL_EMOTION = EmotionalState(
    pleasure=0.0, arousal=0.0, dominance=0.0,
    emotion_label="中立", situation="",
)


_MAX_LABEL_LEN = 50
_MAX_SITUATION_LEN = 200


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sanitize_text(text: str, max_len: int) -> str:
    """Remove control characters and truncate."""
    cleaned = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return cleaned[:max_len]


@dataclass(frozen=True)
class EmotionConfig:
    """Configuration for EmotionEngine."""

    recent_messages_limit: int = 10
    decay_half_life: float = 3600.0


@dataclass(frozen=True)
class EmotionResult:
    """Result of emotion evaluation with trigger information."""

    state: EmotionalState
    trigger_type: str  # "triggered"
    reasons: list[str] = field(default_factory=list)


class EmotionEngine:
    """LLM-based emotion estimation with baseline decay."""

    def __init__(
        self,
        llm: LLMAdapter,
        config: EmotionConfig | None = None,
        model: str | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or EmotionConfig()
        self._model = model

    async def evaluate(
        self,
        personality: Personality,
        messages: list[dict],
        turn_count: int,
        current_state: EmotionalState,
    ) -> EmotionResult:
        """Evaluate emotion by direct LLM estimation every turn.

        Every turn calls estimate() once to get PAD values directly.
        No Tier 0/1 gating -- simplified from the hybrid trigger system.

        Returns:
            EmotionResult with state and trigger_type="triggered".
        """
        state = await self.estimate(personality, messages)
        return EmotionResult(
            state=state,
            trigger_type="triggered",
        )

    async def estimate(
        self,
        personality: Personality,
        messages: list[dict],
    ) -> EmotionalState:
        """Estimate emotional state from conversation via LLM.

        Returns neutral state on any error (malformed JSON, missing fields, LLM exception).
        """
        truncated = messages[-self._config.recent_messages_limit :]

        request = LLMRequest(
            system_prompt=_build_system_prompt(personality),
            messages=truncated,
            model=self._model,
        )

        try:
            response = await self._llm.generate(request)
        except Exception as e:
            logger.warning("LLM generate failed: %s: %s, returning neutral state", type(e).__name__, e)
            return NEUTRAL_EMOTION

        try:
            text = response.content.strip()
            md_match = _MD_CODE_BLOCK_RE.match(text)
            if md_match:
                text = md_match.group(1).strip()
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "Malformed LLM response (%s), raw=%r, returning neutral state",
                e, response.content[:200] if response.content else "<empty>",
            )
            return NEUTRAL_EMOTION

        if not isinstance(data, dict):
            return NEUTRAL_EMOTION

        required = ("pleasure", "arousal", "dominance", "emotion_label", "situation")
        if not all(k in data for k in required):
            return NEUTRAL_EMOTION

        try:
            pleasure = _clamp(float(data["pleasure"]))
            arousal = _clamp(float(data["arousal"]))
            dominance = _clamp(float(data["dominance"]))
        except (ValueError, TypeError):
            return NEUTRAL_EMOTION

        return EmotionalState(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            emotion_label=_sanitize_text(str(data["emotion_label"]), _MAX_LABEL_LEN),
            situation=_sanitize_text(str(data["situation"]), _MAX_SITUATION_LEN),
        )

    def decay_towards_baseline(
        self,
        state: EmotionalState,
        personality: Personality,
        elapsed_seconds: float,
    ) -> EmotionalState:
        """Decay emotional state towards personality baseline."""
        baseline = personality_to_pad_baseline(personality)

        pleasure = exponential_decay(
            state.pleasure, baseline[0], elapsed_seconds, self._config.decay_half_life,
        )
        arousal = exponential_decay(
            state.arousal, baseline[1], elapsed_seconds, self._config.decay_half_life,
        )
        dominance = exponential_decay(
            state.dominance, baseline[2], elapsed_seconds, self._config.decay_half_life,
        )

        label = pad_to_emotion_label(pleasure, arousal, dominance)

        return EmotionalState(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            emotion_label=label,
            situation=state.situation,
        )
