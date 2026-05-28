"""EmotionEngine: LLM-based PAD emotion estimation and lifecycle management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel

from pneuma_core.emotion.baseline import personality_to_pad_baseline
from pneuma_core.emotion.decay import exponential_decay
from pneuma_core.emotion.pad_mapping import pad_to_emotion_label
from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.llm.retry import RetryableLLMCall
from pneuma_core.llm.validation import ResponseValidator, StructuredResponseError
from pneuma_core.models.emotion import EmotionalState, EmotionLabel
from pneuma_core.models.personality import Personality
from pneuma_core.runtime.response_parser import strip_code_fences


class EmotionEvaluation(BaseModel):
    """構造化感情評価レスポンス (Issue #12).

    invariant: structured-output-via-tool-use — schema は Tool Use 経由で強制。
    invariant: llm-response-validated-strict — strict Pydantic 検証。

    Note: pleasure/arousal/dominance は範囲を Field で強制しない。
        LLM がたまに範囲外を返すケースは、_evaluation_to_state() で clamp して
        受容する（既存の degraded-mode 互換）。型 (float) のみ強制。
    """

    pleasure: float
    arousal: float
    dominance: float
    emotion_label: str
    situation: str


# JSON Schema for Tool Use (Anthropic API)
_EMOTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "pleasure": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "快/不快 (-1.0=不快 〜 1.0=快)",
        },
        "arousal": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "覚醒度 (-1.0=沈静 〜 1.0=興奮)",
        },
        "dominance": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "支配感 (-1.0=服従 〜 1.0=支配)",
        },
        "emotion_label": {
            "type": "string",
            "description": "感情ラベル（短い名詞）",
        },
        "situation": {
            "type": "string",
            "description": "現在の状況を1文で記述",
        },
    },
    "required": [
        "pleasure",
        "arousal",
        "dominance",
        "emotion_label",
        "situation",
    ],
}


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


def _evaluation_to_state(ev: EmotionEvaluation) -> EmotionalState:
    """Convert a validated EmotionEvaluation into EmotionalState with sanitisation.

    Pydantic already enforces ranges, but emotion_label / situation may still
    require text sanitisation (control chars / length).
    """
    return EmotionalState(
        pleasure=_clamp(ev.pleasure),
        arousal=_clamp(ev.arousal),
        dominance=_clamp(ev.dominance),
        emotion_label=_sanitize_text(ev.emotion_label, _MAX_LABEL_LEN),
        situation=_sanitize_text(ev.situation, _MAX_SITUATION_LEN),
    )


@dataclass(frozen=True)
class EmotionConfig:
    """Configuration for EmotionEngine.

    Issue #12: retry/backoff for structured-completion failures (Tool Use
    path and generate-fallback path).

    Defaults:
        max_retries=2 keeps the invariant (N=2 retries by default).
        retry_initial_delay=0.0 — emotion failures are usually JSON parse
        issues, not transient network errors, so backoff sleep adds little
        value. Real Claude's Tool Use rarely fails the schema, so this
        retry layer mostly fires on mock paths in tests; 0-delay keeps the
        test suite fast while still satisfying the "retry N times" invariant.
    """

    recent_messages_limit: int = 10
    decay_half_life: float = 3600.0
    max_retries: int = 2
    retry_initial_delay: float = 0.0
    retry_backoff: float = 2.0


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

        Strategy (Issue #12):
            1. If adapter supports structured_completion (real Claude) →
               use Tool Use API with JSON Schema enforcement + retry.
            2. Else (Mock / AsyncMock test path) → use generate() + strip
               code fences + Pydantic strict validate + retry.

        Returns neutral state on any error (malformed JSON, missing fields,
        LLM exception, retries exhausted).

        invariants:
            structured-output-via-tool-use, llm-response-validated-strict,
            llm-response-pre-strip
        """
        truncated = messages[-self._config.recent_messages_limit :]
        system_prompt = _build_system_prompt(personality)

        validator = ResponseValidator(EmotionEvaluation)
        retry = RetryableLLMCall(
            max_retries=self._config.max_retries,
            initial_delay=self._config.retry_initial_delay,
            backoff=self._config.retry_backoff,
        )

        if self._supports_tool_use():
            call = self._make_tool_use_call(
                system_prompt, truncated, validator
            )
        else:
            call = self._make_generate_call(
                system_prompt, truncated, validator
            )

        try:
            evaluation = await retry.run(call)
        except StructuredResponseError as e:
            logger.warning(
                "Emotion estimate failed after retries (%s), returning neutral",
                e,
            )
            return NEUTRAL_EMOTION
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Emotion LLM call failed: %s: %s, returning neutral state",
                type(e).__name__,
                e,
            )
            return NEUTRAL_EMOTION

        return _evaluation_to_state(evaluation)

    def _make_tool_use_call(
        self,
        system_prompt: str,
        messages: list[dict],
        validator: ResponseValidator[EmotionEvaluation],
    ):
        """Build async callable for Tool Use path."""
        async def call() -> EmotionEvaluation:
            payload = await self._llm.structured_completion(  # type: ignore[attr-defined]
                system_prompt=system_prompt,
                messages=messages,
                tool_name="report_emotion",
                tool_description=(
                    "Report the character's current PAD emotional state "
                    "based on the conversation."
                ),
                schema=_EMOTION_SCHEMA,
                model=self._model,
            )
            return validator.validate(payload)
        return call

    def _make_generate_call(
        self,
        system_prompt: str,
        messages: list[dict],
        validator: ResponseValidator[EmotionEvaluation],
    ):
        """Build async callable for generate() fallback path."""
        request = LLMRequest(
            system_prompt=system_prompt,
            messages=messages,
            model=self._model,
        )

        async def call() -> EmotionEvaluation:
            response = await self._llm.generate(request)
            raw = response.content or ""
            stripped = strip_code_fences(raw)
            try:
                data = json.loads(stripped)
            except (json.JSONDecodeError, TypeError) as e:
                raise StructuredResponseError(
                    f"failed to JSON-parse LLM response: {e}; raw={raw[:200]!r}"
                ) from e
            return validator.validate(data)
        return call

    def _supports_tool_use(self) -> bool:
        """Adapter が structured_completion を正式実装しているか判定する.

        AsyncMock / MagicMock は __init__ で属性を自動生成するが
        ``supports_structured_completion`` クラス変数は real adapters のみ
        持つので、これを真値判定の根拠にする.
        """
        marker = getattr(type(self._llm), "supports_structured_completion", None)
        return bool(marker)

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
