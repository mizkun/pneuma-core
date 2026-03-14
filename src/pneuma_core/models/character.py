"""Character identity model."""

from dataclasses import dataclass

from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values


@dataclass(frozen=True)
class Character:
    """キャラクターの Identity.

    不変属性（性格・価値観）と自由記述（プロフィール・外見・口調等）を持つ。
    personality_description / values_description は YAML 直書きまたは LLM 生成（Phase 1.5）。
    """

    id: str
    name: str
    personality: Personality
    values: Values
    profile: str | None = None
    appearance: str | None = None
    speaking_style: str | None = None
    background: str | None = None
    personality_description: str | None = None
    values_description: str | None = None
