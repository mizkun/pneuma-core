"""Middleware Protocol and PipelineContext for RuntimeEngine (#145)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.message import MessageInput, MessageOutput


@dataclass
class PipelineContext:
    """Shared state passed through the middleware chain.

    Middleware can read/write metadata to share data with other middleware.
    """

    character: Character
    emotion: EmotionalState
    goals: GoalTree
    memories: list[EpisodicMemory | SemanticMemory]
    system_prompt: str
    history: list[dict]
    turn_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Middleware(Protocol):
    """Protocol for RuntimeEngine middleware.

    Middleware intercepts the message processing pipeline at two points:
    - pre_process: before LLM generation (can modify input)
    - post_process: after LLM generation (can modify output or trigger side effects)
    """

    async def pre_process(
        self, msg: MessageInput, context: PipelineContext
    ) -> MessageInput:
        """Called before LLM generation. Can modify the input message."""
        ...

    async def post_process(
        self, msg: MessageInput, output: MessageOutput, context: PipelineContext
    ) -> MessageOutput:
        """Called after LLM generation. Can modify output or trigger side effects."""
        ...
