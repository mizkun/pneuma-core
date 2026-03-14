"""Custom middleware example for pneuma-core."""

from __future__ import annotations

from pneuma_core.protocols.middleware import Middleware, PipelineContext
from pneuma_core.models.message import MessageInput, MessageOutput


class LoggingMiddleware:
    """Simple logging middleware that prints messages."""

    async def pre_process(
        self, message: MessageInput, context: PipelineContext
    ) -> None:
        print(f"[LOG] Received: {message.content}")

    async def post_process(
        self, message: MessageInput, output: MessageOutput, context: PipelineContext
    ) -> None:
        print(f"[LOG] Response: {output.content}")
        if output.emotion:
            print(f"[LOG] Emotion: {output.emotion.emotion_label}")
