"""Diagnostic information for debug/tuning mode."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DiagnosticInfo:
    """Diagnostic data collected during message processing."""

    personality: dict
    emotion: dict
    emotion_trigger: dict
    memories_retrieved: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    model: str = ""
