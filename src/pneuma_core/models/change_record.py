"""ChangeRecord: internal state change logging."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChangeRecord:
    """内部状態変化の記録.

    type: "emotion_updated", "memory_added", "goal_updated" など
    before: 変更前の状態（None = 新規追加）
    after: 変更後の状態
    """

    id: str
    character_id: str
    type: str
    before: dict | None
    after: dict
    reason: str
    timestamp: datetime
