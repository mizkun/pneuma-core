"""TodoItem: TODO + 習慣管理のデータモデル."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class TodoItem:
    """TODO アイテム."""

    id: str                          # UUID
    content: str                     # タスク内容
    label: str                       # habit | deadline | this_week | someday
    kind: str                        # must | want
    status: str                      # pending | done | skipped
    priority: int                    # 1(高) 〜 3(低)
    due_date: date | None            # 締切日
    recurrence: str | None           # daily | weekly | weekdays | None
    created_at: datetime
    completed_at: datetime | None = None
    owner_id: str = "user"           # ユーザーまたはキャラクター ID
