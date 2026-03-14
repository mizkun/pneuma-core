"""Task データモデル (#132)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Task:
    """タスク."""

    id: str
    project_id: str
    title: str
    content: str = ""
    status: int = 0  # 0=Normal, 2=Completed（TickTick 準拠）
    priority: int = 0  # 0=None, 1=Low, 3=Medium, 5=High
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)
    repeat: str | None = None
    created_date: datetime | None = None
    completed_date: datetime | None = None


@dataclass
class TaskCreate:
    """タスク作成リクエスト."""

    title: str
    content: str = ""
    priority: int = 0
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TaskUpdate:
    """タスク更新リクエスト（None のフィールドは更新しない）."""

    title: str | None = None
    content: str | None = None
    priority: int | None = None
    due_date: date | None = None
    tags: list[str] | None = None
