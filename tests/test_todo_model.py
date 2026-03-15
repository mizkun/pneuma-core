"""Tests for TodoItem model (Issue #90): TODO + 習慣管理."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from pneuma_core.models.todo import TodoItem


JST = timezone(timedelta(hours=9))


class TestTodoItemCreation:
    """TodoItem データクラスの生成テスト."""

    def test_create_basic_todo(self) -> None:
        """基本的な TodoItem を作成できる."""
        todo = TodoItem(
            id="todo-001",
            content="牛乳を買う",
            label="someday",
            kind="must",
            status="pending",
            priority=2,
            due_date=None,
            recurrence=None,
            created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
        )
        assert todo.id == "todo-001"
        assert todo.content == "牛乳を買う"
        assert todo.label == "someday"
        assert todo.kind == "must"
        assert todo.status == "pending"
        assert todo.priority == 2
        assert todo.due_date is None
        assert todo.recurrence is None
        assert todo.completed_at is None

    def test_create_habit_todo(self) -> None:
        """habit ラベルの TodoItem を作成できる."""
        todo = TodoItem(
            id="todo-002",
            content="朝のストレッチ",
            label="habit",
            kind="want",
            status="pending",
            priority=1,
            due_date=None,
            recurrence="daily",
            created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
        )
        assert todo.label == "habit"
        assert todo.kind == "want"
        assert todo.recurrence == "daily"

    def test_create_deadline_todo(self) -> None:
        """deadline ラベルの TodoItem を作成できる."""
        todo = TodoItem(
            id="todo-003",
            content="レポート提出",
            label="deadline",
            kind="must",
            status="pending",
            priority=1,
            due_date=date(2026, 3, 1),
            recurrence=None,
            created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
        )
        assert todo.label == "deadline"
        assert todo.due_date == date(2026, 3, 1)

    def test_create_this_week_todo(self) -> None:
        """this_week ラベルの TodoItem を作成できる."""
        todo = TodoItem(
            id="todo-004",
            content="コードレビュー",
            label="this_week",
            kind="must",
            status="pending",
            priority=2,
            due_date=None,
            recurrence=None,
            created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
        )
        assert todo.label == "this_week"

    def test_completed_at_default_none(self) -> None:
        """completed_at のデフォルトは None."""
        todo = TodoItem(
            id="todo-005",
            content="テスト",
            label="someday",
            kind="must",
            status="pending",
            priority=2,
            due_date=None,
            recurrence=None,
            created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
        )
        assert todo.completed_at is None

    def test_completed_at_can_be_set(self) -> None:
        """completed_at を明示的に設定できる."""
        now = datetime(2026, 2, 28, 18, 0, 0, tzinfo=JST)
        todo = TodoItem(
            id="todo-006",
            content="完了済み",
            label="someday",
            kind="must",
            status="done",
            priority=2,
            due_date=None,
            recurrence=None,
            created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
            completed_at=now,
        )
        assert todo.completed_at == now
        assert todo.status == "done"

    def test_all_label_values(self) -> None:
        """各ラベル値が正しく設定できる."""
        for label in ("habit", "deadline", "this_week", "someday"):
            todo = TodoItem(
                id=f"todo-{label}",
                content="テスト",
                label=label,
                kind="must",
                status="pending",
                priority=2,
                due_date=None,
                recurrence=None,
                created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
            )
            assert todo.label == label

    def test_all_kind_values(self) -> None:
        """各 kind 値が正しく設定できる."""
        for kind in ("must", "want"):
            todo = TodoItem(
                id=f"todo-{kind}",
                content="テスト",
                label="someday",
                kind=kind,
                status="pending",
                priority=2,
                due_date=None,
                recurrence=None,
                created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
            )
            assert todo.kind == kind

    def test_all_status_values(self) -> None:
        """各 status 値が正しく設定できる."""
        for status in ("pending", "done", "skipped"):
            todo = TodoItem(
                id=f"todo-{status}",
                content="テスト",
                label="someday",
                kind="must",
                status=status,
                priority=2,
                due_date=None,
                recurrence=None,
                created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
            )
            assert todo.status == status

    def test_all_recurrence_values(self) -> None:
        """各 recurrence 値が正しく設定できる."""
        for rec in ("daily", "weekly", "weekdays", None):
            todo = TodoItem(
                id="todo-rec",
                content="テスト",
                label="habit",
                kind="must",
                status="pending",
                priority=2,
                due_date=None,
                recurrence=rec,
                created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
            )
            assert todo.recurrence == rec

    def test_priority_range(self) -> None:
        """priority は 1〜3 の値を受け付ける."""
        for p in (1, 2, 3):
            todo = TodoItem(
                id=f"todo-p{p}",
                content="テスト",
                label="someday",
                kind="must",
                status="pending",
                priority=p,
                due_date=None,
                recurrence=None,
                created_at=datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST),
            )
            assert todo.priority == p
