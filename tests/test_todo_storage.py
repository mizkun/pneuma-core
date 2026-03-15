"""Tests for TodoItem storage CRUD (Issue #90): InMemory + SQLite."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pneuma_core.models.todo import TodoItem
from pneuma_core.storage.backend import StorageBackend
from pneuma_core.storage.in_memory import InMemoryStorageBackend
from pneuma_core.storage.sqlite import SQLiteStorageBackend


JST = timezone(timedelta(hours=9))


# --- Fixtures ---


def _make_todo(
    id: str = "todo-001",
    content: str = "テストタスク",
    label: str = "someday",
    kind: str = "must",
    status: str = "pending",
    priority: int = 2,
    due_date: date | None = None,
    recurrence: str | None = None,
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> TodoItem:
    if created_at is None:
        created_at = datetime(2026, 2, 28, 9, 0, 0, tzinfo=JST)
    return TodoItem(
        id=id,
        content=content,
        label=label,
        kind=kind,
        status=status,
        priority=priority,
        due_date=due_date,
        recurrence=recurrence,
        created_at=created_at,
        completed_at=completed_at,
    )


@pytest.fixture
def inmemory_store() -> InMemoryStorageBackend:
    return InMemoryStorageBackend()


@pytest.fixture
async def sqlite_store(tmp_path: Path) -> SQLiteStorageBackend:
    db_path = tmp_path / "test_todo.db"
    backend = SQLiteStorageBackend(str(db_path))
    await backend.initialize()
    return backend


# === InMemory Todo CRUD ===


class TestInMemoryTodoCRUD:
    """InMemoryStorageBackend の Todo CRUD テスト."""

    @pytest.mark.asyncio
    async def test_save_and_get_todo(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """TodoItem を保存して取得できる."""
        todo = _make_todo()
        await inmemory_store.save_todo(todo)
        result = await inmemory_store.get_todo("todo-001")
        assert result is not None
        assert result.id == "todo-001"
        assert result.content == "テストタスク"

    @pytest.mark.asyncio
    async def test_get_nonexistent_todo(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """存在しない TodoItem を取得すると None."""
        result = await inmemory_store.get_todo("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_todos_all(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """全 TodoItem を取得できる."""
        await inmemory_store.save_todo(_make_todo(id="todo-001"))
        await inmemory_store.save_todo(_make_todo(id="todo-002", content="タスク2"))
        result = await inmemory_store.list_todos()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_todos_filter_by_status(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """status でフィルタリングできる."""
        await inmemory_store.save_todo(
            _make_todo(id="todo-001", status="pending")
        )
        await inmemory_store.save_todo(
            _make_todo(id="todo-002", status="done")
        )
        await inmemory_store.save_todo(
            _make_todo(id="todo-003", status="pending")
        )
        result = await inmemory_store.list_todos(status="pending")
        assert len(result) == 2
        assert all(t.status == "pending" for t in result)

    @pytest.mark.asyncio
    async def test_list_todos_filter_by_label(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """label でフィルタリングできる."""
        await inmemory_store.save_todo(
            _make_todo(id="todo-001", label="habit")
        )
        await inmemory_store.save_todo(
            _make_todo(id="todo-002", label="deadline")
        )
        await inmemory_store.save_todo(
            _make_todo(id="todo-003", label="habit")
        )
        result = await inmemory_store.list_todos(label="habit")
        assert len(result) == 2
        assert all(t.label == "habit" for t in result)

    @pytest.mark.asyncio
    async def test_list_todos_filter_by_status_and_label(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """status + label の複合フィルタリング."""
        await inmemory_store.save_todo(
            _make_todo(id="todo-001", label="habit", status="pending")
        )
        await inmemory_store.save_todo(
            _make_todo(id="todo-002", label="habit", status="done")
        )
        await inmemory_store.save_todo(
            _make_todo(id="todo-003", label="deadline", status="pending")
        )
        result = await inmemory_store.list_todos(status="pending", label="habit")
        assert len(result) == 1
        assert result[0].id == "todo-001"

    @pytest.mark.asyncio
    async def test_list_todos_empty(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """TodoItem が0件のとき空リスト."""
        result = await inmemory_store.list_todos()
        assert result == []

    @pytest.mark.asyncio
    async def test_update_todo(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """TodoItem を更新できる."""
        todo = _make_todo()
        await inmemory_store.save_todo(todo)
        from dataclasses import replace
        updated = replace(todo, status="done", completed_at=datetime.now(JST))
        await inmemory_store.update_todo(updated)
        result = await inmemory_store.get_todo("todo-001")
        assert result is not None
        assert result.status == "done"
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_delete_todo(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """TodoItem を削除できる."""
        todo = _make_todo()
        await inmemory_store.save_todo(todo)
        await inmemory_store.delete_todo("todo-001")
        result = await inmemory_store.get_todo("todo-001")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_todo(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """存在しない TodoItem の削除はエラーにならない."""
        await inmemory_store.delete_todo("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_protocol_conformance(
        self, inmemory_store: InMemoryStorageBackend
    ) -> None:
        """InMemoryStorageBackend は StorageBackend Protocol を満たす."""
        assert isinstance(inmemory_store, StorageBackend)


# === SQLite Todo CRUD ===


class TestSQLiteTodoCRUD:
    """SQLiteStorageBackend の Todo CRUD テスト."""

    @pytest.mark.asyncio
    async def test_todos_table_exists(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """todos テーブルが存在する."""
        tables = await sqlite_store.list_tables()
        assert "todos" in tables

    @pytest.mark.asyncio
    async def test_save_and_get_todo(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """TodoItem を保存して取得できる."""
        todo = _make_todo()
        await sqlite_store.save_todo(todo)
        result = await sqlite_store.get_todo("todo-001")
        assert result is not None
        assert result.id == "todo-001"
        assert result.content == "テストタスク"
        assert result.label == "someday"
        assert result.kind == "must"
        assert result.status == "pending"
        assert result.priority == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_todo(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """存在しない TodoItem を取得すると None."""
        result = await sqlite_store.get_todo("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_todo_with_due_date(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """due_date 付き TodoItem の保存と復元."""
        todo = _make_todo(
            id="todo-dl",
            label="deadline",
            due_date=date(2026, 3, 15),
        )
        await sqlite_store.save_todo(todo)
        result = await sqlite_store.get_todo("todo-dl")
        assert result is not None
        assert result.due_date == date(2026, 3, 15)

    @pytest.mark.asyncio
    async def test_save_todo_with_recurrence(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """recurrence 付き TodoItem の保存と復元."""
        todo = _make_todo(
            id="todo-hab",
            label="habit",
            recurrence="daily",
        )
        await sqlite_store.save_todo(todo)
        result = await sqlite_store.get_todo("todo-hab")
        assert result is not None
        assert result.recurrence == "daily"

    @pytest.mark.asyncio
    async def test_save_todo_with_completed_at(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """completed_at 付き TodoItem の保存と復元."""
        completed = datetime(2026, 2, 28, 18, 0, 0, tzinfo=JST)
        todo = _make_todo(
            id="todo-done",
            status="done",
            completed_at=completed,
        )
        await sqlite_store.save_todo(todo)
        result = await sqlite_store.get_todo("todo-done")
        assert result is not None
        assert result.completed_at is not None
        assert result.status == "done"

    @pytest.mark.asyncio
    async def test_list_todos_all(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """全 TodoItem を取得できる."""
        await sqlite_store.save_todo(_make_todo(id="todo-001"))
        await sqlite_store.save_todo(
            _make_todo(id="todo-002", content="タスク2")
        )
        result = await sqlite_store.list_todos()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_todos_filter_by_status(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """status でフィルタリングできる."""
        await sqlite_store.save_todo(
            _make_todo(id="todo-001", status="pending")
        )
        await sqlite_store.save_todo(
            _make_todo(id="todo-002", status="done")
        )
        await sqlite_store.save_todo(
            _make_todo(id="todo-003", status="pending")
        )
        result = await sqlite_store.list_todos(status="pending")
        assert len(result) == 2
        assert all(t.status == "pending" for t in result)

    @pytest.mark.asyncio
    async def test_list_todos_filter_by_label(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """label でフィルタリングできる."""
        await sqlite_store.save_todo(
            _make_todo(id="todo-001", label="habit")
        )
        await sqlite_store.save_todo(
            _make_todo(id="todo-002", label="deadline")
        )
        await sqlite_store.save_todo(
            _make_todo(id="todo-003", label="habit")
        )
        result = await sqlite_store.list_todos(label="habit")
        assert len(result) == 2
        assert all(t.label == "habit" for t in result)

    @pytest.mark.asyncio
    async def test_list_todos_filter_by_status_and_label(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """status + label の複合フィルタリング."""
        await sqlite_store.save_todo(
            _make_todo(id="todo-001", label="habit", status="pending")
        )
        await sqlite_store.save_todo(
            _make_todo(id="todo-002", label="habit", status="done")
        )
        await sqlite_store.save_todo(
            _make_todo(id="todo-003", label="deadline", status="pending")
        )
        result = await sqlite_store.list_todos(
            status="pending", label="habit"
        )
        assert len(result) == 1
        assert result[0].id == "todo-001"

    @pytest.mark.asyncio
    async def test_list_todos_empty(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """TodoItem が0件のとき空リスト."""
        result = await sqlite_store.list_todos()
        assert result == []

    @pytest.mark.asyncio
    async def test_update_todo(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """TodoItem を更新できる."""
        todo = _make_todo()
        await sqlite_store.save_todo(todo)
        from dataclasses import replace
        now = datetime(2026, 2, 28, 18, 0, 0, tzinfo=JST)
        updated = replace(todo, status="done", completed_at=now)
        await sqlite_store.update_todo(updated)
        result = await sqlite_store.get_todo("todo-001")
        assert result is not None
        assert result.status == "done"
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_delete_todo(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """TodoItem を削除できる."""
        todo = _make_todo()
        await sqlite_store.save_todo(todo)
        await sqlite_store.delete_todo("todo-001")
        result = await sqlite_store.get_todo("todo-001")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_todo(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """存在しない TodoItem の削除はエラーにならない."""
        await sqlite_store.delete_todo("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_null_due_date_round_trip(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """due_date が None のときの保存・復元."""
        todo = _make_todo(due_date=None)
        await sqlite_store.save_todo(todo)
        result = await sqlite_store.get_todo("todo-001")
        assert result is not None
        assert result.due_date is None

    @pytest.mark.asyncio
    async def test_null_recurrence_round_trip(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """recurrence が None のときの保存・復元."""
        todo = _make_todo(recurrence=None)
        await sqlite_store.save_todo(todo)
        result = await sqlite_store.get_todo("todo-001")
        assert result is not None
        assert result.recurrence is None

    @pytest.mark.asyncio
    async def test_null_completed_at_round_trip(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """completed_at が None のときの保存・復元."""
        todo = _make_todo(completed_at=None)
        await sqlite_store.save_todo(todo)
        result = await sqlite_store.get_todo("todo-001")
        assert result is not None
        assert result.completed_at is None

    @pytest.mark.asyncio
    async def test_protocol_conformance(
        self, sqlite_store: SQLiteStorageBackend
    ) -> None:
        """SQLiteStorageBackend は StorageBackend Protocol を満たす."""
        assert isinstance(sqlite_store, StorageBackend)
