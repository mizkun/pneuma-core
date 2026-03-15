"""Tests for Issue #5: GoalTree（Vision → Objective → Task）.

受け入れ基準:
- 3階層の目標構造が正しく表現できる
- ステータス遷移が適切にバリデーションされる
- 全テストがパスする
"""

import pytest

from pneuma_core.models.goals import GoalTree, Objective, Task, Vision


# =============================================
# Vision Tests
# =============================================


class TestVision:
    """長期ビジョンのテスト."""

    def test_create_vision(self):
        """Vision を正しく生成できる."""
        v = Vision(
            id="vis-001",
            character_id="aine-001",
            content="人間と深い信頼関係を築く",
        )
        assert v.id == "vis-001"
        assert v.character_id == "aine-001"
        assert v.content == "人間と深い信頼関係を築く"


# =============================================
# Objective Tests
# =============================================


class TestObjective:
    """中期目標のテスト."""

    def test_create_objective(self):
        """Objective を正しく生成できる."""
        obj = Objective(
            id="obj-001",
            character_id="aine-001",
            vision_id="vis-001",
            content="ユーザーの趣味を理解する",
            status="active",
            progress=0.0,
        )
        assert obj.id == "obj-001"
        assert obj.status == "active"
        assert obj.progress == 0.0

    def test_valid_statuses(self):
        """有効なステータスで生成できる."""
        for status in ("active", "achieved", "abandoned"):
            obj = Objective(
                id="obj-001",
                character_id="aine-001",
                vision_id="vis-001",
                content="テスト",
                status=status,
                progress=0.5,
            )
            assert obj.status == status

    def test_reject_invalid_status(self):
        """無効なステータスでエラー."""
        with pytest.raises(ValueError):
            Objective(
                id="obj-001",
                character_id="aine-001",
                vision_id="vis-001",
                content="テスト",
                status="invalid",
                progress=0.0,
            )

    def test_reject_progress_out_of_range(self):
        """progress が 0.0〜1.0 の範囲外でエラー."""
        with pytest.raises(ValueError):
            Objective(
                id="obj-001",
                character_id="aine-001",
                vision_id="vis-001",
                content="テスト",
                status="active",
                progress=1.5,
            )

    def test_reject_negative_progress(self):
        """progress が負の値でエラー."""
        with pytest.raises(ValueError):
            Objective(
                id="obj-001",
                character_id="aine-001",
                vision_id="vis-001",
                content="テスト",
                status="active",
                progress=-0.1,
            )

    def test_progress_boundary_values(self):
        """progress の境界値 (0.0, 1.0) で生成できる."""
        obj = Objective(
            id="obj-001",
            character_id="aine-001",
            vision_id="vis-001",
            content="テスト",
            status="active",
            progress=1.0,
        )
        assert obj.progress == 1.0


# =============================================
# Task Tests
# =============================================


class TestTask:
    """具体的タスクのテスト."""

    def test_create_task(self):
        """Task を正しく生成できる."""
        task = Task(
            id="task-001",
            character_id="aine-001",
            objective_id="obj-001",
            content="映画の好みを聞く",
            status="pending",
        )
        assert task.id == "task-001"
        assert task.status == "pending"

    def test_valid_statuses(self):
        """有効なステータスで生成できる."""
        for status in ("pending", "in_progress", "completed", "abandoned"):
            task = Task(
                id="task-001",
                character_id="aine-001",
                objective_id="obj-001",
                content="テスト",
                status=status,
            )
            assert task.status == status

    def test_reject_invalid_status(self):
        """無効なステータスでエラー."""
        with pytest.raises(ValueError):
            Task(
                id="task-001",
                character_id="aine-001",
                objective_id="obj-001",
                content="テスト",
                status="done",
            )


# =============================================
# GoalTree Tests
# =============================================


class TestGoalTree:
    """階層目標構造のテスト."""

    def _make_tree(self) -> GoalTree:
        """テスト用の GoalTree を作成."""
        visions = [
            Vision(id="vis-001", character_id="aine-001", content="人間と深い信頼関係を築く"),
        ]
        objectives = [
            Objective(
                id="obj-001",
                character_id="aine-001",
                vision_id="vis-001",
                content="ユーザーの趣味を理解する",
                status="active",
                progress=0.3,
            ),
            Objective(
                id="obj-002",
                character_id="aine-001",
                vision_id="vis-001",
                content="共感力を高める",
                status="active",
                progress=0.0,
            ),
        ]
        tasks = [
            Task(
                id="task-001",
                character_id="aine-001",
                objective_id="obj-001",
                content="映画の好みを聞く",
                status="completed",
            ),
            Task(
                id="task-002",
                character_id="aine-001",
                objective_id="obj-001",
                content="音楽の好みを聞く",
                status="pending",
            ),
            Task(
                id="task-003",
                character_id="aine-001",
                objective_id="obj-002",
                content="感情表現のバリエーションを増やす",
                status="in_progress",
            ),
        ]
        return GoalTree(visions=visions, objectives=objectives, tasks=tasks)

    def test_create_goal_tree(self):
        """GoalTree を正しく生成できる."""
        tree = self._make_tree()
        assert len(tree.visions) == 1
        assert len(tree.objectives) == 2
        assert len(tree.tasks) == 3

    def test_get_objectives_for_vision(self):
        """Vision に紐づく Objectives を取得できる."""
        tree = self._make_tree()
        objs = tree.get_objectives_for_vision("vis-001")
        assert len(objs) == 2

    def test_get_tasks_for_objective(self):
        """Objective に紐づく Tasks を取得できる."""
        tree = self._make_tree()
        tasks = tree.get_tasks_for_objective("obj-001")
        assert len(tasks) == 2
        tasks2 = tree.get_tasks_for_objective("obj-002")
        assert len(tasks2) == 1

    def test_get_objectives_for_nonexistent_vision(self):
        """存在しない Vision の Objectives は空リスト."""
        tree = self._make_tree()
        objs = tree.get_objectives_for_vision("vis-999")
        assert objs == []

    def test_get_tasks_for_nonexistent_objective(self):
        """存在しない Objective の Tasks は空リスト."""
        tree = self._make_tree()
        tasks = tree.get_tasks_for_objective("obj-999")
        assert tasks == []
