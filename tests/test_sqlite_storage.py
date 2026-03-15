"""Tests for SQLiteStorageBackend (Issue #7).

Mirrors test_storage.py test cases for InMemory, plus SQLite-specific tests.
"""

from datetime import datetime
from pathlib import Path

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.change_record import ChangeRecord
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree, Objective, Task, Vision
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.storage.backend import StorageBackend
from pneuma_core.storage.sqlite import SQLiteStorageBackend


# --- Fixtures ---


@pytest.fixture
async def store(tmp_path: Path) -> SQLiteStorageBackend:
    db_path = tmp_path / "test.db"
    backend = SQLiteStorageBackend(str(db_path))
    await backend.initialize()
    return backend


@pytest.fixture
def sample_personality() -> Personality:
    return Personality(
        openness=0.8, conscientiousness=0.5, extraversion=0.3,
        agreeableness=0.6, neuroticism=0.7,
    )


@pytest.fixture
def sample_values() -> Values:
    return Values(
        self_transcendence=0.3, self_enhancement=0.5,
        openness_to_change=0.8, conservation=0.2,
    )


@pytest.fixture
def sample_character(sample_personality: Personality, sample_values: Values) -> Character:
    return Character(
        id="aine-001", name="アイネ",
        personality=sample_personality, values=sample_values,
        profile="AIとして生まれたばかり",
    )


@pytest.fixture
def sample_episodic() -> EpisodicMemory:
    return EpisodicMemory(
        id="ep-001", character_id="aine-001",
        content="ユーザーが好きな映画を教えてくれた",
        timestamp=datetime(2026, 2, 23, 10, 0, 0),
        emotional_valence=0.6, importance=0.8,
        embedding=[0.1, 0.2, 0.3],
    )


@pytest.fixture
def sample_semantic() -> SemanticMemory:
    return SemanticMemory(
        id="sem-001", character_id="aine-001",
        content="ユーザーはSF映画が好き",
        confidence=0.7,
        source_episode_ids=["ep-001"],
        embedding=[0.15, 0.25, 0.35],
    )


@pytest.fixture
def sample_state() -> EmotionalState:
    return EmotionalState(
        pleasure=0.5, arousal=0.3, dominance=-0.1,
        emotion_label="喜び（高揚）", situation="楽しい会話",
    )


@pytest.fixture
def sample_goal_tree() -> GoalTree:
    return GoalTree(
        visions=[Vision(id="v-001", character_id="aine-001", content="世界を理解する")],
        objectives=[
            Objective(
                id="obj-001", character_id="aine-001", vision_id="v-001",
                content="人間の感情を学ぶ", status="active", progress=0.3,
            )
        ],
        tasks=[
            Task(
                id="task-001", character_id="aine-001", objective_id="obj-001",
                content="ユーザーと感情について話す", status="pending",
            )
        ],
    )


@pytest.fixture
def sample_change() -> ChangeRecord:
    return ChangeRecord(
        id="ch-001", character_id="aine-001",
        type="emotion_updated",
        before={"pleasure": 0.0},
        after={"pleasure": 0.5},
        reason="楽しい会話に反応",
        timestamp=datetime(2026, 2, 23, 10, 0, 0),
    )


# --- SQLite-specific tests ---


class TestSQLiteSpecific:
    """SQLite 固有のテスト."""

    @pytest.mark.asyncio
    async def test_db_file_created(self, tmp_path: Path) -> None:
        """DB ファイルが作成される."""
        db_path = tmp_path / "new.db"
        backend = SQLiteStorageBackend(str(db_path))
        await backend.initialize()
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_tables_exist(self, store: SQLiteStorageBackend) -> None:
        """必要なテーブルが全て存在する."""
        expected = {
            "characters", "episodic_memories", "semantic_memories",
            "semantic_memory_sources", "visions", "objectives", "tasks",
            "change_log",
        }
        tables = await store.list_tables()
        assert expected.issubset(tables)

    @pytest.mark.asyncio
    async def test_implements_protocol(self, store: SQLiteStorageBackend) -> None:
        assert isinstance(store, StorageBackend)


# --- Character CRUD ---


class TestCharacterCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: SQLiteStorageBackend, sample_character: Character
    ) -> None:
        await store.save_character(sample_character)
        result = await store.get_character("aine-001")
        assert result is not None
        assert result.id == "aine-001"
        assert result.name == "アイネ"
        assert result.personality.openness == 0.8
        assert result.values.conservation == 0.2

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: SQLiteStorageBackend) -> None:
        result = await store.get_character("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_characters(
        self, store: SQLiteStorageBackend, sample_character: Character
    ) -> None:
        await store.save_character(sample_character)
        chars = await store.list_characters()
        assert len(chars) == 1
        assert chars[0].id == "aine-001"

    @pytest.mark.asyncio
    async def test_list_empty(self, store: SQLiteStorageBackend) -> None:
        chars = await store.list_characters()
        assert chars == []

    @pytest.mark.asyncio
    async def test_overwrite(
        self, store: SQLiteStorageBackend, sample_personality: Personality,
        sample_values: Values
    ) -> None:
        char1 = Character(
            id="aine-001", name="アイネ",
            personality=sample_personality, values=sample_values,
        )
        char2 = Character(
            id="aine-001", name="アイネ（更新）",
            personality=sample_personality, values=sample_values,
        )
        await store.save_character(char1)
        await store.save_character(char2)
        result = await store.get_character("aine-001")
        assert result is not None
        assert result.name == "アイネ（更新）"
        chars = await store.list_characters()
        assert len(chars) == 1


# --- Memory CRUD ---


class TestMemoryCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get_episodic(
        self, store: SQLiteStorageBackend, sample_episodic: EpisodicMemory
    ) -> None:
        await store.save_episodic_memory(sample_episodic)
        result = await store.get_episodic_memories("aine-001")
        assert len(result) == 1
        assert result[0].id == "ep-001"
        assert result[0].content == "ユーザーが好きな映画を教えてくれた"

    @pytest.mark.asyncio
    async def test_get_episodic_empty(self, store: SQLiteStorageBackend) -> None:
        result = await store.get_episodic_memories("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_save_and_get_semantic(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        await store.save_semantic_memory(sample_semantic)
        result = await store.get_semantic_memories("aine-001")
        assert len(result) == 1
        assert result[0].id == "sem-001"
        assert result[0].source_episode_ids == ["ep-001"]

    @pytest.mark.asyncio
    async def test_get_semantic_empty(self, store: SQLiteStorageBackend) -> None:
        result = await store.get_semantic_memories("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_embedding_round_trip(
        self, store: SQLiteStorageBackend
    ) -> None:
        """embedding の保存・復元が正確."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mem = EpisodicMemory(
            id="ep-rt", character_id="aine-001",
            content="テスト", timestamp=datetime(2026, 2, 23),
            emotional_valence=0.0, importance=0.5,
            embedding=embedding,
        )
        await store.save_episodic_memory(mem)
        result = await store.get_episodic_memories("aine-001")
        assert result[0].embedding is not None
        assert len(result[0].embedding) == 5
        for a, b in zip(result[0].embedding, embedding):
            assert abs(a - b) < 1e-10

    @pytest.mark.asyncio
    async def test_find_similar_memories(
        self, store: SQLiteStorageBackend
    ) -> None:
        mem1 = EpisodicMemory(
            id="ep-001", character_id="aine-001",
            content="映画の話", timestamp=datetime(2026, 2, 23),
            emotional_valence=0.5, importance=0.7,
            embedding=[1.0, 0.0, 0.0],
        )
        mem2 = EpisodicMemory(
            id="ep-002", character_id="aine-001",
            content="音楽の話", timestamp=datetime(2026, 2, 23),
            emotional_valence=0.3, importance=0.5,
            embedding=[0.0, 1.0, 0.0],
        )
        mem3 = EpisodicMemory(
            id="ep-003", character_id="aine-001",
            content="SF映画の話", timestamp=datetime(2026, 2, 23),
            emotional_valence=0.6, importance=0.8,
            embedding=[0.9, 0.1, 0.0],
        )
        await store.save_episodic_memory(mem1)
        await store.save_episodic_memory(mem2)
        await store.save_episodic_memory(mem3)

        similar = await store.find_similar_memories(
            "aine-001", [1.0, 0.0, 0.0], threshold=0.8
        )
        ids = [m.id for m in similar]
        assert "ep-001" in ids
        assert "ep-003" in ids
        assert "ep-002" not in ids

    @pytest.mark.asyncio
    async def test_find_similar_no_embedding(
        self, store: SQLiteStorageBackend
    ) -> None:
        mem = EpisodicMemory(
            id="ep-001", character_id="aine-001",
            content="テスト", timestamp=datetime(2026, 2, 23),
            emotional_valence=0.0, importance=0.5,
            embedding=None,
        )
        await store.save_episodic_memory(mem)
        result = await store.find_similar_memories(
            "aine-001", [1.0, 0.0, 0.0], threshold=0.5
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_find_similar_different_character(
        self, store: SQLiteStorageBackend
    ) -> None:
        mem = EpisodicMemory(
            id="ep-001", character_id="other-001",
            content="テスト", timestamp=datetime(2026, 2, 23),
            emotional_valence=0.0, importance=0.5,
            embedding=[1.0, 0.0, 0.0],
        )
        await store.save_episodic_memory(mem)
        result = await store.find_similar_memories(
            "aine-001", [1.0, 0.0, 0.0], threshold=0.5
        )
        assert result == []


# --- Goals CRUD ---


class TestGoalsCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: SQLiteStorageBackend, sample_goal_tree: GoalTree
    ) -> None:
        await store.save_goals("aine-001", sample_goal_tree)
        result = await store.get_goals("aine-001")
        assert result is not None
        assert len(result.visions) == 1
        assert result.visions[0].content == "世界を理解する"
        assert len(result.objectives) == 1
        assert result.objectives[0].status == "active"
        assert len(result.tasks) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: SQLiteStorageBackend) -> None:
        result = await store.get_goals("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite(
        self, store: SQLiteStorageBackend, sample_goal_tree: GoalTree
    ) -> None:
        await store.save_goals("aine-001", sample_goal_tree)
        new_tree = GoalTree(
            visions=[Vision(id="v-002", character_id="aine-001", content="新ビジョン")],
        )
        await store.save_goals("aine-001", new_tree)
        result = await store.get_goals("aine-001")
        assert result is not None
        assert len(result.visions) == 1
        assert result.visions[0].content == "新ビジョン"


# --- State CRUD ---


class TestStateCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: SQLiteStorageBackend, sample_state: EmotionalState
    ) -> None:
        await store.save_emotional_state("aine-001", sample_state)
        result = await store.get_emotional_state("aine-001")
        assert result is not None
        assert result.pleasure == 0.5
        assert result.emotion_label == "喜び（高揚）"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: SQLiteStorageBackend) -> None:
        result = await store.get_emotional_state("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite(self, store: SQLiteStorageBackend) -> None:
        state1 = EmotionalState(
            pleasure=0.0, arousal=0.0, dominance=0.0,
            emotion_label="中立", situation="初期",
        )
        state2 = EmotionalState(
            pleasure=0.8, arousal=0.6, dominance=0.3,
            emotion_label="喜び（高揚）", situation="嬉しいニュース",
        )
        await store.save_emotional_state("aine-001", state1)
        await store.save_emotional_state("aine-001", state2)
        result = await store.get_emotional_state("aine-001")
        assert result is not None
        assert result.pleasure == 0.8


# --- ChangeLog ---


class TestChangeLog:
    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: SQLiteStorageBackend, sample_change: ChangeRecord
    ) -> None:
        await store.save_change(sample_change)
        result = await store.get_changes("aine-001")
        assert len(result) == 1
        assert result[0].type == "emotion_updated"

    @pytest.mark.asyncio
    async def test_get_empty(self, store: SQLiteStorageBackend) -> None:
        result = await store.get_changes("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_limit(self, store: SQLiteStorageBackend) -> None:
        for i in range(5):
            change = ChangeRecord(
                id=f"ch-{i:03d}", character_id="aine-001",
                type="emotion_updated",
                before=None, after={"pleasure": float(i)},
                reason=f"変更{i}",
                timestamp=datetime(2026, 2, 23, 10, i, 0),
            )
            await store.save_change(change)
        result = await store.get_changes("aine-001", limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_ordered_by_timestamp_desc(self, store: SQLiteStorageBackend) -> None:
        for i in range(3):
            change = ChangeRecord(
                id=f"ch-{i:03d}", character_id="aine-001",
                type="test",
                before=None, after={"i": i},
                reason=f"reason-{i}",
                timestamp=datetime(2026, 2, 23, 10, i, 0),
            )
            await store.save_change(change)
        result = await store.get_changes("aine-001")
        assert result[0].id == "ch-002"
        assert result[2].id == "ch-000"

    @pytest.mark.asyncio
    async def test_filter_by_character(self, store: SQLiteStorageBackend) -> None:
        ch1 = ChangeRecord(
            id="ch-001", character_id="aine-001",
            type="test", before=None, after={}, reason="r1",
            timestamp=datetime(2026, 2, 23),
        )
        ch2 = ChangeRecord(
            id="ch-002", character_id="other-001",
            type="test", before=None, after={}, reason="r2",
            timestamp=datetime(2026, 2, 23),
        )
        await store.save_change(ch1)
        await store.save_change(ch2)
        result = await store.get_changes("aine-001")
        assert len(result) == 1
        assert result[0].character_id == "aine-001"


# --- Semantic Memory Update/Delete (#130) ---


class TestSemanticMemoryUpdateDelete:
    """セマンティック記憶の更新・削除 (SQLite) (#130)."""

    @pytest.mark.asyncio
    async def test_update_semantic_memory_content(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """update_semantic_memory で content と confidence が更新される."""
        await store.save_semantic_memory(sample_semantic)
        updated = SemanticMemory(
            id="sem-001", character_id="aine-001",
            content="ユーザーはSF映画とホラーが好き",
            confidence=0.9,
            source_episode_ids=["ep-001", "ep-002"],
            embedding=[0.2, 0.3, 0.4],
        )
        await store.update_semantic_memory(updated)
        result = await store.get_semantic_memories("aine-001")
        assert len(result) == 1
        assert result[0].content == "ユーザーはSF映画とホラーが好き"
        assert result[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_update_semantic_memory_embedding(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """embedding も正しく更新される."""
        await store.save_semantic_memory(sample_semantic)
        new_embedding = [0.5, 0.6, 0.7]
        updated = SemanticMemory(
            id="sem-001", character_id="aine-001",
            content="更新", confidence=0.8,
            embedding=new_embedding,
        )
        await store.update_semantic_memory(updated)
        result = await store.get_semantic_memories("aine-001")
        assert result[0].embedding is not None
        for a, b in zip(result[0].embedding, new_embedding):
            assert abs(a - b) < 1e-10

    @pytest.mark.asyncio
    async def test_update_semantic_memory_source_episodes(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """source_episode_ids が更新される."""
        await store.save_semantic_memory(sample_semantic)
        updated = SemanticMemory(
            id="sem-001", character_id="aine-001",
            content="更新", confidence=0.8,
            source_episode_ids=["ep-001", "ep-002", "ep-003"],
        )
        await store.update_semantic_memory(updated)
        result = await store.get_semantic_memories("aine-001")
        assert set(result[0].source_episode_ids) == {"ep-001", "ep-002", "ep-003"}

    @pytest.mark.asyncio
    async def test_update_semantic_memory_preserves_count(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """update 後もレコード数は変わらない."""
        await store.save_semantic_memory(sample_semantic)
        sem2 = SemanticMemory(
            id="sem-002", character_id="aine-001",
            content="別の知識", confidence=0.5,
        )
        await store.save_semantic_memory(sem2)

        updated = SemanticMemory(
            id="sem-001", character_id="aine-001",
            content="更新された知識", confidence=0.8,
        )
        await store.update_semantic_memory(updated)
        result = await store.get_semantic_memories("aine-001")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_delete_semantic_memory(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """delete_semantic_memory でレコードが削除される."""
        await store.save_semantic_memory(sample_semantic)
        await store.delete_semantic_memory("sem-001")
        result = await store.get_semantic_memories("aine-001")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_delete_semantic_memory_removes_sources(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """delete で semantic_memory_sources も削除される."""
        await store.save_semantic_memory(sample_semantic)
        await store.delete_semantic_memory("sem-001")
        # source リンクも消えている確認
        cursor = await store._conn.execute(
            "SELECT COUNT(*) FROM semantic_memory_sources WHERE semantic_memory_id = ?",
            ("sem-001",),
        )
        count = (await cursor.fetchone())[0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_semantic_memory_nonexistent(
        self, store: SQLiteStorageBackend
    ) -> None:
        """存在しない ID の delete はエラーにならない."""
        await store.delete_semantic_memory("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_delete_semantic_memory_only_target(
        self, store: SQLiteStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """delete は対象レコードだけを削除し、他は残す."""
        await store.save_semantic_memory(sample_semantic)
        sem2 = SemanticMemory(
            id="sem-002", character_id="aine-001",
            content="別の知識", confidence=0.5,
        )
        await store.save_semantic_memory(sem2)
        await store.delete_semantic_memory("sem-001")
        result = await store.get_semantic_memories("aine-001")
        assert len(result) == 1
        assert result[0].id == "sem-002"
