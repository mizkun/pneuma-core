"""Tests for StorageBackend + InMemoryStorageBackend (Issue #6)."""

from datetime import datetime

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree, Objective, Task, Vision
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.storage.backend import StorageBackend
from pneuma_core.storage.in_memory import InMemoryStorageBackend
from pneuma_core.models.change_record import ChangeRecord


# --- Fixtures ---


@pytest.fixture
def store() -> InMemoryStorageBackend:
    return InMemoryStorageBackend()


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
        before={"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
        after={"pleasure": 0.5, "arousal": 0.3, "dominance": -0.1},
        reason="楽しい会話に反応",
        timestamp=datetime(2026, 2, 23, 10, 0, 0),
    )


# --- Protocol conformance ---


class TestProtocolConformance:
    """InMemoryStorageBackend が StorageBackend Protocol を満たすか."""

    def test_implements_protocol(self, store: InMemoryStorageBackend) -> None:
        assert isinstance(store, StorageBackend)


# --- Character CRUD ---


class TestCharacterCRUD:
    """Character の保存・取得・一覧."""

    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: InMemoryStorageBackend, sample_character: Character
    ) -> None:
        await store.save_character(sample_character)
        result = await store.get_character("aine-001")
        assert result is not None
        assert result.id == "aine-001"
        assert result.name == "アイネ"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: InMemoryStorageBackend) -> None:
        result = await store.get_character("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_characters(
        self, store: InMemoryStorageBackend, sample_character: Character
    ) -> None:
        await store.save_character(sample_character)
        chars = await store.list_characters()
        assert len(chars) == 1
        assert chars[0].id == "aine-001"

    @pytest.mark.asyncio
    async def test_list_empty(self, store: InMemoryStorageBackend) -> None:
        chars = await store.list_characters()
        assert chars == []

    @pytest.mark.asyncio
    async def test_overwrite(
        self, store: InMemoryStorageBackend, sample_personality: Personality,
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
    """Memory の保存・取得."""

    @pytest.mark.asyncio
    async def test_save_and_get_episodic(
        self, store: InMemoryStorageBackend, sample_episodic: EpisodicMemory
    ) -> None:
        await store.save_episodic_memory(sample_episodic)
        result = await store.get_episodic_memories("aine-001")
        assert len(result) == 1
        assert result[0].id == "ep-001"

    @pytest.mark.asyncio
    async def test_get_episodic_empty(self, store: InMemoryStorageBackend) -> None:
        result = await store.get_episodic_memories("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_save_and_get_semantic(
        self, store: InMemoryStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        await store.save_semantic_memory(sample_semantic)
        result = await store.get_semantic_memories("aine-001")
        assert len(result) == 1
        assert result[0].id == "sem-001"

    @pytest.mark.asyncio
    async def test_get_semantic_empty(self, store: InMemoryStorageBackend) -> None:
        result = await store.get_semantic_memories("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_find_similar_memories(
        self, store: InMemoryStorageBackend
    ) -> None:
        """コサイン類似度によるフィルタリング."""
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

        # [1.0, 0.0, 0.0] に似た記憶を検索 (threshold=0.8)
        similar = await store.find_similar_memories(
            "aine-001", [1.0, 0.0, 0.0], threshold=0.8
        )
        ids = [m.id for m in similar]
        assert "ep-001" in ids  # 完全一致
        assert "ep-003" in ids  # 高い類似度
        assert "ep-002" not in ids  # 直交

    @pytest.mark.asyncio
    async def test_find_similar_no_embedding(
        self, store: InMemoryStorageBackend
    ) -> None:
        """embedding が None の記憶はスキップ."""
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
        self, store: InMemoryStorageBackend
    ) -> None:
        """別キャラクターの記憶は返さない."""
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
    """Goals の保存・取得."""

    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: InMemoryStorageBackend, sample_goal_tree: GoalTree
    ) -> None:
        await store.save_goals("aine-001", sample_goal_tree)
        result = await store.get_goals("aine-001")
        assert result is not None
        assert len(result.visions) == 1
        assert len(result.objectives) == 1
        assert len(result.tasks) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: InMemoryStorageBackend) -> None:
        result = await store.get_goals("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite(
        self, store: InMemoryStorageBackend, sample_goal_tree: GoalTree
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
    """EmotionalState の保存・取得."""

    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: InMemoryStorageBackend, sample_state: EmotionalState
    ) -> None:
        await store.save_emotional_state("aine-001", sample_state)
        result = await store.get_emotional_state("aine-001")
        assert result is not None
        assert result.pleasure == 0.5
        assert result.emotion_label == "喜び（高揚）"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: InMemoryStorageBackend) -> None:
        result = await store.get_emotional_state("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite(self, store: InMemoryStorageBackend) -> None:
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
    """ChangeRecord の記録・取得."""

    @pytest.mark.asyncio
    async def test_save_and_get(
        self, store: InMemoryStorageBackend, sample_change: ChangeRecord
    ) -> None:
        await store.save_change(sample_change)
        result = await store.get_changes("aine-001")
        assert len(result) == 1
        assert result[0].type == "emotion_updated"

    @pytest.mark.asyncio
    async def test_get_empty(self, store: InMemoryStorageBackend) -> None:
        result = await store.get_changes("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_limit(self, store: InMemoryStorageBackend) -> None:
        """limit パラメータで取得件数を制限."""
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
    async def test_ordered_by_timestamp_desc(self, store: InMemoryStorageBackend) -> None:
        """新しい順で返す."""
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
        assert result[0].id == "ch-002"  # 最新が先頭
        assert result[2].id == "ch-000"

    @pytest.mark.asyncio
    async def test_filter_by_character(self, store: InMemoryStorageBackend) -> None:
        """別キャラクターの ChangeRecord は返さない."""
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


# --- ChangeRecord dataclass ---


class TestChangeRecord:
    """ChangeRecord dataclass のテスト."""

    def test_create(self) -> None:
        cr = ChangeRecord(
            id="ch-001", character_id="aine-001",
            type="emotion_updated",
            before={"pleasure": 0.0},
            after={"pleasure": 0.5},
            reason="会話に反応",
            timestamp=datetime(2026, 2, 23, 10, 0, 0),
        )
        assert cr.id == "ch-001"
        assert cr.type == "emotion_updated"
        assert cr.before == {"pleasure": 0.0}
        assert cr.after == {"pleasure": 0.5}

    def test_before_none(self) -> None:
        cr = ChangeRecord(
            id="ch-001", character_id="aine-001",
            type="memory_added",
            before=None,
            after={"content": "新しい記憶"},
            reason="記憶追加",
            timestamp=datetime(2026, 2, 23),
        )
        assert cr.before is None

    def test_frozen(self) -> None:
        cr = ChangeRecord(
            id="ch-001", character_id="aine-001",
            type="test", before=None, after={},
            reason="r", timestamp=datetime(2026, 2, 23),
        )
        with pytest.raises(AttributeError):
            cr.type = "modified"  # type: ignore[misc]


# --- Semantic Memory Update/Delete (#130) ---


class TestSemanticMemoryUpdateDelete:
    """セマンティック記憶の更新・削除 (#130)."""

    @pytest.mark.asyncio
    async def test_update_semantic_memory_content(
        self, store: InMemoryStorageBackend, sample_semantic: SemanticMemory
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
    async def test_update_semantic_memory_preserves_count(
        self, store: InMemoryStorageBackend, sample_semantic: SemanticMemory
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
        self, store: InMemoryStorageBackend, sample_semantic: SemanticMemory
    ) -> None:
        """delete_semantic_memory でレコードが削除される."""
        await store.save_semantic_memory(sample_semantic)
        await store.delete_semantic_memory("sem-001")
        result = await store.get_semantic_memories("aine-001")
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_delete_semantic_memory_nonexistent(
        self, store: InMemoryStorageBackend
    ) -> None:
        """存在しない ID の delete はエラーにならない."""
        await store.delete_semantic_memory("nonexistent")  # should not raise

    @pytest.mark.asyncio
    async def test_delete_semantic_memory_only_target(
        self, store: InMemoryStorageBackend, sample_semantic: SemanticMemory
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
