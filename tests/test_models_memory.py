"""Tests for Issue #4: EpisodicMemory, SemanticMemory, MemoryStore.

受け入れ基準:
- 各 dataclass が正しくインスタンス化できる
- バリデーションが機能する
- コサイン類似度計算が正確
- MemoryStore Protocol が定義されている
- 全テストがパスする
"""

import math
from datetime import datetime, timezone

import pytest

from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.memory.store import MemoryStore


# =============================================
# EpisodicMemory Tests
# =============================================


class TestEpisodicMemory:
    """エピソード記憶のテスト."""

    def test_create_valid_episodic_memory(self):
        """正常な値で EpisodicMemory を生成できる."""
        mem = EpisodicMemory(
            id="ep-001",
            character_id="aine-001",
            content="ユーザーが好きな映画を教えてくれた",
            timestamp=datetime(2026, 2, 23, tzinfo=timezone.utc),
            emotional_valence=0.5,
            importance=0.8,
        )
        assert mem.id == "ep-001"
        assert mem.character_id == "aine-001"
        assert mem.content == "ユーザーが好きな映画を教えてくれた"
        assert mem.emotional_valence == 0.5
        assert mem.importance == 0.8
        assert mem.embedding is None
        assert mem.conversation_id is None

    def test_create_with_embedding(self):
        """embedding 付きで生成できる."""
        embedding = [0.1] * 1536
        mem = EpisodicMemory(
            id="ep-002",
            character_id="aine-001",
            content="テスト",
            timestamp=datetime(2026, 2, 23, tzinfo=timezone.utc),
            emotional_valence=0.0,
            importance=0.6,
            embedding=embedding,
        )
        assert mem.embedding is not None
        assert len(mem.embedding) == 1536

    def test_reject_emotional_valence_out_of_range(self):
        """emotional_valence が -1.0〜1.0 の範囲外でエラー."""
        with pytest.raises(ValueError):
            EpisodicMemory(
                id="ep-003",
                character_id="aine-001",
                content="テスト",
                timestamp=datetime(2026, 2, 23, tzinfo=timezone.utc),
                emotional_valence=1.5,
                importance=0.5,
            )

    def test_reject_importance_out_of_range(self):
        """importance が 0.0〜1.0 の範囲外でエラー."""
        with pytest.raises(ValueError):
            EpisodicMemory(
                id="ep-004",
                character_id="aine-001",
                content="テスト",
                timestamp=datetime(2026, 2, 23, tzinfo=timezone.utc),
                emotional_valence=0.0,
                importance=-0.1,
            )


# =============================================
# SemanticMemory Tests
# =============================================


class TestSemanticMemory:
    """意味記憶のテスト."""

    def test_create_valid_semantic_memory(self):
        """正常な値で SemanticMemory を生成できる."""
        mem = SemanticMemory(
            id="sem-001",
            character_id="aine-001",
            content="ユーザーは映画好きである",
            confidence=0.8,
            source_episode_ids=["ep-001", "ep-002"],
        )
        assert mem.id == "sem-001"
        assert mem.confidence == 0.8
        assert len(mem.source_episode_ids) == 2
        assert mem.embedding is None

    def test_reject_confidence_out_of_range(self):
        """confidence が 0.0〜1.0 の範囲外でエラー."""
        with pytest.raises(ValueError):
            SemanticMemory(
                id="sem-002",
                character_id="aine-001",
                content="テスト",
                confidence=1.1,
                source_episode_ids=[],
            )

    def test_empty_source_episodes(self):
        """source_episode_ids が空でも生成できる."""
        mem = SemanticMemory(
            id="sem-003",
            character_id="aine-001",
            content="テスト",
            confidence=0.5,
            source_episode_ids=[],
        )
        assert mem.source_episode_ids == []


# =============================================
# Cosine Similarity Tests
# =============================================


class TestCosineSimilarity:
    """コサイン類似度計算のテスト."""

    def test_identical_vectors(self):
        """同一ベクトルの類似度は 1.0."""
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """直交ベクトルの類似度は 0.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """逆方向ベクトルの類似度は -1.0."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_known_similarity(self):
        """既知の類似度を計算できる."""
        v1 = [1.0, 1.0]
        v2 = [1.0, 0.0]
        # cos(45°) = √2/2 ≈ 0.7071
        expected = math.sqrt(2) / 2
        assert cosine_similarity(v1, v2) == pytest.approx(expected, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        """ゼロベクトルの類似度は 0.0."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)


# =============================================
# MemoryStore Protocol Tests
# =============================================


class TestMemoryStoreProtocol:
    """MemoryStore Protocol の型チェック."""

    def test_protocol_is_defined(self):
        """MemoryStore Protocol が定義されている."""
        assert hasattr(MemoryStore, "add_episodic")
        assert hasattr(MemoryStore, "add_semantic")
        assert hasattr(MemoryStore, "get_episodic_by_character")
        assert hasattr(MemoryStore, "get_semantic_by_character")
        assert hasattr(MemoryStore, "find_similar_episodic")
