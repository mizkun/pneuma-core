"""Tests for SemanticConsolidator (Issue #107).

Episodic -> Semantic memory consolidation:
related episodic memories are clustered and generalized into semantic memories.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, call

import pytest

from pneuma_core.memory.semantic_consolidator import (
    SemanticConsolidationConfig,
    SemanticConsolidator,
)
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory

NOW = datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc)
CHARACTER_ID = "mira-001"


def _make_embedding(x: float = 1.0, y: float = 0.0, z: float = 0.0) -> list[float]:
    """簡易 embedding を生成（3次元）."""
    return [x, y, z]


_SENTINEL = object()


def _make_episode(
    id: str,
    content: str,
    embedding: list[float] | None | object = _SENTINEL,
    importance: float = 0.7,
) -> EpisodicMemory:
    """テスト用 EpisodicMemory を生成."""
    if embedding is _SENTINEL:
        embedding = _make_embedding()
    return EpisodicMemory(
        id=id,
        character_id=CHARACTER_ID,
        content=content,
        timestamp=NOW,
        emotional_valence=0.5,
        importance=importance,
        embedding=embedding,
    )


def _make_semantic(
    id: str,
    content: str,
    confidence: float = 0.8,
    embedding: list[float] | None = None,
    source_episode_ids: list[str] | None = None,
) -> SemanticMemory:
    """テスト用 SemanticMemory を生成."""
    return SemanticMemory(
        id=id,
        character_id=CHARACTER_ID,
        content=content,
        confidence=confidence,
        source_episode_ids=source_episode_ids or [],
        embedding=embedding or _make_embedding(),
    )


def _make_llm_response(content: str, confidence: float) -> str:
    """LLM レスポンス JSON を生成."""
    return json.dumps(
        {"content": content, "confidence": confidence}, ensure_ascii=False
    )


# --- SemanticConsolidationConfig ---


class TestSemanticConsolidationConfig:
    """SemanticConsolidationConfig の検証."""

    def test_default_values(self) -> None:
        config = SemanticConsolidationConfig()
        assert config.min_episodes == 3
        assert config.similarity_threshold == 0.7
        assert config.duplicate_threshold == 0.9
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_semantics_per_batch == 5

    def test_custom_values(self) -> None:
        config = SemanticConsolidationConfig(
            min_episodes=5,
            similarity_threshold=0.8,
            duplicate_threshold=0.95,
            model="claude-sonnet-4-20250514",
            max_semantics_per_batch=10,
        )
        assert config.min_episodes == 5
        assert config.similarity_threshold == 0.8
        assert config.duplicate_threshold == 0.95
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_semantics_per_batch == 10


# --- Cluster Episodes ---


class TestClusterEpisodes:
    """エピソード記憶のクラスタリング検証."""

    def _make_consolidator(
        self, config: SemanticConsolidationConfig | None = None
    ) -> SemanticConsolidator:
        llm = AsyncMock()
        embedding_service = AsyncMock()
        return SemanticConsolidator(llm, embedding_service, config)

    def test_similar_episodes_clustered(self) -> None:
        """類似した embedding を持つエピソードが同じクラスタにまとまる."""
        consolidator = self._make_consolidator()
        episodes = [
            _make_episode("ep-1", "なでしことご飯を食べた", _make_embedding(0.9, 0.1, 0.0)),
            _make_episode("ep-2", "なでしことカレーを作った", _make_embedding(0.88, 0.12, 0.0)),
            _make_episode("ep-3", "なでしことBBQした", _make_embedding(0.92, 0.08, 0.0)),
        ]
        clusters = consolidator._cluster_episodes(episodes)
        # All three should be in one cluster (they are very similar)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_dissimilar_episodes_not_clustered(self) -> None:
        """異なる embedding を持つエピソードは別のクラスタになる."""
        consolidator = self._make_consolidator()
        episodes = [
            _make_episode("ep-1", "なでしことご飯を食べた", _make_embedding(1.0, 0.0, 0.0)),
            _make_episode("ep-2", "数学の勉強をした", _make_embedding(0.0, 1.0, 0.0)),
            _make_episode("ep-3", "空を飛ぶ夢を見た", _make_embedding(0.0, 0.0, 1.0)),
        ]
        clusters = consolidator._cluster_episodes(episodes)
        # Each episode is in its own cluster (orthogonal vectors = similarity 0.0)
        # But clusters below min_episodes are filtered, so we expect 0 after filtering
        # _cluster_episodes returns raw clusters before filtering
        assert len(clusters) == 3
        for cluster in clusters:
            assert len(cluster) == 1

    def test_cluster_below_min_episodes_filtered(self) -> None:
        """min_episodes 未満のクラスタは除外される."""
        config = SemanticConsolidationConfig(min_episodes=3)
        consolidator = self._make_consolidator(config)
        episodes = [
            # Cluster of 2 (below threshold of 3)
            _make_episode("ep-1", "本を読んだ", _make_embedding(1.0, 0.0, 0.0)),
            _make_episode("ep-2", "雑誌を読んだ", _make_embedding(0.95, 0.05, 0.0)),
            # Isolated episode
            _make_episode("ep-3", "空を飛ぶ夢を見た", _make_embedding(0.0, 0.0, 1.0)),
        ]
        clusters = consolidator._cluster_episodes(episodes)
        # Filter out small clusters
        filtered = [c for c in clusters if len(c) >= config.min_episodes]
        assert len(filtered) == 0

    def test_overlapping_clusters_merged(self) -> None:
        """重複するクラスタがマージされる."""
        consolidator = self._make_consolidator(
            SemanticConsolidationConfig(similarity_threshold=0.7)
        )
        # A is similar to B, B is similar to C, but A is NOT similar to C
        # They should still be merged through B
        episodes = [
            _make_episode("ep-a", "A", _make_embedding(1.0, 0.3, 0.0)),
            _make_episode("ep-b", "B", _make_embedding(0.7, 0.7, 0.0)),
            _make_episode("ep-c", "C", _make_embedding(0.3, 1.0, 0.0)),
        ]
        clusters = consolidator._cluster_episodes(episodes)
        # All should be merged into one cluster via transitive similarity
        assert len(clusters) == 1
        assert len(clusters[0]) == 3


# --- Consolidate (integration) ---


class TestConsolidate:
    """consolidate() メソッドの統合テスト."""

    def _setup(
        self, config: SemanticConsolidationConfig | None = None
    ) -> tuple[SemanticConsolidator, AsyncMock, AsyncMock]:
        llm = AsyncMock()
        embedding_service = AsyncMock()
        consolidator = SemanticConsolidator(llm, embedding_service, config)
        return consolidator, llm, embedding_service

    @pytest.mark.asyncio
    async def test_consolidate_generates_semantic_memory(self) -> None:
        """3つ以上の関連エピソードから semantic memory が生成される."""
        consolidator, llm, embedding_service = self._setup()

        similar_emb = _make_embedding(0.9, 0.1, 0.0)
        episodes = [
            _make_episode("ep-1", "なでしことご飯を食べた。楽しかった", similar_emb),
            _make_episode("ep-2", "なでしことカレーを作った。美味しかった", _make_embedding(0.88, 0.12, 0.0)),
            _make_episode("ep-3", "なでしことBBQした", _make_embedding(0.92, 0.08, 0.0)),
        ]
        existing_semantics: list[SemanticMemory] = []

        # LLM returns a valid consolidation response
        llm.generate.return_value = AsyncMock(
            content=_make_llm_response("なでしことは一緒に料理するのが楽しい", 0.8)
        )

        # Embedding for the new semantic memory
        new_semantic_emb = _make_embedding(0.91, 0.09, 0.0)
        embedding_service.embed.return_value = new_semantic_emb

        result = await consolidator.consolidate(
            CHARACTER_ID, episodes, existing_semantics
        )

        assert len(result) == 1
        semantic = result[0]
        assert semantic.content == "なでしことは一緒に料理するのが楽しい"
        assert semantic.confidence == 0.8
        assert semantic.character_id == CHARACTER_ID
        assert semantic.embedding == new_semantic_emb

    @pytest.mark.asyncio
    async def test_consolidate_skips_duplicate(self) -> None:
        """既存の semantic memory と類似度が高い場合はスキップされる."""
        config = SemanticConsolidationConfig(duplicate_threshold=0.9)
        consolidator, llm, embedding_service = self._setup(config)

        similar_emb = _make_embedding(0.9, 0.1, 0.0)
        episodes = [
            _make_episode("ep-1", "なでしことご飯", similar_emb),
            _make_episode("ep-2", "なでしことカレー", _make_embedding(0.88, 0.12, 0.0)),
            _make_episode("ep-3", "なでしことBBQ", _make_embedding(0.92, 0.08, 0.0)),
        ]

        # Existing semantic memory that is very similar to what would be generated
        existing_semantics = [
            _make_semantic(
                "sem-existing",
                "なでしことは料理を楽しむ関係",
                embedding=_make_embedding(0.91, 0.09, 0.0),
            ),
        ]

        # LLM generates something similar to existing
        llm.generate.return_value = AsyncMock(
            content=_make_llm_response("なでしことは一緒に料理するのが楽しい", 0.8)
        )

        # New semantic embedding is very similar to existing (> 0.9)
        new_semantic_emb = _make_embedding(0.91, 0.09, 0.0)
        embedding_service.embed.return_value = new_semantic_emb

        result = await consolidator.consolidate(
            CHARACTER_ID, episodes, existing_semantics
        )

        # Should be empty because duplicate was detected
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_consolidate_returns_empty_when_no_clusters(self) -> None:
        """関連するエピソードがない場合は空リストが返る."""
        consolidator, llm, embedding_service = self._setup()

        # All episodes are dissimilar (orthogonal embeddings)
        episodes = [
            _make_episode("ep-1", "ご飯を食べた", _make_embedding(1.0, 0.0, 0.0)),
            _make_episode("ep-2", "数学の勉強をした", _make_embedding(0.0, 1.0, 0.0)),
            _make_episode("ep-3", "空を飛ぶ夢を見た", _make_embedding(0.0, 0.0, 1.0)),
        ]
        existing_semantics: list[SemanticMemory] = []

        result = await consolidator.consolidate(
            CHARACTER_ID, episodes, existing_semantics
        )

        assert len(result) == 0
        # LLM should not have been called since no clusters formed
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidate_sets_source_episode_ids(self) -> None:
        """生成された semantic memory に source_episode_ids が正しく設定される."""
        consolidator, llm, embedding_service = self._setup()

        episodes = [
            _make_episode("ep-1", "なでしことご飯", _make_embedding(0.9, 0.1, 0.0)),
            _make_episode("ep-2", "なでしことカレー", _make_embedding(0.88, 0.12, 0.0)),
            _make_episode("ep-3", "なでしことBBQ", _make_embedding(0.92, 0.08, 0.0)),
        ]

        llm.generate.return_value = AsyncMock(
            content=_make_llm_response("なでしこと料理が楽しい", 0.8)
        )
        embedding_service.embed.return_value = _make_embedding(0.9, 0.1, 0.0)

        result = await consolidator.consolidate(CHARACTER_ID, episodes, [])

        assert len(result) == 1
        # source_episode_ids should contain all episode IDs from the cluster
        assert set(result[0].source_episode_ids) == {"ep-1", "ep-2", "ep-3"}

    @pytest.mark.asyncio
    async def test_consolidate_uses_haiku_model(self) -> None:
        """LLM リクエストに設定されたモデルが使用される."""
        config = SemanticConsolidationConfig(model="claude-haiku-4-5-20251001")
        consolidator, llm, embedding_service = self._setup(config)

        episodes = [
            _make_episode("ep-1", "A", _make_embedding(0.9, 0.1, 0.0)),
            _make_episode("ep-2", "B", _make_embedding(0.88, 0.12, 0.0)),
            _make_episode("ep-3", "C", _make_embedding(0.92, 0.08, 0.0)),
        ]

        llm.generate.return_value = AsyncMock(
            content=_make_llm_response("統合された記憶", 0.7)
        )
        embedding_service.embed.return_value = _make_embedding(0.9, 0.1, 0.0)

        await consolidator.consolidate(CHARACTER_ID, episodes, [])

        # Verify LLM was called with the correct model
        assert llm.generate.called
        request = llm.generate.call_args[0][0]
        assert request.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_consolidate_respects_max_semantics_per_batch(self) -> None:
        """max_semantics_per_batch を超えるクラスタがある場合、制限される."""
        config = SemanticConsolidationConfig(
            min_episodes=2,
            similarity_threshold=0.7,
            max_semantics_per_batch=1,
        )
        consolidator, llm, embedding_service = self._setup(config)

        # Two separate clusters (each with 2+ members)
        episodes = [
            # Cluster 1: cooking
            _make_episode("ep-1", "料理した", _make_embedding(0.9, 0.1, 0.0)),
            _make_episode("ep-2", "カレー作った", _make_embedding(0.88, 0.12, 0.0)),
            # Cluster 2: study
            _make_episode("ep-3", "数学を勉強した", _make_embedding(0.0, 0.9, 0.1)),
            _make_episode("ep-4", "物理を勉強した", _make_embedding(0.0, 0.88, 0.12)),
        ]

        llm.generate.return_value = AsyncMock(
            content=_make_llm_response("統合された記憶", 0.7)
        )
        embedding_service.embed.return_value = _make_embedding(0.9, 0.1, 0.0)

        result = await consolidator.consolidate(CHARACTER_ID, episodes, [])

        # Only 1 semantic memory should be generated (max_semantics_per_batch=1)
        assert len(result) <= 1


# --- LLM Response Parsing ---


class TestLLMParsing:
    """LLM レスポンスのパース検証."""

    def _make_consolidator(self) -> SemanticConsolidator:
        llm = AsyncMock()
        embedding_service = AsyncMock()
        return SemanticConsolidator(llm, embedding_service)

    def test_parse_valid_json(self) -> None:
        """有効な JSON レスポンスが正しくパースされる."""
        consolidator = self._make_consolidator()
        response = '{"content": "なでしことは料理が楽しい", "confidence": 0.8}'
        result = consolidator._parse_llm_response(response)
        assert result is not None
        assert result["content"] == "なでしことは料理が楽しい"
        assert result["confidence"] == 0.8

    def test_parse_markdown_wrapped_json(self) -> None:
        """マークダウンコードブロックで囲まれた JSON がパースされる."""
        consolidator = self._make_consolidator()
        response = '```json\n{"content": "なでしことは料理が楽しい", "confidence": 0.8}\n```'
        result = consolidator._parse_llm_response(response)
        assert result is not None
        assert result["content"] == "なでしことは料理が楽しい"
        assert result["confidence"] == 0.8

    def test_parse_invalid_json_returns_none(self) -> None:
        """不正な JSON レスポンスの場合は None が返る."""
        consolidator = self._make_consolidator()
        result = consolidator._parse_llm_response("this is not json")
        assert result is None

    def test_parse_missing_content_returns_none(self) -> None:
        """content フィールドがない JSON は None が返る."""
        consolidator = self._make_consolidator()
        result = consolidator._parse_llm_response('{"confidence": 0.8}')
        assert result is None

    def test_parse_missing_confidence_returns_none(self) -> None:
        """confidence フィールドがない JSON は None が返る."""
        consolidator = self._make_consolidator()
        result = consolidator._parse_llm_response('{"content": "test"}')
        assert result is None


# --- Edge Cases ---


class TestEdgeCases:
    """エッジケース検証."""

    @pytest.mark.asyncio
    async def test_episodes_without_embeddings_are_skipped(self) -> None:
        """embedding が None のエピソードはスキップされる."""
        llm = AsyncMock()
        embedding_service = AsyncMock()
        consolidator = SemanticConsolidator(llm, embedding_service)

        episodes = [
            _make_episode("ep-1", "なでしことご飯", embedding=None),
            _make_episode("ep-2", "なでしことカレー", embedding=None),
            _make_episode("ep-3", "なでしことBBQ", embedding=None),
        ]

        result = await consolidator.consolidate(CHARACTER_ID, episodes, [])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_empty_episodes_list(self) -> None:
        """空のエピソードリストを渡すと空リストが返る."""
        llm = AsyncMock()
        embedding_service = AsyncMock()
        consolidator = SemanticConsolidator(llm, embedding_service)

        result = await consolidator.consolidate(CHARACTER_ID, [], [])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self) -> None:
        """LLM がエラーを返した場合、そのクラスタはスキップされる."""
        llm = AsyncMock()
        embedding_service = AsyncMock()
        consolidator = SemanticConsolidator(llm, embedding_service)

        episodes = [
            _make_episode("ep-1", "A", _make_embedding(0.9, 0.1, 0.0)),
            _make_episode("ep-2", "B", _make_embedding(0.88, 0.12, 0.0)),
            _make_episode("ep-3", "C", _make_embedding(0.92, 0.08, 0.0)),
        ]

        llm.generate.side_effect = Exception("API Error")

        result = await consolidator.consolidate(CHARACTER_ID, episodes, [])
        assert len(result) == 0
