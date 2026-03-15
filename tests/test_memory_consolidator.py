"""Tests for MemoryConsolidator (Issue #26, #44)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from pneuma_core.memory.consolidator import (
    ConsolidationConfig,
    ConsolidationResult,
    MemoryConsolidator,
)
from pneuma_core.models.memory import EpisodicMemory

NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_llm_response(episodes: list[dict]) -> str:
    """LLM レスポンス JSON を生成."""
    return json.dumps(episodes, ensure_ascii=False)


def _make_embedding(value: float = 0.5) -> list[float]:
    """簡易 embedding を生成（3次元）."""
    return [value, 1.0 - value, 0.0]


def _make_episodic(
    id: str = "ep-existing",
    content: str = "既存の記憶",
    importance: float = 0.7,
    embedding: list[float] | None = None,
    character_id: str = "aine-001",
) -> EpisodicMemory:
    return EpisodicMemory(
        id=id,
        character_id=character_id,
        content=content,
        timestamp=NOW,
        emotional_valence=0.0,
        importance=importance,
        embedding=embedding or _make_embedding(),
    )


# --- ConsolidationConfig ---


class TestConsolidationConfig:
    """ConsolidationConfig の検証."""

    def test_default_values(self) -> None:
        config = ConsolidationConfig()
        assert config.importance_threshold == 0.6
        assert config.max_episodes_per_conversation == 3
        assert config.duplicate_similarity_threshold == 0.95

    def test_custom_values(self) -> None:
        config = ConsolidationConfig(
            importance_threshold=0.7,
            max_episodes_per_conversation=5,
            duplicate_similarity_threshold=0.9,
        )
        assert config.importance_threshold == 0.7
        assert config.max_episodes_per_conversation == 5
        assert config.duplicate_similarity_threshold == 0.9


# --- ConsolidationResult ---


class TestConsolidationResult:
    """ConsolidationResult の検証."""

    def test_result_fields(self) -> None:
        result = ConsolidationResult(
            saved=[],
            filtered_by_importance=2,
            filtered_by_duplicate=1,
            filtered_by_limit=0,
        )
        assert result.saved == []
        assert result.filtered_by_importance == 2
        assert result.filtered_by_duplicate == 1
        assert result.filtered_by_limit == 0


# --- LLM レスポンスのパース ---


class TestEpisodeExtraction:
    """LLM からのエピソード抽出テスト."""

    @pytest.mark.asyncio
    async def test_extract_episodes_from_llm(self) -> None:
        """LLM レスポンスからエピソードを正しくパースできる."""
        episodes_json = [
            {
                "content": "ユーザーと初めて出会った",
                "importance": 0.8,
                "emotional_valence": 0.5,
            },
            {
                "content": "好きな食べ物はカレーだと聞いた",
                "importance": 0.6,
                "emotional_valence": 0.3,
            },
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1), _make_embedding(0.2)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "はじめまして！"},
            ],
            now=NOW,
        )

        assert len(result.saved) == 2
        assert result.saved[0].content == "ユーザーと初めて出会った"
        assert result.saved[0].importance == 0.8
        assert result.saved[0].emotional_valence == 0.5
        assert result.saved[0].character_id == "aine-001"

    @pytest.mark.asyncio
    async def test_empty_llm_response(self) -> None:
        """LLM が空リストを返した場合."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content="[]")
        )

        mock_embedding = AsyncMock()
        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 0
        assert result.filtered_by_importance == 0

    @pytest.mark.asyncio
    async def test_malformed_llm_response_returns_empty(self) -> None:
        """LLM が不正な JSON を返した場合、空の結果を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content="not valid json")
        )

        mock_embedding = AsyncMock()
        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 0


# --- 重要度フィルタ ---


class TestImportanceFilter:
    """重要度フィルタの検証."""

    @pytest.mark.asyncio
    async def test_filter_below_threshold(self) -> None:
        """importance < 0.6 のエピソードはフィルタされる."""
        episodes_json = [
            {"content": "重要な出来事", "importance": 0.8, "emotional_valence": 0.5},
            {"content": "些細な出来事", "importance": 0.3, "emotional_valence": 0.0},
            {"content": "普通の出来事", "importance": 0.5, "emotional_valence": 0.1},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.saved[0].content == "重要な出来事"
        assert result.filtered_by_importance == 2

    @pytest.mark.asyncio
    async def test_threshold_boundary_exact(self) -> None:
        """importance == 0.6 はちょうど通過する."""
        episodes_json = [
            {"content": "境界の出来事", "importance": 0.6, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.filtered_by_importance == 0

    @pytest.mark.asyncio
    async def test_custom_threshold(self) -> None:
        """カスタム閾値 0.8 の場合."""
        episodes_json = [
            {"content": "高重要度", "importance": 0.9, "emotional_valence": 0.5},
            {"content": "中重要度", "importance": 0.7, "emotional_valence": 0.3},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        config = ConsolidationConfig(importance_threshold=0.8)
        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
            config=config,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.saved[0].content == "高重要度"
        assert result.filtered_by_importance == 1


# --- 数量制限 ---


class TestQuantityLimit:
    """数量制限の検証."""

    @pytest.mark.asyncio
    async def test_limit_to_max_episodes(self) -> None:
        """max_episodes_per_conversation を超えた場合、重要度上位のみ保存."""
        episodes_json = [
            {"content": f"出来事{i}", "importance": 0.6 + i * 0.1, "emotional_valence": 0.0}
            for i in range(5)
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(i * 0.1) for i in range(3)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        # max 3 episodes, importance 降順で上位3件
        assert len(result.saved) == 3
        assert result.filtered_by_limit == 2
        importances = [m.importance for m in result.saved]
        assert importances == sorted(importances, reverse=True)

    @pytest.mark.asyncio
    async def test_within_limit_no_trimming(self) -> None:
        """制限以内の場合はトリミングなし."""
        episodes_json = [
            {"content": "出来事1", "importance": 0.8, "emotional_valence": 0.5},
            {"content": "出来事2", "importance": 0.7, "emotional_valence": 0.3},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1), _make_embedding(0.2)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 2
        assert result.filtered_by_limit == 0


# --- 重複チェック ---


class TestDuplicateCheck:
    """重複チェックの検証."""

    @pytest.mark.asyncio
    async def test_skip_duplicate_memory(self) -> None:
        """既存記憶と類似度 >= 0.95 のエピソードはスキップ."""
        # 既存記憶と全く同じ embedding を持つエピソード
        existing_embedding = [1.0, 0.0, 0.0]
        existing_memory = _make_episodic(
            id="ep-existing",
            content="既存の記憶",
            embedding=existing_embedding,
        )

        episodes_json = [
            {"content": "既存の記憶とほぼ同じ", "importance": 0.8, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        # 新エピソードの embedding が既存とほぼ同じ
        mock_embedding.embed_batch = AsyncMock(
            return_value=[existing_embedding]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(
            return_value=[existing_memory]
        )

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 0
        assert result.filtered_by_duplicate == 1

    @pytest.mark.asyncio
    async def test_keep_non_duplicate_memory(self) -> None:
        """類似度 < 0.95 のエピソードは保存する."""
        episodes_json = [
            {"content": "まったく新しい出来事", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        # 既存とは異なる embedding
        mock_embedding.embed_batch = AsyncMock(
            return_value=[[0.0, 1.0, 0.0]]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.filtered_by_duplicate == 0

    @pytest.mark.asyncio
    async def test_duplicate_threshold_boundary(self) -> None:
        """類似度がちょうど 0.95 の場合はスキップ（>= で判定）."""
        existing_memory = _make_episodic(
            id="ep-existing",
            content="既存の記憶",
            embedding=[1.0, 0.0, 0.0],
        )

        episodes_json = [
            {"content": "境界の出来事", "importance": 0.8, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[0.95, 0.3122, 0.0]])

        mock_store = AsyncMock()
        # find_similar_episodic がしきい値判定を担当し、結果を返す
        mock_store.find_similar_episodic = AsyncMock(
            return_value=[existing_memory]
        )

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        # find_similar_episodic が結果を返す → duplicate としてスキップ
        assert len(result.saved) == 0
        assert result.filtered_by_duplicate == 1


# --- ストレージ保存 ---


class TestStorageSave:
    """ストレージへの保存の検証."""

    @pytest.mark.asyncio
    async def test_saved_to_store(self) -> None:
        """フィルタ通過したエピソードが MemoryStore に保存される."""
        episodes_json = [
            {"content": "保存する記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])
        mock_store.add_episodic = AsyncMock()

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert mock_store.add_episodic.call_count == 1
        saved_memory = mock_store.add_episodic.call_args[0][0]
        assert saved_memory.content == "保存する記憶"
        assert saved_memory.importance == 0.8
        assert saved_memory.embedding is not None

    @pytest.mark.asyncio
    async def test_no_save_when_all_filtered(self) -> None:
        """全てフィルタされた場合は保存されない."""
        episodes_json = [
            {"content": "些細", "importance": 0.3, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])
        mock_store.add_episodic = AsyncMock()

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        mock_store.add_episodic.assert_not_called()
        assert len(result.saved) == 0


# --- 統合テスト ---


class TestIntegration:
    """フル統合テスト."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self) -> None:
        """抽出 → フィルタ → 重複チェック → 保存 の全パイプライン."""
        existing_memory = _make_episodic(
            id="ep-existing",
            content="既に覚えてる出来事",
            embedding=[1.0, 0.0, 0.0],
        )

        # 5件のエピソード: 2件は重要度不足、1件は重複、残り2件が保存対象
        episodes_json = [
            {"content": "人生を変える出来事", "importance": 0.95, "emotional_valence": 0.9},
            {"content": "重要な情報", "importance": 0.8, "emotional_valence": 0.5},
            {"content": "既に覚えてる出来事と同じ", "importance": 0.7, "emotional_valence": 0.0},
            {"content": "つまらない出来事", "importance": 0.3, "emotional_valence": 0.0},
            {"content": "些細な出来事", "importance": 0.5, "emotional_valence": -0.1},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        # 3件が重要度フィルタ通過（0.95, 0.8, 0.7）
        # そのうち1件（index 2）が既存と重複する embedding
        mock_embedding.embed_batch = AsyncMock(
            return_value=[
                [0.0, 1.0, 0.0],  # 0.95 - ユニーク
                [0.0, 0.0, 1.0],  # 0.8 - ユニーク
                [1.0, 0.0, 0.0],  # 0.7 - 既存と重複（cosine=1.0）
            ]
        )

        mock_store = AsyncMock()
        # find_similar_episodic: 3件目のみ重複（embedding=[1.0, 0.0, 0.0]が既存と一致）
        mock_store.find_similar_episodic = AsyncMock(
            side_effect=[[], [], [existing_memory]]
        )
        mock_store.add_episodic = AsyncMock()

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[
                {"role": "user", "content": "今日はいろいろあったね"},
                {"role": "assistant", "content": "そうだね、色々あったよ"},
            ],
            now=NOW,
        )

        # 重要度不足: 2件、重複: 1件、保存: 2件
        assert result.filtered_by_importance == 2
        assert result.filtered_by_duplicate == 1
        assert len(result.saved) == 2
        assert mock_store.add_episodic.call_count == 2

    @pytest.mark.asyncio
    async def test_full_pipeline_with_limit(self) -> None:
        """数量制限も含むフルパイプライン（max=2で検証）."""
        episodes_json = [
            {"content": f"出来事{i}", "importance": 0.6 + i * 0.1, "emotional_valence": 0.0}
            for i in range(4)
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(i * 0.1) for i in range(2)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])
        mock_store.add_episodic = AsyncMock()

        config = ConsolidationConfig(max_episodes_per_conversation=2)
        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
            config=config,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 2
        assert result.filtered_by_limit == 2
        # 上位2件が保存されている
        assert result.saved[0].importance >= result.saved[1].importance


# --- バリデーション・エラーハンドリング ---


class TestValidation:
    """LLM 出力バリデーション・例外ハンドリングの検証."""

    @pytest.mark.asyncio
    async def test_invalid_episode_missing_content(self) -> None:
        """content キーがないエピソードはスキップされる."""
        episodes_json = [
            {"importance": 0.8, "emotional_valence": 0.5},
            {"content": "正常な記憶", "importance": 0.7, "emotional_valence": 0.3},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm, embedding_service=mock_embedding, store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.saved[0].content == "正常な記憶"

    @pytest.mark.asyncio
    async def test_invalid_episode_missing_importance(self) -> None:
        """importance キーがないエピソードはスキップされる."""
        episodes_json = [
            {"content": "キーなし", "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm, embedding_service=mock_embedding, store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 0

    @pytest.mark.asyncio
    async def test_invalid_episode_importance_out_of_range(self) -> None:
        """importance が範囲外のエピソードはスキップされる."""
        episodes_json = [
            {"content": "範囲外", "importance": 1.5, "emotional_valence": 0.0},
            {"content": "正常", "importance": 0.8, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm, embedding_service=mock_embedding, store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.saved[0].content == "正常"

    @pytest.mark.asyncio
    async def test_invalid_episode_valence_out_of_range(self) -> None:
        """emotional_valence が範囲外のエピソードはスキップされる."""
        episodes_json = [
            {"content": "範囲外", "importance": 0.8, "emotional_valence": 2.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm, embedding_service=mock_embedding, store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 0

    @pytest.mark.asyncio
    async def test_llm_exception_returns_empty(self) -> None:
        """LLM が例外を投げた場合、空の結果を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("API error"))

        mock_embedding = AsyncMock()
        mock_store = AsyncMock()

        consolidator = MemoryConsolidator(
            llm=mock_llm, embedding_service=mock_embedding, store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 0

    @pytest.mark.asyncio
    async def test_non_dict_episode_skipped(self) -> None:
        """リスト内の非dict要素はスキップされる."""
        episodes_json = [
            "not a dict",
            42,
            {"content": "正常", "importance": 0.8, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm, embedding_service=mock_embedding, store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.saved[0].content == "正常"


# --- Issue #44: conversation_id 対応 ---


class TestConversationId:
    """conversation_id が EpisodicMemory に設定されることの検証 (#44)."""

    @pytest.mark.asyncio
    async def test_conversation_id_set_on_saved_memory(self) -> None:
        """consolidate() に conversation_id を渡すと EpisodicMemory に設定される."""
        episodes_json = [
            {"content": "テスト記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
            conversation_id="conv-001",
        )

        assert len(result.saved) == 1
        assert result.saved[0].conversation_id == "conv-001"

    @pytest.mark.asyncio
    async def test_conversation_id_none_by_default(self) -> None:
        """conversation_id を渡さない場合は None."""
        episodes_json = [
            {"content": "テスト記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.saved[0].conversation_id is None

    @pytest.mark.asyncio
    async def test_conversation_id_passed_to_store(self) -> None:
        """保存時に store.add_episodic に conversation_id 付きの EpisodicMemory が渡される."""
        episodes_json = [
            {"content": "保存する記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])
        mock_store.add_episodic = AsyncMock()

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
            conversation_id="conv-002",
        )

        assert mock_store.add_episodic.call_count == 1
        saved_memory = mock_store.add_episodic.call_args[0][0]
        assert saved_memory.conversation_id == "conv-002"


# --- Issue #44: find_similar_episodic 活用 ---


class TestFindSimilarEpisodic:
    """重複チェックが find_similar_episodic() 経由で行われることの検証 (#44)."""

    @pytest.mark.asyncio
    async def test_uses_find_similar_episodic_for_duplicate_check(self) -> None:
        """重複チェック時に find_similar_episodic() が呼ばれる."""
        episodes_json = [
            {"content": "新しい出来事", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        new_embedding = [0.1, 0.9, 0.0]
        mock_embedding.embed_batch = AsyncMock(return_value=[new_embedding])

        mock_store = AsyncMock()
        # find_similar_episodic が呼ばれるべき（重複なし）
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        # find_similar_episodic が各エピソードに対して呼ばれることを確認
        mock_store.find_similar_episodic.assert_called_once_with(
            "aine-001", new_embedding, consolidator.config.duplicate_similarity_threshold
        )

    @pytest.mark.asyncio
    async def test_does_not_use_get_episodic_by_character(self) -> None:
        """get_episodic_by_character() は重複チェックに使われない."""
        episodes_json = [
            {"content": "新しい出来事", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[0.1, 0.9, 0.0]])

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        # get_episodic_by_character は呼ばれないことを確認
        mock_store.get_episodic_by_character.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_detected_via_find_similar(self) -> None:
        """find_similar_episodic() が結果を返した場合、重複としてスキップされる."""
        existing_memory = _make_episodic(
            id="ep-existing",
            content="既存の記憶",
            embedding=[1.0, 0.0, 0.0],
        )

        episodes_json = [
            {"content": "既存と類似", "importance": 0.8, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[1.0, 0.0, 0.0]])

        mock_store = AsyncMock()
        # find_similar_episodic が既存記憶を返す（= 重複あり）
        mock_store.find_similar_episodic = AsyncMock(return_value=[existing_memory])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 0
        assert result.filtered_by_duplicate == 1

    @pytest.mark.asyncio
    async def test_non_duplicate_via_find_similar(self) -> None:
        """find_similar_episodic() が空を返した場合、保存される."""
        episodes_json = [
            {"content": "新しい出来事", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[0.0, 1.0, 0.0]])

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.filtered_by_duplicate == 0

    @pytest.mark.asyncio
    async def test_multiple_episodes_each_checked_separately(self) -> None:
        """複数エピソードが各々独立に find_similar_episodic() でチェックされる."""
        existing_memory = _make_episodic(
            id="ep-existing",
            content="既存の記憶",
            embedding=[1.0, 0.0, 0.0],
        )

        episodes_json = [
            {"content": "新しい出来事1", "importance": 0.9, "emotional_valence": 0.5},
            {"content": "既存と重複", "importance": 0.8, "emotional_valence": 0.0},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        emb1 = [0.0, 1.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[emb1, emb2])

        mock_store = AsyncMock()
        # 1件目はユニーク、2件目は重複
        mock_store.find_similar_episodic = AsyncMock(
            side_effect=[[], [existing_memory]]
        )

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1
        assert result.saved[0].content == "新しい出来事1"
        assert result.filtered_by_duplicate == 1
        assert mock_store.find_similar_episodic.call_count == 2


# --- Issue #56: モデル選択パラメータ ---


class TestConsolidatorModelSelection:
    """MemoryConsolidator のモデル選択パラメータの検証 (#56)."""

    @pytest.mark.asyncio
    async def test_model_parameter_accepted(self) -> None:
        """MemoryConsolidator が model パラメータを受け取れる."""
        episodes_json = [
            {"content": "テスト記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
            model="claude-haiku-4-20250514",
        )

        result = await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        assert len(result.saved) == 1

    @pytest.mark.asyncio
    async def test_model_set_in_llm_request(self) -> None:
        """model パラメータが LLMRequest.model に設定される."""
        episodes_json = [
            {"content": "テスト記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
            model="claude-haiku-4-20250514",
        )

        await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        call_args = mock_llm.generate.call_args[0][0]
        assert call_args.model == "claude-haiku-4-20250514"

    @pytest.mark.asyncio
    async def test_model_none_by_default(self) -> None:
        """model を指定しない場合は LLMRequest.model が None."""
        episodes_json = [
            {"content": "テスト記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
        )

        await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        call_args = mock_llm.generate.call_args[0][0]
        assert call_args.model is None

    @pytest.mark.asyncio
    async def test_model_with_config(self) -> None:
        """model と config を同時に指定できる."""
        episodes_json = [
            {"content": "テスト記憶", "importance": 0.8, "emotional_valence": 0.5},
        ]

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=AsyncMock(content=_make_llm_response(episodes_json))
        )

        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(
            return_value=[_make_embedding(0.1)]
        )

        mock_store = AsyncMock()
        mock_store.find_similar_episodic = AsyncMock(return_value=[])

        config = ConsolidationConfig(importance_threshold=0.7)
        consolidator = MemoryConsolidator(
            llm=mock_llm,
            embedding_service=mock_embedding,
            store=mock_store,
            config=config,
            model="claude-haiku-4-20250514",
        )

        await consolidator.consolidate(
            character_id="aine-001",
            conversation_history=[{"role": "user", "content": "hi"}],
            now=NOW,
        )

        call_args = mock_llm.generate.call_args[0][0]
        assert call_args.model == "claude-haiku-4-20250514"
