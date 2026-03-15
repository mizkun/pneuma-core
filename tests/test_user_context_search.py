"""Tests for UserContextSearchEngine (Issue #73).

Implements vector search over user context chunks using cosine similarity.
Uses mock EmbeddingService to avoid real API calls.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.runtime.user_context import UserContextChunk
from pneuma_core.runtime.user_context_search import (
    UserContextSearchConfig,
    UserContextSearchEngine,
    UserContextSearchResult,
)


# --- Fixtures ---


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """Create a mock EmbeddingService that returns deterministic embeddings.

    Each text maps to a unique 4-dimensional unit-ish vector for testing.
    """
    service = AsyncMock()

    # Simple deterministic embedding: hash text to produce a small vector
    def _make_embedding(text: str) -> list[float]:
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, 8, 2)]

    async def embed(text: str) -> list[float]:
        return _make_embedding(text)

    async def embed_batch(texts: list[str]) -> list[list[float]]:
        return [_make_embedding(t) for t in texts]

    service.embed = embed
    service.embed_batch = embed_batch
    return service


@pytest.fixture
def sample_chunks() -> list[UserContextChunk]:
    """Create sample chunks for testing."""
    return [
        UserContextChunk(
            content="Name: Kyohei, Age: 30, Location: Tokyo",
            layer=1,
            source_file="identity.md",
        ),
        UserContextChunk(
            content="Career vision: Build meaningful AI products",
            layer=2,
            source_file="values.md",
        ),
        UserContextChunk(
            content="Wife: Yuki, childhood friend: Taro",
            layer=3,
            source_file="glossary.md",
        ),
        UserContextChunk(
            content="Pneuma: AI character system with inner life",
            layer=4,
            source_file="projects/pneuma.md",
        ),
        UserContextChunk(
            content="2026-02-20: Worked on user context feature",
            layer=5,
            source_file="diary/2026-02-20.md",
            metadata={"date": "2026-02-20"},
        ),
    ]


# --- UserContextSearchConfig ---


class TestUserContextSearchConfig:
    """Tests for search configuration dataclass."""

    def test_default_top_k(self) -> None:
        config = UserContextSearchConfig()
        assert config.top_k == 5

    def test_custom_top_k(self) -> None:
        config = UserContextSearchConfig(top_k=10)
        assert config.top_k == 10

    def test_default_values(self) -> None:
        config = UserContextSearchConfig()
        assert config.top_k == 5


# --- UserContextSearchResult ---


class TestUserContextSearchResult:
    """Tests for search result dataclass."""

    def test_result_has_chunk(self) -> None:
        chunk = UserContextChunk(
            content="test", layer=1, source_file="identity.md"
        )
        result = UserContextSearchResult(chunk=chunk, score=0.85)
        assert result.chunk is chunk

    def test_result_has_score(self) -> None:
        chunk = UserContextChunk(
            content="test", layer=1, source_file="identity.md"
        )
        result = UserContextSearchResult(chunk=chunk, score=0.85)
        assert result.score == 0.85

    def test_result_chunk_has_layer(self) -> None:
        chunk = UserContextChunk(
            content="test", layer=3, source_file="glossary.md"
        )
        result = UserContextSearchResult(chunk=chunk, score=0.5)
        assert result.chunk.layer == 3

    def test_result_chunk_has_source_file(self) -> None:
        chunk = UserContextChunk(
            content="test",
            layer=5,
            source_file="diary/2026-02-20.md",
            metadata={"date": "2026-02-20"},
        )
        result = UserContextSearchResult(chunk=chunk, score=0.7)
        assert result.chunk.source_file == "diary/2026-02-20.md"
        assert result.chunk.metadata == {"date": "2026-02-20"}


# --- UserContextSearchEngine initialization ---


class TestUserContextSearchEngineInit:
    """Tests for search engine initialization."""

    def test_init_with_embedding_service(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        assert engine is not None

    def test_init_with_custom_config(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        config = UserContextSearchConfig(top_k=3)
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
            config=config,
        )
        assert engine.config.top_k == 3

    def test_init_default_config(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        assert engine.config.top_k == 5


# --- Embedding generation (build_index) ---


class TestBuildIndex:
    """Tests for building the search index (embedding generation)."""

    @pytest.mark.asyncio
    async def test_build_index_generates_embeddings(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """build_index should generate embeddings for all chunks."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        # After building, the engine should have embeddings for all chunks
        assert engine.index_size == len(sample_chunks)

    @pytest.mark.asyncio
    async def test_build_index_empty_chunks(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """build_index with empty list should produce empty index."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index([])
        assert engine.index_size == 0

    @pytest.mark.asyncio
    async def test_build_index_calls_embed_batch(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """build_index should use embed_batch for efficiency."""
        call_count = 0
        original_embed_batch = mock_embedding_service.embed_batch

        async def tracking_embed_batch(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            return await original_embed_batch(texts)

        mock_embedding_service.embed_batch = tracking_embed_batch
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_rebuild_index_replaces_old(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """Rebuilding index should replace the old index completely."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        assert engine.index_size == 5

        # Rebuild with fewer chunks
        fewer = sample_chunks[:2]
        await engine.build_index(fewer)
        assert engine.index_size == 2


# --- Search ---


class TestSearch:
    """Tests for searching user context chunks."""

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """search should return results as UserContextSearchResult."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("Tokyo location")
        assert len(results) > 0
        assert all(isinstance(r, UserContextSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score_desc(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """Results should be sorted by score in descending order."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("AI products")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_respects_top_k(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """search should return at most top_k results."""
        config = UserContextSearchConfig(top_k=2)
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
            config=config,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("anything")
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_on_empty_index(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """search on empty index should return empty list."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index([])
        results = await engine.search("some query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_result_has_score_and_metadata(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """Each result should contain score and chunk metadata."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("diary entry")
        for result in results:
            assert isinstance(result.score, float)
            assert result.chunk.layer >= 1
            assert result.chunk.source_file != ""

    @pytest.mark.asyncio
    async def test_search_with_exact_match_gets_high_score(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """Searching with exact chunk content should yield max score for that chunk."""
        chunks = [
            UserContextChunk(
                content="unique special content ABC123",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="completely different text XYZ789",
                layer=2,
                source_file="values.md",
            ),
        ]
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks)
        # Search with exact content - should be the top result with score 1.0
        results = await engine.search("unique special content ABC123")
        assert len(results) > 0
        assert results[0].chunk.content == "unique special content ABC123"
        assert results[0].score == pytest.approx(1.0, abs=0.001)


# --- Layer filtering ---


class TestLayerFiltering:
    """Tests for filtering search results by layer."""

    @pytest.mark.asyncio
    async def test_search_filter_by_single_layer(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """search with layers filter should only return chunks from those layers."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("query", layers=[1])
        assert all(r.chunk.layer == 1 for r in results)

    @pytest.mark.asyncio
    async def test_search_filter_by_multiple_layers(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """Filtering by multiple layers should return chunks from those layers."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("query", layers=[1, 2])
        assert all(r.chunk.layer in [1, 2] for r in results)

    @pytest.mark.asyncio
    async def test_search_no_filter_returns_all_layers(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """search without layers filter should return chunks from all layers."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("query")
        layers_in_results = {r.chunk.layer for r in results}
        # Should have results from multiple layers
        assert len(layers_in_results) > 1

    @pytest.mark.asyncio
    async def test_search_filter_nonexistent_layer(
        self,
        mock_embedding_service: AsyncMock,
        sample_chunks: list[UserContextChunk],
    ) -> None:
        """Filtering by a layer that doesn't exist should return empty."""
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(sample_chunks)
        results = await engine.search("query", layers=[99])
        assert results == []


# --- Embedding cache (hash-based change detection) ---


class TestEmbeddingCache:
    """Tests for embedding cache with file hash-based change detection."""

    @pytest.mark.asyncio
    async def test_build_index_with_cache_skips_unchanged(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """Unchanged chunks should not be re-embedded."""
        chunks = [
            UserContextChunk(
                content="stable content",
                layer=1,
                source_file="identity.md",
            ),
        ]

        embed_call_count = 0
        original_embed_batch = mock_embedding_service.embed_batch

        async def counting_embed_batch(texts: list[str]) -> list[list[float]]:
            nonlocal embed_call_count
            embed_call_count += len(texts)
            return await original_embed_batch(texts)

        mock_embedding_service.embed_batch = counting_embed_batch

        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks)
        first_count = embed_call_count

        # Build again with same chunks - should use cache
        embed_call_count = 0
        await engine.build_index(chunks)
        assert embed_call_count == 0, "Unchanged chunks should not be re-embedded"

    @pytest.mark.asyncio
    async def test_build_index_cache_detects_content_change(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """Changed chunk content should trigger re-embedding."""
        chunks_v1 = [
            UserContextChunk(
                content="original content",
                layer=1,
                source_file="identity.md",
            ),
        ]
        chunks_v2 = [
            UserContextChunk(
                content="updated content",
                layer=1,
                source_file="identity.md",
            ),
        ]

        embed_call_count = 0
        original_embed_batch = mock_embedding_service.embed_batch

        async def counting_embed_batch(texts: list[str]) -> list[list[float]]:
            nonlocal embed_call_count
            embed_call_count += len(texts)
            return await original_embed_batch(texts)

        mock_embedding_service.embed_batch = counting_embed_batch

        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks_v1)
        first_count = embed_call_count

        # Build again with modified content - should re-embed
        embed_call_count = 0
        await engine.build_index(chunks_v2)
        assert embed_call_count > 0, "Changed content should trigger re-embedding"

    @pytest.mark.asyncio
    async def test_cache_handles_mixed_changed_unchanged(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """When some chunks change and some don't, only changed ones are re-embedded."""
        chunks_v1 = [
            UserContextChunk(
                content="stable chunk",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="will change",
                layer=2,
                source_file="values.md",
            ),
        ]
        chunks_v2 = [
            UserContextChunk(
                content="stable chunk",  # same
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="has changed now",  # different
                layer=2,
                source_file="values.md",
            ),
        ]

        embedded_texts: list[str] = []
        original_embed_batch = mock_embedding_service.embed_batch

        async def tracking_embed_batch(texts: list[str]) -> list[list[float]]:
            embedded_texts.extend(texts)
            return await original_embed_batch(texts)

        mock_embedding_service.embed_batch = tracking_embed_batch

        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks_v1)

        embedded_texts.clear()
        await engine.build_index(chunks_v2)

        # Only the changed chunk should be re-embedded
        assert len(embedded_texts) == 1
        assert "has changed now" in embedded_texts[0]

    @pytest.mark.asyncio
    async def test_cache_handles_new_chunks(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """New chunks added to the index should be embedded."""
        chunks_v1 = [
            UserContextChunk(
                content="existing chunk",
                layer=1,
                source_file="identity.md",
            ),
        ]
        chunks_v2 = [
            UserContextChunk(
                content="existing chunk",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="brand new chunk",
                layer=2,
                source_file="values.md",
            ),
        ]

        embedded_texts: list[str] = []
        original_embed_batch = mock_embedding_service.embed_batch

        async def tracking_embed_batch(texts: list[str]) -> list[list[float]]:
            embedded_texts.extend(texts)
            return await original_embed_batch(texts)

        mock_embedding_service.embed_batch = tracking_embed_batch

        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks_v1)

        embedded_texts.clear()
        await engine.build_index(chunks_v2)

        # Only the new chunk should be embedded
        assert len(embedded_texts) == 1
        assert "brand new chunk" in embedded_texts[0]

    @pytest.mark.asyncio
    async def test_cache_removes_stale_entries(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """Chunks that no longer exist should be removed from cache."""
        chunks_v1 = [
            UserContextChunk(
                content="chunk A",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="chunk B",
                layer=2,
                source_file="values.md",
            ),
        ]
        chunks_v2 = [
            UserContextChunk(
                content="chunk A",
                layer=1,
                source_file="identity.md",
            ),
            # chunk B removed
        ]

        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks_v1)
        assert engine.index_size == 2

        await engine.build_index(chunks_v2)
        assert engine.index_size == 1


# --- Empty chunk handling ---


class TestEmptyChunkHandling:
    """Tests for skipping empty-content chunks in build_index."""

    @pytest.mark.asyncio
    async def test_build_index_skips_empty_content_chunks(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """空の content を持つチャンクはインデックスに含まれない."""
        chunks = [
            UserContextChunk(
                content="valid content",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="",
                layer=2,
                source_file="values.md",
            ),
            UserContextChunk(
                content="another valid",
                layer=3,
                source_file="glossary.md",
            ),
        ]
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks)
        assert engine.index_size == 2

    @pytest.mark.asyncio
    async def test_build_index_skips_whitespace_only_chunks(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """ホワイトスペースのみの content を持つチャンクもスキップされる."""
        chunks = [
            UserContextChunk(
                content="valid content",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="   \n\t  ",
                layer=2,
                source_file="values.md",
            ),
        ]
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks)
        assert engine.index_size == 1

    @pytest.mark.asyncio
    async def test_build_index_all_empty_chunks_produces_empty_index(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """全てのチャンクが空の場合、インデックスは空になる."""
        chunks = [
            UserContextChunk(
                content="",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="   ",
                layer=2,
                source_file="values.md",
            ),
        ]
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks)
        assert engine.index_size == 0

    @pytest.mark.asyncio
    async def test_build_index_does_not_embed_empty_chunks(
        self, mock_embedding_service: AsyncMock
    ) -> None:
        """空チャンクは embedding API に渡されない."""
        embedded_texts: list[str] = []
        original_embed_batch = mock_embedding_service.embed_batch

        async def tracking_embed_batch(texts: list[str]) -> list[list[float]]:
            embedded_texts.extend(texts)
            return await original_embed_batch(texts)

        mock_embedding_service.embed_batch = tracking_embed_batch

        chunks = [
            UserContextChunk(
                content="real content",
                layer=1,
                source_file="identity.md",
            ),
            UserContextChunk(
                content="",
                layer=2,
                source_file="values.md",
            ),
            UserContextChunk(
                content="  ",
                layer=3,
                source_file="glossary.md",
            ),
        ]
        engine = UserContextSearchEngine(
            embedding_service=mock_embedding_service,
        )
        await engine.build_index(chunks)
        # 空文字列は embed_batch に渡されないことを確認
        assert "" not in embedded_texts
        assert "  " not in embedded_texts
        assert "real content" in embedded_texts
