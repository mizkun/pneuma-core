"""UserContextSearchEngine: vector search over user context chunks.

Uses cosine similarity to find chunks relevant to a query.
Supports hash-based caching to avoid re-embedding unchanged chunks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pneuma_core.llm.embedding import EmbeddingService
from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.runtime.user_context import UserContextChunk


@dataclass(frozen=True)
class UserContextSearchConfig:
    """Configuration for user context search."""

    top_k: int = 5


@dataclass
class UserContextSearchResult:
    """A single search result containing a chunk and its relevance score."""

    chunk: UserContextChunk
    score: float


def _chunk_hash(chunk: UserContextChunk) -> str:
    """Compute a content hash for cache invalidation."""
    key = f"{chunk.source_file}:{chunk.layer}:{chunk.content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class UserContextSearchEngine:
    """Vector search engine over user context chunks.

    Builds an in-memory index of chunk embeddings and supports
    cosine-similarity-based retrieval with optional layer filtering.

    Embedding cache: chunks whose content hash hasn't changed
    since the last build_index call are not re-embedded.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: UserContextSearchConfig | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self.config = config or UserContextSearchConfig()

        # Index: list of (chunk, embedding) pairs
        self._index: list[tuple[UserContextChunk, list[float]]] = []

        # Cache: hash -> embedding (persists across build_index calls)
        self._cache: dict[str, list[float]] = {}

    @property
    def index_size(self) -> int:
        """Number of chunks currently in the index."""
        return len(self._index)

    async def build_index(
        self, chunks: list[UserContextChunk]
    ) -> None:
        """Build (or rebuild) the search index from chunks.

        Uses a content-hash cache so that unchanged chunks are not
        re-embedded. Stale cache entries (chunks no longer present)
        are pruned.
        """
        # Filter out chunks with empty or whitespace-only content
        chunks = [c for c in chunks if c.content.strip()]

        if not chunks:
            self._index = []
            self._cache = {}
            return

        # Compute hashes for all incoming chunks
        chunk_hashes = [_chunk_hash(c) for c in chunks]
        current_hash_set = set(chunk_hashes)

        # Identify which chunks need new embeddings
        to_embed: list[tuple[int, str]] = []  # (index, text)
        for i, (chunk, h) in enumerate(zip(chunks, chunk_hashes)):
            if h not in self._cache:
                to_embed.append((i, chunk.content))

        # Batch-embed new/changed chunks
        if to_embed:
            indices, texts = zip(*to_embed)
            embeddings = await self._embedding_service.embed_batch(list(texts))
            for idx, emb in zip(indices, embeddings):
                self._cache[chunk_hashes[idx]] = emb

        # Prune stale cache entries
        stale_keys = set(self._cache.keys()) - current_hash_set
        for key in stale_keys:
            del self._cache[key]

        # Build the index from cache
        self._index = [
            (chunk, self._cache[h])
            for chunk, h in zip(chunks, chunk_hashes)
        ]

    async def search(
        self,
        query: str,
        layers: list[int] | None = None,
    ) -> list[UserContextSearchResult]:
        """Search for chunks relevant to the query.

        Args:
            query: The search query text.
            layers: Optional list of layer numbers to filter by.

        Returns:
            List of UserContextSearchResult sorted by score descending,
            limited to config.top_k results.
        """
        if not self._index:
            return []

        query_embedding = await self._embedding_service.embed(query)

        scored: list[UserContextSearchResult] = []
        for chunk, embedding in self._index:
            # Apply layer filter
            if layers is not None and chunk.layer not in layers:
                continue

            score = cosine_similarity(query_embedding, embedding)
            scored.append(UserContextSearchResult(chunk=chunk, score=score))

        # Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)

        return scored[: self.config.top_k]
