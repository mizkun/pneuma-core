"""Semantic memory consolidation: episodic -> semantic generalization.

Clusters related episodic memories and uses LLM to extract generalized
knowledge (semantic memories) from them.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.llm.embedding import EmbeddingService
from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory

CONSOLIDATION_SYSTEM_PROMPT = """\
あなたは記憶の統合を行うシステムです。
以下のエピソード記憶群から共通するパターン・理解・一般的な知識を抽出してください。

エピソード:
{episodes}

以下のJSON形式で応答してください:
{{
  "content": "統合された理解（1-2文）",
  "confidence": 0.0-1.0（確信度。エピソード数や一貫性から判断）
}}
"""


@dataclass(frozen=True)
class SemanticConsolidationConfig:
    """Tunable parameters for semantic consolidation."""

    min_episodes: int = 3
    similarity_threshold: float = 0.7
    duplicate_threshold: float = 0.9
    model: str = "claude-haiku-4-5-20251001"
    max_semantics_per_batch: int = 5


class SemanticConsolidator:
    """Consolidate episodic memories into semantic memories.

    Pipeline:
        1. Filter episodes without embeddings
        2. Cluster related episodes by embedding similarity
        3. Filter clusters with < min_episodes
        4. For each cluster, call LLM to generate semantic memory
        5. Check for duplicates against existing semantics
        6. Embed the new semantic memories
        7. Return new SemanticMemory objects
    """

    def __init__(
        self,
        llm: LLMAdapter,
        embedding_service: EmbeddingService,
        config: SemanticConsolidationConfig | None = None,
    ) -> None:
        self.llm = llm
        self.embedding_service = embedding_service
        self.config = config or SemanticConsolidationConfig()

    async def consolidate(
        self,
        character_id: str,
        episodes: list[EpisodicMemory],
        existing_semantics: list[SemanticMemory],
    ) -> list[SemanticMemory]:
        """Consolidate episodic memories into semantic memories.

        1. Cluster related episodes by embedding similarity
        2. Filter clusters with < min_episodes
        3. For each cluster, call LLM to generate semantic memory
        4. Check for duplicates against existing semantics
        5. Embed the new semantic memories
        6. Return new SemanticMemory objects
        """
        # Filter episodes without embeddings
        episodes_with_emb = [ep for ep in episodes if ep.embedding is not None]
        if not episodes_with_emb:
            return []

        # 1. Cluster related episodes
        raw_clusters = self._cluster_episodes(episodes_with_emb)

        # 2. Filter clusters below min_episodes
        valid_clusters = [
            c for c in raw_clusters if len(c) >= self.config.min_episodes
        ]
        if not valid_clusters:
            return []

        # Limit to max_semantics_per_batch
        valid_clusters = valid_clusters[: self.config.max_semantics_per_batch]

        # 3-6. Process each cluster
        results: list[SemanticMemory] = []
        for cluster in valid_clusters:
            semantic = await self._process_cluster(
                character_id, cluster, existing_semantics
            )
            if semantic is not None:
                results.append(semantic)
                # Add to existing_semantics to prevent duplicates within batch
                existing_semantics = [*existing_semantics, semantic]

        return results

    def _cluster_episodes(
        self, episodes: list[EpisodicMemory]
    ) -> list[list[EpisodicMemory]]:
        """Cluster episodes by embedding similarity using greedy merging.

        Algorithm:
        1. Build adjacency: for each pair, check if similarity > threshold
        2. Use Union-Find to merge overlapping clusters
        3. Return list of clusters
        """
        n = len(episodes)
        if n == 0:
            return []

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build adjacency and merge
        threshold = self.config.similarity_threshold
        for i in range(n):
            for j in range(i + 1, n):
                assert episodes[i].embedding is not None
                assert episodes[j].embedding is not None
                sim = cosine_similarity(episodes[i].embedding, episodes[j].embedding)
                if sim >= threshold:
                    union(i, j)

        # Group by root
        clusters_map: dict[int, list[EpisodicMemory]] = {}
        for i in range(n):
            root = find(i)
            if root not in clusters_map:
                clusters_map[root] = []
            clusters_map[root].append(episodes[i])

        return list(clusters_map.values())

    async def _process_cluster(
        self,
        character_id: str,
        cluster: list[EpisodicMemory],
        existing_semantics: list[SemanticMemory],
    ) -> SemanticMemory | None:
        """Process a single cluster: LLM consolidation, dedup, embedding."""
        # Call LLM to generate semantic memory
        try:
            episodes_text = "\n".join(
                f"{i + 1}. {ep.content}" for i, ep in enumerate(cluster)
            )
            system_prompt = CONSOLIDATION_SYSTEM_PROMPT.format(episodes=episodes_text)

            request = LLMRequest(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": "上記のエピソードを統合してください。"}],
                model=self.config.model,
                temperature=0.3,
                max_tokens=512,
            )
            response = await self.llm.generate(request)
        except Exception:
            return None

        # Parse LLM response
        parsed = self._parse_llm_response(response.content)
        if parsed is None:
            return None

        content = parsed["content"]
        confidence = float(parsed["confidence"])

        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))

        # Generate embedding for the new semantic memory
        try:
            embedding = await self.embedding_service.embed(content)
        except Exception:
            return None

        # Check for duplicates against existing semantics
        for existing in existing_semantics:
            if existing.embedding is not None:
                sim = cosine_similarity(embedding, existing.embedding)
                if sim >= self.config.duplicate_threshold:
                    return None

        # Build and return SemanticMemory
        source_ids = [ep.id for ep in cluster]
        return SemanticMemory(
            id=f"sem-{uuid.uuid4().hex[:12]}",
            character_id=character_id,
            content=content,
            confidence=confidence,
            source_episode_ids=source_ids,
            embedding=embedding,
        )

    def _parse_llm_response(self, response: str) -> dict | None:
        """Parse LLM response JSON, handling markdown code blocks."""
        text = response.strip()

        # Strip markdown code blocks
        md_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
        if md_match:
            text = md_match.group(1).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(data, dict):
            return None
        if "content" not in data or "confidence" not in data:
            return None

        return data
