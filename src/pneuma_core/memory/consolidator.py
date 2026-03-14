"""Memory consolidation: selective episode extraction from conversations."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest, LLMResponse
from pneuma_core.llm.embedding import EmbeddingService
from pneuma_core.memory.store import MemoryStore
from pneuma_core.models.memory import EpisodicMemory

EXTRACTION_SYSTEM_PROMPT = """\
あなたは会話からエピソード記憶を抽出するアシスタントです。
以下の会話履歴を読み、キャラクターにとって記憶すべき重要な出来事を抽出してください。

出力は JSON 配列で、各要素は以下の形式です:
[
  {
    "content": "記憶する出来事の説明（1〜2文）",
    "importance": 0.0〜1.0の数値,
    "emotional_valence": -1.0〜1.0の数値（不快〜快）
  }
]

重要度の基準:
- 0.9-1.0: 人生を変える出来事、決定的な約束
- 0.7-0.8: 強く印象に残る、重要情報の開示
- 0.6: 記憶の最低ライン
- 0.6未満: 保存しない些細な出来事（含めてよいが重要度を正直に）

JSON 配列のみを出力してください。説明文は不要です。
"""


@dataclass(frozen=True)
class ConsolidationConfig:
    """Tunable parameters for memory consolidation."""

    importance_threshold: float = 0.6
    max_episodes_per_conversation: int = 3
    duplicate_similarity_threshold: float = 0.95


@dataclass
class ConsolidationResult:
    """Result of memory consolidation."""

    saved: list[EpisodicMemory] = field(default_factory=list)
    filtered_by_importance: int = 0
    filtered_by_duplicate: int = 0
    filtered_by_limit: int = 0


class MemoryConsolidator:
    """Selective episode extraction and storage engine.

    Pipeline:
        1. LLM extracts episodes from conversation history
        2. Filter by importance threshold
        3. Limit quantity (top N by importance)
        4. Check duplicates against existing memories
        5. Generate embeddings and save to store
    """

    def __init__(
        self,
        llm: LLMAdapter,
        embedding_service: EmbeddingService,
        store: MemoryStore,
        config: ConsolidationConfig | None = None,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.embedding_service = embedding_service
        self.store = store
        self.config = config or ConsolidationConfig()
        self._model = model

    async def consolidate(
        self,
        character_id: str,
        conversation_history: list[dict],
        now: datetime,
        conversation_id: str | None = None,
    ) -> ConsolidationResult:
        """Extract, filter, and save episodic memories from a conversation."""
        result = ConsolidationResult()

        # 1. LLM でエピソード抽出
        raw_episodes = await self._extract_episodes(conversation_history)
        if not raw_episodes:
            return result

        # 2. 重要度フィルタ
        filtered = []
        for ep in raw_episodes:
            if ep["importance"] >= self.config.importance_threshold:
                filtered.append(ep)
            else:
                result.filtered_by_importance += 1

        if not filtered:
            return result

        # 3. 数量制限（importance 降順で top N）
        filtered.sort(key=lambda x: x["importance"], reverse=True)
        max_n = self.config.max_episodes_per_conversation
        if len(filtered) > max_n:
            result.filtered_by_limit = len(filtered) - max_n
            filtered = filtered[:max_n]

        # 4. Embedding 生成
        contents = [ep["content"] for ep in filtered]
        embeddings = await self.embedding_service.embed_batch(contents)

        # 5. 重複チェック（find_similar_episodic 経由）
        non_duplicate = []
        for ep, emb in zip(filtered, embeddings):
            similar = await self.store.find_similar_episodic(
                character_id, emb, self.config.duplicate_similarity_threshold
            )
            if similar:
                result.filtered_by_duplicate += 1
            else:
                non_duplicate.append((ep, emb))

        # 6. EpisodicMemory 作成と保存
        for ep, emb in non_duplicate:
            memory = EpisodicMemory(
                id=f"ep-{uuid.uuid4().hex[:12]}",
                character_id=character_id,
                content=ep["content"],
                timestamp=now,
                emotional_valence=ep.get("emotional_valence", 0.0),
                importance=ep["importance"],
                conversation_id=conversation_id,
                embedding=emb,
            )
            await self.store.add_episodic(memory)
            result.saved.append(memory)

        return result

    async def _extract_episodes(
        self, conversation_history: list[dict]
    ) -> list[dict]:
        """Use LLM to extract episodes from conversation history."""
        request = LLMRequest(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            messages=conversation_history,
            model=self._model,
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            response: LLMResponse = await self.llm.generate(request)
        except Exception:
            return []

        try:
            episodes = json.loads(response.content)
            if not isinstance(episodes, list):
                return []
            return [ep for ep in episodes if self._validate_episode(ep)]
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _validate_episode(ep: object) -> bool:
        """Validate that an episode dict has required fields with correct types."""
        if not isinstance(ep, dict):
            return False
        if "content" not in ep or not isinstance(ep["content"], str):
            return False
        if "importance" not in ep:
            return False
        try:
            importance = float(ep["importance"])
        except (TypeError, ValueError):
            return False
        if not (0.0 <= importance <= 1.0):
            return False
        if "emotional_valence" in ep:
            try:
                valence = float(ep["emotional_valence"])
            except (TypeError, ValueError):
                return False
            if not (-1.0 <= valence <= 1.0):
                return False
        return True

