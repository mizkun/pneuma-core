"""MemoryStore protocol: interface for memory persistence."""

from typing import Protocol

from pneuma_core.models.memory import EpisodicMemory, SemanticMemory


class MemoryStore(Protocol):
    """記憶ストアのプロトコル.

    StorageBackend から分離された記憶専用インターフェース。
    記憶はベクトル類似検索を必要とし、他のデータとはアクセスパターンが異なるため独立。
    """

    async def add_episodic(self, memory: EpisodicMemory) -> None:
        """エピソード記憶を追加."""
        ...

    async def add_semantic(self, memory: SemanticMemory) -> None:
        """意味記憶を追加."""
        ...

    async def update_semantic(self, memory: SemanticMemory) -> None:
        """意味記憶を更新."""
        ...

    async def delete_semantic(self, memory_id: str) -> None:
        """意味記憶を削除."""
        ...

    async def get_episodic_by_character(self, character_id: str) -> list[EpisodicMemory]:
        """キャラクターのエピソード記憶を全件取得."""
        ...

    async def get_semantic_by_character(self, character_id: str) -> list[SemanticMemory]:
        """キャラクターの意味記憶を全件取得."""
        ...

    async def find_similar_episodic(
        self, character_id: str, embedding: list[float], threshold: float
    ) -> list[EpisodicMemory]:
        """類似度がしきい値以上のエピソード記憶を検索."""
        ...
