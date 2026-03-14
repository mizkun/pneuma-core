"""SessionEndPipeline: セッション終了時の統合分析 (#130).

Opus 1回で episodic/semantic/relationship を統合更新する。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pneuma_core.llm.adapter import LLMAdapter
    from pneuma_core.llm.embedding import EmbeddingService
    from pneuma_core.models.memory import SemanticMemory
    from pneuma_core.models.relation import Relation
    from pneuma_core.runtime.session import ConversationSession
    from pneuma_core.storage.backend import StorageBackend

from pathlib import Path

from pneuma_core.llm.adapter import LLMRequest
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory as SemanticMemoryModel

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

_MD_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)

_SYSTEM_PROMPT = """\
あなたはAIキャラクターの記憶管理アシスタントです。
以下の会話セッションを分析し、保存すべき情報をJSON形式で出力してください。

## 出力フォーマット
```json
{
  "episodic_memories": [
    {
      "content": "具体的な出来事の記述",
      "emotional_valence": -1.0〜1.0,
      "importance": 0.0〜1.0
    }
  ],
  "semantic_updates": [
    {
      "action": "add" | "modify" | "delete",
      "content": "汎化された知識（add/modify時）",
      "confidence": 0.0〜1.0（add/modify時）,
      "memory_id": "既存ID（modify/delete時）",
      "reason": "変更理由（delete時）"
    }
  ],
  "user_context_updates": [
    {
      "file": "identity.md | values.md | core_experiences.md | glossary.md | relationships.md | projects/*.md",
      "section": "## セクション名（rewrite時のみ必須）",
      "action": "rewrite | append",
      "new_content": "新しいセクション内容またはファイル末尾に追記する内容",
      "reason": "更新理由の簡潔な説明"
    }
  ],
  "relationship_changes": [
    {
      "relation_id": "関係性ID",
      "closeness_delta": -0.1〜0.1,
      "trust_delta": -0.1〜0.1
    }
  ]
}
```

## ルール
- episodic_memories: このセッションで起きた重要な出来事を記録。些細すぎる内容は省略。
- semantic_updates: 会話から得られた汎化的な知識。既存知識の修正や、古くなった知識の削除も含む。
- user_context_updates: ユーザーの生活・仕事・趣味・価値観などに関する重要な変化があった場合のみ更新。些細な変化は無視する。rewrite はマークダウンセクションを丸ごと書き換える。append はファイル末尾に追記する。
- relationship_changes: 会話の内容から関係性の変化を判断。変化がなければ空配列。
- 該当がない場合は空配列を返すこと。
"""


@dataclass
class SessionEndResult:
    """セッション終了パイプラインの実行結果."""

    success: bool
    episodic_memories_saved: int = 0
    semantic_updates_applied: int = 0
    user_context_updates: int = 0
    relationship_changes: int = 0


class SessionEndPipeline:
    """セッション終了時の統合分析パイプライン.

    Opus 1回呼び出しで4つの出力を生成し、各ストアに適用する。
    """

    def __init__(
        self,
        llm: LLMAdapter,
        embedding_service: EmbeddingService,
        memory_store: StorageBackend,
        storage: StorageBackend,
        user_context_dir: str | None = None,
        model: str = "claude-opus-4-6",
    ) -> None:
        self._llm = llm
        self._embedding = embedding_service
        self._memory_store = memory_store
        self._storage = storage
        self._user_context_dir = user_context_dir
        self._model = model

    async def run(
        self,
        session: ConversationSession,
        character_id: str,
        existing_semantics: list[SemanticMemory] | None = None,
        existing_relations: list[Relation] | None = None,
    ) -> SessionEndResult:
        """セッション終了分析を実行する."""
        # Empty session → skip LLM
        if not session.messages:
            return SessionEndResult(success=True)

        try:
            analysis = await self._call_llm(
                session, character_id, existing_semantics, existing_relations,
            )
        except Exception:
            logger.warning("Session-end pipeline LLM call failed", exc_info=True)
            return SessionEndResult(success=False)

        # Apply results
        ep_count = await self._apply_episodic(
            analysis.get("episodic_memories", []),
            character_id,
            session.session_id,
        )
        sem_count = await self._apply_semantic(
            analysis.get("semantic_updates", []),
            character_id,
        )
        uc_count = await self._apply_user_context(
            analysis.get("user_context_updates", []),
        )
        rel_count = await self._apply_relationships(
            analysis.get("relationship_changes", []),
            existing_relations or [],
        )

        return SessionEndResult(
            success=True,
            episodic_memories_saved=ep_count,
            semantic_updates_applied=sem_count,
            user_context_updates=uc_count,
            relationship_changes=rel_count,
        )

    async def _call_llm(
        self,
        session: ConversationSession,
        character_id: str,
        existing_semantics: list[SemanticMemory] | None,
        existing_relations: list[Relation] | None,
    ) -> dict:
        """LLM を呼び出して分析結果を取得."""
        system_prompt = _SYSTEM_PROMPT

        # Inject UserContext files
        if self._user_context_dir:
            uc_text = self._load_user_context_text()
            if uc_text:
                system_prompt += "\n\n## ユーザーコンテキスト（現在の内容）\n" + uc_text

        # Inject existing semantics
        if existing_semantics:
            sem_lines = [
                f"- [{m.id}] {m.content} (confidence={m.confidence})"
                for m in existing_semantics
            ]
            system_prompt += "\n\n## 既存のセマンティック記憶\n" + "\n".join(sem_lines)

        # Inject existing relations
        if existing_relations:
            rel_lines = [
                f"- [{r.id}] {r.target_name} ({r.relationship_type}): "
                f"closeness={r.closeness}, trust={r.trust}"
                for r in existing_relations
            ]
            system_prompt += "\n\n## 既存の関係性\n" + "\n".join(rel_lines)

        request = LLMRequest(
            system_prompt=system_prompt,
            messages=session.messages,
            model=self._model,
            temperature=0.3,
            max_tokens=2048,
        )

        response = await self._llm.generate(request)
        return self._parse_json(response.content)

    @staticmethod
    def _parse_json(content: str) -> dict:
        """LLM 応答から JSON をパースする."""
        # Strip markdown code blocks
        match = _MD_CODE_BLOCK_RE.match(content.strip())
        if match:
            content = match.group(1)

        return json.loads(content)

    async def _apply_episodic(
        self,
        memories: list[dict],
        character_id: str,
        session_id: str,
    ) -> int:
        """エピソード記憶を保存."""
        if not memories:
            return 0

        contents = [m["content"] for m in memories]
        embeddings = await self._embedding.embed_batch(contents)

        now = datetime.now(JST)
        count = 0
        for mem_data, embedding in zip(memories, embeddings):
            memory = EpisodicMemory(
                id=f"ep-{uuid.uuid4().hex[:12]}",
                character_id=character_id,
                content=mem_data["content"],
                timestamp=now,
                emotional_valence=float(mem_data.get("emotional_valence", 0.0)),
                importance=float(mem_data.get("importance", 0.5)),
                conversation_id=session_id,
                embedding=embedding,
            )
            await self._memory_store.save_episodic_memory(memory)
            count += 1

        return count

    async def _apply_semantic(
        self,
        updates: list[dict],
        character_id: str,
    ) -> int:
        """セマンティック記憶を追加/更新/削除."""
        count = 0
        for update in updates:
            action = update.get("action", "add")

            if action == "add":
                content = update["content"]
                embeddings = await self._embedding.embed_batch([content])
                memory = SemanticMemoryModel(
                    id=f"sem-{uuid.uuid4().hex[:12]}",
                    character_id=character_id,
                    content=content,
                    confidence=float(update.get("confidence", 0.5)),
                    embedding=embeddings[0],
                )
                await self._memory_store.save_semantic_memory(memory)
                count += 1

            elif action == "modify":
                memory_id = update["memory_id"]
                content = update["content"]
                confidence = float(update.get("confidence", 0.5))
                updated = SemanticMemoryModel(
                    id=memory_id,
                    character_id=character_id,
                    content=content,
                    confidence=confidence,
                )
                await self._memory_store.update_semantic_memory(updated)
                count += 1

            elif action == "delete":
                memory_id = update["memory_id"]
                await self._memory_store.delete_semantic_memory(memory_id)
                count += 1

        return count

    def _load_user_context_text(self) -> str:
        """UserContext ディレクトリからファイルを読み込みテキスト化."""
        if not self._user_context_dir:
            return ""

        uc_dir = Path(self._user_context_dir)
        if not uc_dir.exists() or not uc_dir.is_dir():
            return ""

        parts: list[str] = []

        # Read top-level .md files
        for md_file in sorted(uc_dir.iterdir()):
            if md_file.is_file() and md_file.suffix == ".md":
                try:
                    content = md_file.read_text(encoding="utf-8")
                    parts.append(f"### {md_file.name}\n{content}")
                except OSError:
                    continue

        # Read projects/ subdirectory
        projects_dir = uc_dir / "projects"
        if projects_dir.exists() and projects_dir.is_dir():
            for md_file in sorted(projects_dir.iterdir()):
                if md_file.is_file() and md_file.suffix == ".md":
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        parts.append(f"### projects/{md_file.name}\n{content}")
                    except OSError:
                        continue

        return "\n\n".join(parts)

    async def _apply_user_context(self, updates: list[dict]) -> int:
        """UserContext 更新を適用する."""
        if not updates or not self._user_context_dir:
            return 0

        from pneuma_core.runtime.user_context_writer import UserContextWriter

        writer = UserContextWriter(self._user_context_dir)
        try:
            return await writer.apply(updates)
        except Exception:
            logger.warning("UserContext update failed", exc_info=True)
            return 0

    async def _apply_relationships(
        self,
        changes: list[dict],
        existing_relations: list[Relation],
    ) -> int:
        """関係性の delta を適用."""
        if not changes:
            return 0

        rel_map = {r.id: r for r in existing_relations}
        count = 0

        for change in changes:
            rel_id = change["relation_id"]
            rel = rel_map.get(rel_id)
            if rel is None:
                continue

            closeness_delta = float(change.get("closeness_delta", 0.0))
            trust_delta = float(change.get("trust_delta", 0.0))

            new_closeness = max(0.0, min(1.0, rel.closeness + closeness_delta))
            new_trust = max(0.0, min(1.0, rel.trust + trust_delta))

            rel.closeness = new_closeness
            rel.trust = new_trust
            rel.updated_at = datetime.now(JST)

            await self._storage.save_relation(rel)
            count += 1

        return count
