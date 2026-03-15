"""Tests for SessionEndPipeline: セッション終了時の統合分析 (#130)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pneuma_core.llm.adapter import LLMRequest, LLMResponse
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.relation import Relation
from pneuma_core.runtime.session import ConversationSession
from pneuma_core.runtime.session_end_pipeline import SessionEndPipeline, SessionEndResult
from pneuma_core.storage.in_memory import InMemoryStorageBackend

NOW = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)


# --- Helpers ---


def _make_session(messages: list[dict] | None = None) -> ConversationSession:
    """テスト用 ConversationSession を作る."""
    session = ConversationSession(
        session_id="sess-test-001",
        character_id="mira-001",
        user_id="user-001",
        channel_id="ch-001",
        started_at=NOW,
        last_active_at=NOW + timedelta(minutes=10),
    )
    if messages:
        session.messages.extend(messages)
    return session


def _make_llm_response(data: dict) -> LLMResponse:
    """LLM レスポンスを JSON 文字列で構築."""
    return LLMResponse(
        content=json.dumps(data, ensure_ascii=False),
        model="claude-opus-4-6",
        usage={"input_tokens": 100, "output_tokens": 200},
    )


SAMPLE_MESSAGES = [
    {"role": "user", "content": "今日はいい天気だね"},
    {"role": "assistant", "content": "そうだね！散歩に行きたいな"},
    {"role": "user", "content": "じゃあ公園に行こうか"},
    {"role": "assistant", "content": "うん、行こう！楽しみ"},
]


# --- Fixtures ---


@pytest.fixture
def store() -> InMemoryStorageBackend:
    return InMemoryStorageBackend()


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_embedding() -> AsyncMock:
    mock = AsyncMock()
    mock.embed_batch.return_value = [[0.1, 0.2, 0.3]]
    return mock


@pytest.fixture
def pipeline(mock_llm: AsyncMock, mock_embedding: AsyncMock, store: InMemoryStorageBackend) -> SessionEndPipeline:
    return SessionEndPipeline(
        llm=mock_llm,
        embedding_service=mock_embedding,
        memory_store=store,
        storage=store,
    )


# --- Basic Lifecycle ---


class TestSessionEndPipelineLifecycle:
    """パイプラインの基本ライフサイクル."""

    @pytest.mark.asyncio
    async def test_returns_session_end_result(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock
    ) -> None:
        """run() は SessionEndResult を返す."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })
        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(session, "mira-001")
        assert isinstance(result, SessionEndResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_session_skips_llm(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock
    ) -> None:
        """空セッション → LLM 呼ばず空結果."""
        session = _make_session([])  # no messages
        result = await pipeline.run(session, "mira-001")
        assert result.success is True
        assert result.episodic_memories_saved == 0
        assert result.semantic_updates_applied == 0
        assert result.relationship_changes == 0
        mock_llm.generate.assert_not_called()


# --- Episodic Memory Extraction ---


class TestEpisodicMemoryExtraction:
    """エピソード記憶の抽出と保存."""

    @pytest.mark.asyncio
    async def test_extracts_and_saves_episodic_memories(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        mock_embedding: AsyncMock, store: InMemoryStorageBackend,
    ) -> None:
        """Opus がエピソード返却 → DB 保存確認."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [
                {
                    "content": "ユーザーと一緒に公園に散歩に行った",
                    "emotional_valence": 0.7,
                    "importance": 0.6,
                },
            ],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })
        mock_embedding.embed_batch.return_value = [[0.1, 0.2, 0.3]]

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(session, "mira-001")

        assert result.episodic_memories_saved == 1
        memories = await store.get_episodic_memories("mira-001")
        assert len(memories) == 1
        assert "公園" in memories[0].content
        assert memories[0].conversation_id == "sess-test-001"

    @pytest.mark.asyncio
    async def test_embeddings_generated_for_episodic(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        mock_embedding: AsyncMock, store: InMemoryStorageBackend,
    ) -> None:
        """新記憶に embedding 付与確認."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [
                {
                    "content": "散歩の記憶",
                    "emotional_valence": 0.5,
                    "importance": 0.5,
                },
            ],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })
        mock_embedding.embed_batch.return_value = [[0.4, 0.5, 0.6]]

        session = _make_session(SAMPLE_MESSAGES)
        await pipeline.run(session, "mira-001")

        memories = await store.get_episodic_memories("mira-001")
        assert memories[0].embedding == [0.4, 0.5, 0.6]
        mock_embedding.embed_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_episodic_memories(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        mock_embedding: AsyncMock, store: InMemoryStorageBackend,
    ) -> None:
        """複数のエピソード記憶を一度に保存."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [
                {"content": "記憶1", "emotional_valence": 0.3, "importance": 0.5},
                {"content": "記憶2", "emotional_valence": 0.8, "importance": 0.9},
            ],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })
        mock_embedding.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(session, "mira-001")

        assert result.episodic_memories_saved == 2
        memories = await store.get_episodic_memories("mira-001")
        assert len(memories) == 2


# --- Semantic Memory Updates ---


class TestSemanticMemoryUpdates:
    """セマンティック記憶の追加・修正・削除."""

    @pytest.mark.asyncio
    async def test_adds_new_semantic_memory(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        mock_embedding: AsyncMock, store: InMemoryStorageBackend,
    ) -> None:
        """新規セマンティック追加."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [
                {
                    "action": "add",
                    "content": "ユーザーは公園が好き",
                    "confidence": 0.7,
                },
            ],
            "user_context_updates": [],
            "relationship_changes": [],
        })
        mock_embedding.embed_batch.return_value = [[0.1, 0.2, 0.3]]

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(session, "mira-001")

        assert result.semantic_updates_applied == 1
        semantics = await store.get_semantic_memories("mira-001")
        assert len(semantics) == 1
        assert "公園" in semantics[0].content

    @pytest.mark.asyncio
    async def test_modifies_existing_semantic_memory(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        mock_embedding: AsyncMock, store: InMemoryStorageBackend,
    ) -> None:
        """既存セマンティック修正."""
        # 事前に既存記憶を登録
        existing = SemanticMemory(
            id="sem-existing-001", character_id="mira-001",
            content="ユーザーは散歩が好き",
            confidence=0.5,
        )
        await store.save_semantic_memory(existing)

        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [
                {
                    "action": "modify",
                    "memory_id": "sem-existing-001",
                    "content": "ユーザーは公園での散歩が特に好き",
                    "confidence": 0.8,
                },
            ],
            "user_context_updates": [],
            "relationship_changes": [],
        })

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(
            session, "mira-001",
            existing_semantics=[existing],
        )

        assert result.semantic_updates_applied == 1
        semantics = await store.get_semantic_memories("mira-001")
        assert len(semantics) == 1
        assert "公園での散歩" in semantics[0].content
        assert semantics[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_deletes_semantic_memory(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        store: InMemoryStorageBackend,
    ) -> None:
        """既存セマンティック削除."""
        existing = SemanticMemory(
            id="sem-existing-001", character_id="mira-001",
            content="古い情報",
            confidence=0.3,
        )
        await store.save_semantic_memory(existing)

        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [
                {
                    "action": "delete",
                    "memory_id": "sem-existing-001",
                    "reason": "情報が古くなった",
                },
            ],
            "user_context_updates": [],
            "relationship_changes": [],
        })

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(
            session, "mira-001",
            existing_semantics=[existing],
        )

        assert result.semantic_updates_applied == 1
        semantics = await store.get_semantic_memories("mira-001")
        assert len(semantics) == 0


# --- Relationship Changes ---


class TestRelationshipChanges:
    """関係性の更新."""

    @pytest.mark.asyncio
    async def test_updates_relationship_closeness(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        store: InMemoryStorageBackend,
    ) -> None:
        """closeness delta を適用."""
        existing_rel = Relation(
            id="rel-001", owner_id="mira-001", target_id="user-001",
            target_name="きょうへい", relationship_type="partner",
            description="大切なパートナー",
            closeness=0.6, trust=0.7,
            updated_at=NOW,
        )
        await store.save_relation(existing_rel)

        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [
                {
                    "relation_id": "rel-001",
                    "closeness_delta": 0.05,
                    "trust_delta": 0.03,
                },
            ],
        })

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(
            session, "mira-001",
            existing_relations=[existing_rel],
        )

        assert result.relationship_changes == 1
        updated_rel = await store.get_relation("rel-001")
        assert updated_rel is not None
        assert abs(updated_rel.closeness - 0.65) < 1e-10
        assert abs(updated_rel.trust - 0.73) < 1e-10

    @pytest.mark.asyncio
    async def test_clamps_relationship_values(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
        store: InMemoryStorageBackend,
    ) -> None:
        """closeness/trust は 0.0-1.0 にクランプされる."""
        existing_rel = Relation(
            id="rel-001", owner_id="mira-001", target_id="user-001",
            target_name="きょうへい", relationship_type="partner",
            description="大切なパートナー",
            closeness=0.95, trust=0.05,
            updated_at=NOW,
        )
        await store.save_relation(existing_rel)

        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [
                {
                    "relation_id": "rel-001",
                    "closeness_delta": 0.1,   # 0.95 + 0.1 = 1.05 → 1.0
                    "trust_delta": -0.1,       # 0.05 - 0.1 = -0.05 → 0.0
                },
            ],
        })

        session = _make_session(SAMPLE_MESSAGES)
        await pipeline.run(
            session, "mira-001",
            existing_relations=[existing_rel],
        )

        updated_rel = await store.get_relation("rel-001")
        assert updated_rel is not None
        assert updated_rel.closeness == 1.0
        assert updated_rel.trust == 0.0


# --- Error Handling ---


class TestErrorHandling:
    """エラーハンドリング."""

    @pytest.mark.asyncio
    async def test_graceful_failure_on_llm_error(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock
    ) -> None:
        """Opus 失敗 → success=False, クラッシュしない."""
        mock_llm.generate.side_effect = Exception("LLM API error")

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(session, "mira-001")

        assert result.success is False
        assert result.episodic_memories_saved == 0

    @pytest.mark.asyncio
    async def test_graceful_failure_on_invalid_json(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock
    ) -> None:
        """不正 JSON → success=False."""
        mock_llm.generate.return_value = LLMResponse(
            content="これはJSONではない",
            model="claude-opus-4-6",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(session, "mira-001")

        assert result.success is False


# --- LLM Call Verification ---


class TestLLMCallVerification:
    """LLM 呼び出しの検証."""

    @pytest.mark.asyncio
    async def test_llm_called_with_conversation_history(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock
    ) -> None:
        """LLM にセッションの会話履歴が渡される."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })

        session = _make_session(SAMPLE_MESSAGES)
        await pipeline.run(session, "mira-001")

        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args[0][0]
        assert isinstance(call_args, LLMRequest)
        # 会話履歴がプロンプトに含まれている
        prompt_text = call_args.system_prompt + str(call_args.messages)
        assert "公園" in prompt_text

    @pytest.mark.asyncio
    async def test_existing_semantics_included_in_prompt(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock
    ) -> None:
        """既存セマンティック記憶がプロンプトに含まれる."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })

        existing = SemanticMemory(
            id="sem-001", character_id="mira-001",
            content="ユーザーはコーヒーが好き",
            confidence=0.8,
        )

        session = _make_session(SAMPLE_MESSAGES)
        await pipeline.run(
            session, "mira-001",
            existing_semantics=[existing],
        )

        call_args = mock_llm.generate.call_args[0][0]
        prompt_text = call_args.system_prompt
        assert "コーヒー" in prompt_text


# --- UserContext Integration (#136) ---


class TestUserContextIntegration:
    """UserContext 自動更新の統合テスト (#136)."""

    @pytest.mark.asyncio
    async def test_user_context_updates_applied(
        self, mock_llm: AsyncMock, mock_embedding: AsyncMock,
        store: InMemoryStorageBackend, tmp_path: Path,
    ) -> None:
        """user_context_updates が _apply_user_context() に渡される."""
        ctx_dir = tmp_path / "user_context"
        ctx_dir.mkdir()
        (ctx_dir / "identity.md").write_text(
            "# プロフィール\n\n## 仕事\n\n- 職業: エンジニア\n",
            encoding="utf-8",
        )

        pipeline = SessionEndPipeline(
            llm=mock_llm,
            embedding_service=mock_embedding,
            memory_store=store,
            storage=store,
            user_context_dir=str(ctx_dir),
        )

        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [
                {
                    "file": "identity.md",
                    "section": "## 仕事",
                    "action": "rewrite",
                    "new_content": "- 職業: VP of AI",
                    "reason": "転職を反映",
                }
            ],
            "relationship_changes": [],
        })

        session = _make_session(SAMPLE_MESSAGES)

        with patch("pneuma_core.runtime.user_context_writer.subprocess.run"):
            result = await pipeline.run(session, "mira-001")

        assert result.user_context_updates == 1

        content = (ctx_dir / "identity.md").read_text(encoding="utf-8")
        assert "VP of AI" in content

    @pytest.mark.asyncio
    async def test_user_context_files_in_llm_prompt(
        self, mock_llm: AsyncMock, mock_embedding: AsyncMock,
        store: InMemoryStorageBackend, tmp_path: Path,
    ) -> None:
        """UserContext ファイルが LLM プロンプトに含まれる."""
        ctx_dir = tmp_path / "user_context"
        ctx_dir.mkdir()
        (ctx_dir / "identity.md").write_text(
            "# プロフィール\n\n## 基本情報\n\n- 名前: テスト太郎\n",
            encoding="utf-8",
        )
        (ctx_dir / "values.md").write_text(
            "# 価値観\n\n## 大切にしていること\n\n- 誠実さ\n",
            encoding="utf-8",
        )

        pipeline = SessionEndPipeline(
            llm=mock_llm,
            embedding_service=mock_embedding,
            memory_store=store,
            storage=store,
            user_context_dir=str(ctx_dir),
        )

        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })

        session = _make_session(SAMPLE_MESSAGES)
        await pipeline.run(session, "mira-001")

        call_args = mock_llm.generate.call_args[0][0]
        prompt_text = call_args.system_prompt
        # UserContext ファイルの内容がプロンプトに含まれている
        assert "テスト太郎" in prompt_text
        assert "誠実さ" in prompt_text

    @pytest.mark.asyncio
    async def test_no_user_context_dir_skips_updates(
        self, pipeline: SessionEndPipeline, mock_llm: AsyncMock,
    ) -> None:
        """user_context_dir 未設定時は user_context_updates=0."""
        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [
                {
                    "file": "identity.md",
                    "section": "## 仕事",
                    "action": "rewrite",
                    "new_content": "- 職業: CTO",
                    "reason": "昇進",
                }
            ],
            "relationship_changes": [],
        })

        session = _make_session(SAMPLE_MESSAGES)
        result = await pipeline.run(session, "mira-001")

        # user_context_dir が None なので更新はスキップされる
        assert result.user_context_updates == 0

    @pytest.mark.asyncio
    async def test_user_context_updates_schema_in_prompt(
        self, mock_llm: AsyncMock, mock_embedding: AsyncMock,
        store: InMemoryStorageBackend, tmp_path: Path,
    ) -> None:
        """user_context_updates のスキーマがプロンプトに含まれる."""
        ctx_dir = tmp_path / "user_context"
        ctx_dir.mkdir()

        pipeline = SessionEndPipeline(
            llm=mock_llm,
            embedding_service=mock_embedding,
            memory_store=store,
            storage=store,
            user_context_dir=str(ctx_dir),
        )

        mock_llm.generate.return_value = _make_llm_response({
            "episodic_memories": [],
            "semantic_updates": [],
            "user_context_updates": [],
            "relationship_changes": [],
        })

        session = _make_session(SAMPLE_MESSAGES)
        await pipeline.run(session, "mira-001")

        call_args = mock_llm.generate.call_args[0][0]
        prompt_text = call_args.system_prompt
        # スキーマがプロンプトに含まれている
        assert "rewrite" in prompt_text
        assert "append" in prompt_text
        assert "user_context_updates" in prompt_text
