"""正史の連続性 + RAG 想起の結合テスト (Issue #31).

invariants:
- canon-continuity: storage を渡すと、セッション終了で各キャラの記憶/感情/関係が
  キャラ ID 単位で永続化され、次セッション開始時にロードされる（正史は 1 本）。
- memory-rag-recall: storage を渡すと、発話プロンプトの想起記憶は episodic の
  単純スライスではなく MemorySearchEngine の性格バイアス検索結果を使う。

後方互換: storage=None なら従来どおり in-memory のみ（既存挙動を壊さない）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.memory import EpisodicMemory
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession
from pneuma_core.multi_agent.session_end import MultiAgentSessionEndPipeline
from pneuma_core.storage.in_memory import InMemoryStorageBackend


# ── テスト用の決定論的 embedding ──────────────────────────────────────
# キーワードの有無を次元に符号化した素朴なベクトルで、トピック関連性を
# 決定論的に検証できるようにする（real LLM / OpenAI 不要）。
_VOCAB = ["キャンプ", "カレー", "テント", "焚き火", "宿題", "テスト", "勉強"]


class _KeywordEmbedding:
    """語彙の出現有無を 0/1 で並べた決定論的な擬似 embedding."""

    async def embed(self, text: str) -> list[float]:
        return [1.0 if w in (text or "") else 0.0 for w in _VOCAB]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _make_char(
    char_id: str, name: str, extraversion: float = 0.6
) -> Character:
    return Character(
        id=char_id,
        name=name,
        personality=Personality(
            openness=0.6, conscientiousness=0.5, extraversion=extraversion,
            agreeableness=0.7, neuroticism=0.4,
        ),
        values=Values(
            self_transcendence=0.5, self_enhancement=0.5,
            openness_to_change=0.5, conservation=0.5,
        ),
        profile="テストキャラ",
        speaking_style="普通",
    )


def _make_session(
    chars: list[Character],
    storage: InMemoryStorageBackend | None,
    embedding,
    *,
    turn_limit: int = 6,
    seed: int = 7,
) -> MultiAgentSession:
    conv = Conversation(participants=chars)
    cb = CircuitBreaker(CircuitBreakerConfig(
        max_turns=turn_limit, max_elapsed_seconds=600,
    ))
    cb.start()
    return MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=seed),
        shared_context="部室で雑談",
        circuit_breaker=cb,
        storage=storage,
        embedding_service=embedding,
    )


async def _run_all_turns(session: MultiAgentSession) -> None:
    while True:
        result = await session.run_turn()
        if result is None:
            break


# ── 段階1: 正史の連続性（セッション間引き継ぎ）─────────────────────────

@pytest.mark.asyncio
async def test_canon_continuity_persists_and_reloads_state() -> None:
    """AC-1 (canon-continuity): storage を渡すと、セッション A の記憶/感情/関係が
    永続化され、同じ storage で開始したセッション B のキャラに引き継がれる.
    """
    storage = InMemoryStorageBackend()
    embedding = _KeywordEmbedding()
    chars = [
        _make_char("nade", "なでしこ"),
        _make_char("chia", "千明"),
        _make_char("aoi", "あおい"),
    ]
    for ch in chars:
        await storage.save_character(ch)

    # ── セッション A: 会話して出来事を作る → SessionEnd で永続化 ──
    session_a = _make_session(chars, storage, embedding)
    await _run_all_turns(session_a)

    pipeline = MultiAgentSessionEndPipeline(llm=MockLLMAdapter(seed=99))
    await pipeline.run(session_a)
    # SessionEnd の永続化（正史を storage に書き戻す）
    await session_a.persist_canon()

    # storage に各キャラの状態が積まれている（キャラ ID 単位）
    saved_emotions = {
        ch.id: await storage.get_emotional_state(ch.id) for ch in chars
    }
    assert any(e is not None for e in saved_emotions.values()), (
        "セッション終了後、最終 emotion が storage に永続化されていない"
    )

    spoke_ids = [
        ch.id for ch in chars
        if session_a.character_states[ch.id].speak_count > 0
    ]
    assert spoke_ids, "テスト前提: 誰かは発話しているはず"

    # 発話したキャラは episodic を storage に持つ（正史が積み上がる）
    for cid in spoke_ids:
        eps = await storage.get_episodic_memories(cid)
        assert eps, f"{cid} の episodic が storage に永続化されていない"

    # 関係性も Relation として永続化される
    all_relations = await storage.list_relations()
    assert all_relations, "関係性が Relation として永続化されていない"

    # ── セッション B: 同じ storage で開始 → 前回状態がロードされる ──
    chars_b = [
        _make_char("nade", "なでしこ"),
        _make_char("chia", "千明"),
        _make_char("aoi", "あおい"),
    ]
    session_b = _make_session(chars_b, storage, embedding)
    await session_b.load_canon()

    # セッション A の最終 emotion がセッション B の初期 emotion に引き継がれる
    for cid, saved in saved_emotions.items():
        if saved is None:
            continue
        st = session_b.character_states[cid]
        assert st.emotion.pleasure == pytest.approx(saved.pleasure)
        assert st.emotion.arousal == pytest.approx(saved.arousal)
        assert st.emotion.dominance == pytest.approx(saved.dominance)
        assert st.emotion.emotion_label == saved.emotion_label

    # セッション A の episodic がセッション B のキャラに引き継がれる
    for cid in spoke_ids:
        saved_eps = await storage.get_episodic_memories(cid)
        loaded = session_b.character_states[cid].episodic
        assert loaded, f"{cid} の前回 episodic がセッション B にロードされていない"
        saved_contents = {m.content for m in saved_eps}
        assert any(item in saved_contents for item in loaded), (
            f"{cid} のロードされた episodic が storage の内容と一致しない"
        )

    # セッション A の関係性がセッション B のキャラに引き継がれる
    for cid in spoke_ids:
        rels_b = session_b.character_states[cid].relations
        assert rels_b, f"{cid} の関係性がセッション B にロードされていない"


@pytest.mark.asyncio
async def test_storage_none_keeps_inmemory_behavior() -> None:
    """AC-3 (後方互換): storage=None なら従来どおり in-memory のみで動く.

    load_canon / persist_canon は no-op で、例外なく完走する。
    """
    chars = [
        _make_char("nade", "なでしこ"),
        _make_char("chia", "千明"),
    ]
    conv = Conversation(participants=chars)
    cb = CircuitBreaker(CircuitBreakerConfig(max_turns=4, max_elapsed_seconds=600))
    cb.start()
    session = MultiAgentSession(
        conversation=conv,
        llm=MockLLMAdapter(seed=3),
        shared_context="部室で雑談",
        circuit_breaker=cb,
    )

    # storage 未指定でも load/persist は no-op で安全
    await session.load_canon()
    await _run_all_turns(session)
    await session.persist_canon()

    # 従来どおりセッション内に状態が存在する
    snap = session.snapshot()
    assert snap["turn_index"] > 0
    assert len(snap["characters"]) == 2


# ── 段階2: RAG 想起（発話プロンプトへの記憶注入）──────────────────────

@pytest.mark.asyncio
async def test_rag_recall_uses_memory_search_not_slice() -> None:
    """AC-2 (memory-rag-recall): storage を渡すと、想起記憶は episodic スライス
    ではなく MemorySearchEngine の性格バイアス検索の結果になる.

    話題に強く関連する過去記憶（キャンプ）は recall されるが、関連しない記憶
    （宿題）は recall されない（単純な直近 N 件スライスでは区別できない）。
    """
    storage = InMemoryStorageBackend()
    embedding = _KeywordEmbedding()
    chars = [
        _make_char("nade", "なでしこ"),
        _make_char("chia", "千明"),
    ]
    for ch in chars:
        await storage.save_character(ch)

    now = datetime.now(timezone.utc)
    # なでしこに、トピック関連（キャンプ）と無関連（宿題）の過去記憶を仕込む
    relevant = EpisodicMemory(
        id="ep-camp",
        character_id="nade",
        content="前回みんなでキャンプの計画を立てて盛り上がった",
        timestamp=now,
        emotional_valence=0.7,
        importance=0.8,
        embedding=await embedding.embed("キャンプ テント 焚き火"),
    )
    irrelevant = EpisodicMemory(
        id="ep-homework",
        character_id="nade",
        content="先週は宿題のテスト勉強に追われていた",
        timestamp=now,
        emotional_valence=-0.3,
        importance=0.6,
        embedding=await embedding.embed("宿題 テスト 勉強"),
    )
    await storage.save_episodic_memory(relevant)
    await storage.save_episodic_memory(irrelevant)

    session = _make_session(chars, storage, embedding, turn_limit=8)
    await session.load_canon()

    # キャンプ話題で会話を駆動する
    session.shared_context = "今度のキャンプ、テントどうしよう"

    recalled_for_nade: list[str] = []
    for _ in range(8):
        result = await session.run_turn()
        if result is None:
            break
        if result.speaker.id == "nade":
            recalled_for_nade.extend(result.utterance.recalled_memories)

    assert recalled_for_nade, (
        "storage 指定時、なでしこの発話で過去記憶が RAG 想起されていない"
    )
    # 関連記憶（キャンプ）は想起され、無関連記憶（宿題）は想起されない
    assert any("キャンプ" in m for m in recalled_for_nade), (
        "話題に関連する過去記憶がスコアで上位に来ていない（RAG 検索が効いていない）"
    )
    assert not any("宿題" in m for m in recalled_for_nade), (
        "話題に無関連の記憶まで想起されている（単純スライスのままで RAG 化されていない）"
    )
