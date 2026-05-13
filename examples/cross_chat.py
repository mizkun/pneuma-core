"""Cross-character autonomous chat demo.

2 体のキャラクター (アイネ・リン) が交互にメッセージを投げ合い、
ターンごとに各キャラクターの感情 (PAD) と記憶/関係性の変化を観測する。

セッション中: emotion (PAD) がターンごとに変化
セッション終了時: SessionEndPipeline で episodic_memory / semantic_memory /
                 relation が更新される

実行:
    export ANTHROPIC_API_KEY=...  # LLM 用 (Claude)
    export OPENAI_API_KEY=...     # Embedding 用 (text-embedding-3-small)
    .venv/bin/python examples/cross_chat.py [TURNS]   # 既定 6 ターン

注意:
    - 最低 2 ターン × 2 キャラクター = 4 LLM 呼び出し + session-end で +2 呼び出し
    - 6 ターンだと合計 14 呼び出しくらい (cost を意識して短めに)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.llm.claude import ClaudeAdapter
from pneuma_core.llm.embedding import OpenAIEmbeddingService
from pneuma_core.models.message import MessageInput, MessageOutput
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.runtime.session import ConversationSession
from pneuma_core.runtime.session_end_pipeline import SessionEndPipeline
from pneuma_core.storage.sqlite import SQLiteStorageBackend


# ──────────────────────────────────────────────────────────────────────
# Snapshot helpers
# ──────────────────────────────────────────────────────────────────────

EXAMPLES_DIR = Path(__file__).parent


async def _snapshot(
    storage: SQLiteStorageBackend, character_id: str, name: str
) -> str:
    """1 行に圧縮した PAD / memory / relation のスナップショット."""
    state = await storage.get_emotional_state(character_id)
    pad = (
        f"P={state.pleasure:+.2f} A={state.arousal:+.2f} D={state.dominance:+.2f} "
        f"({state.emotion_label})"
        if state
        else "(uninitialised)"
    )
    eps = await storage.get_episodic_memories(character_id)
    sems = await storage.get_semantic_memories(character_id)
    rels = await storage.list_relations(owner_id=character_id)
    rel_part = ""
    if rels:
        r = rels[0]
        rel_part = f" rel→{r.target_name} close={r.closeness:.2f} trust={r.trust:.2f}"
    return f"{name:>4}: {pad}  ep={len(eps)} sem={len(sems)}{rel_part}"


async def _setup_character(
    storage: SQLiteStorageBackend,
    sheet_path: Path,
) -> tuple[str, str]:
    """YAML をロードしてキャラクター・goals・初期感情を storage に保存."""
    sheet = CharacterSheet.load(sheet_path)
    char = sheet.character
    await storage.save_character(char)
    if sheet.goal_tree is not None:
        await storage.save_goals(char.id, sheet.goal_tree)
    if sheet.initial_state is not None:
        await storage.save_emotional_state(char.id, sheet.initial_state)
    return char.id, char.name


# ──────────────────────────────────────────────────────────────────────
# Engine factory
# ──────────────────────────────────────────────────────────────────────


def _make_engine(
    *,
    character_id: str,
    storage: SQLiteStorageBackend,
    llm: ClaudeAdapter,
    embedding: OpenAIEmbeddingService,
) -> RuntimeEngine:
    """RuntimeEngine を1体ぶん生成。storage は MemoryStore も兼ねる (SQLite)."""
    return RuntimeEngine(
        character_id=character_id,
        storage=storage,
        llm=llm,
        embedding_service=embedding,
        memory_store=storage,  # SQLiteStorageBackend は MemoryStore Protocol も満たす
    )


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────


async def cross_chat(turns: int) -> None:
    # Optional env keys — check up-front so the user doesn't spend time only to crash later.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY が設定されていません。", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY が設定されていません。", file=sys.stderr)
        sys.exit(1)

    # 一時 sqlite ファイル。終了時に削除する。
    tmpfile = tempfile.NamedTemporaryFile(
        prefix="pneuma_cross_", suffix=".db", delete=False
    )
    tmpfile.close()
    db_path = tmpfile.name
    print(f"# storage: {db_path}\n")

    storage = SQLiteStorageBackend(db_path)
    await storage.initialize()
    try:
        # 1. キャラクター 2 体をロード
        aine_id, aine_name = await _setup_character(
            storage, EXAMPLES_DIR / "aine.character.yaml"
        )
        rin_id, rin_name = await _setup_character(
            storage, EXAMPLES_DIR / "rin.character.yaml"
        )

        # 2. アダプタ + Runtime
        llm = ClaudeAdapter.from_env()
        embedding = OpenAIEmbeddingService.from_env()
        aine_engine = _make_engine(
            character_id=aine_id, storage=storage, llm=llm, embedding=embedding
        )
        rin_engine = _make_engine(
            character_id=rin_id, storage=storage, llm=llm, embedding=embedding
        )

        # 3. 初期スナップショット
        print("─" * 70)
        print(f"INITIAL  {await _snapshot(storage, aine_id, aine_name)}")
        print(f"         {await _snapshot(storage, rin_id, rin_name)}")
        print("─" * 70)

        # 4. キックオフ発話 (人間の "システム" がキッカケを与える)
        opener = "はじめまして。よかったら、今ハマってることを教えて。"
        last_speaker_id, last_speaker_name = "system", "system"
        last_content = opener
        print(f"\n  [opener  → 全員] {opener}\n")

        # session bookkeeping (session_end_pipeline 用)
        session_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        aine_session = ConversationSession(
            session_id=session_id,
            character_id=aine_id,
            user_id=rin_id,
            channel_id="cross-chat",
            started_at=now,
            last_active_at=now,
        )
        rin_session = ConversationSession(
            session_id=session_id,
            character_id=rin_id,
            user_id=aine_id,
            channel_id="cross-chat",
            started_at=now,
            last_active_at=now,
        )

        # 5. ターンを交互に進める
        speakers = [
            (aine_engine, aine_id, aine_name, aine_session, rin_session),
            (rin_engine, rin_id, rin_name, rin_session, aine_session),
        ]
        for turn in range(1, turns + 1):
            engine, char_id, char_name, own_session, peer_session = speakers[
                (turn - 1) % 2
            ]
            msg_in = MessageInput(
                content=last_content,
                sender_id=last_speaker_id,
                sender_name=last_speaker_name,
                sender_type="character" if last_speaker_id != "system" else "system",
            )
            output: MessageOutput = await engine.process_message(msg_in)
            # session に詰める (両側で記録)
            own_session.messages.append(
                {"role": "user", "content": last_content,
                 "sender_id": last_speaker_id, "sender_name": last_speaker_name}
            )
            own_session.messages.append(
                {"role": "assistant", "content": output.content,
                 "sender_id": char_id, "sender_name": char_name}
            )
            peer_session.messages.append(
                {"role": "assistant", "content": last_content,
                 "sender_id": last_speaker_id, "sender_name": last_speaker_name}
            )
            peer_session.messages.append(
                {"role": "user", "content": output.content,
                 "sender_id": char_id, "sender_name": char_name}
            )

            # 表示
            print(f"  [turn {turn} {char_name}] {output.content}")
            print(f"          {await _snapshot(storage, char_id, char_name)}\n")

            last_speaker_id = char_id
            last_speaker_name = char_name
            last_content = output.content

        # 6. session 終了処理を両キャラに適用 (Opus 1 回ずつ呼び出し)
        print("─" * 70)
        print("SESSION END pipeline (LLM analysis → episodic / semantic / relation)")
        print("─" * 70)
        pipeline = SessionEndPipeline(
            llm=llm,
            embedding_service=embedding,
            memory_store=storage,
            storage=storage,
        )
        for engine, char_id, char_name, own_session, _peer in speakers:
            existing_sem = await storage.get_semantic_memories(char_id)
            existing_rel = await storage.list_relations(owner_id=char_id)
            result = await pipeline.run(
                session=own_session,
                character_id=char_id,
                existing_semantics=existing_sem,
                existing_relations=existing_rel,
            )
            print(
                f"  {char_name}: success={result.success} "
                f"episodic={result.episodic_memories_saved} "
                f"semantic={result.semantic_updates_applied} "
                f"relations={result.relationship_changes}"
            )

        # 7. 最終スナップショット
        print("─" * 70)
        print(f"FINAL    {await _snapshot(storage, aine_id, aine_name)}")
        print(f"         {await _snapshot(storage, rin_id, rin_name)}")
        print("─" * 70)

        # 8. memory / relation の詳細ダンプ
        print("\n## エピソード記憶")
        for char_id, char_name in [(aine_id, aine_name), (rin_id, rin_name)]:
            eps = await storage.get_episodic_memories(char_id)
            print(f"\n  {char_name} ({len(eps)} 件):")
            for ep in eps:
                print(
                    f"    [{ep.importance:.2f}] valence={ep.emotional_valence:+.2f} "
                    f"{ep.content}"
                )

        print("\n## セマンティック記憶")
        for char_id, char_name in [(aine_id, aine_name), (rin_id, rin_name)]:
            sems = await storage.get_semantic_memories(char_id)
            print(f"\n  {char_name} ({len(sems)} 件):")
            for sem in sems:
                print(f"    [conf={sem.confidence:.2f}] {sem.content}")

        print("\n## 関係性")
        for char_id, char_name in [(aine_id, aine_name), (rin_id, rin_name)]:
            rels = await storage.list_relations(owner_id=char_id)
            for r in rels:
                print(
                    f"  {char_name} → {r.target_name}: closeness={r.closeness:.2f} "
                    f"trust={r.trust:.2f} type={r.relationship_type}"
                )
    finally:
        await storage.close()
        try:
            os.unlink(db_path)
        except OSError:
            pass


def main() -> None:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if turns < 1:
        print("turns must be >= 1", file=sys.stderr)
        sys.exit(1)
    asyncio.run(cross_chat(turns))


if __name__ == "__main__":
    main()
