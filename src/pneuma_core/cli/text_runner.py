"""Text-mode multi-agent runner (Issue #7 / Phase 0 C2).

「ゆるキャン野クル組 3 人がテキストで会話していて、内部パラメータ
（PAD・関係性）が変化していくのが localhost の暫定 UI で観察できる」
の "テキストでの動作確認" 部分。

Usage::

    python -m pneuma_core.cli.text_runner --duration 5 \
        --context "今日は文化祭の準備中"

Args:
    --duration MIN     : 何分動かすか (default: 5)
    --turn-limit N     : ターン上限 (default: 50)
    --context STR      : 共有コンテクスト (default: 部室で雑談)
    --use-mock-llm     : ANTHROPIC_API_KEY があっても mock を強制
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone

from pneuma_core._packaged_examples.yurucamp import load_yurucamp_sheets
from pneuma_core.llm.adapter import LLMAdapter
from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.fun_config import FunEngineConfig
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession
from pneuma_core.multi_agent.session_end import MultiAgentSessionEndPipeline


def _build_llm(use_mock: bool) -> tuple[LLMAdapter, str]:
    """LLM adapter を構築。返り値は (adapter, 種別ラベル)."""
    if use_mock or not os.environ.get("ANTHROPIC_API_KEY"):
        return MockLLMAdapter(), "mock-llm"
    from pneuma_core.llm.claude import ClaudeAdapter

    return ClaudeAdapter.from_env(), "claude (real)"


async def _build_persistence(persist_path: str | None, chars):
    """--persist 指定時、正史の連続性 (Issue #31) 用の storage + embedding を構築.

    返り値は (storage, embedding_service)。persist_path が None なら (None, None)
    で、従来どおり in-memory モード（正史を積まない）。

    storage は SQLite。embedding は OPENAI_API_KEY があれば OpenAI、無ければ None
    （RAG は embedding 無しのフォールバック = 直近スライスになるが、状態引き継ぎ
    自体は embedding 無しでも機能する）。
    """
    if not persist_path:
        return None, None

    from pneuma_core.storage.sqlite import SQLiteStorageBackend

    storage = SQLiteStorageBackend(persist_path)
    await storage.initialize()
    # キャラ identity を storage に登録（get_character で参照できるように）
    for ch in chars:
        await storage.save_character(ch)

    embedding = None
    if os.environ.get("OPENAI_API_KEY"):
        from pneuma_core.llm.embedding import OpenAIEmbeddingService

        embedding = OpenAIEmbeddingService.from_env()

    return storage, embedding


def _fmt_pad(p: float, a: float, d: float) -> str:
    return f"P={p:+.2f} A={a:+.2f} D={d:+.2f}"


async def run(
    duration_minutes: float,
    context: str,
    turn_limit: int,
    use_mock: bool,
    loop_delay_seconds: float = 0.5,
    save_log: bool = True,
    log_dir: str = ".vibe/references",
    fun_config: FunEngineConfig | None = None,
    persist_path: str | None = None,
) -> int:
    """テキストランナー本体.

    save_log=True のとき、会話トランスクリプトを log_dir に Markdown 保存する
    (Issue #7 Phase0-C2 の AC「実行ログを保存」)。

    fun_config で面白さエンジンの各トグル (Issue #22) を指定する。None なら
    デフォルト（enable_intent のみ True）= #20 の挙動。

    persist_path を指定すると、正史の連続性 (Issue #31) が有効になる。SQLite に
    各キャラの記憶/感情/関係を永続化し、次回起動時に同じ DB を指せば前回状態を
    引き継いで会話が始まる（正史が積み上がる）。None なら従来どおり in-memory。
    """
    fun_config = fun_config or FunEngineConfig()
    llm, llm_label = _build_llm(use_mock)
    sheets = load_yurucamp_sheets()
    chars = [s.character for s in sheets]
    conv = Conversation(participants=chars)

    # 正史の連続性 (Issue #31): --persist 指定時のみ storage + embedding を用意。
    storage, embedding = await _build_persistence(persist_path, chars)

    cb = CircuitBreaker(CircuitBreakerConfig(
        max_turns=turn_limit,
        max_elapsed_seconds=duration_minutes * 60,
    ))
    cb.start()
    session = MultiAgentSession(
        conversation=conv,
        llm=llm,
        shared_context=context,
        circuit_breaker=cb,
        fun_config=fun_config,
        storage=storage,
        embedding_service=embedding,
    )
    # 前回までの正史をロード（storage 未指定なら no-op）
    await session.load_canon()

    # print と同時にトランスクリプトへ収集する emit ヘルパー
    transcript: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        transcript.append(line)

    emit("=" * 70)
    emit("# Pneuma multi-agent text runner")
    emit(f"# LLM   : {llm_label}")
    emit("# Cast  : " + " / ".join(c.name for c in chars))
    emit(f"# Ctx   : {context}")
    emit(f"# Limit : {turn_limit} turns or {duration_minutes:.1f} min")
    on_flags = [
        name.removeprefix("enable_")
        for name, val in vars(fun_config).items() if val
    ]
    emit(f"# Fun   : {', '.join(on_flags) if on_flags else '(none)'}")
    emit("=" * 70)

    # ──── 思惑（短期ゴール）の生成と表示 (Issue #20) ────
    await session.ensure_intents()
    if fun_config.enable_intent:
        emit("\n# 思惑（短期ゴール）— 長期欲求 + 状況から創発")
        for ch in chars:
            intent = session.intents.get(ch.id)
            if intent is None:
                continue
            hidden = intent.hidden_goal or "（なし）"
            emit(
                f"  {ch.name:<8} 思惑(表): {intent.surface_goal}\n"
                f"  {'':<8} 本音(裏): {hidden}  [強さ={intent.intensity:.2f}]"
            )

    started = time.monotonic()
    while True:
        result = await session.run_turn()
        if result is None:
            emit(f"\n[circuit breaker tripped] {cb.trip_reason}")
            break

        speaker = result.speaker.name
        u = result.utterance
        my_emo = session.character_states[result.speaker.id].emotion
        recalled = ", ".join(u.recalled_memories) if u.recalled_memories else "(none)"
        thought = u.thought or "（なし）"
        emit(
            f"\n[turn {result.turn_index:>3}] {speaker}\n"
            f"  speech : {u.speech}\n"
            f"  本音   : {thought}\n"
            f"  action : {u.action}\n"
            f"  PAD    : {_fmt_pad(my_emo.pleasure, my_emo.arousal, my_emo.dominance)} "
            f"emotion={my_emo.emotion_label}\n"
            f"  recalled: {recalled}"
        )

        # 他キャラの emotion 速報
        peers = [
            (cid, em) for cid, em in result.emotion_changes.items()
            if cid != result.speaker.id
        ]
        if peers:
            for cid, em in peers:
                ch_name = session.character_states[cid].character.name
                emit(
                    f"    [obs ] {ch_name}: "
                    f"{_fmt_pad(em.pleasure, em.arousal, em.dominance)} "
                    f"({em.emotion_label})"
                )

        # 進行速度を抑える（リアル LLM だと自然と遅い）
        if loop_delay_seconds > 0:
            await asyncio.sleep(loop_delay_seconds)

    # ──── Session End ────
    emit("\n" + "=" * 70)
    emit("# Session End — running per-character analysis ...")
    emit("=" * 70)
    end_pipeline = MultiAgentSessionEndPipeline(llm=llm)
    end_result = await end_pipeline.run(session)

    # 正史を前に進める (Issue #31): SessionEnd で積まれた記憶・最終感情・関係を
    # storage に永続化する。次回起動時に load_canon が読み戻す（storage 未指定なら no-op）。
    await session.persist_canon()
    if storage is not None and persist_path:
        emit(f"\n[canon persisted] {persist_path}")

    for per in end_result.per_character:
        emit(f"\n## {per.character_name}")
        if not per.success:
            emit("  (analysis failed)")
            continue
        if per.episodic:
            emit("  episodic:")
            for ep in per.episodic:
                v = ep.get("emotional_valence", 0)
                i = ep.get("importance", 0)
                emit(f"    - [imp={i:.2f} val={v:+.2f}] {ep.get('content','')}")
        if per.semantic:
            emit("  semantic:")
            for sm in per.semantic:
                c = sm.get("confidence", 0)
                emit(f"    - [conf={c:.2f}] {sm.get('content','')}")

    # ──── Final summary ────
    emit("\n" + "=" * 70)
    emit("# Final Snapshot")
    emit("=" * 70)
    snap = session.snapshot()
    for ch in snap["characters"]:
        e = ch["emotion"]
        emit(
            f"\n{ch['name']:<8} | "
            f"{_fmt_pad(e['pleasure'], e['arousal'], e['dominance'])} "
            f"({e['emotion_label']})  "
            f"spoke={ch['speak_count']}"
        )
        if ch["relations"]:
            emit("  relations:")
            for tid, rel in ch["relations"].items():
                emit(
                    f"    → {rel['target_name']:<8} "
                    f"closeness={rel['closeness']:.3f} "
                    f"trust={rel['trust']:.3f}"
                )
        if ch["recent_episodic"]:
            emit("  recent_episodic:")
            for ep in ch["recent_episodic"]:
                emit(f"    - {ep}")

    elapsed = time.monotonic() - started
    emit("\n" + "=" * 70)
    emit(
        f"# Done: {session.snapshot()['turn_index']} turns in {elapsed:.1f}s "
        f"(circuit={cb.state.value})"
    )
    emit("=" * 70)

    # ──── ログ保存 ────
    if save_log:
        from pathlib import Path

        out_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        log_path = out_dir / f"text-runner-trial-{ts}.md"
        log_path.write_text("\n".join(transcript) + "\n", encoding="utf-8")
        print(f"\n[log saved] {log_path}")

    # storage を閉じる（SQLite 接続のクリーンアップ）
    if storage is not None:
        await storage.close()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pneuma multi-agent text-mode runner",
    )
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Run for this many minutes (default: 5)")
    parser.add_argument("--turn-limit", type=int, default=50,
                        help="Max turns (default: 50)")
    parser.add_argument("--context", type=str,
                        default="部室で何気ない雑談中。窓から夕方の光が差している。",
                        help="Shared session context")
    parser.add_argument("--use-mock-llm", action="store_true",
                        help="Force MockLLMAdapter even if ANTHROPIC_API_KEY is set")
    parser.add_argument("--loop-delay", type=float, default=0.1,
                        help="Sleep between turns in seconds (default: 0.1)")
    parser.add_argument("--no-save-log", action="store_true",
                        help="Disable saving the transcript to .vibe/references/")
    parser.add_argument("--log-dir", type=str, default=".vibe/references",
                        help="Directory to save transcript log (default: .vibe/references)")
    parser.add_argument("--persist", type=str, default=None, metavar="DB_PATH",
                        help="正史の連続性 (Issue #31): SQLite パスを指定すると記憶/感情/"
                             "関係を永続化し、次回同じ DB を指せば前回状態を引き継ぐ")

    # ──── 面白さエンジンのトグル (Issue #22) ────
    # 指定しなければデフォルト（intent のみ）= #20 の挙動。
    fun = parser.add_argument_group("fun engine toggles (Issue #22)")
    fun.add_argument("--no-intent", action="store_true",
                     help="思惑（#20）を無効化する（デフォルトは有効）")
    fun.add_argument("--quirk", action="store_true",
                     help="キャラの思考のクセを発話に注入する")
    fun.add_argument("--terse", action="store_true",
                     help="発話を短く・大喜利的にする（テンポ）")
    fun.add_argument("--lateral", action="store_true",
                     help="水平思考的飛躍（連想・アナロジー）を促す")
    fun.add_argument("--emotion-dynamics", action="store_true",
                     help="感情を性格 baseline に引き戻し温度差を作る")
    fun.add_argument("--structured-ending", action="store_true",
                     help="終盤に締切/決定の外圧で収束を促す（構造的オチ）")

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    fun_config = FunEngineConfig(
        enable_intent=not args.no_intent,
        enable_quirk=args.quirk,
        enable_terse=args.terse,
        enable_lateral_thinking=args.lateral,
        enable_emotion_dynamics=args.emotion_dynamics,
        enable_structured_ending=args.structured_ending,
    )

    try:
        rc = asyncio.run(run(
            duration_minutes=args.duration,
            context=args.context,
            turn_limit=args.turn_limit,
            use_mock=args.use_mock_llm,
            loop_delay_seconds=args.loop_delay,
            save_log=not args.no_save_log,
            log_dir=args.log_dir,
            fun_config=fun_config,
            persist_path=args.persist,
        ))
    except KeyboardInterrupt:
        print("\n^C interrupted", file=sys.stderr)
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
