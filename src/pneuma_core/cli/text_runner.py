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
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession
from pneuma_core.multi_agent.session_end import MultiAgentSessionEndPipeline


def _build_llm(use_mock: bool) -> tuple[LLMAdapter, str]:
    """LLM adapter を構築。返り値は (adapter, 種別ラベル)."""
    if use_mock or not os.environ.get("ANTHROPIC_API_KEY"):
        return MockLLMAdapter(), "mock-llm"
    from pneuma_core.llm.claude import ClaudeAdapter

    return ClaudeAdapter.from_env(), "claude (real)"


def _fmt_pad(p: float, a: float, d: float) -> str:
    return f"P={p:+.2f} A={a:+.2f} D={d:+.2f}"


async def run(
    duration_minutes: float,
    context: str,
    turn_limit: int,
    use_mock: bool,
    loop_delay_seconds: float = 0.5,
) -> int:
    """テキストランナー本体."""
    llm, llm_label = _build_llm(use_mock)
    sheets = load_yurucamp_sheets()
    chars = [s.character for s in sheets]
    conv = Conversation(participants=chars)

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
    )

    print("=" * 70)
    print(f"# Pneuma multi-agent text runner")
    print(f"# LLM   : {llm_label}")
    print(f"# Cast  : " + " / ".join(c.name for c in chars))
    print(f"# Ctx   : {context}")
    print(f"# Limit : {turn_limit} turns or {duration_minutes:.1f} min")
    print("=" * 70)

    started = time.monotonic()
    while True:
        result = await session.run_turn()
        if result is None:
            print(f"\n[circuit breaker tripped] {cb.trip_reason}")
            break

        speaker = result.speaker.name
        u = result.utterance
        my_emo = session.character_states[result.speaker.id].emotion
        recalled = ", ".join(u.recalled_memories) if u.recalled_memories else "(none)"
        print(
            f"\n[turn {result.turn_index:>3}] {speaker}\n"
            f"  speech : {u.speech}\n"
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
                print(
                    f"    [obs ] {ch_name}: "
                    f"{_fmt_pad(em.pleasure, em.arousal, em.dominance)} "
                    f"({em.emotion_label})"
                )

        # 進行速度を抑える（リアル LLM だと自然と遅い）
        if loop_delay_seconds > 0:
            await asyncio.sleep(loop_delay_seconds)

    # ──── Session End ────
    print("\n" + "=" * 70)
    print("# Session End — running per-character analysis ...")
    print("=" * 70)
    end_pipeline = MultiAgentSessionEndPipeline(llm=llm)
    end_result = await end_pipeline.run(session)

    for per in end_result.per_character:
        print(f"\n## {per.character_name}")
        if not per.success:
            print("  (analysis failed)")
            continue
        if per.episodic:
            print("  episodic:")
            for ep in per.episodic:
                v = ep.get("emotional_valence", 0)
                i = ep.get("importance", 0)
                print(f"    - [imp={i:.2f} val={v:+.2f}] {ep.get('content','')}")
        if per.semantic:
            print("  semantic:")
            for sm in per.semantic:
                c = sm.get("confidence", 0)
                print(f"    - [conf={c:.2f}] {sm.get('content','')}")

    # ──── Final summary ────
    print("\n" + "=" * 70)
    print("# Final Snapshot")
    print("=" * 70)
    snap = session.snapshot()
    for ch in snap["characters"]:
        e = ch["emotion"]
        print(
            f"\n{ch['name']:<8} | "
            f"{_fmt_pad(e['pleasure'], e['arousal'], e['dominance'])} "
            f"({e['emotion_label']})  "
            f"spoke={ch['speak_count']}"
        )
        if ch["relations"]:
            print("  relations:")
            for tid, rel in ch["relations"].items():
                print(
                    f"    → {rel['target_name']:<8} "
                    f"closeness={rel['closeness']:.3f} "
                    f"trust={rel['trust']:.3f}"
                )
        if ch["recent_episodic"]:
            print("  recent_episodic:")
            for ep in ch["recent_episodic"]:
                print(f"    - {ep}")

    elapsed = time.monotonic() - started
    print("\n" + "=" * 70)
    print(
        f"# Done: {session.snapshot()['turn_index']} turns in {elapsed:.1f}s "
        f"(circuit={cb.state.value})"
    )
    print("=" * 70)
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
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    try:
        rc = asyncio.run(run(
            duration_minutes=args.duration,
            context=args.context,
            turn_limit=args.turn_limit,
            use_mock=args.use_mock_llm,
            loop_delay_seconds=args.loop_delay,
        ))
    except KeyboardInterrupt:
        print("\n^C interrupted", file=sys.stderr)
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
