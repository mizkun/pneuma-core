"""Minimal web UI for observing multi-agent conversation (Issue #7 / Phase 0 C2).

stdlib のみで動かす（FastAPI / uvicorn なし）。

- `GET /`        : 単一ファイル HTML（インライン CSS + JS）
- `GET /events`  : Server-Sent Events で会話イベントを push
- `GET /state`   : 現在の snapshot を一発で返す

会話セッションは別スレッドで asyncio loop を回し続ける。

Usage::

    python -m pneuma_core.cli.web_server
    # → http://localhost:8000

Args:
    --port            : default 8000
    --use-mock-llm    : 強制 mock
    --context STR     : 共有コンテクスト
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pneuma_core._packaged_examples.yurucamp import load_yurucamp_sheets
from pneuma_core.llm.adapter import LLMAdapter
from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.fun_config import FunEngineConfig
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter
from pneuma_core.multi_agent.session import MultiAgentSession, TurnResult
from pneuma_core.multi_agent.session_end import MultiAgentSessionEndPipeline

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Background session driver
# ──────────────────────────────────────────────────────────────────────


class SessionDriver:
    """別スレッドの asyncio loop で session を回し、イベントを subscribers に流す."""

    def __init__(
        self,
        llm: LLMAdapter,
        llm_label: str,
        context: str,
        use_mock: bool,
        turn_limit_per_session: int = 50,
        inter_turn_seconds: float = 2.5,
        inter_session_seconds: float = 8.0,
        fun_config: FunEngineConfig | None = None,
        persist_path: str | None = None,
    ) -> None:
        self._llm = llm
        self._llm_label = llm_label
        self._context = context
        self._use_mock = use_mock
        self._turn_limit_per_session = turn_limit_per_session
        self._inter_turn_seconds = inter_turn_seconds
        self._inter_session_seconds = inter_session_seconds
        self._fun_config = fun_config or FunEngineConfig()
        # 正史の連続性 (Issue #31): --persist 指定時、storage を別スレッドの
        # asyncio loop 内で初期化する。各セッションが load_canon → 会話 →
        # persist_canon で 1 本の正史を積み上げる。
        self._persist_path = persist_path
        self._storage: Any = None
        self._embedding: Any = None

        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._current_session: MultiAgentSession | None = None
        self._latest_snapshot: dict[str, Any] | None = None
        self._next_session_at: datetime | None = None
        self._stop_event = threading.Event()

    def latest_snapshot(self) -> dict[str, Any]:
        snap = dict(self._latest_snapshot or self._bootstrap_snapshot())
        snap["llm_label"] = self._llm_label
        snap["context"] = self._context
        if self._next_session_at:
            snap["next_session_at"] = self._next_session_at.isoformat()
        return snap

    def _bootstrap_snapshot(self) -> dict[str, Any]:
        """Session が始まる前に返す初期 snapshot."""
        sheets = load_yurucamp_sheets()
        return {
            "session_id": None,
            "turn_index": 0,
            "circuit_state": "closed",
            "circuit_remaining_turns": self._turn_limit_per_session,
            "characters": [
                {
                    "id": s.character.id,
                    "name": s.character.name,
                    "personality": {
                        "openness": s.character.personality.openness,
                        "conscientiousness": s.character.personality.conscientiousness,
                        "extraversion": s.character.personality.extraversion,
                        "agreeableness": s.character.personality.agreeableness,
                        "neuroticism": s.character.personality.neuroticism,
                    },
                    "emotion": {
                        "pleasure": s.initial_state.pleasure if s.initial_state else 0.0,
                        "arousal": s.initial_state.arousal if s.initial_state else 0.0,
                        "dominance": s.initial_state.dominance if s.initial_state else 0.0,
                        "emotion_label": (s.initial_state.emotion_label
                                          if s.initial_state else "neutral"),
                    },
                    "speak_count": 0,
                    "relations": {},
                    "recent_episodic": [],
                    "intent": {
                        "surface_goal": "",
                        "hidden_goal": None,
                        "intensity": 0.0,
                    },
                    "recent_thought": "",
                }
                for s in sheets
            ],
            "history_tail": [],
        }

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.append(q)
        # initial state
        try:
            q.put_nowait({"type": "snapshot", "data": self.latest_snapshot()})
        except queue.Full:
            pass
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event_type: str, data: dict) -> None:
        payload = {"type": event_type, "data": data}
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                # drop oldest, push newest
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass

    def stop(self) -> None:
        self._stop_event.set()

    def run_forever_in_thread(self) -> threading.Thread:
        th = threading.Thread(target=self._thread_target, daemon=True)
        th.start()
        return th

    def _thread_target(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        # 正史の連続性 (Issue #31): storage はこのスレッドの event loop 内で初期化。
        await self._init_persistence()
        session_count = 0
        while not self._stop_event.is_set():
            session_count += 1
            await self._run_one_session(session_count)
            if self._stop_event.is_set():
                break
            self._next_session_at = datetime.now(timezone.utc)
            self._broadcast("inter_session", {
                "delay_seconds": self._inter_session_seconds,
                "session_count": session_count,
            })
            await asyncio.sleep(self._inter_session_seconds)

    async def _init_persistence(self) -> None:
        """--persist 指定時、SQLite storage + embedding を初期化する (Issue #31)."""
        if not self._persist_path:
            return
        from pneuma_core.storage.sqlite import SQLiteStorageBackend

        storage = SQLiteStorageBackend(self._persist_path)
        await storage.initialize()
        sheets = load_yurucamp_sheets()
        for s in sheets:
            await storage.save_character(s.character)
        self._storage = storage

        if os.environ.get("OPENAI_API_KEY"):
            from pneuma_core.llm.embedding import OpenAIEmbeddingService

            self._embedding = OpenAIEmbeddingService.from_env()

    async def _run_one_session(self, idx: int) -> None:
        sheets = load_yurucamp_sheets()
        chars = [s.character for s in sheets]
        conv = Conversation(participants=chars)
        cb = CircuitBreaker(CircuitBreakerConfig(
            max_turns=self._turn_limit_per_session,
            max_elapsed_seconds=15 * 60,
        ))
        cb.start()
        session = MultiAgentSession(
            conversation=conv,
            llm=self._llm,
            shared_context=self._context,
            circuit_breaker=cb,
            fun_config=self._fun_config,
            storage=self._storage,
            embedding_service=self._embedding,
        )
        self._current_session = session
        # apply initial emotions from sheets（正史が無い初回のみ）
        for s in sheets:
            if s.initial_state is not None:
                session.character_states[s.character.id].emotion = s.initial_state
        # 正史の連続性 (Issue #31): 前回までの状態をロード（storage 未指定なら no-op）。
        # ロード後の emotion がシートの初期値を上書きする（前回から引き継ぐ）。
        await session.load_canon()

        self._broadcast("session_start", {
            "session_id": session.session_id,
            "session_count": idx,
            "context": self._context,
        })
        self._latest_snapshot = session.snapshot()
        self._broadcast("snapshot", self._latest_snapshot)

        while not self._stop_event.is_set():
            try:
                result = await session.run_turn()
            except Exception as e:
                logger.exception("turn failed: %s", e)
                self._broadcast("error", {"message": str(e)})
                break
            if result is None:
                break

            self._broadcast("turn", self._turn_payload(result, session))
            self._latest_snapshot = session.snapshot()
            self._broadcast("snapshot", self._latest_snapshot)

            await asyncio.sleep(self._inter_turn_seconds)

        # session end
        try:
            pipeline = MultiAgentSessionEndPipeline(llm=self._llm)
            end_result = await pipeline.run(session)
            self._broadcast("session_end", {
                "per_character": [
                    {
                        "character_id": p.character_id,
                        "character_name": p.character_name,
                        "episodic": p.episodic,
                        "semantic": p.semantic,
                        "success": p.success,
                    }
                    for p in end_result.per_character
                ],
                "total_episodic": end_result.total_episodic,
                "total_semantic": end_result.total_semantic,
            })
            # 正史を前に進める (Issue #31): SessionEnd で積まれた記憶・最終感情・
            # 関係を storage に永続化。次セッションの load_canon が読み戻す
            # （storage 未指定なら no-op）。
            await session.persist_canon()
            self._latest_snapshot = session.snapshot()
            self._broadcast("snapshot", self._latest_snapshot)
        except Exception as e:
            logger.exception("session_end failed: %s", e)
            self._broadcast("error", {"message": f"session_end: {e}"})

    @staticmethod
    def _turn_payload(result: TurnResult, session: MultiAgentSession) -> dict:
        u = result.utterance
        return {
            "turn_index": result.turn_index,
            "speaker": {
                "id": result.speaker.id,
                "name": result.speaker.name,
            },
            "utterance": {
                "speech": u.speech,
                "thought": u.thought,
                "action": u.action,
                "emotion_label": u.emotion_label,
                "pad": list(u.pad),
            },
            "emotion_changes": {
                cid: {
                    "pleasure": em.pleasure,
                    "arousal": em.arousal,
                    "dominance": em.dominance,
                    "emotion_label": em.emotion_label,
                }
                for cid, em in result.emotion_changes.items()
            },
            "recalled_memories": u.recalled_memories,
            "snapshot": session.snapshot(),
        }


# ──────────────────────────────────────────────────────────────────────
# HTTP handler
# ──────────────────────────────────────────────────────────────────────


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Pneuma Multi-Agent Observatory</title>
<style>
  :root {
    --bg: #0e0f17;
    --panel: #161826;
    --text: #d8d8e2;
    --dim: #8b8ea3;
    --accent: #ff7eb6;
    --good: #6fcf97;
    --bad: #eb5757;
    --mid: #f2c94c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "Hiragino Sans", "Yu Gothic", sans-serif;
    font-size: 13px;
    height: 100vh;
    overflow: hidden;
  }
  header {
    padding: 8px 12px;
    background: #060710;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #21243a;
    font-size: 12px;
  }
  header h1 { margin: 0; font-size: 14px; color: var(--accent); }
  header .meta { color: var(--dim); }
  main {
    display: grid;
    grid-template-columns: 320px 1fr 320px;
    gap: 8px;
    padding: 8px;
    height: calc(100vh - 40px);
  }
  .col { overflow-y: auto; }
  .panel {
    background: var(--panel);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    border: 1px solid #21243a;
  }
  h2 { font-size: 13px; margin: 0 0 8px; color: var(--accent); }
  .char-card { margin-bottom: 8px; }
  .char-name {
    font-size: 16px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .char-emoji {
    font-size: 24px;
    width: 36px; height: 36px;
    display: inline-flex;
    align-items: center; justify-content: center;
    background: #21243a;
    border-radius: 50%;
    margin-right: 8px;
  }
  .char-row {
    display: flex;
    align-items: center;
    margin-bottom: 4px;
  }
  .pad-bar {
    position: relative;
    height: 6px;
    background: #21243a;
    border-radius: 3px;
    margin: 2px 0 4px;
    overflow: hidden;
  }
  .pad-bar .fill {
    position: absolute; top: 0;
    height: 100%;
    background: var(--accent);
    transition: width 0.4s, left 0.4s;
  }
  .pad-bar.neg .fill { background: var(--bad); }
  .pad-label {
    font-size: 10px;
    color: var(--dim);
    display: flex;
    justify-content: space-between;
  }
  .badge {
    display: inline-block;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    background: #21243a;
    color: var(--dim);
    margin-left: 4px;
  }
  .badge.happy { background: #2d4a30; color: #6fcf97; }
  .badge.teasing { background: #4a3a2d; color: #f2c94c; }
  .badge.surprised { background: #3a2d4a; color: #bb6bd9; }
  .badge.embarrassed { background: #4a2d3a; color: #ff7eb6; }
  .badge.sad_lite { background: #2d3a4a; color: #6789eb; }
  .badge.neutral { background: #21243a; color: #8b8ea3; }
  .big-five-row {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--dim);
    margin-top: 4px;
  }
  .conv {
    height: calc(100vh - 60px);
    overflow-y: auto;
    padding-right: 8px;
  }
  .utt {
    background: var(--panel);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    border-left: 3px solid var(--dim);
    animation: pop 0.4s ease-out;
  }
  .utt .speaker {
    font-weight: 600;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .utt .speech { font-size: 15px; line-height: 1.6; }
  .utt .action {
    font-size: 11px;
    color: var(--dim);
    margin-top: 4px;
    font-style: italic;
  }
  .utt.silent { opacity: 0.6; }
  .utt.nade { border-left-color: #ff7eb6; }
  .utt.chia { border-left-color: #f2c94c; }
  .utt.aoi  { border-left-color: #6fcf97; }
  .relation-grid {
    display: grid;
    grid-template-columns: 100px repeat(3, 1fr);
    gap: 4px;
    font-size: 11px;
  }
  .relation-grid .head { color: var(--dim); }
  .relation-grid .cell {
    background: #1c1f30;
    padding: 4px;
    border-radius: 4px;
    text-align: center;
  }
  .recalled {
    font-size: 11px;
    color: var(--dim);
    margin-top: 8px;
  }
  .recalled .item {
    padding: 4px 8px;
    background: #1c1f30;
    border-radius: 4px;
    margin-bottom: 2px;
  }
  @keyframes pop {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .session-divider {
    text-align: center;
    color: var(--dim);
    font-size: 11px;
    margin: 12px 0;
    border-top: 1px dashed #21243a;
    padding-top: 6px;
  }
</style>
</head>
<body>

<header>
  <h1>Pneuma Multi-Agent Observatory</h1>
  <div class="meta">
    LLM: <span id="llm-label">…</span>
    | Session: <span id="session-idx">0</span>
    | Turn: <span id="turn-idx">0</span>
    | Circuit: <span id="circuit-state">closed</span>
    <span id="next-session"></span>
  </div>
</header>

<main>
  <div class="col" id="characters-col">
    <div class="panel">
      <h2>キャラクター</h2>
      <div id="characters"></div>
    </div>
  </div>

  <div class="col">
    <div class="panel" style="background: transparent; border: none;">
      <h2>会話ログ</h2>
      <div class="conv" id="conv"></div>
    </div>
  </div>

  <div class="col">
    <div class="panel">
      <h2>関係性マトリクス</h2>
      <div id="relations"></div>
    </div>
    <div class="panel">
      <h2>想起記憶</h2>
      <div id="recalled"></div>
    </div>
    <div class="panel">
      <h2>セッション分析（直近）</h2>
      <div id="session-end-info"></div>
    </div>
  </div>
</main>

<script>
const CHAR_EMOJI = {
  "nadeshiko-001": "🍙",
  "chiaki-001": "😤",
  "aoi-001": "🍵",
};
const CHAR_KIND = {
  "nadeshiko-001": "nade",
  "chiaki-001": "chia",
  "aoi-001": "aoi",
};

function padBar(value) {
  // value in [-1, 1]
  const pct = (value + 1) * 50; // map to 0..100
  const isNeg = value < 0;
  return `<div class="pad-bar ${isNeg ? 'neg' : ''}">
    <div class="fill" style="left:${isNeg ? pct : 50}%; width:${Math.abs(value)*50}%"></div>
  </div>`;
}

function badge(label) {
  return `<span class="badge ${label}">${label}</span>`;
}

function renderIntent(c) {
  // 思惑（短期ゴール）の表示 (Issue #20)。surface=表のゴール / hidden=裏の本音。
  const it = c.intent;
  if (!it || !it.surface_goal) return "";
  const surface = escapeHtml(it.surface_goal);
  const hidden = it.hidden_goal ? escapeHtml(it.hidden_goal) : "（なし）";
  const intensity = (typeof it.intensity === "number") ? it.intensity.toFixed(2) : "?";
  const thought = c.recent_thought ? escapeHtml(c.recent_thought) : "";
  const thoughtHtml = thought
    ? `<div style="font-size:10px; color:var(--dim); margin-top:2px;">本音: ${thought}</div>`
    : "";
  return `<div style="font-size:11px; margin-bottom:6px; padding:4px 6px;
      background:rgba(255,255,255,0.04); border-radius:4px;">
    <div>思惑(表): ${surface} <span style="color:var(--dim)">[強さ${intensity}]</span></div>
    <div style="color:var(--dim)">裏: ${hidden}</div>
    ${thoughtHtml}
  </div>`;
}

function renderCharacters(snap) {
  const div = document.getElementById("characters");
  div.innerHTML = (snap.characters || []).map(c => {
    const e = c.emotion;
    const p = c.personality;
    const kind = CHAR_KIND[c.id] || "";
    return `<div class="char-card ${kind}">
      <div class="char-name">
        <span><span class="char-emoji">${CHAR_EMOJI[c.id] || "🎭"}</span>${c.name}</span>
        ${badge(e.emotion_label || "neutral")}
      </div>
      <div style="font-size: 11px; color: var(--dim); margin-bottom: 6px;">
        spoke=${c.speak_count}
      </div>
      ${renderIntent(c)}
      <div class="pad-label"><span>Pleasure</span><span>${e.pleasure.toFixed(2)}</span></div>
      ${padBar(e.pleasure)}
      <div class="pad-label"><span>Arousal</span><span>${e.arousal.toFixed(2)}</span></div>
      ${padBar(e.arousal)}
      <div class="pad-label"><span>Dominance</span><span>${e.dominance.toFixed(2)}</span></div>
      ${padBar(e.dominance)}
      <div class="big-five-row">
        <span>O:${p.openness.toFixed(1)}</span>
        <span>C:${p.conscientiousness.toFixed(1)}</span>
        <span>E:${p.extraversion.toFixed(1)}</span>
        <span>A:${p.agreeableness.toFixed(1)}</span>
        <span>N:${p.neuroticism.toFixed(1)}</span>
      </div>
    </div>`;
  }).join("");
}

function renderRelations(snap) {
  const div = document.getElementById("relations");
  const chars = snap.characters || [];
  const head = `<div class="head"></div>` + chars.map(c => `<div class="head">${c.name}</div>`).join("");
  const rows = chars.map(c => {
    const rels = c.relations || {};
    const row = `<div class="head">${c.name} →</div>` + chars.map(p => {
      if (p.id === c.id) return `<div class="cell" style="opacity:0.3">—</div>`;
      const r = rels[p.id];
      if (!r) return `<div class="cell">—</div>`;
      return `<div class="cell">
        <div>c=${r.closeness.toFixed(2)}</div>
        <div>t=${r.trust.toFixed(2)}</div>
      </div>`;
    }).join("");
    return row;
  }).join("");
  div.innerHTML = `<div class="relation-grid">${head}${rows}</div>`;
}

function renderRecalled(snap) {
  const div = document.getElementById("recalled");
  div.innerHTML = (snap.characters || []).map(c => {
    if (!c.recent_episodic || c.recent_episodic.length === 0) {
      return `<div class="recalled"><b>${c.name}</b>: (まだ無し)</div>`;
    }
    return `<div class="recalled"><b>${c.name}</b>:` +
      c.recent_episodic.map(ep => `<div class="item">${ep}</div>`).join("") +
      `</div>`;
  }).join("");
}

function renderSnapshot(snap) {
  document.getElementById("turn-idx").textContent = snap.turn_index;
  document.getElementById("circuit-state").textContent = snap.circuit_state;
  if (snap.llm_label) document.getElementById("llm-label").textContent = snap.llm_label;
  if (snap.next_session_at) {
    document.getElementById("next-session").textContent =
      " | Next session ETA: " + new Date(snap.next_session_at).toLocaleTimeString();
  } else {
    document.getElementById("next-session").textContent = "";
  }
  renderCharacters(snap);
  renderRelations(snap);
  renderRecalled(snap);
}

function appendUtterance(payload) {
  const c = document.getElementById("conv");
  const kind = CHAR_KIND[payload.speaker.id] || "";
  const u = payload.utterance;
  const silent = !u.speech ? "silent" : "";
  const speech = u.speech || "（沈黙）";
  const action = u.action ? `<div class="action">（${u.action}）</div>` : "";
  // 本音 (thought) を speech と分けて表示 (Issue #20: speech-thought-separation)
  const thought = u.thought
    ? `<div class="action" style="color:var(--dim); font-style:italic;">本音: ${escapeHtml(u.thought)}</div>`
    : "";
  const padTxt = `P=${u.pad[0].toFixed(2)} A=${u.pad[1].toFixed(2)} D=${u.pad[2].toFixed(2)}`;
  const html = `<div class="utt ${kind} ${silent}">
    <div class="speaker">
      <span class="char-emoji">${CHAR_EMOJI[payload.speaker.id] || ""}</span>
      ${payload.speaker.name}
      ${badge(u.emotion_label || "neutral")}
      <span class="badge">${padTxt}</span>
    </div>
    <div class="speech">${escapeHtml(speech)}</div>
    ${thought}
    ${action}
  </div>`;
  c.insertAdjacentHTML("beforeend", html);
  c.scrollTop = c.scrollHeight;

  // trim to last 100 utterances
  while (c.children.length > 100) c.removeChild(c.firstChild);
}

function appendDivider(text) {
  const c = document.getElementById("conv");
  c.insertAdjacentHTML("beforeend", `<div class="session-divider">— ${text} —</div>`);
  c.scrollTop = c.scrollHeight;
}

function appendSessionEnd(payload) {
  const div = document.getElementById("session-end-info");
  div.innerHTML = (payload.per_character || []).map(p => {
    const eps = (p.episodic || []).map(e => `<div class="item">${escapeHtml(e.content || "")}</div>`).join("");
    return `<div class="recalled"><b>${p.character_name}</b>${eps}</div>`;
  }).join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

const evtSrc = new EventSource("/events");
evtSrc.onmessage = (ev) => {
  try {
    const msg = JSON.parse(ev.data);
    if (msg.type === "snapshot") {
      renderSnapshot(msg.data);
    } else if (msg.type === "turn") {
      appendUtterance(msg.data);
      renderSnapshot(msg.data.snapshot);
    } else if (msg.type === "session_start") {
      document.getElementById("session-idx").textContent = msg.data.session_count;
      appendDivider(`SESSION ${msg.data.session_count} START — ${msg.data.context || ""}`);
    } else if (msg.type === "session_end") {
      appendDivider(`SESSION END — analyzed ${msg.data.total_episodic} episodic memories`);
      appendSessionEnd(msg.data);
    } else if (msg.type === "inter_session") {
      appendDivider(`待機中 (${msg.data.delay_seconds}s)`);
    } else if (msg.type === "error") {
      appendDivider("ERROR: " + msg.data.message);
    }
  } catch (e) {
    console.error("parse failed", e, ev.data);
  }
};
evtSrc.onerror = (e) => console.error("SSE error", e);

// initial state fetch
fetch("/state").then(r => r.json()).then(renderSnapshot);
</script>
</body>
</html>
"""


def _make_handler(driver: SessionDriver):
    class Handler(BaseHTTPRequestHandler):
        # silence default logging spam
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                return self._send_html(_HTML_TEMPLATE)
            if self.path.startswith("/state"):
                return self._send_json(driver.latest_snapshot())
            if self.path.startswith("/events"):
                return self._send_sse(driver)
            self.send_error(404, "Not Found")

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_sse(self, driver: SessionDriver) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            q = driver.subscribe()
            try:
                # heartbeat
                last_hb = time.time()
                while True:
                    try:
                        payload = q.get(timeout=1.0)
                        line = "data: " + json.dumps(
                            payload, ensure_ascii=False
                        ) + "\n\n"
                        self.wfile.write(line.encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        if time.time() - last_hb > 15:
                            try:
                                self.wfile.write(b": heartbeat\n\n")
                                self.wfile.flush()
                                last_hb = time.time()
                            except (BrokenPipeError, ConnectionResetError):
                                return
                    except (BrokenPipeError, ConnectionResetError):
                        return
            finally:
                driver.unsubscribe(q)

    return Handler


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def _build_llm(use_mock: bool) -> tuple[LLMAdapter, str]:
    if use_mock or not os.environ.get("ANTHROPIC_API_KEY"):
        return MockLLMAdapter(latency_seconds=0.4), "mock-llm"
    from pneuma_core.llm.claude import ClaudeAdapter

    return ClaudeAdapter.from_env(), "claude (real)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pneuma multi-agent web UI",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--use-mock-llm", action="store_true")
    parser.add_argument(
        "--context", type=str,
        default="部室で何気ない雑談中。窓から夕方の光が差している。",
    )
    parser.add_argument("--turn-limit", type=int, default=40,
                        help="Max turns per session (default: 40)")
    parser.add_argument("--inter-turn", type=float, default=2.5,
                        help="Seconds between turns (default: 2.5)")
    parser.add_argument("--inter-session", type=float, default=8.0,
                        help="Seconds between sessions (default: 8.0)")
    parser.add_argument("--persist", type=str, default=None, metavar="DB_PATH",
                        help="正史の連続性 (Issue #31): SQLite パスを指定すると各セッションの"
                             "記憶/感情/関係を永続化し、セッション間で引き継ぐ（正史が積み上がる）")
    # 面白さエンジンのトグル (Issue #22) — text_runner と同じ
    parser.add_argument("--no-intent", action="store_true",
                        help="思惑（#20）を無効化する（デフォルトは有効）")
    parser.add_argument("--quirk", action="store_true",
                        help="キャラの思考のクセを発話に注入する")
    parser.add_argument("--terse", action="store_true",
                        help="発話を短く・大喜利的にする（テンポ）")
    parser.add_argument("--lateral", action="store_true",
                        help="水平思考的飛躍（連想・アナロジー）を促す")
    parser.add_argument("--emotion-dynamics", action="store_true",
                        help="感情を性格 baseline に引き戻し温度差を作る")
    parser.add_argument("--structured-ending", action="store_true",
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

    llm, llm_label = _build_llm(args.use_mock_llm)
    driver = SessionDriver(
        llm=llm,
        llm_label=llm_label,
        context=args.context,
        use_mock=args.use_mock_llm,
        turn_limit_per_session=args.turn_limit,
        inter_turn_seconds=args.inter_turn,
        inter_session_seconds=args.inter_session,
        fun_config=fun_config,
        persist_path=args.persist,
    )

    print("=" * 60)
    print(f"Pneuma multi-agent web UI")
    print(f"  LLM   : {llm_label}")
    print(f"  Cast  : なでしこ / 千明 / あおい")
    print(f"  Ctx   : {args.context}")
    print(f"  Open  : http://{args.host}:{args.port}/")
    print(f"  Next  : starting first session immediately...")
    print("=" * 60)

    driver.run_forever_in_thread()
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(driver))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down ...")
        driver.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
