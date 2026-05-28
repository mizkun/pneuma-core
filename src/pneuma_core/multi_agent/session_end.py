"""MultiAgentSessionEndPipeline: N 体対応のセッション終了統合分析.

invariant: multi-agent-session-end — N 体ぶんの episodic/semantic/relation を
一括統合更新する。

Phase 0 C の最小実装:
- 各キャラごとに簡易 LLM 分析を呼んで episodic/semantic を生成
- 関係性は MultiAgentSession 内ですでに微増更新されているので、ここでは
  「session 全体での delta」として PAD 履歴の平均から再評価
- storage 永続化はオプショナル（テキストランナーは in-memory）

Issue #12: 構造化レスポンスは Tool Use 経由で取得（real Claude）。
モック / fallback 経路では strip_code_fences → json.loads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.multi_agent.session import MultiAgentSession
from pneuma_core.runtime.response_parser import strip_code_fences

logger = logging.getLogger(__name__)


# JSON Schema for Tool Use (Anthropic API). Issue #12 invariant.
_SESSION_END_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "episodic_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "emotional_valence": {
                        "type": "number",
                        "minimum": -1.0,
                        "maximum": 1.0,
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["content", "emotional_valence", "importance"],
            },
        },
        "semantic_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "content": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["action", "content", "confidence"],
            },
        },
        "relationship_changes": {
            "type": "array",
            "items": {"type": "object"},
        },
    },
    "required": ["episodic_memories", "semantic_updates", "relationship_changes"],
}

_SYSTEM_PROMPT = """\
あなたはAIキャラクターの記憶管理アシスタントです。
キャラクター「{name}」視点で、以下の会話セッションを分析し、保存すべき情報を
JSON形式で出力してください。

## 出力フォーマット
```json
{{
  "episodic_memories": [
    {{
      "content": "{name} 視点での具体的な出来事の記述（1文）",
      "emotional_valence": -1.0〜1.0,
      "importance": 0.0〜1.0
    }}
  ],
  "semantic_updates": [
    {{
      "action": "add",
      "content": "{name} がこのセッションから学んだ汎化的な知識（1文）",
      "confidence": 0.0〜1.0
    }}
  ],
  "relationship_changes": []
}}
```

## ルール
- episodic_memories は 1〜3 件を {name} の主観で記述
- semantic_updates は 0〜2 件
- 些細すぎる内容は省略
- relationship_changes は基本空配列（pipeline 側で別途更新）
"""


@dataclass
class CharacterEndResult:
    """1 キャラぶんの終了分析結果."""

    character_id: str
    character_name: str
    episodic: list[dict] = field(default_factory=list)
    semantic: list[dict] = field(default_factory=list)
    success: bool = True


@dataclass
class MultiAgentEndResult:
    """N 体ぶんの終了分析結果."""

    per_character: list[CharacterEndResult]
    total_episodic: int = 0
    total_semantic: int = 0


class MultiAgentSessionEndPipeline:
    """N 体対応セッション終了パイプライン.

    各キャラ視点で 1 回ずつ LLM 分析を呼び、結果を CharacterRuntimeState に
    積み戻す。
    """

    def __init__(self, llm: LLMAdapter, model: str | None = None) -> None:
        self._llm = llm
        self._model = model

    async def run(self, session: MultiAgentSession) -> MultiAgentEndResult:
        per_results: list[CharacterEndResult] = []
        total_ep = 0
        total_sem = 0

        for ch in session.conversation.participants:
            state = session.character_states[ch.id]
            view = session.conversation.history_as_messages(viewer_id=ch.id)
            if not view:
                per_results.append(CharacterEndResult(
                    character_id=ch.id,
                    character_name=ch.name,
                ))
                continue

            try:
                analysis = await self._analyze(ch.name, view)
            except Exception as e:
                logger.warning(
                    "session-end analysis failed for %s: %s", ch.name, e
                )
                per_results.append(CharacterEndResult(
                    character_id=ch.id,
                    character_name=ch.name,
                    success=False,
                ))
                continue

            ep_list = analysis.get("episodic_memories", []) or []
            sem_list = analysis.get("semantic_updates", []) or []

            # 内部状態に積み戻す
            for ep in ep_list:
                content = ep.get("content", "")
                if content:
                    state.episodic.append(content)

            per_results.append(CharacterEndResult(
                character_id=ch.id,
                character_name=ch.name,
                episodic=ep_list,
                semantic=sem_list,
            ))
            total_ep += len(ep_list)
            total_sem += len(sem_list)

        return MultiAgentEndResult(
            per_character=per_results,
            total_episodic=total_ep,
            total_semantic=total_sem,
        )

    async def _analyze(self, name: str, view: list[dict]) -> dict:
        prompt = _SYSTEM_PROMPT.format(name=name)

        # Issue #12: real Claude → Tool Use（schema 強制）; mock / fallback → generate().
        if self._supports_tool_use():
            try:
                return await self._llm.structured_completion(  # type: ignore[attr-defined]
                    system_prompt=prompt,
                    messages=view[-30:],
                    tool_name="record_session_memories",
                    tool_description=(
                        "Record episodic memories, semantic updates, and "
                        "relationship changes from the session."
                    ),
                    schema=_SESSION_END_SCHEMA,
                    model=self._model,
                    temperature=0.3,
                    max_tokens=1024,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "session-end structured_completion failed: %s, "
                    "falling back to text response",
                    e,
                )
                # Fall through to generate() path

        response = await self._llm.generate(
            LLMRequest(
                system_prompt=prompt,
                messages=view[-30:],
                model=self._model,
                temperature=0.3,
                max_tokens=1024,
            )
        )
        return self._parse_json(response.content)

    def _supports_tool_use(self) -> bool:
        """Adapter が structured_completion を正式実装しているか判定."""
        marker = getattr(type(self._llm), "supports_structured_completion", None)
        return bool(marker)

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Pre-strip code fences then json.loads (invariant: llm-response-pre-strip)."""
        stripped = strip_code_fences(content or "")
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return {}
