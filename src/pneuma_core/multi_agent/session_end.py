"""MultiAgentSessionEndPipeline: N 体対応のセッション終了統合分析.

invariant: multi-agent-session-end — N 体ぶんの episodic/semantic/relation を
一括統合更新する。

Phase 0 C の最小実装:
- 各キャラごとに簡易 LLM 分析を呼んで episodic/semantic を生成
- 関係性は MultiAgentSession 内ですでに微増更新されているので、ここでは
  「session 全体での delta」として PAD 履歴の平均から再評価
- storage 永続化はオプショナル（テキストランナーは in-memory）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.multi_agent.session import MultiAgentSession

logger = logging.getLogger(__name__)

_MD_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)

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

    @staticmethod
    def _parse_json(content: str) -> dict:
        text = content.strip()
        m = _MD_CODE_BLOCK_RE.match(text)
        if m:
            text = m.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
