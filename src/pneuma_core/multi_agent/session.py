"""MultiAgentSession: N 体 ↔ 1 セッションの実行ループ.

invariants:
- thinking-on-every-turn: 各ターンで全キャラの内部状態は更新される（PAD推定）
- observation-turn-state-update: 他キャラの発話は自分の history に積まれる
- utterance-action-pair: 出力は (speech, action) のペア
- intent-driven-utterance: 各キャラの思惑（surface_goal）が発話プロンプトに注入され
  会話を駆動する。FloorController も intent_relevance を加味する。
- speech-thought-separation: speech（表）と thought（裏の本音）を分離記録する。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState, EmotionLabel
from pneuma_core.models.message import StructuredResponse
from pneuma_core.multi_agent.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.floor_controller import (
    FloorController,
    Utterance,
)
from pneuma_core.multi_agent.intent import Intent, IntentGenerator
from pneuma_core.runtime.emotion_engine import EmotionEngine, pad_to_label
from pneuma_core.runtime.response_parser import parse_structured_response

logger = logging.getLogger(__name__)


@dataclass
class CharacterRuntimeState:
    """1 キャラぶんのセッション内部状態."""

    character: Character
    emotion: EmotionalState
    # キャラごとの簡易 episodic 記憶（このセッション中の出来事）
    episodic: list[str] = field(default_factory=list)
    # キャラごとの関係性（target_id → closeness, trust）
    relations: dict[str, dict[str, float]] = field(default_factory=dict)
    # PAD の履歴（時系列観察用）
    pad_history: list[tuple[float, float, float, str]] = field(default_factory=list)
    speak_count: int = 0
    # 直近の内心（speech-thought-separation の観測用）
    last_thought: str = ""


@dataclass
class TurnResult:
    """1 ターンの実行結果."""

    turn_index: int
    speaker: Character
    utterance: Utterance
    emotion_changes: dict[str, EmotionalState]  # char_id → 更新後 emotion
    recalled_memories: list[str]
    llm_call_count: int = 0
    tokens_used: int = 0


_SPEECH_SYSTEM_PROMPT = """\
あなたは「{name}」というキャラクターです。

## プロフィール
{profile}

## 話し方
{speaking_style}

## 性格 (Big Five, 0〜1)
- 開放性 (Openness): {openness}
- 誠実性 (Conscientiousness): {conscientiousness}
- 外向性 (Extraversion): {extraversion}
- 協調性 (Agreeableness): {agreeableness}
- 神経症傾向 (Neuroticism): {neuroticism}

## 現在の状況
{situation}

## あなたの現在の気分 (PAD)
- pleasure: {pleasure} ({emotion_label})
- arousal: {arousal}
- dominance: {dominance}

## 共有コンテクスト
{context}

## あなたの今の思惑（短期ゴール）
表のゴール: {surface_goal}
裏の本音: {hidden_goal}
思惑の強さ: {intensity}

## 出力ルール
- 必ず以下の JSON 形式で 1 回だけ出力すること
- speech は 1〜2 文以内。長文は禁止。沈黙してもよい（その場合は ""）
- speech（表の発言）は、上の「表のゴール（思惑）」に向かって自然に会話を進めること。
  ただし視聴者を意識した演技はしない。あくまで自分の思惑のため。
- thought（心の中の声）は、上の「裏の本音」を踏まえた内心を書く。
  表の speech と食い違ってよい（例: 表では強気だが内心は引けないだけ）。
- action は 1 文以内。例: 「笑顔で頷く」「下を向く」「両手を握る」
- 自分らしい口調・性格を保つこと

```json
{{"speech": "<セリフ>", "thought": "<心の中の声>", "action": "<身振り>"}}
```
"""


class MultiAgentSession:
    """N 体マルチエージェントの会話セッション実行器.

    Conversation + 各キャラの CharacterRuntimeState を管理し、
    FloorController で発話順を決めながら 1 ターンずつ進める。
    """

    def __init__(
        self,
        conversation: Conversation,
        llm: LLMAdapter,
        floor_controller: FloorController | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        emotion_engine: EmotionEngine | None = None,
        intent_generator: IntentGenerator | None = None,
        shared_context: str = "",
        session_id: str | None = None,
    ) -> None:
        self.conversation = conversation
        self._llm = llm
        self._floor = floor_controller or FloorController()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._emotion_engine = emotion_engine or EmotionEngine(llm=llm)
        self._intent_generator = intent_generator or IntentGenerator(llm=llm)
        self.shared_context = shared_context
        self.session_id = session_id or f"masess-{uuid.uuid4().hex[:12]}"
        self.started_at = datetime.now(timezone.utc)

        # キャラごとの思惑（短期ゴール）。ensure_intents() で遅延生成する。
        self.intents: dict[str, Intent] = {}

        # キャラごとの内部状態
        self.character_states: dict[str, CharacterRuntimeState] = {}
        for ch in conversation.participants:
            self.character_states[ch.id] = CharacterRuntimeState(
                character=ch,
                emotion=EmotionalState(
                    pleasure=0.0, arousal=0.0, dominance=0.0,
                    emotion_label="neutral", situation=shared_context,
                ),
            )

        # 関係性初期化（cross relation）
        for ch in conversation.participants:
            for peer in conversation.participants:
                if peer.id == ch.id:
                    continue
                self.character_states[ch.id].relations[peer.id] = {
                    "closeness": 0.5,
                    "trust": 0.5,
                    "target_name": peer.name,
                }

        self._turn_index = 0

    async def ensure_intents(self) -> dict[str, Intent]:
        """各キャラの思惑（短期ゴール）を生成して保持する（未生成時のみ）.

        invariant: intent-driven-utterance — 会話開始前に全キャラの思惑を
        長期欲求 + shared_context から創発させる。冪等（既に生成済みなら再生成
        しない）。

        Returns:
            char_id → Intent の dict（self.intents と同一参照）。
        """
        for ch in self.conversation.participants:
            if ch.id in self.intents:
                continue
            self.intents[ch.id] = await self._intent_generator.generate(
                character=ch,
                shared_context=self.shared_context,
            )
        return self.intents

    async def run_turn(self) -> TurnResult | None:
        """1 ターン進める.

        Returns:
            TurnResult（成功）、CircuitBreaker tripped なら None。
        """
        if not self.circuit_breaker.allow():
            return None

        # 思惑を未生成なら生成（会話を駆動する短期ゴール）
        await self.ensure_intents()

        self._turn_index += 1
        self.circuit_breaker.record_turn()

        # 1. FloorController で発話者決定（思惑を加味）
        speaker = self._floor.next_speaker(
            history=self.conversation.history,
            participants=self.conversation.participants,
            intents=self.intents,
        )
        state = self.character_states[speaker.id]

        # 2. LLM 呼び出し → speech/action 生成
        utterance = await self._generate_utterance(speaker, state)

        # 3. Conversation history に積む
        self.conversation.record_utterance(utterance)
        state.speak_count += 1

        # 4. 全キャラの感情更新（thinking-on-every-turn）
        emotion_changes: dict[str, EmotionalState] = {}
        for ch in self.conversation.participants:
            ch_state = self.character_states[ch.id]
            try:
                # 他キャラ視点で見た会話履歴で PAD 推定
                view = self.conversation.history_as_messages(viewer_id=ch.id)
                new_emotion = await self._emotion_engine.estimate(
                    personality=ch.personality,
                    messages=view[-8:],  # 直近 8 ターンぶん
                )
                # PAD → EmotionLabel に正規化（emotion_engine.estimate の
                # emotion_label は LLM 自由出力なので統一する）
                label = pad_to_label(new_emotion)
                normalized = EmotionalState(
                    pleasure=new_emotion.pleasure,
                    arousal=new_emotion.arousal,
                    dominance=new_emotion.dominance,
                    emotion_label=label.value,
                    situation=new_emotion.situation or self.shared_context,
                )
                ch_state.emotion = normalized
                emotion_changes[ch.id] = normalized
                ch_state.pad_history.append(
                    (normalized.pleasure, normalized.arousal,
                     normalized.dominance, label.value)
                )

                # 関係性のクロス更新（簡易: 喋った人とリスナーの closeness を
                # 微増、PAD pleasure が高ければさらに trust+）
                if ch.id != speaker.id:
                    rel = ch_state.relations.get(speaker.id)
                    if rel is not None:
                        rel["closeness"] = min(
                            1.0, rel["closeness"] + 0.005
                        )
                        if normalized.pleasure > 0.3:
                            rel["trust"] = min(
                                1.0, rel["trust"] + 0.003
                            )
                        elif normalized.pleasure < -0.2:
                            rel["trust"] = max(
                                0.0, rel["trust"] - 0.002
                            )
            except Exception as e:
                logger.warning(
                    "emotion estimation failed for %s: %s", ch.name, e
                )

        return TurnResult(
            turn_index=self._turn_index,
            speaker=speaker,
            utterance=utterance,
            emotion_changes=emotion_changes,
            recalled_memories=list(state.episodic[-3:]),
            llm_call_count=1 + len(self.conversation.participants),
        )

    async def _generate_utterance(
        self,
        speaker: Character,
        state: CharacterRuntimeState,
    ) -> Utterance:
        """LLM を呼んで 1 ターンぶんの speech/thought/action を生成.

        invariants: intent-driven-utterance（思惑を注入）,
        speech-thought-separation（thought を別フィールドで記録）。
        """
        intent = self.intents.get(speaker.id)
        system_prompt = _SPEECH_SYSTEM_PROMPT.format(
            name=speaker.name,
            profile=speaker.profile or "",
            speaking_style=speaker.speaking_style or "自由な口調",
            openness=speaker.personality.openness,
            conscientiousness=speaker.personality.conscientiousness,
            extraversion=speaker.personality.extraversion,
            agreeableness=speaker.personality.agreeableness,
            neuroticism=speaker.personality.neuroticism,
            situation=state.emotion.situation or self.shared_context,
            pleasure=round(state.emotion.pleasure, 2),
            emotion_label=state.emotion.emotion_label,
            arousal=round(state.emotion.arousal, 2),
            dominance=round(state.emotion.dominance, 2),
            context=self.shared_context or "（特になし）",
            surface_goal=intent.surface_goal if intent else "（特になし）",
            hidden_goal=(intent.hidden_goal if intent and intent.hidden_goal
                         else "（なし）"),
            intensity=round(intent.intensity, 2) if intent else 0.0,
        )

        # speaker 視点で history を渡す
        messages = self.conversation.history_as_messages(viewer_id=speaker.id)
        if not messages:
            # キックオフ
            messages = [{
                "role": "user",
                "content": f"今、{self.shared_context or '部室で'} です。何か話して。",
            }]

        request = LLMRequest(
            system_prompt=system_prompt,
            messages=messages[-12:],
            temperature=0.9,
            max_tokens=256,
        )
        try:
            response = await self._llm.generate(request)
            self.circuit_breaker.record_tokens(
                int(response.usage.get("input_tokens", 0))
                + int(response.usage.get("output_tokens", 0))
            )
            structured = parse_structured_response(response.content)
        except Exception as e:
            logger.warning("LLM generate failed for %s: %s", speaker.name, e)
            structured = StructuredResponse(speech="…", action="黙る")

        speech = structured.speech or ""
        action = structured.action or ""
        thought = structured.thought or ""

        # speech-thought-separation: 内心を runtime state に保持（観測用）
        state.last_thought = thought

        return Utterance(
            speaker_id=speaker.id,
            speaker_name=speaker.name,
            speech=speech,
            action=action,
            thought=thought,
            emotion_label=state.emotion.emotion_label,
            pad=(state.emotion.pleasure, state.emotion.arousal,
                 state.emotion.dominance),
            recalled_memories=list(state.episodic[-3:]),
        )

    def snapshot(self) -> dict[str, Any]:
        """現在のセッション状態をシリアライズ可能な dict にまとめる.

        Web UI / CLI / テストでの可視化に使う。
        """
        return {
            "session_id": self.session_id,
            "turn_index": self._turn_index,
            "circuit_state": self.circuit_breaker.state.value,
            "circuit_remaining_turns": self.circuit_breaker.remaining_turns(),
            "characters": [
                {
                    "id": st.character.id,
                    "name": st.character.name,
                    "personality": {
                        "openness": st.character.personality.openness,
                        "conscientiousness": st.character.personality.conscientiousness,
                        "extraversion": st.character.personality.extraversion,
                        "agreeableness": st.character.personality.agreeableness,
                        "neuroticism": st.character.personality.neuroticism,
                    },
                    "emotion": {
                        "pleasure": st.emotion.pleasure,
                        "arousal": st.emotion.arousal,
                        "dominance": st.emotion.dominance,
                        "emotion_label": st.emotion.emotion_label,
                    },
                    "speak_count": st.speak_count,
                    "relations": {
                        pid: {
                            "target_name": rel["target_name"],
                            "closeness": round(rel["closeness"], 3),
                            "trust": round(rel["trust"], 3),
                        }
                        for pid, rel in st.relations.items()
                    },
                    "recent_episodic": st.episodic[-5:],
                    "intent": self._intent_snapshot(st.character.id),
                    "recent_thought": st.last_thought,
                }
                for st in self.character_states.values()
            ],
            "history_tail": [
                {
                    "speaker_id": u.speaker_id,
                    "speaker_name": u.speaker_name,
                    "speech": u.speech,
                    "thought": u.thought,
                    "action": u.action,
                    "emotion_label": u.emotion_label,
                    "pad": list(u.pad),
                }
                for u in self.conversation.history[-20:]
            ],
        }

    def _intent_snapshot(self, char_id: str) -> dict[str, Any]:
        """1 キャラの思惑を snapshot 用 dict にする（未生成なら空）."""
        intent = self.intents.get(char_id)
        if intent is None:
            return {"surface_goal": "", "hidden_goal": None, "intensity": 0.0}
        return {
            "surface_goal": intent.surface_goal,
            "hidden_goal": intent.hidden_goal,
            "intensity": round(intent.intensity, 2),
        }
