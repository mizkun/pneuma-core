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

from pneuma_core.emotion.baseline import personality_to_pad_baseline
from pneuma_core.llm.adapter import LLMAdapter, LLMRequest
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.message import StructuredResponse
from pneuma_core.multi_agent.circuit_breaker import CircuitBreaker
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.floor_controller import (
    FloorController,
    Utterance,
)
from pneuma_core.multi_agent.emotion_text_consistency import (
    resolve_emotion_text_consistency,
)
from pneuma_core.multi_agent.fun_config import FunEngineConfig
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


# ベースの発話プロンプト（思惑・各創発要素のセクションは下で動的に挿入する）。
# {fun_sections} に FunEngineConfig 由来のセクションが入る。
_SPEECH_SYSTEM_PROMPT = """\
あなたは「{name}」というキャラクターです。
{cast_roster}
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
{fun_sections}
## 出力ルール
- 必ず以下の JSON 形式で 1 回だけ出力すること
- speech は 1〜2 文以内。長文は禁止。沈黙してもよい（その場合は ""）
- thought（心の中の声）は内心を書く。表の speech と食い違ってよい
  （例: 表では強気だが内心は引けないだけ）。
- action は 1 文以内。例: 「笑顔で頷く」「下を向く」「両手を握る」
- 自分らしい口調・性格を保つこと
- 視聴者を意識した演技はしない。あくまで自分自身として話す。

```json
{{"speech": "<セリフ>", "thought": "<心の中の声>", "action": "<身振り>"}}
```
"""


# ── 各創発要素のプロンプトセクション (Issue #22) ──
# いずれも「傾向・性質を与える」だけで、出力（セリフ）は LLM の自律に委ねる。

_INTENT_SECTION = """\

## あなたの今の思惑（短期ゴール）
表のゴール: {surface_goal}
裏の本音: {hidden_goal}
思惑の強さ: {intensity}
- speech（表の発言）は、上の「表のゴール（思惑）」に向かって自然に会話を進める。
  あくまで自分の思惑のため（視聴者向けの演技ではない）。
- thought は上の「裏の本音」を踏まえる。
"""

_QUIRK_SECTION = """\

## あなたの思考のクセ
{quirk}
- お題や話題を、上のクセの角度に引き寄せて捉えること。
- ロジカルで優等生的な深掘り回答は避ける。賢く無難にまとめず、クセのある角度で返す。
"""

_TERSE_SECTION = """\

## テンポ
- 発話は短く、大喜利のようにテンポよく。長い説明・前置きはしない。
- 一言〜短い一文で、リズムよく返すこと。
"""

_LATERAL_SECTION = """\

## 水平思考的な飛躍
- 開放性が高いときは、お題を一段抽象化して、一見無関係な別のことと結びつける
  連想・アナロジー（「◯◯も△△と一緒だよね」）を時々する。
- 飛ばしすぎて意味不明にならず、言われてみれば確かにと共感できる絶妙な距離で。
  どんなアナロジーにするかは自分で考えること（中身は決めつけない）。
"""

_ENDING_SECTION = """\

## そろそろ終盤（締切・決定の外圧）
- 会話の残り時間が少ない。そろそろ結論・決着に向かう時間だ。
- 何に決まるか・どう着地するかは自分たちで決めてよい（結末は指定しない）。
  ただし、いつまでも広げず、収束に向けて一歩進める発言をすること。
"""


# 構造的オチ（structured_ending）の「終盤」判定: 残りターンの割合がこれ以下なら終盤。
_ENDGAME_REMAINING_RATIO = 0.5
# 感情ダイナミクス（emotion_dynamics）ON 時、推定 PAD を性格 baseline 側へ
# 引き戻す重み（0=引き戻さない 〜 1=baseline そのもの）。
# 0.5 で「他者につられた高止まり」を半分、自分の性格起点に引き戻す。
_EMOTION_BASELINE_PULL = 0.5


# キャスト呼称表 (Issue #23 バグ3, cast-naming-fixed)。
# 本名 → このキャラを他キャラが呼ぶときの愛称・呼び方。発話プロンプト先頭に
# 固定表として明示し、旧デモの名前（「リン」等）への逸脱を防ぐ。
# ここに無いキャラ（テスト用など）は本名のみを呼称として載せる。
_CAST_NICKNAMES: dict[str, list[str]] = {
    "なでしこ": ["なでしこ"],
    "千明": ["ちかちゃん", "千明"],
    "あおい": ["あおいちゃん", "あおい"],
}


def build_cast_roster_section(participants: list[Character]) -> str:
    """発話プロンプトに載せるキャスト呼称表を組み立てる (Issue #23 バグ3).

    invariant: cast-naming-fixed — このセッションに実在するキャストの本名と
    呼び方だけを明示し、それ以外の名前（旧デモ「リン」「アイネ」等）を会話に
    出さないよう固定する。

    Args:
        participants: このセッションの参加キャラクター。

    Returns:
        プロンプトに差し込むキャスト表テキスト（末尾改行付き）。
    """
    lines = ["", "## このセッションの登場人物（この名前以外は使わない）"]
    for ch in participants:
        nicks = _CAST_NICKNAMES.get(ch.name, [ch.name])
        # 本名は必ず含める（重複は除く）
        names = list(dict.fromkeys([ch.name, *nicks]))
        call = " / ".join(names)
        lines.append(f"- {ch.name}（呼び方: {call}）")
    lines.append(
        "- 上記にいない名前（他作品・旧デモのキャラ名など）は絶対に使わないこと。"
    )
    return "\n".join(lines) + "\n"


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
        fun_config: FunEngineConfig | None = None,
    ) -> None:
        self.conversation = conversation
        self._llm = llm
        self._floor = floor_controller or FloorController()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._emotion_engine = emotion_engine or EmotionEngine(llm=llm)
        self._intent_generator = intent_generator or IntentGenerator(llm=llm)
        self.shared_context = shared_context
        # 面白さエンジンのトグル (Issue #22)。デフォルトは enable_intent のみ True
        # で #20 の挙動を保つ（fun-engine-toggleable）。FunEngineConfig は frozen
        # なので共有デフォルトでも安全。
        self.fun_config = fun_config or FunEngineConfig()
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

        fun_config.enable_intent が False のときは思惑を生成しない
        （fun-engine-toggleable: OFF 時は注入もスキップ）。

        Returns:
            char_id → Intent の dict（self.intents と同一参照）。
        """
        if not self.fun_config.enable_intent:
            return self.intents
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
                # 感情ダイナミクス (Issue #22, emotion-temperature-diversity):
                # 推定 PAD を各キャラの性格 baseline 側に引き戻し、「全員が同方向に
                # 高止まり」を解消して温度差を作る。
                if self.fun_config.enable_emotion_dynamics:
                    new_emotion = self._apply_emotion_dynamics(
                        ch, new_emotion
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
        Issue #22: fun_config に応じてクセ/テンポ/水平思考/構造的オチの
        プロンプトセクションを注入する（傾向の付与のみ。出力は LLM の自律）。
        """
        system_prompt = _SPEECH_SYSTEM_PROMPT.format(
            name=speaker.name,
            cast_roster=build_cast_roster_section(
                self.conversation.participants
            ),
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
            fun_sections=self._build_fun_sections(speaker),
        )

        # speaker 視点で history を渡す
        messages = self.conversation.history_as_messages(viewer_id=speaker.id)
        if not messages:
            # キックオフ
            messages = [{
                "role": "user",
                "content": f"今、{self.shared_context or '部室で'} です。何か話して。",
            }]

        # テンポ (enable_terse): speech を短くするのはプロンプト指示 (_TERSE_SECTION)
        # で担保する。max_tokens は speech/thought/action の JSON 構造を壊さない値を
        # 常に確保する（terse で絞ると JSON が途中で切れてパース失敗し、生 JSON が
        # speech に漏れて thought/action が消えるため。Issue #22 QA で発覚）。
        max_tokens = 256

        request = LLMRequest(
            system_prompt=system_prompt,
            messages=messages[-12:],
            temperature=0.9,
            max_tokens=max_tokens,
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

        # pad-text-consistency (Issue #23 バグ1): 確定済み emotion_label と、
        # 生成された thought（内心）の感情方向が正反対なら label を thought 側に
        # 合わせる。生成順序は「PAD/emotion 確定 → プロンプト注入 → 生成 → 整合」で
        # 固定する。structured_ending（終盤収束）も感情を強制上書きしない
        # （speech/thought だけ収束を促し、emotion はこの整合を通る）。
        emotion_label, label_changed = resolve_emotion_text_consistency(
            state.emotion.emotion_label, thought
        )
        if label_changed:
            logger.info(
                "pad-text-consistency: %s の emotion_label を %r→%r に補正"
                "（thought と整合）",
                speaker.name, state.emotion.emotion_label, emotion_label,
            )

        return Utterance(
            speaker_id=speaker.id,
            speaker_name=speaker.name,
            speech=speech,
            action=action,
            thought=thought,
            emotion_label=emotion_label,
            pad=(state.emotion.pleasure, state.emotion.arousal,
                 state.emotion.dominance),
            recalled_memories=list(state.episodic[-3:]),
        )

    def _apply_emotion_dynamics(
        self, character: Character, estimated: EmotionalState
    ) -> EmotionalState:
        """推定 PAD を性格 baseline 側に引き戻す (Issue #22).

        invariant: emotion-temperature-diversity — 全キャラが同方向に高止まり
        するのを解消する。各キャラの性格から導いた PAD baseline は個体差が大きい
        ため、推定値（他者につられた値）を baseline に向けてブレンドすることで、
        会話が進んでも「自分の性格起点の温度」が残り、温度差・収束が生まれる。

        blend(estimated, baseline) = (1-w)*estimated + w*baseline
        （w=_EMOTION_BASELINE_PULL）。感情そのものを消すのではなく、収束しすぎを
        防ぐ「減衰」として効かせる（出力 = 結末は決めない、創発）。
        """
        base = personality_to_pad_baseline(character.personality)
        w = _EMOTION_BASELINE_PULL

        def pull(value: float, target: float) -> float:
            return (1.0 - w) * value + w * target

        return EmotionalState(
            pleasure=pull(estimated.pleasure, base[0]),
            arousal=pull(estimated.arousal, base[1]),
            dominance=pull(estimated.dominance, base[2]),
            emotion_label=estimated.emotion_label,
            situation=estimated.situation,
        )

    def _build_fun_sections(self, speaker: Character) -> str:
        """fun_config に応じて発話プロンプトに足す創発セクションを組み立てる.

        各セクションは「傾向・性質を与える」だけで、出力（セリフ）は決めない
        （台本化しない）。OFF のフラグは何も足さない（OFF 時は挙動不変）。
        """
        cfg = self.fun_config
        sections: list[str] = []

        # 思惑（#20）— enable_intent ON のときだけ注入（intent-driven-utterance）
        if cfg.enable_intent:
            intent = self.intents.get(speaker.id)
            sections.append(_INTENT_SECTION.format(
                surface_goal=intent.surface_goal if intent else "（特になし）",
                hidden_goal=(intent.hidden_goal
                             if intent and intent.hidden_goal else "（なし）"),
                intensity=round(intent.intensity, 2) if intent else 0.0,
            ))

        # クセ（quirk-emergent）— quirk テキストがあるときだけ
        if cfg.enable_quirk and speaker.quirk:
            sections.append(_QUIRK_SECTION.format(quirk=speaker.quirk.strip()))

        # テンポ（メカニカル + プロンプト指示）
        if cfg.enable_terse:
            sections.append(_TERSE_SECTION)

        # 水平思考的飛躍（lateral-thinking-promoted）
        if cfg.enable_lateral_thinking:
            sections.append(_LATERAL_SECTION)

        # 構造的オチ（structured-ending-external）— 終盤にのみ外圧を注入
        if cfg.enable_structured_ending and self._is_endgame():
            sections.append(_ENDING_SECTION)

        return "".join(sections)

    def _is_endgame(self) -> bool:
        """構造的オチの「終盤」判定 (Issue #22, structured-ending-external).

        CircuitBreaker の残りターン数を使う（メカニカルな制約）。残りターンが
        max_turns の一定割合以下になったら「終盤」とみなす。max_turns が
        無限大相当（巨大）なら終盤にはならない。
        """
        cb = self.circuit_breaker
        max_turns = cb.max_turns
        if max_turns <= 0:
            return False
        remaining_ratio = cb.remaining_turns() / max_turns
        return remaining_ratio <= _ENDGAME_REMAINING_RATIO

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
