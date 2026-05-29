"""FloorController: 次の発話者を決めるスコアリング.

invariants:
- turn-rotation-base: 基本は順次キャラクターを巡る (A→B→C→A...)
- floor-controller-decision: 性格（外向性）・直前発話への関連度・
  直近の発話頻度（喋りすぎ抑制）をスコアリングする
- intent-driven-utterance: 直前発話が自分の思惑（intent）に関わるキャラは
  発話しやすくなる（intent_relevance を加味）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pneuma_core.models.character import Character

if TYPE_CHECKING:
    from pneuma_core.multi_agent.intent import Intent


@dataclass(frozen=True)
class Utterance:
    """1 ターンの発話 (speech, thought, action のセット).

    invariant: utterance-action-pair — speech 空 = 沈黙、action 空 = 動かない。
    両方空も許容（完全沈黙ターン）。
    invariant: speech-thought-separation — thought は表の speech と分離された
    裏の本音。視聴者は thought を観測できるが会話相手には見えない。
    """

    speaker_id: str
    speaker_name: str
    speech: str = ""
    action: str = ""
    thought: str = ""
    emotion_label: str = "neutral"
    pad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    recalled_memories: list[str] = field(default_factory=list)


_DEFAULT_RECENT_WINDOW = 4
_W_EXTRAVERSION = 0.4
_W_RELEVANCE = 0.3
_W_RECENT_PENALTY = 0.5
# 直前発話が自分の思惑に関わるときのスコア寄与（intensity でスケール）
_W_INTENT_RELEVANCE = 0.4

# 直前の発話者を即連続させないハード抑制（同一連続防止）
_LAST_SPEAKER_REPEAT_PENALTY = 0.9


class FloorController:
    """次の発話者を決めるスコアリング器.

    score = (extraversion * W_E)
          + (relevance * W_R)
          - (recent_speak_penalty * W_RP)
          - (last_speaker_repeat_penalty if 直前と同一)
    """

    def __init__(
        self,
        recent_window: int = _DEFAULT_RECENT_WINDOW,
        w_extraversion: float = _W_EXTRAVERSION,
        w_relevance: float = _W_RELEVANCE,
        w_recent_penalty: float = _W_RECENT_PENALTY,
        w_intent_relevance: float = _W_INTENT_RELEVANCE,
    ) -> None:
        self._recent_window = recent_window
        self._w_e = w_extraversion
        self._w_r = w_relevance
        self._w_rp = w_recent_penalty
        self._w_ir = w_intent_relevance

    def next_speaker(
        self,
        history: list[Utterance],
        participants: list[Character],
        intents: dict[str, Intent] | None = None,
    ) -> Character:
        """次の発話者を返す.

        - 最初のターン (history 空): 外向性最大のキャラ
        - 以降: スコアリング最大のキャラ

        Args:
            intents: char_id → Intent。渡すと intent_relevance を加味する。
        """
        if not participants:
            raise ValueError("participants must not be empty")

        if not history:
            return max(participants, key=lambda c: c.personality.extraversion)

        # スコアリング
        scored: list[tuple[float, Character]] = []
        for ch in participants:
            score = self.score(ch, history, intents=intents)
            scored.append((score, ch))

        # 最高スコアのキャラ（同点は順序で）
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def score(
        self,
        character: Character,
        history: list[Utterance],
        intents: dict[str, Intent] | None = None,
    ) -> float:
        """指定キャラの「次に喋る」スコアを計算する.

        invariant: intent-driven-utterance — intents が渡されたとき、直前発話が
        自分の思惑に関わるキャラのスコアを intensity でスケールして加点する。
        """
        # 1. 外向性
        s = character.personality.extraversion * self._w_e

        # 2. 関連度（簡易: 直前発話に自分への言及があればボーナス）
        if history:
            last = history[-1]
            relevance = self._relevance(character, last)
            s += relevance * self._w_r

        # 3. 直近 N ターンの発話頻度（喋りすぎ抑制）
        recent = history[-self._recent_window :]
        my_recent = sum(1 for u in recent if u.speaker_id == character.id)
        recent_penalty = my_recent / max(1, self._recent_window)
        s -= recent_penalty * self._w_rp

        # 4. 直前と同一発話者なら強くペナルティ
        if history and history[-1].speaker_id == character.id:
            s -= _LAST_SPEAKER_REPEAT_PENALTY

        # 5. 思惑への関連（intent_relevance）— 直前発話が自分の思惑に絡むなら加点
        if history and intents:
            intent = intents.get(character.id)
            if intent is not None:
                ir = self._intent_relevance(intent, history[-1])
                s += ir * intent.intensity * self._w_ir

        return s

    @staticmethod
    def _relevance(character: Character, utterance: Utterance) -> float:
        """直前発話への関連度（0.0〜1.0）.

        簡易実装: 自分の名前が含まれていれば 1.0、それ以外は 0.3。
        将来は embedding cosine 等に差し替え予定。
        """
        text = (utterance.speech or "") + " " + (utterance.action or "")
        if character.name and character.name in text:
            return 1.0
        return 0.3

    @staticmethod
    def _intent_relevance(intent: Intent, utterance: Utterance) -> float:
        """直前発話が自分の思惑に関わる度合い（0.0〜1.0）.

        簡易実装: 思惑テキスト（surface/hidden）の語と直前発話の語の重なりで
        判定する。重なりがあれば 1.0、なければ 0.0。
        将来は embedding cosine 等に差し替え予定。
        """
        text = (utterance.speech or "") + " " + (utterance.action or "")
        goal_text = intent.surface_goal + " " + (intent.hidden_goal or "")
        tokens = _content_tokens(goal_text)
        if not tokens:
            return 0.0
        for tok in tokens:
            if tok in text:
                return 1.0
        return 0.0


_MIN_TOKEN_LEN = 2


def _content_tokens(text: str) -> list[str]:
    """思惑テキストから内容語の核（漢字/カタカナ/英数字の塊）を抜く.

    日本語は分かち書きしないので、形態素解析を使わずに「ひらがな（≒助詞・
    活用語尾）で区切る」近似でノイズを落とす。たとえば「お化け屋敷をやりたい」
    は ['化け屋敷'] に、「カフェで料理を褒められたい」は ['カフェ', '料理']
    になる。語の重なり検出（substring 一致）に十分な粒度を狙う。
    """
    # ひらがな以外（漢字・カタカナ・英数字）の連続塊を「内容語の核」として拾う。
    # 先頭に付くことのある「お」「ご」等の接頭ひらがなは塊に含めないが、
    # 内部の漢字塊が拾えれば一致するので問題ない。
    raw = re.findall(r"[一-鿿々〆ヵヶァ-ヴーA-Za-z0-9]+", text)
    out: list[str] = []
    for tok in raw:
        if len(tok) < _MIN_TOKEN_LEN:
            continue
        out.append(tok)
    return out
