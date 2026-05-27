"""FloorController: 次の発話者を決めるスコアリング.

invariants:
- turn-rotation-base: 基本は順次キャラクターを巡る (A→B→C→A...)
- floor-controller-decision: 性格（外向性）・直前発話への関連度・
  直近の発話頻度（喋りすぎ抑制）をスコアリングする
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pneuma_core.models.character import Character


@dataclass(frozen=True)
class Utterance:
    """1 ターンの発話 (speech, action のペア).

    invariant: utterance-action-pair — speech 空 = 沈黙、action 空 = 動かない。
    両方空も許容（完全沈黙ターン）。
    """

    speaker_id: str
    speaker_name: str
    speech: str = ""
    action: str = ""
    emotion_label: str = "neutral"
    pad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    recalled_memories: list[str] = field(default_factory=list)


_DEFAULT_RECENT_WINDOW = 4
_W_EXTRAVERSION = 0.4
_W_RELEVANCE = 0.3
_W_RECENT_PENALTY = 0.5

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
    ) -> None:
        self._recent_window = recent_window
        self._w_e = w_extraversion
        self._w_r = w_relevance
        self._w_rp = w_recent_penalty

    def next_speaker(
        self,
        history: list[Utterance],
        participants: list[Character],
    ) -> Character:
        """次の発話者を返す.

        - 最初のターン (history 空): 外向性最大のキャラ
        - 以降: スコアリング最大のキャラ
        """
        if not participants:
            raise ValueError("participants must not be empty")

        if not history:
            return max(participants, key=lambda c: c.personality.extraversion)

        # スコアリング
        scored: list[tuple[float, Character]] = []
        for ch in participants:
            score = self.score(ch, history)
            scored.append((score, ch))

        # 最高スコアのキャラ（同点は順序で）
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def score(self, character: Character, history: list[Utterance]) -> float:
        """指定キャラの「次に喋る」スコアを計算する."""
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
