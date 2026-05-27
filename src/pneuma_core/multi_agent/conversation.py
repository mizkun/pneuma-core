"""Conversation: N 体の participants と共通 history を保持する.

invariants:
- observation-turn-state-update — 自分のターンでなくても他キャラの発話は
  自分の history に積まれる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pneuma_core.models.character import Character
from pneuma_core.multi_agent.floor_controller import Utterance


@dataclass
class Conversation:
    """N 体参加の 1 つの会話セッション.

    participants: 参加キャラクター（2 体以上）
    history: 時系列の発話列。各 viewer から見ると role が assistant/user に
             変わる（history_as_messages 参照）。
    """

    participants: list[Character]
    history: list[Utterance] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.participants) < 2:
            raise ValueError(
                "Conversation requires at least 2 participants, "
                f"got {len(self.participants)}"
            )
        ids = {p.id for p in self.participants}
        if len(ids) != len(self.participants):
            raise ValueError("participant ids must be unique")

    def get_participant(self, char_id: str) -> Character:
        for p in self.participants:
            if p.id == char_id:
                return p
        raise KeyError(f"Participant not found: {char_id}")

    def record_utterance(self, utterance: Utterance) -> None:
        """history に発話を追加."""
        self.history.append(utterance)

    def history_as_messages(self, viewer_id: str) -> list[dict]:
        """viewer 視点で history を OpenAI/Anthropic 形式の messages 配列に変換.

        - viewer 自身の発話 → role="assistant"
        - 他キャラの発話 → role="user"  + "[name] speech (action)"

        observation-turn-state-update invariant の実体。
        """
        messages: list[dict] = []
        for u in self.history:
            text = u.speech or ""
            if u.action:
                text = f"{text} ({u.action})" if text else f"({u.action})"
            if not text:
                # 完全沈黙ターン — 観察対象として残す
                text = "（沈黙）"

            if u.speaker_id == viewer_id:
                messages.append({"role": "assistant", "content": text})
            else:
                messages.append({
                    "role": "user",
                    "content": f"[{u.speaker_name}] {text}",
                })
        return messages
