"""Tests for FloorController (Issue #6 / Phase 0 C).

invariants:
- floor-controller-decision: 性格（外向性）・直前発話への関連度・
  直近の発話頻度をスコアリングして次ターンの発話者を決める。
"""

from __future__ import annotations

from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.floor_controller import (
    FloorController,
    Utterance,
)


def _make_char(
    char_id: str, name: str, extraversion: float = 0.5
) -> Character:
    return Character(
        id=char_id,
        name=name,
        personality=Personality(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=extraversion,
            agreeableness=0.5,
            neuroticism=0.5,
        ),
        values=Values(
            self_transcendence=0.5,
            self_enhancement=0.5,
            openness_to_change=0.5,
            conservation=0.5,
        ),
    )


class TestFloorControllerFirstTurn:
    """AC: 最初のターンは外向性最大のキャラが発話する."""

    def test_first_turn_picks_highest_extraversion(self) -> None:
        chars = [
            _make_char("a", "A", extraversion=0.3),
            _make_char("b", "B", extraversion=0.9),
            _make_char("c", "C", extraversion=0.6),
        ]
        fc = FloorController()
        speaker = fc.next_speaker(history=[], participants=chars)
        assert speaker.id == "b"


class TestFloorControllerPenalizesRecent:
    """AC: 直近で喋ったキャラほどスコアが下がる（喋りすぎ抑制）."""

    def test_recent_speaker_is_deprioritized(self) -> None:
        chars = [
            _make_char("a", "A", extraversion=0.9),
            _make_char("b", "B", extraversion=0.9),
            _make_char("c", "C", extraversion=0.5),
        ]
        # A が直近 3 ターン連続で喋った
        history = [
            Utterance(speaker_id="a", speaker_name="A", speech="hi 1"),
            Utterance(speaker_id="a", speaker_name="A", speech="hi 2"),
            Utterance(speaker_id="a", speaker_name="A", speech="hi 3"),
        ]
        fc = FloorController()
        # ペナルティで A は選ばれにくくなる → B or C
        speaker = fc.next_speaker(history=history, participants=chars)
        assert speaker.id != "a"

    def test_last_speaker_not_immediate_repeat(self) -> None:
        """AC: 直前の発話者は次ターンの発話者にならない（同一連続抑制）."""
        chars = [
            _make_char("a", "A", extraversion=0.9),
            _make_char("b", "B", extraversion=0.4),
        ]
        history = [
            Utterance(speaker_id="a", speaker_name="A", speech="hi"),
        ]
        fc = FloorController()
        speaker = fc.next_speaker(history=history, participants=chars)
        assert speaker.id == "b"


class TestFloorControllerRotation:
    """AC: 単純なローテーション（A→B→C→A→...）に近い結果を出す."""

    def test_three_participants_eventually_all_speak(self) -> None:
        chars = [
            _make_char("a", "A", extraversion=0.6),
            _make_char("b", "B", extraversion=0.6),
            _make_char("c", "C", extraversion=0.6),
        ]
        fc = FloorController()
        history: list[Utterance] = []
        speakers_seen = set()
        for _ in range(9):
            sp = fc.next_speaker(history=history, participants=chars)
            speakers_seen.add(sp.id)
            history.append(
                Utterance(speaker_id=sp.id, speaker_name=sp.name, speech="x")
            )
        assert speakers_seen == {"a", "b", "c"}

    def test_yurucamp_extraversion_gap_still_rotates(self) -> None:
        """AC: 実際のゆるキャン野クル組（外向性 0.95/0.85/0.5）でも、
        15 ターンあれば全員が一度は発話する."""
        chars = [
            _make_char("nade", "なでしこ", extraversion=0.95),
            _make_char("chia", "千明", extraversion=0.85),
            _make_char("aoi", "あおい", extraversion=0.5),
        ]
        fc = FloorController()
        history: list[Utterance] = []
        speakers_seen = set()
        for _ in range(15):
            sp = fc.next_speaker(history=history, participants=chars)
            speakers_seen.add(sp.id)
            history.append(
                Utterance(speaker_id=sp.id, speaker_name=sp.name, speech="x")
            )
        assert speakers_seen == {"nade", "chia", "aoi"}


class TestFloorControllerScoring:
    """スコアリング関数の単体テスト."""

    def test_score_higher_for_extraverted(self) -> None:
        ch_high = _make_char("h", "H", extraversion=0.9)
        ch_low = _make_char("l", "L", extraversion=0.2)
        fc = FloorController()
        s_high = fc.score(ch_high, history=[])
        s_low = fc.score(ch_low, history=[])
        assert s_high > s_low

    def test_score_penalty_for_consecutive_speaking(self) -> None:
        ch = _make_char("a", "A", extraversion=0.8)
        fc = FloorController()
        s_no_history = fc.score(ch, history=[])
        history_with_self = [
            Utterance(speaker_id="a", speaker_name="A", speech="x")
            for _ in range(3)
        ]
        s_self_recent = fc.score(ch, history=history_with_self)
        assert s_no_history > s_self_recent
