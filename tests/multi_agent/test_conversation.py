"""Tests for Conversation (Issue #6 / Phase 0 C).

invariant: observation-turn-state-update — 自分のターンでなくても、
他キャラの発話・行動は自分の history に積まれる。
"""

from __future__ import annotations

from pneuma_core.models.character import Character
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.multi_agent.conversation import Conversation
from pneuma_core.multi_agent.floor_controller import Utterance


def _make_char(char_id: str, name: str) -> Character:
    return Character(
        id=char_id,
        name=name,
        personality=Personality(
            openness=0.5, conscientiousness=0.5, extraversion=0.5,
            agreeableness=0.5, neuroticism=0.5,
        ),
        values=Values(
            self_transcendence=0.5, self_enhancement=0.5,
            openness_to_change=0.5, conservation=0.5,
        ),
    )


class TestConversationParticipants:
    """AC: N 体の participants を保持できる."""

    def test_three_participants(self) -> None:
        chars = [_make_char("a", "A"), _make_char("b", "B"), _make_char("c", "C")]
        conv = Conversation(participants=chars)
        assert len(conv.participants) == 3
        assert conv.get_participant("b").name == "B"

    def test_minimum_two_participants(self) -> None:
        # 1 体だけ → エラー（マルチエージェントの意味がない）
        import pytest
        with pytest.raises(ValueError):
            Conversation(participants=[_make_char("a", "A")])


class TestConversationHistory:
    """AC: 共通 history を保持し、record_utterance で積める."""

    def test_record_and_get_history(self) -> None:
        chars = [_make_char("a", "A"), _make_char("b", "B")]
        conv = Conversation(participants=chars)
        conv.record_utterance(
            Utterance(speaker_id="a", speaker_name="A", speech="hi")
        )
        conv.record_utterance(
            Utterance(speaker_id="b", speaker_name="B", speech="yo")
        )
        assert len(conv.history) == 2
        assert conv.history[0].speech == "hi"
        assert conv.history[1].speaker_id == "b"

    def test_observation_view_for_self(self) -> None:
        """AC: 自分視点では 自分=assistant / 他人=user としてラベリングされる."""
        chars = [_make_char("a", "A"), _make_char("b", "B")]
        conv = Conversation(participants=chars)
        conv.record_utterance(
            Utterance(speaker_id="a", speaker_name="A", speech="hi from A")
        )
        conv.record_utterance(
            Utterance(speaker_id="b", speaker_name="B", speech="hi from B")
        )
        view_a = conv.history_as_messages(viewer_id="a")
        # A 視点: A の発話は assistant, B の発話は user
        assert view_a[0]["role"] == "assistant"
        assert "hi from A" in view_a[0]["content"]
        assert view_a[1]["role"] == "user"
        assert "[B]" in view_a[1]["content"]
