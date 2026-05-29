"""Tests for ゆるキャン野クル組 3 キャラ character.yaml dataset."""

from __future__ import annotations

from pneuma_core._packaged_examples.yurucamp import load_yurucamp_sheets


def test_three_characters_load() -> None:
    sheets = load_yurucamp_sheets()
    names = [s.character.name for s in sheets]
    assert names == ["なでしこ", "千明", "あおい"]


def test_extraversion_ordering() -> None:
    """AC: なでしこ > 千明 > あおい の外向性順 (FloorController で自然な発話順)."""
    sheets = load_yurucamp_sheets()
    extras = [s.character.personality.extraversion for s in sheets]
    assert extras[0] > extras[1] > extras[2]
    assert extras[0] == 0.95
    assert extras[2] == 0.5


def test_initial_emotion_labels_are_english() -> None:
    """AC (Issue #9): 全キャラの initial_state.emotion_label は EmotionLabel 値."""
    sheets = load_yurucamp_sheets()
    valid = {"neutral", "happy", "teasing", "surprised", "embarrassed", "sad_lite"}
    for s in sheets:
        assert s.initial_state is not None
        assert s.initial_state.emotion_label in valid, (
            f"{s.character.name}: {s.initial_state.emotion_label} not in {valid}"
        )


def test_each_character_has_goals() -> None:
    sheets = load_yurucamp_sheets()
    for s in sheets:
        assert s.goal_tree is not None
        assert len(s.goal_tree.visions) >= 1


def test_each_character_has_quirk() -> None:
    """AC-3 (Issue #22): 3 体それぞれに思考のクセ (quirk) が設定される."""
    sheets = load_yurucamp_sheets()
    by_name = {s.character.name: s.character for s in sheets}
    # quirk は非空（思考傾向を与える。セリフは台本化しない）
    assert by_name["千明"].quirk
    assert by_name["なでしこ"].quirk
    assert by_name["あおい"].quirk
    # 設計どおりの傾向が入っている（中身の語で確認）
    assert "勝負" in by_name["千明"].quirk or "競争" in by_name["千明"].quirk
    assert "食" in by_name["なでしこ"].quirk or "お腹" in by_name["なでしこ"].quirk
    assert "達観" in by_name["あおい"].quirk or "ことわざ" in by_name["あおい"].quirk
