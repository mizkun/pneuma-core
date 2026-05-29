"""Tests for Issue #2: Character, Personality, Values データモデル.

受け入れ基準:
- dataclass としてインスタンス化できる
- 範囲外の値で適切なエラーが出る
- 全テストがパスする
"""

import pytest

from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.models.character import Character


# =============================================
# Personality Tests
# =============================================


class TestPersonality:
    """Big Five 性格モデルのテスト."""

    def test_create_valid_personality(self):
        """正常な値で Personality を生成できる."""
        p = Personality(
            openness=0.8,
            conscientiousness=0.5,
            extraversion=0.3,
            agreeableness=0.6,
            neuroticism=0.7,
        )
        assert p.openness == 0.8
        assert p.conscientiousness == 0.5
        assert p.extraversion == 0.3
        assert p.agreeableness == 0.6
        assert p.neuroticism == 0.7

    def test_create_boundary_values(self):
        """境界値 (0.0, 1.0) で生成できる."""
        p = Personality(
            openness=0.0,
            conscientiousness=1.0,
            extraversion=0.0,
            agreeableness=1.0,
            neuroticism=0.0,
        )
        assert p.openness == 0.0
        assert p.conscientiousness == 1.0

    def test_reject_negative_value(self):
        """負の値はエラーになる."""
        with pytest.raises(ValueError):
            Personality(
                openness=-0.1,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            )

    def test_reject_over_one(self):
        """1.0 を超える値はエラーになる."""
        with pytest.raises(ValueError):
            Personality(
                openness=0.5,
                conscientiousness=1.1,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            )

    def test_is_high(self):
        """0.7 以上は 'high' と判定される."""
        p = Personality(
            openness=0.7,
            conscientiousness=0.3,
            extraversion=0.5,
            agreeableness=0.8,
            neuroticism=0.2,
        )
        assert p.is_high("openness") is True
        assert p.is_high("agreeableness") is True
        assert p.is_high("conscientiousness") is False

    def test_is_low(self):
        """0.3 未満は 'low' と判定される."""
        p = Personality(
            openness=0.7,
            conscientiousness=0.3,
            extraversion=0.5,
            agreeableness=0.8,
            neuroticism=0.2,
        )
        assert p.is_low("neuroticism") is True
        assert p.is_low("conscientiousness") is False  # 0.3 は「低い」ではない
        assert p.is_low("openness") is False


# =============================================
# Values Tests
# =============================================


class TestValues:
    """Schwartz 4 カテゴリ価値観モデルのテスト."""

    def test_create_valid_values(self):
        """正常な値で Values を生成できる."""
        v = Values(
            self_transcendence=0.3,
            self_enhancement=0.5,
            openness_to_change=0.8,
            conservation=0.2,
        )
        assert v.self_transcendence == 0.3
        assert v.openness_to_change == 0.8

    def test_reject_out_of_range(self):
        """範囲外の値はエラーになる."""
        with pytest.raises(ValueError):
            Values(
                self_transcendence=1.5,
                self_enhancement=0.5,
                openness_to_change=0.5,
                conservation=0.5,
            )

    def test_is_important(self):
        """0.6 以上で「重視する」と判定される."""
        v = Values(
            self_transcendence=0.3,
            self_enhancement=0.5,
            openness_to_change=0.8,
            conservation=0.6,
        )
        assert v.is_important("openness_to_change") is True
        assert v.is_important("conservation") is True
        assert v.is_important("self_transcendence") is False
        assert v.is_important("self_enhancement") is False


# =============================================
# Character Tests
# =============================================


class TestCharacter:
    """キャラクター Identity のテスト."""

    def test_create_character(self):
        """Character を正しく生成できる."""
        personality = Personality(
            openness=0.8,
            conscientiousness=0.5,
            extraversion=0.3,
            agreeableness=0.6,
            neuroticism=0.7,
        )
        values = Values(
            self_transcendence=0.3,
            self_enhancement=0.5,
            openness_to_change=0.8,
            conservation=0.2,
        )
        char = Character(
            id="aine-001",
            name="アイネ",
            personality=personality,
            values=values,
        )
        assert char.id == "aine-001"
        assert char.name == "アイネ"
        assert char.personality.openness == 0.8
        assert char.values.openness_to_change == 0.8

    def test_optional_fields_default_none(self):
        """オプショナルフィールドはデフォルトで None."""
        char = Character(
            id="test-001",
            name="テスト",
            personality=Personality(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
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
        assert char.profile is None
        assert char.appearance is None
        assert char.speaking_style is None
        assert char.background is None
        assert char.personality_description is None
        assert char.values_description is None
        # Issue #22: quirk（思考のクセ）はデフォルト空文字
        assert char.quirk == ""

    def test_character_with_descriptions(self):
        """自由記述フィールドを設定できる."""
        char = Character(
            id="aine-001",
            name="アイネ",
            personality=Personality(
                openness=0.8,
                conscientiousness=0.5,
                extraversion=0.3,
                agreeableness=0.6,
                neuroticism=0.7,
            ),
            values=Values(
                self_transcendence=0.3,
                self_enhancement=0.5,
                openness_to_change=0.8,
                conservation=0.2,
            ),
            profile="AIとして生まれたばかり（0歳）。",
            appearance="ダークブルーの髪。ボブ〜ウルフくらいの短め。",
            speaking_style="...（沈黙）とかはよく使う。タメ口気味。",
            background="背景テキスト",
            personality_description="カスタム性格記述",
            values_description="カスタム価値観記述",
        )
        assert char.profile == "AIとして生まれたばかり（0歳）。"
        assert char.personality_description == "カスタム性格記述"
