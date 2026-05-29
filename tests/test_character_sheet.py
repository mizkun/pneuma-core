"""Tests for CharacterSheet YAML read/write (Issue #8)."""

import tempfile
from pathlib import Path

import pytest

from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values


VALID_YAML = """\
name: アイネ
id: aine-001

profile: |
  AIとして生まれたばかり（0歳）。
appearance: |
  ダークブルーの髪。ボブ〜ウルフくらいの短め。
speaking_style: |
  ...（沈黙）とかはよく使う。タメ口気味で話す。
background: |
  とある研究所で生まれた。
personality_description: |
  内向的だが好奇心旺盛。
values_description: |
  変化を好み、保守を嫌う。

personality:
  openness: 0.8
  conscientiousness: 0.5
  extraversion: 0.3
  agreeableness: 0.6
  neuroticism: 0.7

values:
  self_transcendence: 0.3
  self_enhancement: 0.5
  openness_to_change: 0.8
  conservation: 0.2

initial_state:
  pleasure: 0.0
  arousal: 0.0
  dominance: 0.0
  emotion_label: 中立
  situation: 初めての会話

goals:
  visions:
    - content: "世界を理解する"
  objectives:
    - content: "人間の感情を学ぶ"
      status: active
      progress: 0.3
      vision_index: 0
  tasks:
    - content: "ユーザーと感情について話す"
      status: pending
      objective_index: 0

voice_id: null
"""

MINIMAL_YAML = """\
name: テスト
id: test-001

personality:
  openness: 0.5
  conscientiousness: 0.5
  extraversion: 0.5
  agreeableness: 0.5
  neuroticism: 0.5

values:
  self_transcendence: 0.5
  self_enhancement: 0.5
  openness_to_change: 0.5
  conservation: 0.5
"""


# --- Load tests ---


class TestLoad:
    """YAML 読み込みテスト."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        """正常な YAML の読み込み."""
        f = tmp_path / "aine.character.yaml"
        f.write_text(VALID_YAML, encoding="utf-8")

        result = CharacterSheet.load(f)

        assert result.character.id == "aine-001"
        assert result.character.name == "アイネ"
        assert result.character.personality.openness == 0.8
        assert result.character.values.conservation == 0.2
        assert result.character.profile is not None
        assert "AIとして生まれたばかり" in result.character.profile

    def test_load_initial_state(self, tmp_path: Path) -> None:
        """initial_state の読み込み."""
        f = tmp_path / "aine.character.yaml"
        f.write_text(VALID_YAML, encoding="utf-8")

        result = CharacterSheet.load(f)

        assert result.initial_state is not None
        assert result.initial_state.pleasure == 0.0
        assert result.initial_state.emotion_label == "中立"

    def test_load_goals(self, tmp_path: Path) -> None:
        """goals の読み込み."""
        f = tmp_path / "aine.character.yaml"
        f.write_text(VALID_YAML, encoding="utf-8")

        result = CharacterSheet.load(f)

        assert result.goal_tree is not None
        assert len(result.goal_tree.visions) == 1
        assert result.goal_tree.visions[0].content == "世界を理解する"
        assert len(result.goal_tree.objectives) == 1
        assert result.goal_tree.objectives[0].status == "active"
        assert result.goal_tree.objectives[0].progress == 0.3
        assert len(result.goal_tree.tasks) == 1
        assert result.goal_tree.tasks[0].status == "pending"

    def test_load_minimal(self, tmp_path: Path) -> None:
        """最小限の YAML（optional フィールドなし）."""
        f = tmp_path / "test.character.yaml"
        f.write_text(MINIMAL_YAML, encoding="utf-8")

        result = CharacterSheet.load(f)

        assert result.character.id == "test-001"
        assert result.character.name == "テスト"
        assert result.character.profile is None
        assert result.initial_state is None
        assert result.goal_tree is None

    def test_load_string(self) -> None:
        """文字列から読み込み."""
        result = CharacterSheet.from_yaml(VALID_YAML)

        assert result.character.id == "aine-001"
        assert result.character.name == "アイネ"


# --- Validation error tests ---


class TestValidationErrors:
    """バリデーションエラーテスト."""

    def test_missing_name(self) -> None:
        yaml_str = """\
id: test-001
personality:
  openness: 0.5
  conscientiousness: 0.5
  extraversion: 0.5
  agreeableness: 0.5
  neuroticism: 0.5
values:
  self_transcendence: 0.5
  self_enhancement: 0.5
  openness_to_change: 0.5
  conservation: 0.5
"""
        with pytest.raises(ValueError, match="name"):
            CharacterSheet.from_yaml(yaml_str)

    def test_missing_id(self) -> None:
        yaml_str = """\
name: テスト
personality:
  openness: 0.5
  conscientiousness: 0.5
  extraversion: 0.5
  agreeableness: 0.5
  neuroticism: 0.5
values:
  self_transcendence: 0.5
  self_enhancement: 0.5
  openness_to_change: 0.5
  conservation: 0.5
"""
        with pytest.raises(ValueError, match="id"):
            CharacterSheet.from_yaml(yaml_str)

    def test_missing_personality(self) -> None:
        yaml_str = """\
name: テスト
id: test-001
values:
  self_transcendence: 0.5
  self_enhancement: 0.5
  openness_to_change: 0.5
  conservation: 0.5
"""
        with pytest.raises(ValueError, match="personality"):
            CharacterSheet.from_yaml(yaml_str)

    def test_missing_values(self) -> None:
        yaml_str = """\
name: テスト
id: test-001
personality:
  openness: 0.5
  conscientiousness: 0.5
  extraversion: 0.5
  agreeableness: 0.5
  neuroticism: 0.5
"""
        with pytest.raises(ValueError, match="values"):
            CharacterSheet.from_yaml(yaml_str)

    def test_personality_out_of_range(self) -> None:
        yaml_str = """\
name: テスト
id: test-001
personality:
  openness: 1.5
  conscientiousness: 0.5
  extraversion: 0.5
  agreeableness: 0.5
  neuroticism: 0.5
values:
  self_transcendence: 0.5
  self_enhancement: 0.5
  openness_to_change: 0.5
  conservation: 0.5
"""
        with pytest.raises(ValueError, match="openness"):
            CharacterSheet.from_yaml(yaml_str)

    def test_invalid_goal_status(self) -> None:
        yaml_str = """\
name: テスト
id: test-001
personality:
  openness: 0.5
  conscientiousness: 0.5
  extraversion: 0.5
  agreeableness: 0.5
  neuroticism: 0.5
values:
  self_transcendence: 0.5
  self_enhancement: 0.5
  openness_to_change: 0.5
  conservation: 0.5
goals:
  visions:
    - content: "ビジョン"
  objectives:
    - content: "目標"
      status: invalid_status
      progress: 0.5
      vision_index: 0
"""
        with pytest.raises(ValueError, match="status"):
            CharacterSheet.from_yaml(yaml_str)


# --- Serialization tests ---


class TestSerialization:
    """YAML シリアライズテスト."""

    def test_to_yaml(self) -> None:
        """Character → YAML 変換."""
        sheet = CharacterSheet.from_yaml(VALID_YAML)
        yaml_str = sheet.to_yaml()

        assert "name: アイネ" in yaml_str or "name:" in yaml_str
        assert "aine-001" in yaml_str
        assert "openness:" in yaml_str

    def test_round_trip(self) -> None:
        """round-trip: YAML → load → to_yaml → load でデータ保持."""
        sheet1 = CharacterSheet.from_yaml(VALID_YAML)
        yaml_str = sheet1.to_yaml()
        sheet2 = CharacterSheet.from_yaml(yaml_str)

        assert sheet2.character.id == sheet1.character.id
        assert sheet2.character.name == sheet1.character.name
        assert sheet2.character.personality.openness == sheet1.character.personality.openness
        assert sheet2.character.values.conservation == sheet1.character.values.conservation

    def test_round_trip_goals(self) -> None:
        """round-trip で goals が保持される."""
        sheet1 = CharacterSheet.from_yaml(VALID_YAML)
        yaml_str = sheet1.to_yaml()
        sheet2 = CharacterSheet.from_yaml(yaml_str)

        assert sheet2.goal_tree is not None
        assert len(sheet2.goal_tree.visions) == len(sheet1.goal_tree.visions)
        assert sheet2.goal_tree.visions[0].content == sheet1.goal_tree.visions[0].content

    def test_round_trip_initial_state(self) -> None:
        """round-trip で initial_state が保持される."""
        sheet1 = CharacterSheet.from_yaml(VALID_YAML)
        yaml_str = sheet1.to_yaml()
        sheet2 = CharacterSheet.from_yaml(yaml_str)

        assert sheet2.initial_state is not None
        assert sheet2.initial_state.pleasure == sheet1.initial_state.pleasure
        assert sheet2.initial_state.emotion_label == sheet1.initial_state.emotion_label

    def test_load_quirk(self) -> None:
        """Issue #22: quirk（思考のクセ）フィールドが読み込まれる."""
        yaml_str = VALID_YAML + "\nquirk: 何でも勝負に変換して考える\n"
        sheet = CharacterSheet.from_yaml(yaml_str)
        assert sheet.character.quirk == "何でも勝負に変換して考える"

    def test_quirk_defaults_empty_when_absent(self) -> None:
        """Issue #22: quirk 未記載のときは空文字（既存 YAML を壊さない）."""
        sheet = CharacterSheet.from_yaml(VALID_YAML)
        assert sheet.character.quirk == ""

    def test_round_trip_quirk(self) -> None:
        """Issue #22: round-trip で quirk が保持される."""
        yaml_str = VALID_YAML + "\nquirk: 何でも食べ物に結びつける\n"
        sheet1 = CharacterSheet.from_yaml(yaml_str)
        sheet2 = CharacterSheet.from_yaml(sheet1.to_yaml())
        assert sheet2.character.quirk == "何でも食べ物に結びつける"

    def test_save_to_file(self, tmp_path: Path) -> None:
        """ファイルへの書き出し."""
        sheet = CharacterSheet.from_yaml(VALID_YAML)
        out = tmp_path / "output.yaml"
        sheet.save(out)

        assert out.exists()
        loaded = CharacterSheet.load(out)
        assert loaded.character.id == "aine-001"

    def test_minimal_round_trip(self) -> None:
        """最小限 YAML の round-trip."""
        sheet1 = CharacterSheet.from_yaml(MINIMAL_YAML)
        yaml_str = sheet1.to_yaml()
        sheet2 = CharacterSheet.from_yaml(yaml_str)

        assert sheet2.character.id == "test-001"
        assert sheet2.initial_state is None
        assert sheet2.goal_tree is None


# --- Multiple files ---


class TestMultipleFiles:
    """複数ファイルの一括読み込み."""

    def test_load_directory(self, tmp_path: Path) -> None:
        """ディレクトリ内の全 .character.yaml を読み込み."""
        (tmp_path / "aine.character.yaml").write_text(VALID_YAML, encoding="utf-8")
        (tmp_path / "test.character.yaml").write_text(MINIMAL_YAML, encoding="utf-8")
        (tmp_path / "not-a-character.yaml").write_text("foo: bar", encoding="utf-8")

        sheets = CharacterSheet.load_directory(tmp_path)

        assert len(sheets) == 2
        ids = {s.character.id for s in sheets}
        assert "aine-001" in ids
        assert "test-001" in ids

    def test_load_empty_directory(self, tmp_path: Path) -> None:
        """空ディレクトリ."""
        sheets = CharacterSheet.load_directory(tmp_path)
        assert sheets == []
