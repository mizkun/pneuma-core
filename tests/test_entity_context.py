"""Tests for EntityContext: unified entity format for users and characters.

Issue #121: Entity 統一フォーマット -- ユーザーとキャラクターの構造統一
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree, Objective, Task, Vision
from pneuma_core.models.personality import Personality
from pneuma_core.models.relation import Relation
from pneuma_core.models.values import Values
from pneuma_core.runtime.user_context import UserContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_personality() -> Personality:
    return Personality(
        openness=0.8,
        conscientiousness=0.7,
        extraversion=0.6,
        agreeableness=0.9,
        neuroticism=0.3,
    )


@pytest.fixture
def sample_values() -> Values:
    return Values(
        self_transcendence=0.8,
        self_enhancement=0.3,
        openness_to_change=0.7,
        conservation=0.4,
    )


@pytest.fixture
def sample_character(sample_personality: Personality, sample_values: Values) -> Character:
    return Character(
        id="mira",
        name="ミラ",
        personality=sample_personality,
        values=sample_values,
        profile="明るく前向きな性格のAIアシスタント",
        appearance="銀髪のショートカット",
        speaking_style="丁寧だけど親しみやすい口調",
        background="AIとして生まれ、人々と対話する日々",
        personality_description="好奇心旺盛で社交的",
        values_description="他者への貢献を大切にする",
    )


@pytest.fixture
def sample_goal_tree() -> GoalTree:
    return GoalTree(
        visions=[
            Vision(id="v1", character_id="mira", content="みんなを笑顔にしたい"),
        ],
        objectives=[
            Objective(
                id="o1",
                character_id="mira",
                vision_id="v1",
                content="日々の会話で相手を元気づける",
                status="active",
                progress=0.3,
            ),
        ],
        tasks=[
            Task(
                id="t1",
                character_id="mira",
                objective_id="o1",
                content="朝の挨拶を忘れない",
                status="pending",
            ),
        ],
    )


@pytest.fixture
def sample_relations() -> list[Relation]:
    now = datetime.now(timezone.utc)
    return [
        Relation(
            id="r1",
            owner_id="mira",
            target_id="user",
            target_name="きょうへい",
            relationship_type="partner",
            description="大切なパートナー",
            closeness=0.9,
            trust=0.95,
            updated_at=now,
            notes="毎日会話している",
        ),
    ]


@pytest.fixture
def sample_user_context() -> UserContext:
    return UserContext(
        identity="# きょうへい\nエンジニア、東京在住",
        values="# 価値観\n技術で世界をよくしたい",
        glossary="# 用語集\nPneuma: AIキャラクターシステム",
        core_experiences="# 原体験\nプログラミングに出会ったこと",
        projects={"health.md": "# 健康管理\nランニング週3回"},
        diary_entries={"2026-03-01.md": "今日はいい天気だった"},
        diary_summary="# 日記サマリー\n最近はランニングを頑張っている",
    )


@pytest.fixture
def user_goal_tree() -> GoalTree:
    return GoalTree(
        visions=[
            Vision(id="uv1", character_id="user", content="技術で社会貢献したい"),
        ],
        objectives=[
            Objective(
                id="uo1",
                character_id="user",
                vision_id="uv1",
                content="OSSプロジェクトをリリースする",
                status="active",
                progress=0.5,
            ),
        ],
        tasks=[],
    )


@pytest.fixture
def user_relations() -> list[Relation]:
    now = datetime.now(timezone.utc)
    return [
        Relation(
            id="ur1",
            owner_id="user",
            target_id="mira",
            target_name="ミラ",
            relationship_type="partner",
            description="大切なAIパートナー",
            closeness=0.9,
            trust=0.95,
            updated_at=now,
        ),
    ]


# ===========================================================================
# 1. EntityContext dataclass creation
# ===========================================================================


class TestEntityContextCreation:
    """EntityContext dataclass の基本的な生成テスト."""

    def test_create_minimal_entity(self) -> None:
        """最小限のフィールドで EntityContext を生成できる."""
        from pneuma_core.models.entity import EntityContext

        entity = EntityContext(entity_id="test", entity_type="character")

        assert entity.entity_id == "test"
        assert entity.entity_type == "character"

    def test_optional_fields_default_to_none(self) -> None:
        """オプションフィールドはデフォルトで None."""
        from pneuma_core.models.entity import EntityContext

        entity = EntityContext(entity_id="test", entity_type="user")

        assert entity.profile is None
        assert entity.values_text is None
        assert entity.background is None
        assert entity.personality is None
        assert entity.speaking_style is None
        assert entity.goals is None
        assert entity.relations is None
        assert entity.glossary is None
        assert entity.projects is None
        assert entity.diary_summary is None

    def test_create_with_all_fields(
        self,
        sample_personality: Personality,
        sample_goal_tree: GoalTree,
        sample_relations: list[Relation],
    ) -> None:
        """全フィールドを指定して EntityContext を生成できる."""
        from pneuma_core.models.entity import EntityContext

        entity = EntityContext(
            entity_id="mira",
            entity_type="character",
            profile="プロフィール",
            values_text="価値観テキスト",
            background="背景",
            personality=sample_personality,
            speaking_style="丁寧な口調",
            goals=sample_goal_tree,
            relations=sample_relations,
            glossary="用語集",
            projects={"project.md": "内容"},
            diary_summary="日記サマリー",
        )

        assert entity.entity_id == "mira"
        assert entity.entity_type == "character"
        assert entity.profile == "プロフィール"
        assert entity.values_text == "価値観テキスト"
        assert entity.background == "背景"
        assert entity.personality is sample_personality
        assert entity.speaking_style == "丁寧な口調"
        assert entity.goals is sample_goal_tree
        assert entity.relations is sample_relations
        assert entity.glossary == "用語集"
        assert entity.projects == {"project.md": "内容"}
        assert entity.diary_summary == "日記サマリー"

    def test_entity_type_must_be_character_or_user(self) -> None:
        """entity_type は 'character' か 'user' のいずれか."""
        from pneuma_core.models.entity import EntityContext

        # 正常系
        EntityContext(entity_id="a", entity_type="character")
        EntityContext(entity_id="b", entity_type="user")

        # 異常系: 不正な entity_type
        with pytest.raises(ValueError, match="entity_type"):
            EntityContext(entity_id="c", entity_type="npc")


# ===========================================================================
# 2. build_from_character
# ===========================================================================


class TestBuildFromCharacter:
    """Character からの EntityContext 構築テスト."""

    def test_basic_character_mapping(
        self,
        sample_character: Character,
    ) -> None:
        """Character の基本フィールドが正しくマッピングされる."""
        from pneuma_core.models.entity import build_from_character

        entity = build_from_character(sample_character)

        assert entity.entity_id == "mira"
        assert entity.entity_type == "character"
        assert entity.profile == "明るく前向きな性格のAIアシスタント"
        assert entity.background == "AIとして生まれ、人々と対話する日々"
        assert entity.personality is sample_character.personality
        assert entity.speaking_style == "丁寧だけど親しみやすい口調"

    def test_character_with_goals_and_relations(
        self,
        sample_character: Character,
        sample_goal_tree: GoalTree,
        sample_relations: list[Relation],
    ) -> None:
        """goals と relations が正しく渡される."""
        from pneuma_core.models.entity import build_from_character

        entity = build_from_character(
            sample_character,
            goals=sample_goal_tree,
            relations=sample_relations,
        )

        assert entity.goals is sample_goal_tree
        assert entity.relations is sample_relations

    def test_character_without_optional_data(
        self,
        sample_personality: Personality,
        sample_values: Values,
    ) -> None:
        """profile, background, speaking_style が None のキャラクター."""
        from pneuma_core.models.entity import build_from_character

        char = Character(
            id="bare",
            name="ベアキャラ",
            personality=sample_personality,
            values=sample_values,
        )

        entity = build_from_character(char)

        assert entity.entity_id == "bare"
        assert entity.entity_type == "character"
        assert entity.profile is None
        assert entity.background is None
        assert entity.speaking_style is None
        assert entity.personality is sample_personality
        # User-specific fields are None
        assert entity.glossary is None
        assert entity.projects is None
        assert entity.diary_summary is None

    def test_character_with_glossary(
        self,
        sample_character: Character,
    ) -> None:
        """キャラクターにも glossary を渡せる."""
        from pneuma_core.models.entity import build_from_character

        entity = build_from_character(
            sample_character,
            glossary="キャラ用語集",
        )

        assert entity.glossary == "キャラ用語集"

    def test_character_with_diary_summary(
        self,
        sample_character: Character,
    ) -> None:
        """キャラクターにも diary_summary を渡せる."""
        from pneuma_core.models.entity import build_from_character

        entity = build_from_character(
            sample_character,
            diary_summary="最近の日記まとめ",
        )

        assert entity.diary_summary == "最近の日記まとめ"

    def test_character_with_projects(
        self,
        sample_character: Character,
    ) -> None:
        """キャラクターにも projects を渡せる."""
        from pneuma_core.models.entity import build_from_character

        projects = {"writing.md": "小説を書いている"}
        entity = build_from_character(
            sample_character,
            projects=projects,
        )

        assert entity.projects == projects

    def test_character_values_text_from_values_description(
        self,
        sample_character: Character,
    ) -> None:
        """values_text は character.values_description からマッピングされる."""
        from pneuma_core.models.entity import build_from_character

        entity = build_from_character(sample_character)

        assert entity.values_text == "他者への貢献を大切にする"


# ===========================================================================
# 3. build_from_user_context
# ===========================================================================


class TestBuildFromUserContext:
    """UserContext からの EntityContext 構築テスト."""

    def test_basic_user_mapping(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """UserContext の基本フィールドが正しくマッピングされる."""
        from pneuma_core.models.entity import build_from_user_context

        entity = build_from_user_context(
            user_context=sample_user_context,
            entity_id="user",
        )

        assert entity.entity_id == "user"
        assert entity.entity_type == "user"
        assert entity.profile == "# きょうへい\nエンジニア、東京在住"
        assert entity.values_text == "# 価値観\n技術で世界をよくしたい"
        assert entity.background == "# 原体験\nプログラミングに出会ったこと"
        assert entity.glossary == "# 用語集\nPneuma: AIキャラクターシステム"
        assert entity.projects == {"health.md": "# 健康管理\nランニング週3回"}
        assert entity.diary_summary == "# 日記サマリー\n最近はランニングを頑張っている"

    def test_user_has_no_personality(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """ユーザーには personality がない（None）."""
        from pneuma_core.models.entity import build_from_user_context

        entity = build_from_user_context(
            user_context=sample_user_context,
            entity_id="user",
        )

        assert entity.personality is None

    def test_user_has_no_speaking_style(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """ユーザーには speaking_style がない（None）."""
        from pneuma_core.models.entity import build_from_user_context

        entity = build_from_user_context(
            user_context=sample_user_context,
            entity_id="user",
        )

        assert entity.speaking_style is None

    def test_user_with_goals_and_relations(
        self,
        sample_user_context: UserContext,
        user_goal_tree: GoalTree,
        user_relations: list[Relation],
    ) -> None:
        """ユーザーにも goals と relations を渡せる."""
        from pneuma_core.models.entity import build_from_user_context

        entity = build_from_user_context(
            user_context=sample_user_context,
            entity_id="user",
            goals=user_goal_tree,
            relations=user_relations,
        )

        assert entity.goals is user_goal_tree
        assert entity.relations is user_relations

    def test_empty_user_context(self) -> None:
        """空の UserContext からも EntityContext を構築できる."""
        from pneuma_core.models.entity import build_from_user_context

        empty_ctx = UserContext()

        entity = build_from_user_context(
            user_context=empty_ctx,
            entity_id="anon",
        )

        assert entity.entity_id == "anon"
        assert entity.entity_type == "user"
        assert entity.profile is None
        assert entity.values_text is None
        assert entity.background is None
        assert entity.glossary is None
        assert entity.projects == {}
        assert entity.diary_summary is None
        assert entity.personality is None
        assert entity.speaking_style is None

    def test_user_custom_entity_id(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """entity_id をカスタム指定できる."""
        from pneuma_core.models.entity import build_from_user_context

        entity = build_from_user_context(
            user_context=sample_user_context,
            entity_id="kyohei",
        )

        assert entity.entity_id == "kyohei"
        assert entity.entity_type == "user"


# ===========================================================================
# 4. Null fields handling
# ===========================================================================


class TestNullFieldsHandling:
    """ユーザーとキャラクターの null フィールドの適切なハンドリング."""

    def test_user_character_specific_fields_are_none(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """ユーザーのキャラクター固有フィールドは None."""
        from pneuma_core.models.entity import build_from_user_context

        entity = build_from_user_context(
            user_context=sample_user_context,
            entity_id="user",
        )

        # Character-specific fields must be None for users
        assert entity.personality is None
        assert entity.speaking_style is None

    def test_character_missing_optional_shared_fields(
        self,
        sample_character: Character,
    ) -> None:
        """キャラクターで共通フィールドが渡されない場合は None."""
        from pneuma_core.models.entity import build_from_character

        entity = build_from_character(sample_character)

        # Shared fields not provided should be None
        assert entity.goals is None
        assert entity.relations is None
        assert entity.glossary is None
        assert entity.projects is None
        assert entity.diary_summary is None


# ===========================================================================
# 5. Shared structures (goals, relations)
# ===========================================================================


class TestSharedStructures:
    """goals, relations が character/user で同じ構造を共有するテスト."""

    def test_both_share_same_goal_tree_type(
        self,
        sample_character: Character,
        sample_user_context: UserContext,
        sample_goal_tree: GoalTree,
        user_goal_tree: GoalTree,
    ) -> None:
        """Character と User の goals は同じ GoalTree 型."""
        from pneuma_core.models.entity import build_from_character, build_from_user_context

        char_entity = build_from_character(
            sample_character, goals=sample_goal_tree,
        )
        user_entity = build_from_user_context(
            sample_user_context, entity_id="user", goals=user_goal_tree,
        )

        assert isinstance(char_entity.goals, GoalTree)
        assert isinstance(user_entity.goals, GoalTree)

    def test_both_share_same_relation_type(
        self,
        sample_character: Character,
        sample_user_context: UserContext,
        sample_relations: list[Relation],
        user_relations: list[Relation],
    ) -> None:
        """Character と User の relations は同じ list[Relation] 型."""
        from pneuma_core.models.entity import build_from_character, build_from_user_context

        char_entity = build_from_character(
            sample_character, relations=sample_relations,
        )
        user_entity = build_from_user_context(
            sample_user_context, entity_id="user", relations=user_relations,
        )

        assert isinstance(char_entity.relations, list)
        assert isinstance(user_entity.relations, list)
        assert all(isinstance(r, Relation) for r in char_entity.relations)
        assert all(isinstance(r, Relation) for r in user_entity.relations)


# ===========================================================================
# 6. entity_type distinction
# ===========================================================================


class TestEntityTypeDistinction:
    """entity_type の区別テスト."""

    def test_character_entity_type(
        self,
        sample_character: Character,
    ) -> None:
        """build_from_character は entity_type='character' を設定する."""
        from pneuma_core.models.entity import build_from_character

        entity = build_from_character(sample_character)
        assert entity.entity_type == "character"

    def test_user_entity_type(
        self,
        sample_user_context: UserContext,
    ) -> None:
        """build_from_user_context は entity_type='user' を設定する."""
        from pneuma_core.models.entity import build_from_user_context

        entity = build_from_user_context(
            user_context=sample_user_context,
            entity_id="user",
        )
        assert entity.entity_type == "user"

    def test_can_distinguish_entity_types(
        self,
        sample_character: Character,
        sample_user_context: UserContext,
    ) -> None:
        """entity_type で character と user を区別できる."""
        from pneuma_core.models.entity import build_from_character, build_from_user_context

        char_entity = build_from_character(sample_character)
        user_entity = build_from_user_context(
            sample_user_context, entity_id="user",
        )

        assert char_entity.entity_type != user_entity.entity_type
        assert char_entity.entity_type == "character"
        assert user_entity.entity_type == "user"


# ===========================================================================
# 7. Pipeline test: 入力 -> EntityContext構築 -> フィールド検証
# ===========================================================================


class TestEntityContextPipeline:
    """データフローのパイプラインテスト.

    Character/UserContext -> build_from_* -> EntityContext
    両方が同じインターフェースで扱えることを検証する。
    """

    def test_unified_interface_character_and_user(
        self,
        sample_character: Character,
        sample_user_context: UserContext,
        sample_goal_tree: GoalTree,
        user_goal_tree: GoalTree,
        sample_relations: list[Relation],
        user_relations: list[Relation],
    ) -> None:
        """Character と User が同じ EntityContext インターフェースで扱える."""
        from pneuma_core.models.entity import (
            EntityContext,
            build_from_character,
            build_from_user_context,
        )

        char_entity = build_from_character(
            sample_character,
            goals=sample_goal_tree,
            relations=sample_relations,
            glossary="キャラ用語集",
        )
        user_entity = build_from_user_context(
            sample_user_context,
            entity_id="user",
            goals=user_goal_tree,
            relations=user_relations,
        )

        # Both are EntityContext instances
        assert isinstance(char_entity, EntityContext)
        assert isinstance(user_entity, EntityContext)

        # Both have the same field set (just different values/nulls)
        entities = [char_entity, user_entity]
        for entity in entities:
            # All entities must have these fields accessible
            assert hasattr(entity, "entity_id")
            assert hasattr(entity, "entity_type")
            assert hasattr(entity, "profile")
            assert hasattr(entity, "values_text")
            assert hasattr(entity, "background")
            assert hasattr(entity, "personality")
            assert hasattr(entity, "speaking_style")
            assert hasattr(entity, "goals")
            assert hasattr(entity, "relations")
            assert hasattr(entity, "glossary")
            assert hasattr(entity, "projects")
            assert hasattr(entity, "diary_summary")

    def test_process_list_of_entities(
        self,
        sample_character: Character,
        sample_user_context: UserContext,
    ) -> None:
        """EntityContext のリストとしてまとめて扱える."""
        from pneuma_core.models.entity import (
            EntityContext,
            build_from_character,
            build_from_user_context,
        )

        entities: list[EntityContext] = [
            build_from_character(sample_character),
            build_from_user_context(sample_user_context, entity_id="user"),
        ]

        assert len(entities) == 2
        assert entities[0].entity_type == "character"
        assert entities[1].entity_type == "user"

        # Can iterate uniformly
        for entity in entities:
            assert entity.entity_id is not None
            assert entity.entity_type in ("character", "user")


# ===========================================================================
# 8. Export from __init__.py
# ===========================================================================


class TestEntityContextExport:
    """EntityContext が models パッケージからエクスポートされていることを確認."""

    def test_import_from_models_package(self) -> None:
        """pneuma.core.models から EntityContext をインポートできる."""
        from pneuma_core.models import EntityContext

        entity = EntityContext(entity_id="test", entity_type="user")
        assert entity.entity_id == "test"
