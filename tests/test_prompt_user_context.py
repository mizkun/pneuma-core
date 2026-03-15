"""Tests for PromptBuilder UserContext 3-tier integration (Issue #69).

3-tier structure:
  Tier 1 (always, ~500 tokens): identity summary + top 3 projects + last session summary
  Tier 2 (session, ~1000 tokens): recent diary summary + all projects list
  Tier 3 (RAG, ~2000 tokens): search results from UserContextSearchEngine
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality
from pneuma_core.models.values import Values
from pneuma_core.runtime.prompt_builder import PromptBuilder
from pneuma_core.runtime.prompt_cache import CachedPrompt, PromptCache
from pneuma_core.runtime.user_context import UserContext, UserContextChunk
from pneuma_core.runtime.user_context_search import UserContextSearchResult

NOW = datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc)


# --- Test helpers ---


def _make_character(**kwargs) -> Character:
    defaults = dict(
        id="aine-001",
        name="アイネ",
        personality=Personality(
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.7,
            agreeableness=0.9,
            neuroticism=0.3,
        ),
        values=Values(
            self_transcendence=0.8,
            self_enhancement=0.3,
            openness_to_change=0.7,
            conservation=0.4,
        ),
        profile="好奇心旺盛な AI キャラクター",
        speaking_style="丁寧だけど親しみやすい口調",
        background="研究者に作られた AI",
        personality_description="明るく前向き",
        values_description="他者への思いやり",
    )
    defaults.update(kwargs)
    return Character(**defaults)


def _make_emotional_state(
    pleasure: float = 0.5,
    arousal: float = 0.3,
    dominance: float = 0.1,
    emotion_label: str = "喜び",
    situation: str = "楽しい会話中",
) -> EmotionalState:
    return EmotionalState(
        pleasure=pleasure,
        arousal=arousal,
        dominance=dominance,
        emotion_label=emotion_label,
        situation=situation,
    )


def _make_user_context(
    identity: str | None = None,
    projects: dict[str, str] | None = None,
    diary_entries: dict[str, str] | None = None,
    diary_summary: str | None = None,
    values: str | None = None,
    glossary: str | None = None,
    core_experiences: str | None = None,
) -> UserContext:
    return UserContext(
        identity=identity,
        values=values,
        glossary=glossary,
        core_experiences=core_experiences,
        projects=projects or {},
        diary_entries=diary_entries or {},
        diary_summary=diary_summary,
    )


def _make_search_results(
    contents: list[str],
    scores: list[float] | None = None,
) -> list[UserContextSearchResult]:
    """Create mock search results."""
    if scores is None:
        scores = [0.9 - i * 0.1 for i in range(len(contents))]
    results = []
    for content, score in zip(contents, scores):
        chunk = UserContextChunk(
            content=content,
            layer=3,
            source_file="glossary.md",
        )
        results.append(UserContextSearchResult(chunk=chunk, score=score))
    return results


# --- _build_user_context_section method existence ---


class TestUserContextSectionMethodExists:
    """_build_user_context_section メソッドの存在確認."""

    def test_method_exists(self) -> None:
        """PromptBuilder に _build_user_context_section が存在する."""
        builder = PromptBuilder()
        assert hasattr(builder, "_build_user_context_section")
        assert callable(builder._build_user_context_section)


# --- Tier 1: Always (identity + top projects + session summary) ---


class TestTier1Always:
    """Tier 1: 常時載せる情報の検証."""

    def test_identity_summary_included(self) -> None:
        """identity の要約がプロンプトに含まれる."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            identity="# プロフィール\n## 基本情報\n名前: 田中太郎\n年齢: 30歳\n職業: エンジニア",
        )
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=ctx,
        )
        assert "田中太郎" in prompt

    def test_top_projects_included(self) -> None:
        """projects のトップ3要約がプロンプトに含まれる."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            projects={
                "project-alpha.md": "# Project Alpha\nAI開発プロジェクト",
                "project-beta.md": "# Project Beta\nWebアプリ開発",
                "project-gamma.md": "# Project Gamma\nデータ分析基盤",
                "project-delta.md": "# Project Delta\nモバイルアプリ",
            },
        )
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=ctx,
        )
        # At least first 3 projects should be present
        assert "Project Alpha" in prompt
        assert "Project Beta" in prompt
        assert "Project Gamma" in prompt

    def test_diary_summary_included(self) -> None:
        """diary_summary (前回セッション要約的な位置づけ) が含まれる."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            diary_summary="## 先週のまとめ\n仕事が忙しかった。新しいプロジェクトを始めた。",
        )
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=ctx,
        )
        assert "先週のまとめ" in prompt


# --- Tier 2: Session start (recent diary + all projects list) ---


class TestTier2Session:
    """Tier 2: セッション開始時の情報の検証."""

    def test_recent_diary_entries_included(self) -> None:
        """直近日記要約がプロンプトに含まれる."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            diary_entries={
                "2026-02-24.md": "今日はコーディングに集中した。",
                "2026-02-23.md": "チームミーティングがあった。",
                "2026-02-22.md": "新機能のデザインレビュー。",
            },
        )
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=ctx,
        )
        assert "コーディング" in prompt

    def test_all_projects_list_included(self) -> None:
        """全 projects の1行要約一覧がプロンプトに含まれる."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            projects={
                "project-alpha.md": "# Project Alpha\nAI開発プロジェクト",
                "project-beta.md": "# Project Beta\nWebアプリ開発",
                "project-gamma.md": "# Project Gamma\nデータ分析基盤",
                "project-delta.md": "# Project Delta\nモバイルアプリ",
            },
        )
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=ctx,
        )
        # All projects should appear (Tier 2 has the full list)
        assert "Project Alpha" in prompt
        assert "Project Delta" in prompt


# --- Tier 3: RAG search results ---


class TestTier3RAG:
    """Tier 3: RAG 検索結果の検証."""

    def test_search_results_included(self) -> None:
        """RAG 検索結果がプロンプトに含まれる."""
        builder = PromptBuilder()
        search_results = _make_search_results([
            "田中太郎は山田花子と同僚で、一緒にProject Alphaを進めている。",
            "趣味は登山で、毎月第一土曜日に行く。",
        ])
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=_make_user_context(),
            user_context_search_results=search_results,
        )
        assert "田中太郎は山田花子と同僚" in prompt
        assert "趣味は登山" in prompt

    def test_no_search_results_no_rag_section(self) -> None:
        """検索結果がない場合、RAG セクションは生成されない."""
        builder = PromptBuilder()
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=_make_user_context(identity="テストユーザー"),
            user_context_search_results=[],
        )
        # identity should be present but no RAG content
        assert "テストユーザー" in prompt


# --- Backward compatibility: UserContext is None ---


class TestBackwardCompatibility:
    """UserContext が None の場合のテスト（後方互換）."""

    def test_none_user_context_no_section(self) -> None:
        """user_context=None でセクションが生成されない."""
        builder = PromptBuilder()
        prompt_without = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
        )
        # Should not contain user context heading
        assert "ユーザーについて" not in prompt_without

    def test_none_user_context_same_as_before(self) -> None:
        """user_context=None の場合、既存の build() と同じ結果."""
        builder = PromptBuilder()
        char = _make_character()
        state = _make_emotional_state()
        goals = GoalTree()
        memories: list[EpisodicMemory | SemanticMemory] = []

        prompt_default = builder.build(char, state, goals, memories)
        prompt_explicit_none = builder.build(
            char, state, goals, memories, user_context=None
        )
        assert prompt_default == prompt_explicit_none

    def test_empty_user_context_no_section(self) -> None:
        """空の UserContext でもセクションが生成されない."""
        builder = PromptBuilder()
        empty_ctx = UserContext()
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=empty_ctx,
        )
        assert "ユーザーについて" not in prompt


# --- build() method signature ---


class TestBuildSignature:
    """build() メソッドが user_context パラメータを受け取る."""

    def test_build_accepts_user_context_kwarg(self) -> None:
        """build() が user_context キーワード引数を受け取る."""
        builder = PromptBuilder()
        # Should not raise
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=_make_user_context(identity="テスト"),
        )
        assert isinstance(prompt, str)

    def test_build_accepts_search_results_kwarg(self) -> None:
        """build() が user_context_search_results キーワード引数を受け取る."""
        builder = PromptBuilder()
        results = _make_search_results(["テスト結果"])
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            [],
            user_context=_make_user_context(),
            user_context_search_results=results,
        )
        assert isinstance(prompt, str)


# --- PromptCache integration ---


class TestPromptCacheIntegration:
    """PromptCache での UserContext 統合テスト."""

    def test_tier1_in_static_section(self) -> None:
        """Tier 1 (identity) は static セクションに含まれる."""
        cache = PromptCache()
        ctx = _make_user_context(
            identity="名前: 田中太郎、30歳、エンジニア",
        )
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=ctx,
        )
        assert "田中太郎" in cached.static_section

    def test_tier2_in_dynamic_section(self) -> None:
        """Tier 2 (diary) は dynamic セクションに含まれる."""
        cache = PromptCache()
        ctx = _make_user_context(
            diary_entries={
                "2026-02-24.md": "今日はコーディングに集中した。",
            },
        )
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=ctx,
        )
        assert "コーディング" in cached.dynamic_section

    def test_tier3_in_dynamic_section(self) -> None:
        """Tier 3 (RAG) は dynamic セクションに含まれる."""
        cache = PromptCache()
        ctx = _make_user_context()
        search_results = _make_search_results(["関連情報: ユーザーは猫好き"])
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=ctx,
            user_context_search_results=search_results,
        )
        assert "猫好き" in cached.dynamic_section

    def test_cache_backward_compatible(self) -> None:
        """PromptCache が user_context なしでも動作する."""
        cache = PromptCache()
        cached = cache.build(
            character=_make_character(),
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert isinstance(cached, CachedPrompt)
        assert len(cached.full_prompt) > 0


# --- Token budget configuration ---


class TestTokenBudget:
    """トークン予算の設定テスト."""

    def test_token_budget_configurable(self) -> None:
        """UserContextConfig でトークン予算を設定できる."""
        from pneuma_core.runtime.prompt_builder import UserContextConfig

        config = UserContextConfig(
            tier1_max_tokens=300,
            tier2_max_tokens=800,
            tier3_max_tokens=1500,
        )
        assert config.tier1_max_tokens == 300
        assert config.tier2_max_tokens == 800
        assert config.tier3_max_tokens == 1500

    def test_default_token_budget(self) -> None:
        """デフォルトのトークン予算が設定されている."""
        from pneuma_core.runtime.prompt_builder import UserContextConfig

        config = UserContextConfig()
        assert config.tier1_max_tokens == 500
        assert config.tier2_max_tokens == 1000
        assert config.tier3_max_tokens == 2000


# --- Section ordering ---


class TestSectionOrdering:
    """セクション順序のテスト."""

    def test_user_context_after_values_before_memory(self) -> None:
        """UserContext は Values の後、Memory の前に配置される."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            identity="名前: テストユーザー",
        )
        memories = [
            EpisodicMemory(
                id="ep-1",
                character_id="aine-001",
                content="テスト記憶データ",
                timestamp=NOW,
                emotional_valence=0.5,
                importance=0.8,
            )
        ]
        prompt = builder.build(
            _make_character(),
            _make_emotional_state(),
            GoalTree(),
            memories,
            user_context=ctx,
        )
        values_pos = prompt.find("価値観")
        user_ctx_pos = prompt.find("テストユーザー")
        memory_pos = prompt.find("テスト記憶データ")

        assert values_pos < user_ctx_pos, "UserContext should come after Values"
        assert user_ctx_pos < memory_pos, "UserContext should come before Memory"


# --- build_static_sections / build_dynamic_sections with user_context ---


class TestStaticDynamicSectionsWithUserContext:
    """build_static_sections / build_dynamic_sections に user_context を渡す."""

    def test_static_sections_include_tier1(self) -> None:
        """build_static_sections に Tier 1 が含まれる."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            identity="名前: 田中太郎、エンジニア",
        )
        result = builder.build_static_sections(
            _make_character(),
            user_context=ctx,
        )
        assert "田中太郎" in result

    def test_dynamic_sections_include_tier2(self) -> None:
        """build_dynamic_sections に Tier 2 が含まれる."""
        builder = PromptBuilder()
        ctx = _make_user_context(
            diary_entries={"2026-02-24.md": "今日の日記内容"},
        )
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context=ctx,
        )
        assert "今日の日記内容" in result

    def test_dynamic_sections_include_tier3(self) -> None:
        """build_dynamic_sections に Tier 3 が含まれる."""
        builder = PromptBuilder()
        search_results = _make_search_results(["RAG検索結果テスト"])
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
            user_context_search_results=search_results,
        )
        assert "RAG検索結果テスト" in result

    def test_static_sections_backward_compatible(self) -> None:
        """build_static_sections が user_context なしでも動作する."""
        builder = PromptBuilder()
        result = builder.build_static_sections(_make_character())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_dynamic_sections_backward_compatible(self) -> None:
        """build_dynamic_sections が user_context なしでも動作する."""
        builder = PromptBuilder()
        result = builder.build_dynamic_sections(
            emotional_state=_make_emotional_state(),
            goal_tree=GoalTree(),
            memories=[],
        )
        assert isinstance(result, str)
        assert len(result) > 0
