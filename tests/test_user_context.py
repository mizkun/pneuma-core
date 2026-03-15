"""Tests for UserContextLoader (Issue #68).

5-layer user context structure:
  Layer 1: identity.md - Basic profile
  Layer 2: values.md - Values, career vision
  Layer 3a: glossary.md - Glossary
  Layer 3b: core_experiences.md - Core experiences (no summarization)
  Layer 4: projects/*.md - Active projects
  Layer 5: diary/YYYY-MM-DD.md - Diary entries
  Layer 6: diary_summary.md - Past diary summary

Note: relationships.md は relations.yaml に統合済み (#123)。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

from pneuma_core.runtime.user_context import (
    UserContext,
    UserContextChunk,
    UserContextLoader,
)


# --- Fixtures ---


@pytest.fixture
def tmp_context_dir(tmp_path: Path) -> Path:
    """Create a temporary user_context directory with sample files."""
    ctx = tmp_path / "user_context"
    ctx.mkdir()
    return ctx


@pytest.fixture
def populated_context_dir(tmp_context_dir: Path) -> Path:
    """Create a fully populated user_context directory."""
    # Layer 1: identity.md
    (tmp_context_dir / "identity.md").write_text(
        "# Identity\n\n## Basic Info\nName: Kyohei\nAge: 30\n\n## Location\nTokyo, Japan\n",
        encoding="utf-8",
    )

    # Layer 2: values.md
    (tmp_context_dir / "values.md").write_text(
        "# Values\n\n## Career Vision\nBuild meaningful AI products\n\n## Life Goals\nCreate positive impact\n",
        encoding="utf-8",
    )

    # Layer 3a: glossary.md
    (tmp_context_dir / "glossary.md").write_text(
        "# Glossary\n\n## Company\nPneuma Inc.\n\n## Terms\nRAG: Retrieval Augmented Generation\n",
        encoding="utf-8",
    )

    # Layer 3c: core_experiences.md
    (tmp_context_dir / "core_experiences.md").write_text(
        "# Core Experiences\n\nThe day I decided to become an engineer was transformative.\nI still remember the feeling vividly.\n",
        encoding="utf-8",
    )

    # Layer 4: projects/
    projects = tmp_context_dir / "projects"
    projects.mkdir()
    (projects / "pneuma.md").write_text(
        "# Pneuma Project\nAI character system with inner life\n",
        encoding="utf-8",
    )
    (projects / "book.md").write_text(
        "# Book Project\nWriting a book about AI consciousness\n",
        encoding="utf-8",
    )

    # Layer 5: diary/
    diary = tmp_context_dir / "diary"
    diary.mkdir()
    (diary / "2026-02-20.md").write_text(
        "# 2026-02-20\nWorked on Pneuma user context feature.\n",
        encoding="utf-8",
    )
    (diary / "2026-02-22.md").write_text(
        "# 2026-02-22\nHad a great brainstorming session.\n",
        encoding="utf-8",
    )
    (diary / "2026-02-21.md").write_text(
        "# 2026-02-21\nDebugged memory search scoring.\n",
        encoding="utf-8",
    )

    # Layer 6: diary_summary.md
    (tmp_context_dir / "diary_summary.md").write_text(
        "# Diary Summary\n\n## February 2026\nFocused on Pneuma development.\n",
        encoding="utf-8",
    )

    return tmp_context_dir


# --- UserContext dataclass ---


class TestUserContextDataclass:
    """UserContext dataclass definition tests."""

    def test_user_context_has_identity(self) -> None:
        ctx = UserContext()
        assert ctx.identity is None

    def test_user_context_has_values(self) -> None:
        ctx = UserContext()
        assert ctx.values is None

    def test_user_context_no_relationships_field(self) -> None:
        """relationships フィールドは削除済み (#123)."""
        ctx = UserContext()
        assert not hasattr(ctx, "relationships")

    def test_user_context_has_glossary(self) -> None:
        ctx = UserContext()
        assert ctx.glossary is None

    def test_user_context_has_core_experiences(self) -> None:
        ctx = UserContext()
        assert ctx.core_experiences is None

    def test_user_context_has_projects(self) -> None:
        ctx = UserContext()
        assert ctx.projects == {}

    def test_user_context_has_diary_entries(self) -> None:
        ctx = UserContext()
        assert ctx.diary_entries == {}

    def test_user_context_has_diary_summary(self) -> None:
        ctx = UserContext()
        assert ctx.diary_summary is None

    def test_user_context_with_all_fields(self) -> None:
        ctx = UserContext(
            identity="Name: Test",
            values="Build things",
            glossary="RAG: Retrieval Augmented Generation",
            core_experiences="A life-changing moment",
            projects={"project_a.md": "# Project A"},
            diary_entries={"2026-02-20.md": "Good day"},
            diary_summary="February was productive",
        )
        assert ctx.identity == "Name: Test"
        assert ctx.values == "Build things"
        assert ctx.glossary == "RAG: Retrieval Augmented Generation"
        assert ctx.core_experiences == "A life-changing moment"
        assert ctx.projects == {"project_a.md": "# Project A"}
        assert ctx.diary_entries == {"2026-02-20.md": "Good day"}
        assert ctx.diary_summary == "February was productive"


# --- UserContextChunk dataclass ---


class TestUserContextChunkDataclass:
    """UserContextChunk dataclass definition tests."""

    def test_chunk_has_content(self) -> None:
        chunk = UserContextChunk(
            content="test content",
            layer=1,
            source_file="identity.md",
        )
        assert chunk.content == "test content"

    def test_chunk_has_layer(self) -> None:
        chunk = UserContextChunk(
            content="test",
            layer=3,
            source_file="glossary.md",
        )
        assert chunk.layer == 3

    def test_chunk_has_source_file(self) -> None:
        chunk = UserContextChunk(
            content="test",
            layer=1,
            source_file="identity.md",
        )
        assert chunk.source_file == "identity.md"

    def test_chunk_has_metadata(self) -> None:
        chunk = UserContextChunk(
            content="test",
            layer=5,
            source_file="diary/2026-02-20.md",
            metadata={"date": "2026-02-20"},
        )
        assert chunk.metadata == {"date": "2026-02-20"}

    def test_chunk_metadata_default_empty(self) -> None:
        chunk = UserContextChunk(
            content="test",
            layer=1,
            source_file="identity.md",
        )
        assert chunk.metadata == {}


# --- UserContextLoader initialization ---


class TestUserContextLoaderInit:
    """UserContextLoader initialization tests."""

    def test_loader_accepts_path(self, tmp_context_dir: Path) -> None:
        loader = UserContextLoader(tmp_context_dir)
        assert loader.base_path == tmp_context_dir

    def test_loader_accepts_string_path(self, tmp_context_dir: Path) -> None:
        loader = UserContextLoader(str(tmp_context_dir))
        assert loader.base_path == tmp_context_dir

    def test_loader_nonexistent_directory(self, tmp_path: Path) -> None:
        """Nonexistent directory should not raise on init."""
        nonexistent = tmp_path / "does_not_exist"
        loader = UserContextLoader(nonexistent)
        assert loader.base_path == nonexistent


# --- File reading ---


class TestFileReading:
    """Tests for reading individual files."""

    @pytest.mark.asyncio
    async def test_load_identity(self, populated_context_dir: Path) -> None:
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert ctx.identity is not None
        assert "Kyohei" in ctx.identity

    @pytest.mark.asyncio
    async def test_load_values(self, populated_context_dir: Path) -> None:
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert ctx.values is not None
        assert "meaningful AI products" in ctx.values

    @pytest.mark.asyncio
    async def test_relationships_md_not_loaded(self, populated_context_dir: Path) -> None:
        """relationships.md は読み込まない (#123)."""
        # relationships.md がディレクトリに存在してもロードされないことを確認
        (populated_context_dir / "relationships.md").write_text(
            "# Relationships\n\n## Family\nWife: Yuki\n",
            encoding="utf-8",
        )
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert not hasattr(ctx, "relationships")

    @pytest.mark.asyncio
    async def test_load_glossary(self, populated_context_dir: Path) -> None:
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert ctx.glossary is not None
        assert "Retrieval Augmented Generation" in ctx.glossary

    @pytest.mark.asyncio
    async def test_load_core_experiences(self, populated_context_dir: Path) -> None:
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert ctx.core_experiences is not None
        assert "transformative" in ctx.core_experiences

    @pytest.mark.asyncio
    async def test_load_diary_summary(self, populated_context_dir: Path) -> None:
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert ctx.diary_summary is not None
        assert "Pneuma development" in ctx.diary_summary

    @pytest.mark.asyncio
    async def test_missing_file_returns_none(self, tmp_context_dir: Path) -> None:
        """Missing optional files should result in None."""
        loader = UserContextLoader(tmp_context_dir)
        ctx = await loader.load()
        assert ctx.identity is None
        assert ctx.values is None
        assert ctx.glossary is None
        assert ctx.core_experiences is None
        assert ctx.diary_summary is None


# --- Directory reading ---


class TestDirectoryReading:
    """Tests for reading projects/ and diary/ directories."""

    @pytest.mark.asyncio
    async def test_load_projects(self, populated_context_dir: Path) -> None:
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert len(ctx.projects) == 2
        assert "pneuma.md" in ctx.projects
        assert "book.md" in ctx.projects
        assert "AI character system" in ctx.projects["pneuma.md"]

    @pytest.mark.asyncio
    async def test_load_diary_entries(self, populated_context_dir: Path) -> None:
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        assert len(ctx.diary_entries) == 3
        assert "2026-02-20.md" in ctx.diary_entries
        assert "2026-02-21.md" in ctx.diary_entries
        assert "2026-02-22.md" in ctx.diary_entries

    @pytest.mark.asyncio
    async def test_diary_sorted_descending(self, populated_context_dir: Path) -> None:
        """Diary entries should be sorted by date descending."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        dates = list(ctx.diary_entries.keys())
        assert dates == ["2026-02-22.md", "2026-02-21.md", "2026-02-20.md"]

    @pytest.mark.asyncio
    async def test_missing_projects_dir(self, tmp_context_dir: Path) -> None:
        """Missing projects/ directory should result in empty dict."""
        loader = UserContextLoader(tmp_context_dir)
        ctx = await loader.load()
        assert ctx.projects == {}

    @pytest.mark.asyncio
    async def test_missing_diary_dir(self, tmp_context_dir: Path) -> None:
        """Missing diary/ directory should result in empty dict."""
        loader = UserContextLoader(tmp_context_dir)
        ctx = await loader.load()
        assert ctx.diary_entries == {}

    @pytest.mark.asyncio
    async def test_projects_only_md_files(self, tmp_context_dir: Path) -> None:
        """Only .md files in projects/ should be loaded."""
        projects = tmp_context_dir / "projects"
        projects.mkdir()
        (projects / "valid.md").write_text("# Valid", encoding="utf-8")
        (projects / "ignored.txt").write_text("ignored", encoding="utf-8")
        (projects / ".hidden").write_text("hidden", encoding="utf-8")

        loader = UserContextLoader(tmp_context_dir)
        ctx = await loader.load()
        assert "valid.md" in ctx.projects
        assert "ignored.txt" not in ctx.projects
        assert ".hidden" not in ctx.projects


# --- Nonexistent base directory ---


class TestNonexistentDirectory:
    """Tests for handling nonexistent base directory."""

    @pytest.mark.asyncio
    async def test_nonexistent_dir_returns_empty_context(self, tmp_path: Path) -> None:
        """Nonexistent directory should return empty UserContext without error."""
        nonexistent = tmp_path / "does_not_exist"
        loader = UserContextLoader(nonexistent)
        ctx = await loader.load()
        assert ctx.identity is None
        assert ctx.values is None
        assert ctx.glossary is None
        assert ctx.core_experiences is None
        assert ctx.projects == {}
        assert ctx.diary_entries == {}
        assert ctx.diary_summary is None


# --- .obsidian/ directory ignore ---


class TestObsidianIgnore:
    """Tests for ignoring .obsidian/ directory."""

    @pytest.mark.asyncio
    async def test_obsidian_dir_ignored(self, tmp_context_dir: Path) -> None:
        """Files in .obsidian/ should not be loaded."""
        obsidian = tmp_context_dir / ".obsidian"
        obsidian.mkdir()
        (obsidian / "config.json").write_text("{}", encoding="utf-8")
        (obsidian / "workspace.md").write_text("# workspace", encoding="utf-8")

        loader = UserContextLoader(tmp_context_dir)
        ctx = await loader.load()
        # .obsidian content should not appear anywhere
        assert ctx.identity is None
        assert ctx.projects == {}


# --- Chunk splitting ---


class TestChunkSplitting:
    """Tests for chunk splitting functionality."""

    @pytest.mark.asyncio
    async def test_chunks_from_identity_by_section(
        self, populated_context_dir: Path
    ) -> None:
        """Identity file should be split by markdown sections."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        identity_chunks = [c for c in chunks if c.source_file == "identity.md"]
        assert len(identity_chunks) >= 2  # "Basic Info" and "Location" sections
        assert all(c.layer == 1 for c in identity_chunks)

    @pytest.mark.asyncio
    async def test_chunks_from_values_by_section(
        self, populated_context_dir: Path
    ) -> None:
        """Values file should be split by markdown sections."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        values_chunks = [c for c in chunks if c.source_file == "values.md"]
        assert len(values_chunks) >= 2  # "Career Vision" and "Life Goals"
        assert all(c.layer == 2 for c in values_chunks)

    @pytest.mark.asyncio
    async def test_diary_one_chunk_per_entry(
        self, populated_context_dir: Path
    ) -> None:
        """Each diary entry should become one chunk."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        diary_chunks = [c for c in chunks if c.source_file.startswith("diary/")]
        assert len(diary_chunks) == 3
        assert all(c.layer == 5 for c in diary_chunks)

    @pytest.mark.asyncio
    async def test_diary_chunks_have_date_metadata(
        self, populated_context_dir: Path
    ) -> None:
        """Diary chunks should have date in metadata."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        diary_chunks = [c for c in chunks if c.source_file.startswith("diary/")]
        for chunk in diary_chunks:
            assert "date" in chunk.metadata

    @pytest.mark.asyncio
    async def test_projects_one_chunk_per_file(
        self, populated_context_dir: Path
    ) -> None:
        """Each project file should become one chunk."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        project_chunks = [c for c in chunks if c.source_file.startswith("projects/")]
        assert len(project_chunks) == 2
        assert all(c.layer == 4 for c in project_chunks)

    @pytest.mark.asyncio
    async def test_chunk_metadata_has_layer_and_source(
        self, populated_context_dir: Path
    ) -> None:
        """All chunks should have layer and source_file set."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.layer >= 1
            assert chunk.layer <= 6
            assert chunk.source_file != ""
            assert chunk.content != ""

    @pytest.mark.asyncio
    async def test_empty_context_produces_no_chunks(self, tmp_path: Path) -> None:
        """Empty UserContext should produce no chunks."""
        nonexistent = tmp_path / "does_not_exist"
        loader = UserContextLoader(nonexistent)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_core_experiences_single_chunk(
        self, populated_context_dir: Path
    ) -> None:
        """Core experiences should be a single chunk (no summarization)."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        ce_chunks = [c for c in chunks if c.source_file == "core_experiences.md"]
        # core_experiences is kept as a single chunk to preserve full context
        assert len(ce_chunks) == 1
        assert ce_chunks[0].layer == 3

    @pytest.mark.asyncio
    async def test_glossary_single_chunk(
        self, populated_context_dir: Path
    ) -> None:
        """Glossary should be a single chunk."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        glossary_chunks = [c for c in chunks if c.source_file == "glossary.md"]
        assert len(glossary_chunks) == 1
        assert glossary_chunks[0].layer == 3

    @pytest.mark.asyncio
    async def test_no_relationships_chunks(
        self, populated_context_dir: Path
    ) -> None:
        """relationships.md のチャンクは生成されない (#123)."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        rel_chunks = [c for c in chunks if c.source_file == "relationships.md"]
        assert len(rel_chunks) == 0


# --- Long content splitting ---


class TestLongContentSplitting:
    """Tests for splitting long content into fixed-size chunks."""

    @pytest.mark.asyncio
    async def test_long_section_is_split(self, tmp_context_dir: Path) -> None:
        """Sections exceeding max_chunk_chars should be split."""
        # Create a very long identity file with one huge section
        long_text = "A" * 5000
        (tmp_context_dir / "identity.md").write_text(
            f"# Identity\n\n## Bio\n{long_text}\n",
            encoding="utf-8",
        )

        loader = UserContextLoader(tmp_context_dir, max_chunk_chars=1000)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        identity_chunks = [c for c in chunks if c.source_file == "identity.md"]
        # The long section should be split into multiple chunks
        assert len(identity_chunks) >= 5

    @pytest.mark.asyncio
    async def test_split_chunks_preserve_content(self, tmp_context_dir: Path) -> None:
        """Split chunks should together contain all original content."""
        content = "Word " * 200  # ~1000 chars
        (tmp_context_dir / "identity.md").write_text(
            f"# Identity\n\n## Bio\n{content}\n",
            encoding="utf-8",
        )

        loader = UserContextLoader(tmp_context_dir, max_chunk_chars=500)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)
        identity_chunks = [c for c in chunks if c.source_file == "identity.md"]
        combined = "".join(c.content for c in identity_chunks)
        # All words should be preserved
        assert combined.count("Word") == 200


# --- Layer assignment ---


class TestLayerAssignment:
    """Tests for correct layer number assignment."""

    @pytest.mark.asyncio
    async def test_all_layers_assigned(self, populated_context_dir: Path) -> None:
        """Each file type should have the correct layer number."""
        loader = UserContextLoader(populated_context_dir)
        ctx = await loader.load()
        chunks = loader.to_chunks(ctx)

        # Collect layers per source file
        layer_map: dict[str, set[int]] = {}
        for chunk in chunks:
            layer_map.setdefault(chunk.source_file, set()).add(chunk.layer)

        assert layer_map["identity.md"] == {1}
        assert layer_map["values.md"] == {2}
        assert layer_map["glossary.md"] == {3}
        assert layer_map["core_experiences.md"] == {3}
        assert all(
            layer_map[f] == {4}
            for f in layer_map
            if f.startswith("projects/")
        )
        assert all(
            layer_map[f] == {5}
            for f in layer_map
            if f.startswith("diary/")
        )
        assert layer_map["diary_summary.md"] == {6}


# --- ユーザーゴール読み込みテスト ---


class TestLoadUserGoals:
    """vault/user/goals.yaml からユーザーの GoalTree を読み込む."""

    def test_load_goals_from_yaml(self, tmp_path: Path) -> None:
        """goals.yaml が存在すれば GoalTree が返る."""
        from pneuma_core.runtime.user_context import load_user_goals

        goals_yaml = tmp_path / "goals.yaml"
        goals_yaml.write_text(
            """\
visions:
  - id: uv-1
    content: 健康的な生活を送る

objectives:
  - id: uo-1
    vision_id: uv-1
    content: 毎日運動する
    status: active
    progress: 0.3

tasks:
  - id: ut-1
    objective_id: uo-1
    content: 筋トレを始める
    status: pending
""",
            encoding="utf-8",
        )

        result = load_user_goals(goals_yaml)
        assert result is not None
        assert len(result.visions) == 1
        assert result.visions[0].content == "健康的な生活を送る"
        assert len(result.objectives) == 1
        assert result.objectives[0].content == "毎日運動する"
        assert len(result.tasks) == 1
        assert result.tasks[0].content == "筋トレを始める"

    def test_load_goals_file_not_found(self, tmp_path: Path) -> None:
        """goals.yaml が存在しなければ None."""
        from pneuma_core.runtime.user_context import load_user_goals

        result = load_user_goals(tmp_path / "nonexistent.yaml")
        assert result is None

    def test_load_goals_all_character_id_is_user(self, tmp_path: Path) -> None:
        """読み込まれたゴールの character_id は全て 'user'."""
        from pneuma_core.runtime.user_context import load_user_goals

        goals_yaml = tmp_path / "goals.yaml"
        goals_yaml.write_text(
            """\
visions:
  - id: uv-1
    content: テスト

objectives:
  - id: uo-1
    vision_id: uv-1
    content: テスト目標
    status: active
    progress: 0.0

tasks:
  - id: ut-1
    objective_id: uo-1
    content: テストタスク
    status: pending
""",
            encoding="utf-8",
        )

        result = load_user_goals(goals_yaml)
        assert result is not None
        assert all(v.character_id == "user" for v in result.visions)
        assert all(o.character_id == "user" for o in result.objectives)
        assert all(t.character_id == "user" for t in result.tasks)
