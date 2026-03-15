"""Tests for UserContextWriter: UserContext ファイル自動更新 (#136)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pneuma_core.runtime.user_context_writer import UserContextWriter


# --- Fixtures ---


@pytest.fixture
def tmp_user_context(tmp_path: Path) -> Path:
    """テスト用の user_context ディレクトリを作成."""
    ctx_dir = tmp_path / "user_context"
    ctx_dir.mkdir()
    return ctx_dir


@pytest.fixture
def writer(tmp_user_context: Path) -> UserContextWriter:
    """UserContextWriter インスタンスを作成."""
    return UserContextWriter(tmp_user_context)


# --- Sample Markdown ---

SAMPLE_IDENTITY_MD = """\
# プロフィール

## 基本情報

- 名前: テスト太郎
- 年齢: 30歳

## 仕事

- 職業: エンジニア
- 会社: テスト株式会社

## 趣味

- 読書
- 映画鑑賞
"""

SAMPLE_VALUES_MD = """\
# 価値観

## 大切にしていること

- 誠実さ
- 成長

## キャリアビジョン

- AIエンジニアとして成長したい
"""


# --- Section Rewrite Tests ---


class TestSectionRewrite:
    """セクション書き換えのテスト."""

    @pytest.mark.asyncio
    async def test_rewrite_section_by_header(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """指定ヘッダーのセクションを書き換えられる."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 仕事",
                "action": "rewrite",
                "new_content": "- 職業: VP of AI\n- 会社: 新会社株式会社",
                "reason": "転職を反映",
            }
        ]

        count = await writer.apply(updates)
        assert count == 1

        content = (tmp_user_context / "identity.md").read_text(encoding="utf-8")
        assert "VP of AI" in content
        assert "新会社株式会社" in content

    @pytest.mark.asyncio
    async def test_rewrite_preserves_other_sections(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """書き換え対象外のセクションは保持される."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 仕事",
                "action": "rewrite",
                "new_content": "- 職業: データサイエンティスト",
                "reason": "転職",
            }
        ]

        await writer.apply(updates)

        content = (tmp_user_context / "identity.md").read_text(encoding="utf-8")
        # 基本情報セクションが残っている
        assert "テスト太郎" in content
        assert "30歳" in content
        # 趣味セクションが残っている
        assert "読書" in content
        assert "映画鑑賞" in content
        # 古い仕事情報は消えている
        assert "テスト株式会社" not in content

    @pytest.mark.asyncio
    async def test_rewrite_section_with_subsections(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """サブセクション(###)を含むセクションも正しく書き換える."""
        md_with_subsections = """\
# プロフィール

## 仕事

- 職業: エンジニア

### 職務経歴

| 期間 | 会社 |
|---|---|
| 2020〜 | テスト株式会社 |

### 退職経緯

- 転職のため

## 趣味

- 読書
"""
        (tmp_user_context / "identity.md").write_text(md_with_subsections, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 仕事",
                "action": "rewrite",
                "new_content": "- 職業: VP of AI\n\n### 職務経歴\n\n新しい経歴",
                "reason": "転職",
            }
        ]

        await writer.apply(updates)

        content = (tmp_user_context / "identity.md").read_text(encoding="utf-8")
        assert "VP of AI" in content
        assert "新しい経歴" in content
        # 趣味セクションは残っている
        assert "読書" in content
        # 古い内容は消えている
        assert "テスト株式会社" not in content

    @pytest.mark.asyncio
    async def test_rewrite_last_section(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """ファイル末尾のセクションも正しく書き換える."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 趣味",
                "action": "rewrite",
                "new_content": "- プログラミング\n- ゲーム",
                "reason": "趣味更新",
            }
        ]

        await writer.apply(updates)

        content = (tmp_user_context / "identity.md").read_text(encoding="utf-8")
        assert "プログラミング" in content
        assert "ゲーム" in content
        # 古い趣味は消えている
        assert "読書" not in content
        assert "映画鑑賞" not in content
        # 他のセクションは保持
        assert "テスト太郎" in content


# --- Append Tests ---


class TestAppend:
    """ファイル末尾追記のテスト."""

    @pytest.mark.asyncio
    async def test_append_to_existing_file(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """既存ファイルの末尾に追記できる."""
        (tmp_user_context / "core_experiences.md").write_text(
            "# 重要な体験\n\n## 体験1\n\n内容1\n", encoding="utf-8"
        )

        updates = [
            {
                "file": "core_experiences.md",
                "action": "append",
                "new_content": "## 新体験\n\n新しい内容",
                "reason": "重要な体験を記録",
            }
        ]

        count = await writer.apply(updates)
        assert count == 1

        content = (tmp_user_context / "core_experiences.md").read_text(encoding="utf-8")
        assert "新体験" in content
        assert "新しい内容" in content
        # 既存コンテンツも残っている
        assert "体験1" in content
        assert "内容1" in content

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_file(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """存在しないファイルへの append は新規作成."""
        updates = [
            {
                "file": "relationships.md",
                "action": "append",
                "new_content": "## ミラ\n\nAIキャラクター",
                "reason": "新しい関係性",
            }
        ]

        count = await writer.apply(updates)
        assert count == 1

        content = (tmp_user_context / "relationships.md").read_text(encoding="utf-8")
        assert "ミラ" in content


# --- Multiple Updates ---


class TestMultipleUpdates:
    """複数更新の適用テスト."""

    @pytest.mark.asyncio
    async def test_multiple_updates_applied(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """複数の更新が正しく適用される."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")
        (tmp_user_context / "values.md").write_text(SAMPLE_VALUES_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 仕事",
                "action": "rewrite",
                "new_content": "- 職業: CTO",
                "reason": "昇進",
            },
            {
                "file": "values.md",
                "action": "append",
                "new_content": "## 新しい価値観\n\n- チームワーク",
                "reason": "新しい気づき",
            },
        ]

        count = await writer.apply(updates)
        assert count == 2

        identity_content = (tmp_user_context / "identity.md").read_text(encoding="utf-8")
        assert "CTO" in identity_content

        values_content = (tmp_user_context / "values.md").read_text(encoding="utf-8")
        assert "チームワーク" in values_content


# --- Git Commit Tests ---


class TestGitCommit:
    """Git コミットのテスト."""

    @pytest.mark.asyncio
    async def test_git_commit_after_updates(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """更新後に git commit が実行される."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 仕事",
                "action": "rewrite",
                "new_content": "- 職業: CTO",
                "reason": "昇進",
            }
        ]

        with patch("pneuma_core.runtime.user_context_writer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await writer.apply(updates)

            # git add と git commit が呼ばれている
            calls = mock_run.call_args_list
            assert len(calls) == 2
            # git add .
            assert calls[0][0][0] == ["git", "add", "."]
            # git commit -m "..."
            assert calls[1][0][0][0:2] == ["git", "commit"]
            assert "昇進" in calls[1][0][0][3]

    @pytest.mark.asyncio
    async def test_no_commit_if_no_updates(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """更新がない場合は git commit しない."""
        updates: list[dict] = []

        with patch("pneuma_core.runtime.user_context_writer.subprocess.run") as mock_run:
            count = await writer.apply(updates)

            assert count == 0
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_commit_if_all_updates_fail(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """全更新が失敗した場合は git commit しない."""
        # パストラバーサル → 失敗
        updates = [
            {
                "file": "../../../etc/passwd",
                "section": "## Evil",
                "action": "rewrite",
                "new_content": "hacked",
                "reason": "attack",
            }
        ]

        with patch("pneuma_core.runtime.user_context_writer.subprocess.run") as mock_run:
            count = await writer.apply(updates)
            assert count == 0
            mock_run.assert_not_called()


# --- Error Handling ---


class TestErrorHandling:
    """エラーハンドリングのテスト."""

    @pytest.mark.asyncio
    async def test_invalid_file_path_ignored(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """不正なファイルパス(パストラバーサル)は安全に無視される."""
        updates = [
            {
                "file": "../../../etc/passwd",
                "section": "## Evil",
                "action": "rewrite",
                "new_content": "hacked",
                "reason": "attack",
            }
        ]

        count = await writer.apply(updates)
        assert count == 0

    @pytest.mark.asyncio
    async def test_unknown_action_ignored(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """不明な action は安全に無視される."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "action": "delete_everything",
                "reason": "不明なアクション",
            }
        ]

        count = await writer.apply(updates)
        assert count == 0

    @pytest.mark.asyncio
    async def test_rewrite_nonexistent_section_appends(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """存在しないセクションへの rewrite は append にフォールバック."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 存在しないセクション",
                "action": "rewrite",
                "new_content": "新しい内容",
                "reason": "新セクション追加",
            }
        ]

        count = await writer.apply(updates)
        assert count == 1

        content = (tmp_user_context / "identity.md").read_text(encoding="utf-8")
        # 新セクションが追加されている
        assert "## 存在しないセクション" in content
        assert "新しい内容" in content
        # 既存コンテンツも残っている
        assert "テスト太郎" in content

    @pytest.mark.asyncio
    async def test_file_outside_allowed_list_ignored(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """許可リスト外のファイルは無視される."""
        updates = [
            {
                "file": "secret.txt",
                "action": "append",
                "new_content": "secret data",
                "reason": "不正",
            }
        ]

        count = await writer.apply(updates)
        assert count == 0

    @pytest.mark.asyncio
    async def test_projects_subdirectory_allowed(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """projects/ サブディレクトリ内のファイルは許可される."""
        projects_dir = tmp_user_context / "projects"
        projects_dir.mkdir()
        (projects_dir / "pneuma.md").write_text("# Pneuma\n\n内容\n", encoding="utf-8")

        updates = [
            {
                "file": "projects/pneuma.md",
                "action": "append",
                "new_content": "## 進捗\n\n新しい進捗",
                "reason": "プロジェクト更新",
            }
        ]

        count = await writer.apply(updates)
        assert count == 1

        content = (projects_dir / "pneuma.md").read_text(encoding="utf-8")
        assert "新しい進捗" in content

    @pytest.mark.asyncio
    async def test_git_commit_failure_does_not_crash(
        self, writer: UserContextWriter, tmp_user_context: Path
    ) -> None:
        """git commit が失敗してもクラッシュしない."""
        (tmp_user_context / "identity.md").write_text(SAMPLE_IDENTITY_MD, encoding="utf-8")

        updates = [
            {
                "file": "identity.md",
                "section": "## 仕事",
                "action": "rewrite",
                "new_content": "- 職業: CTO",
                "reason": "昇進",
            }
        ]

        with patch(
            "pneuma_core.runtime.user_context_writer.subprocess.run",
            side_effect=subprocess.SubprocessError("git error"),
        ):
            # クラッシュしない
            count = await writer.apply(updates)
            assert count == 1
