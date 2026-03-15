"""Tests for UserContextConsolidator (Issue #74)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pneuma_core.runtime.user_context_consolidator import (
    ContextUpdate,
    ContextUpdateRisk,
    ConsolidationResult,
    UserContextConsolidator,
)


# --- Helpers ---


def _make_llm_response(updates: list[dict]) -> str:
    """LLM レスポンス JSON を生成."""
    return json.dumps(updates, ensure_ascii=False)


def _make_conversation() -> list[dict]:
    """テスト用の会話履歴を生成."""
    return [
        {"role": "user", "content": "最近引っ越して渋谷に住み始めたんだ"},
        {"role": "assistant", "content": "渋谷に引っ越したんですね！新生活はどうですか？"},
        {"role": "user", "content": "カフェイン断ち3日目で頑張ってる"},
        {"role": "assistant", "content": "3日目！順調ですね。体調に変化はありますか？"},
    ]


def _make_mock_llm(response_json: list[dict]) -> AsyncMock:
    """LLMAdapter のモックを生成."""
    mock = AsyncMock()
    mock.generate.return_value = AsyncMock(
        content=_make_llm_response(response_json),
        model="claude-haiku-4-5-20251001",
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    return mock


def _setup_user_context_dir(tmp_path: Path) -> Path:
    """テスト用のユーザーコンテキストディレクトリを作成."""
    ctx_dir = tmp_path / "user_context"
    ctx_dir.mkdir()

    # identity.md
    (ctx_dir / "identity.md").write_text(
        "# Identity\n\n## 基本情報\n名前: テストユーザー\n居住地: 東京\n\n## 職業\nエンジニア\n",
        encoding="utf-8",
    )

    # values.md
    (ctx_dir / "values.md").write_text(
        "# Values\n\n## 大切にしていること\n誠実さ\n",
        encoding="utf-8",
    )

    # glossary.md
    (ctx_dir / "glossary.md").write_text(
        "# Glossary\n\n## 用語\n- Pneuma: AIキャラクターシステム\n",
        encoding="utf-8",
    )

    # core_experiences.md
    (ctx_dir / "core_experiences.md").write_text(
        "# Core Experiences\n\n## 原体験\n- 大学時代にプログラミングに出会った\n",
        encoding="utf-8",
    )

    # projects/
    projects_dir = ctx_dir / "projects"
    projects_dir.mkdir()
    (projects_dir / "health.md").write_text(
        "# Health\n\n## カフェイン断ち\nステータス: 開始前\n",
        encoding="utf-8",
    )

    return ctx_dir


# --- ContextUpdateRisk ---


class TestContextUpdateRisk:
    """ContextUpdateRisk enum の検証."""

    def test_risk_levels_exist(self) -> None:
        assert ContextUpdateRisk.LOW is not None
        assert ContextUpdateRisk.MEDIUM is not None
        assert ContextUpdateRisk.HIGH is not None

    def test_risk_values(self) -> None:
        assert ContextUpdateRisk.LOW.value == "low"
        assert ContextUpdateRisk.MEDIUM.value == "medium"
        assert ContextUpdateRisk.HIGH.value == "high"


# --- ContextUpdate ---


class TestContextUpdate:
    """ContextUpdate dataclass の検証."""

    def test_create_context_update(self) -> None:
        update = ContextUpdate(
            target_file="projects/health.md",
            update_type="modify",
            content="## カフェイン断ち\nステータス: 3日目\n",
            risk_level=ContextUpdateRisk.LOW,
            reason="ユーザーがカフェイン断ち3日目と報告",
        )
        assert update.target_file == "projects/health.md"
        assert update.update_type == "modify"
        assert update.risk_level == ContextUpdateRisk.LOW
        assert "3日目" in update.content
        assert update.reason is not None

    def test_update_types(self) -> None:
        """add, modify, replace の3種類がサポートされる."""
        for update_type in ["add", "modify", "replace"]:
            update = ContextUpdate(
                target_file="test.md",
                update_type=update_type,
                content="test",
                risk_level=ContextUpdateRisk.LOW,
                reason="test",
            )
            assert update.update_type == update_type

    def test_section_field(self) -> None:
        """セクション指定フィールドがある."""
        update = ContextUpdate(
            target_file="identity.md",
            update_type="modify",
            content="居住地: 渋谷",
            risk_level=ContextUpdateRisk.HIGH,
            reason="引っ越し報告",
            section="## 基本情報",
        )
        assert update.section == "## 基本情報"

    def test_section_default_none(self) -> None:
        """section のデフォルトは None."""
        update = ContextUpdate(
            target_file="test.md",
            update_type="add",
            content="test",
            risk_level=ContextUpdateRisk.LOW,
            reason="test",
        )
        assert update.section is None


# --- ConsolidationResult ---


class TestConsolidationResult:
    """ConsolidationResult dataclass の検証."""

    def test_default_values(self) -> None:
        result = ConsolidationResult()
        assert result.applied == []
        assert result.pending_approval == []
        assert result.change_log == []

    def test_with_values(self) -> None:
        update_low = ContextUpdate(
            target_file="projects/health.md",
            update_type="modify",
            content="ステータス: 3日目",
            risk_level=ContextUpdateRisk.LOW,
            reason="カフェイン断ち進捗",
        )
        update_high = ContextUpdate(
            target_file="identity.md",
            update_type="modify",
            content="居住地: 渋谷",
            risk_level=ContextUpdateRisk.HIGH,
            reason="引っ越し報告",
        )
        result = ConsolidationResult(
            applied=[update_low],
            pending_approval=[update_high],
            change_log=["projects/health.md: カフェイン断ち進捗を更新"],
        )
        assert len(result.applied) == 1
        assert len(result.pending_approval) == 1
        assert len(result.change_log) == 1


# --- UserContextConsolidator Initialization ---


class TestUserContextConsolidatorInit:
    """UserContextConsolidator の初期化テスト."""

    def test_init(self) -> None:
        mock_llm = AsyncMock()
        consolidator = UserContextConsolidator(llm=mock_llm)
        assert consolidator.llm is mock_llm

    def test_init_with_model(self) -> None:
        mock_llm = AsyncMock()
        consolidator = UserContextConsolidator(
            llm=mock_llm, model="claude-haiku-4-5-20251001"
        )
        assert consolidator._model == "claude-haiku-4-5-20251001"


# --- LLM Analysis ---


class TestLLMAnalysis:
    """LLM による会話分析テスト."""

    @pytest.mark.asyncio
    async def test_analyze_conversation_returns_updates(self, tmp_path: Path) -> None:
        """会話履歴からコンテキスト更新を検出する."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n",
                "risk_level": "low",
                "reason": "カフェイン断ち3日目の報告",
                "section": "## カフェイン断ち",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert mock_llm.generate.called
        assert len(result.applied) == 1
        assert result.applied[0].target_file == "projects/health.md"

    @pytest.mark.asyncio
    async def test_no_updates_returns_empty(self, tmp_path: Path) -> None:
        """更新が不要な場合は空リストを返す."""
        mock_llm = _make_mock_llm([])
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert result.applied == []
        assert result.pending_approval == []
        assert result.change_log == []

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self, tmp_path: Path) -> None:
        """LLM がエラーを返した場合は空の結果を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("API Error")
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert result.applied == []
        assert result.pending_approval == []

    @pytest.mark.asyncio
    async def test_llm_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        """LLM が不正な JSON を返した場合は空の結果を返す."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = AsyncMock(
            content="This is not JSON",
            model="claude-haiku-4-5-20251001",
            usage={},
        )
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert result.applied == []
        assert result.pending_approval == []

    @pytest.mark.asyncio
    async def test_llm_markdown_wrapped_json(self, tmp_path: Path) -> None:
        """LLM がマークダウンコードブロックで囲んだ JSON を返した場合もパースできる."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n",
                "risk_level": "low",
                "reason": "カフェイン断ち3日目の報告",
            },
        ]
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = AsyncMock(
            content=f"```json\n{json.dumps(updates_json, ensure_ascii=False)}\n```",
            model="claude-haiku-4-5-20251001",
            usage={},
        )
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.applied) == 1


# --- Risk Level Classification ---


class TestRiskLevelClassification:
    """リスクレベル別の処理テスト."""

    @pytest.mark.asyncio
    async def test_low_risk_auto_applied(self, tmp_path: Path) -> None:
        """低リスク（projects/）は自動的にファイルを更新する."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n",
                "risk_level": "low",
                "reason": "カフェイン断ち進捗更新",
                "section": "## カフェイン断ち",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.applied) == 1
        assert result.applied[0].risk_level == ContextUpdateRisk.LOW
        assert len(result.pending_approval) == 0
        # ファイルが実際に更新されていることを確認
        content = (ctx_dir / "projects" / "health.md").read_text(encoding="utf-8")
        assert "3日目" in content

    @pytest.mark.asyncio
    async def test_medium_risk_auto_applied_with_summary(
        self, tmp_path: Path
    ) -> None:
        """中リスク（glossary）は自動更新 + サマリーを返す."""
        updates_json = [
            {
                "target_file": "glossary.md",
                "update_type": "add",
                "content": "- 渋谷: ユーザーの新しい居住エリア\n",
                "risk_level": "medium",
                "reason": "新しい用語の追加",
                "section": "## 用語",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.applied) == 1
        assert result.applied[0].risk_level == ContextUpdateRisk.MEDIUM
        # ファイルが実際に更新されている
        content = (ctx_dir / "glossary.md").read_text(encoding="utf-8")
        assert "渋谷" in content
        # 変更サマリーが change_log に記録されている
        assert len(result.change_log) > 0

    @pytest.mark.asyncio
    async def test_high_risk_pending_approval(self, tmp_path: Path) -> None:
        """高リスク（identity, values, core_experiences）は提案のみ、ファイル更新しない."""
        updates_json = [
            {
                "target_file": "identity.md",
                "update_type": "modify",
                "content": "居住地: 渋谷",
                "risk_level": "high",
                "reason": "居住地の変更",
                "section": "## 基本情報",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)
        original_content = (ctx_dir / "identity.md").read_text(encoding="utf-8")

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.pending_approval) == 1
        assert result.pending_approval[0].risk_level == ContextUpdateRisk.HIGH
        assert len(result.applied) == 0
        # ファイルは変更されていないことを確認
        current_content = (ctx_dir / "identity.md").read_text(encoding="utf-8")
        assert current_content == original_content

    @pytest.mark.asyncio
    async def test_mixed_risk_levels(self, tmp_path: Path) -> None:
        """複数のリスクレベルが混在する場合の処理."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n",
                "risk_level": "low",
                "reason": "カフェイン断ち進捗",
                "section": "## カフェイン断ち",
            },
            {
                "target_file": "glossary.md",
                "update_type": "add",
                "content": "- 渋谷: ユーザーの居住エリア\n",
                "risk_level": "medium",
                "reason": "用語追加",
                "section": "## 用語",
            },
            {
                "target_file": "identity.md",
                "update_type": "modify",
                "content": "居住地: 渋谷",
                "risk_level": "high",
                "reason": "居住地変更",
                "section": "## 基本情報",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        # LOW + MEDIUM は applied に
        assert len(result.applied) == 2
        # HIGH は pending_approval に
        assert len(result.pending_approval) == 1
        # change_log は LOW + MEDIUM 分
        assert len(result.change_log) >= 2

    @pytest.mark.asyncio
    async def test_glossary_add_is_medium_risk(self, tmp_path: Path) -> None:
        """glossary.md への追加は中リスク."""
        updates_json = [
            {
                "target_file": "glossary.md",
                "update_type": "add",
                "content": "- 田中さん: 職場の同僚\n",
                "risk_level": "medium",
                "reason": "新しい人物の追加",
                "section": "## 用語",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.applied) == 1
        assert result.applied[0].risk_level == ContextUpdateRisk.MEDIUM
        content = (ctx_dir / "glossary.md").read_text(encoding="utf-8")
        assert "田中さん" in content

    @pytest.mark.asyncio
    async def test_values_is_high_risk(self, tmp_path: Path) -> None:
        """values.md の更新は高リスク."""
        updates_json = [
            {
                "target_file": "values.md",
                "update_type": "modify",
                "content": "## 大切にしていること\n誠実さ\n自由\n",
                "risk_level": "high",
                "reason": "価値観の追加",
                "section": "## 大切にしていること",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)
        original = (ctx_dir / "values.md").read_text(encoding="utf-8")

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.pending_approval) == 1
        assert result.pending_approval[0].risk_level == ContextUpdateRisk.HIGH
        current = (ctx_dir / "values.md").read_text(encoding="utf-8")
        assert current == original

    @pytest.mark.asyncio
    async def test_core_experiences_is_high_risk(self, tmp_path: Path) -> None:
        """core_experiences.md の更新は高リスク."""
        updates_json = [
            {
                "target_file": "core_experiences.md",
                "update_type": "add",
                "content": "- 渋谷への引っ越し\n",
                "risk_level": "high",
                "reason": "新しい重要体験",
                "section": "## 原体験",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)
        original = (ctx_dir / "core_experiences.md").read_text(encoding="utf-8")

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.pending_approval) == 1
        current = (ctx_dir / "core_experiences.md").read_text(encoding="utf-8")
        assert current == original


# --- File Operations ---


class TestFileOperations:
    """ファイル操作テスト."""

    @pytest.mark.asyncio
    async def test_section_update_modifies_only_section(
        self, tmp_path: Path
    ) -> None:
        """セクション指定で部分更新する（ファイル全体を上書きしない）."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n目標: 1週間\n",
                "risk_level": "low",
                "reason": "ステータス更新",
                "section": "## カフェイン断ち",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        # ファイルの先頭タイトルは保持される
        consolidator = UserContextConsolidator(llm=mock_llm)
        await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        content = (ctx_dir / "projects" / "health.md").read_text(encoding="utf-8")
        # タイトル行が保持されている
        assert "# Health" in content
        # セクションが更新されている
        assert "3日目" in content
        assert "目標: 1週間" in content

    @pytest.mark.asyncio
    async def test_add_new_section(self, tmp_path: Path) -> None:
        """新しいセクションを追加する."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "add",
                "content": "## 運動習慣\nステータス: 検討中\n",
                "risk_level": "low",
                "reason": "新しいプロジェクト情報",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        content = (ctx_dir / "projects" / "health.md").read_text(encoding="utf-8")
        # 既存のセクションが保持されている
        assert "# Health" in content
        assert "カフェイン断ち" in content
        # 新しいセクションが追加されている
        assert "## 運動習慣" in content

    @pytest.mark.asyncio
    async def test_create_new_file(self, tmp_path: Path) -> None:
        """存在しないファイルを新規作成する."""
        updates_json = [
            {
                "target_file": "projects/work.md",
                "update_type": "replace",
                "content": "# Work\n\n## 転職活動\nステータス: 情報収集中\n",
                "risk_level": "low",
                "reason": "新しいプロジェクトファイル作成",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        new_file = ctx_dir / "projects" / "work.md"
        assert new_file.exists()
        content = new_file.read_text(encoding="utf-8")
        assert "# Work" in content
        assert "転職活動" in content

    @pytest.mark.asyncio
    async def test_create_file_with_new_directory(self, tmp_path: Path) -> None:
        """存在しないディレクトリ配下にファイルを作成する（ディレクトリも作成）."""
        updates_json = [
            {
                "target_file": "projects/side_projects.md",
                "update_type": "replace",
                "content": "# Side Projects\n\n## ブログ\nステータス: 開始\n",
                "risk_level": "low",
                "reason": "新規プロジェクト追加",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)

        # projects/ ディレクトリがない状態のコンテキストディレクトリ
        ctx_dir = tmp_path / "user_context_no_projects"
        ctx_dir.mkdir()

        consolidator = UserContextConsolidator(llm=mock_llm)
        await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        new_file = ctx_dir / "projects" / "side_projects.md"
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_replace_entire_file(self, tmp_path: Path) -> None:
        """replace タイプでファイル全体を置換する."""
        new_content = "# Health\n\n## カフェイン断ち\nステータス: 完了！\n達成日: 2026-02-25\n"
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "replace",
                "content": new_content,
                "risk_level": "low",
                "reason": "ファイル全体更新",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        content = (ctx_dir / "projects" / "health.md").read_text(encoding="utf-8")
        assert content == new_content


# --- Change Log ---


class TestChangeLog:
    """変更ログのテスト."""

    @pytest.mark.asyncio
    async def test_change_log_recorded(self, tmp_path: Path) -> None:
        """変更ログが ConsolidationResult に記録される."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n",
                "risk_level": "low",
                "reason": "カフェイン断ち進捗",
                "section": "## カフェイン断ち",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert len(result.change_log) == 1
        log_entry = result.change_log[0]
        assert "projects/health.md" in log_entry
        assert "カフェイン断ち進捗" in log_entry

    @pytest.mark.asyncio
    async def test_change_log_includes_reason(self, tmp_path: Path) -> None:
        """変更ログに reason が含まれる."""
        updates_json = [
            {
                "target_file": "glossary.md",
                "update_type": "add",
                "content": "- 渋谷: 居住エリア\n",
                "risk_level": "medium",
                "reason": "新用語追加: 渋谷",
                "section": "## 用語",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert any("新用語追加" in log for log in result.change_log)

    @pytest.mark.asyncio
    async def test_update_log_file_written(self, tmp_path: Path) -> None:
        """変更ログが .update_log.md ファイルに書き出される."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n",
                "risk_level": "low",
                "reason": "カフェイン断ち進捗",
                "section": "## カフェイン断ち",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        log_file = ctx_dir / ".update_log.md"
        assert log_file.exists()
        log_content = log_file.read_text(encoding="utf-8")
        assert "projects/health.md" in log_content

    @pytest.mark.asyncio
    async def test_update_log_appends(self, tmp_path: Path) -> None:
        """変更ログは追記される（上書きしない）."""
        updates_json_1 = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "## カフェイン断ち\nステータス: 3日目\n",
                "risk_level": "low",
                "reason": "1回目の更新",
                "section": "## カフェイン断ち",
            },
        ]
        updates_json_2 = [
            {
                "target_file": "glossary.md",
                "update_type": "add",
                "content": "- テスト: テスト用語\n",
                "risk_level": "medium",
                "reason": "2回目の更新",
                "section": "## 用語",
            },
        ]

        ctx_dir = _setup_user_context_dir(tmp_path)

        # 1回目
        mock_llm_1 = _make_mock_llm(updates_json_1)
        consolidator = UserContextConsolidator(llm=mock_llm_1)
        await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        # 2回目
        mock_llm_2 = _make_mock_llm(updates_json_2)
        consolidator2 = UserContextConsolidator(llm=mock_llm_2)
        await consolidator2.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        log_file = ctx_dir / ".update_log.md"
        log_content = log_file.read_text(encoding="utf-8")
        assert "1回目の更新" in log_content
        assert "2回目の更新" in log_content

    @pytest.mark.asyncio
    async def test_high_risk_also_logged(self, tmp_path: Path) -> None:
        """高リスクの提案もログに記録される（ファイルは変更しないが）."""
        updates_json = [
            {
                "target_file": "identity.md",
                "update_type": "modify",
                "content": "居住地: 渋谷",
                "risk_level": "high",
                "reason": "居住地変更提案",
                "section": "## 基本情報",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        # change_log に高リスクの提案も記録される
        assert any("identity.md" in log for log in result.change_log)
        assert any("提案" in log or "pending" in log.lower() for log in result.change_log)


# --- Validation ---


class TestValidation:
    """入力バリデーションテスト."""

    @pytest.mark.asyncio
    async def test_invalid_update_type_skipped(self, tmp_path: Path) -> None:
        """不正な update_type の更新はスキップされる."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "invalid_type",
                "content": "test",
                "risk_level": "low",
                "reason": "test",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert result.applied == []
        assert result.pending_approval == []

    @pytest.mark.asyncio
    async def test_invalid_risk_level_skipped(self, tmp_path: Path) -> None:
        """不正な risk_level の更新はスキップされる."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                "update_type": "modify",
                "content": "test",
                "risk_level": "invalid",
                "reason": "test",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert result.applied == []
        assert result.pending_approval == []

    @pytest.mark.asyncio
    async def test_missing_required_fields_skipped(self, tmp_path: Path) -> None:
        """必須フィールドが欠落している更新はスキップされる."""
        updates_json = [
            {
                "target_file": "projects/health.md",
                # update_type が欠落
                "content": "test",
                "risk_level": "low",
                "reason": "test",
            },
        ]
        mock_llm = _make_mock_llm(updates_json)
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=_make_conversation(),
            user_context_dir=ctx_dir,
        )

        assert result.applied == []
        assert result.pending_approval == []

    @pytest.mark.asyncio
    async def test_empty_conversation_returns_empty(self, tmp_path: Path) -> None:
        """空の会話履歴では空の結果を返す."""
        mock_llm = _make_mock_llm([])
        ctx_dir = _setup_user_context_dir(tmp_path)

        consolidator = UserContextConsolidator(llm=mock_llm)
        result = await consolidator.consolidate(
            conversation_history=[],
            user_context_dir=ctx_dir,
        )

        assert result.applied == []
        assert result.pending_approval == []
