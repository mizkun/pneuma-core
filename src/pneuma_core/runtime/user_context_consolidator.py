"""UserContextConsolidator: analyzes conversations and updates user context files.

Analyzes conversation history using LLM (Haiku) to detect context updates,
then applies them based on risk level:
  - LOW (projects/): auto-apply
  - MEDIUM (glossary): auto-apply + report summary
  - HIGH (identity, values, core_experiences): proposal only, pending user approval

File updates use section-level partial modification (not full file overwrite).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest

_VALID_UPDATE_TYPES = {"add", "modify", "replace"}

_SECTION_RE = re.compile(r"(?=^## )", re.MULTILINE)

ANALYSIS_SYSTEM_PROMPT = """\
あなたはユーザーとAIキャラクターの会話を分析し、ユーザーコンテキストの更新提案を生成するアシスタントです。

ユーザーコンテキストは以下の層構造で管理されています:
- identity.md: 基本プロフィール（名前、居住地、年齢など）
- values.md: 価値観、キャリアビジョン
- glossary.md: 用語集（ユーザー固有の用語や略語）
- core_experiences.md: 原体験（人生を変えた経験）
- projects/*.md: 進行中のプロジェクト（健康管理、仕事など）

会話から検出された情報を、適切なファイルへの更新提案として出力してください。

リスクレベル:
- "low": projects/ 配下のステータス更新（自動適用される）
- "medium": glossary への追加（自動適用 + 報告される）
- "high": identity, values, core_experiences の変更（ユーザー承認が必要）

update_type:
- "add": 既存ファイルの末尾にセクションを追加（section フィールド不要）
- "modify": 指定セクションの内容を差し替え（section フィールドでセクション見出しを指定）
- "replace": ファイル全体を置換（新規ファイル作成を含む）

重要:
- 音声入力の誤字は文脈から修正して反映（例：「しぶや」→「渋谷」）
- 些細な情報や更新不要な場合は空配列を返す
- section フィールドは "## セクション名" の形式で指定

出力は JSON 配列で、各要素は以下の形式です:
[
  {
    "target_file": "projects/health.md",
    "update_type": "modify",
    "content": "## カフェイン断ち\\nステータス: 3日目\\n",
    "risk_level": "low",
    "reason": "ユーザーがカフェイン断ち3日目と報告",
    "section": "## カフェイン断ち"
  }
]

JSON 配列のみを出力してください。説明文は不要です。
"""


class ContextUpdateRisk(Enum):
    """Risk level for context updates."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ContextUpdate:
    """A single proposed update to a user context file."""

    target_file: str
    update_type: str  # "add", "modify", "replace"
    content: str
    risk_level: ContextUpdateRisk
    reason: str
    section: str | None = None


@dataclass
class ConsolidationResult:
    """Result of user context consolidation."""

    applied: list[ContextUpdate] = field(default_factory=list)
    pending_approval: list[ContextUpdate] = field(default_factory=list)
    change_log: list[str] = field(default_factory=list)


class UserContextConsolidator:
    """Analyzes conversations and updates user context files.

    Pipeline:
        1. LLM analyzes conversation history
        2. Parse and validate update proposals
        3. Classify by risk level
        4. Apply LOW/MEDIUM updates to files
        5. Keep HIGH updates as pending proposals
        6. Record change log
    """

    def __init__(
        self,
        llm: LLMAdapter,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self._model = model

    async def consolidate(
        self,
        conversation_history: list[dict],
        user_context_dir: Path,
    ) -> ConsolidationResult:
        """Analyze conversation and update user context files.

        Args:
            conversation_history: List of conversation messages.
            user_context_dir: Path to user context directory.

        Returns:
            ConsolidationResult with applied updates, pending approvals,
            and change log entries.
        """
        result = ConsolidationResult()

        if not conversation_history:
            return result

        # 1. LLM で会話分析
        raw_updates = await self._analyze_conversation(conversation_history)
        if not raw_updates:
            return result

        # 2. バリデーションとパース
        updates = self._parse_updates(raw_updates)
        if not updates:
            return result

        # 3. リスクレベル別に処理
        for update in updates:
            if update.risk_level == ContextUpdateRisk.HIGH:
                result.pending_approval.append(update)
                result.change_log.append(
                    f"[pending] {update.target_file}: {update.reason}"
                )
            else:
                # LOW / MEDIUM: ファイルに適用
                self._apply_update(update, user_context_dir)
                result.applied.append(update)
                result.change_log.append(
                    f"{update.target_file}: {update.reason}"
                )

        # 4. 変更ログをファイルに書き出し
        if result.change_log:
            self._write_update_log(result.change_log, user_context_dir)

        return result

    async def _analyze_conversation(
        self, conversation_history: list[dict]
    ) -> list[dict]:
        """Use LLM to analyze conversation and propose updates."""
        request = LLMRequest(
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            messages=conversation_history,
            model=self._model,
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            response = await self.llm.generate(request)
        except Exception:
            return []

        return self._parse_json_response(response.content)

    @staticmethod
    def _parse_json_response(content: str) -> list[dict]:
        """Parse JSON response, handling markdown code block wrapping."""
        text = content.strip()

        # Strip markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                return []
            return parsed
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _parse_updates(raw_updates: list[dict]) -> list[ContextUpdate]:
        """Validate and convert raw dicts to ContextUpdate objects."""
        updates: list[ContextUpdate] = []

        for raw in raw_updates:
            if not isinstance(raw, dict):
                continue

            # Check required fields
            required = {"target_file", "update_type", "content", "risk_level", "reason"}
            if not required.issubset(raw.keys()):
                continue

            # Validate update_type
            if raw["update_type"] not in _VALID_UPDATE_TYPES:
                continue

            # Validate risk_level
            try:
                risk = ContextUpdateRisk(raw["risk_level"])
            except ValueError:
                continue

            updates.append(
                ContextUpdate(
                    target_file=raw["target_file"],
                    update_type=raw["update_type"],
                    content=raw["content"],
                    risk_level=risk,
                    reason=raw["reason"],
                    section=raw.get("section"),
                )
            )

        return updates

    @staticmethod
    def _apply_update(update: ContextUpdate, base_dir: Path) -> None:
        """Apply a single update to the filesystem."""
        filepath = base_dir / update.target_file

        if update.update_type == "replace":
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(update.content, encoding="utf-8")
            return

        if update.update_type == "add":
            if filepath.exists():
                existing = filepath.read_text(encoding="utf-8")
                # Append new content with a blank line separator
                new_content = existing.rstrip() + "\n\n" + update.content
                filepath.write_text(new_content, encoding="utf-8")
            else:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(update.content, encoding="utf-8")
            return

        if update.update_type == "modify":
            if not filepath.exists():
                # If file doesn't exist, create it with the content
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(update.content, encoding="utf-8")
                return

            existing = filepath.read_text(encoding="utf-8")

            if update.section:
                new_content = _replace_section(
                    existing, update.section, update.content
                )
                filepath.write_text(new_content, encoding="utf-8")
            else:
                # No section specified: replace entire file
                filepath.write_text(update.content, encoding="utf-8")

    @staticmethod
    def _write_update_log(
        log_entries: list[str], base_dir: Path
    ) -> None:
        """Append change log entries to .update_log.md."""
        log_file = base_dir / ".update_log.md"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        entry = f"\n## {now}\n"
        for log in log_entries:
            entry += f"- {log}\n"

        if log_file.exists():
            existing = log_file.read_text(encoding="utf-8")
            log_file.write_text(existing + entry, encoding="utf-8")
        else:
            header = "# User Context Update Log\n"
            log_file.write_text(header + entry, encoding="utf-8")


def _replace_section(
    document: str, section_heading: str, new_content: str
) -> str:
    """Replace a specific section in a markdown document.

    Finds the section matching section_heading (e.g. "## カフェイン断ち")
    and replaces it with new_content. The rest of the document is preserved.

    If the section is not found, the new content is appended at the end.
    """
    sections = _SECTION_RE.split(document)

    if not sections:
        return new_content

    # Find and replace the matching section
    found = False
    new_sections: list[str] = []

    for section in sections:
        stripped = section.strip()
        if stripped.startswith(section_heading.strip()):
            new_sections.append(new_content)
            found = True
        else:
            new_sections.append(section)

    if not found:
        # Section not found: append at end
        new_sections.append(new_content + "\n")

    # Reconstruct the document
    result_parts: list[str] = []
    for i, section in enumerate(new_sections):
        stripped = section.strip()
        if not stripped:
            continue
        if i == 0 and not stripped.startswith("##"):
            # First part is likely the title (# Title)
            result_parts.append(stripped)
        else:
            result_parts.append(stripped)

    return "\n\n".join(result_parts) + "\n"
