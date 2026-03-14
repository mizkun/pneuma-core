"""UserContextWriter: UserContext ファイル自動更新 (#136).

セッション終了時に LLM が出力した user_context_updates を
実際のファイルに適用し、git commit する。
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# 許可されたファイル名 / プレフィックス
_ALLOWED_FILES = {
    "identity.md",
    "values.md",
    "core_experiences.md",
    "glossary.md",
    "relationships.md",
}
_ALLOWED_PREFIXES = ("projects/",)


class UserContextWriter:
    """UserContext ファイルへの更新適用.

    LLM が出力した user_context_updates (list[dict]) を受け取り、
    対象ファイルの rewrite / append を実行し、git commit する。
    """

    def __init__(self, user_context_dir: Path | str) -> None:
        self._dir = Path(user_context_dir)

    async def apply(self, updates: list[dict]) -> int:
        """更新を適用し、成功した件数を返す.

        Args:
            updates: LLM が出力した user_context_updates リスト。

        Returns:
            適用成功した件数。
        """
        if not updates:
            return 0

        applied = 0
        reasons: list[str] = []

        for update in updates:
            file_name = update.get("file", "")
            action = update.get("action", "")
            new_content = update.get("new_content", "")
            reason = update.get("reason", "")

            # ファイルパスのバリデーション
            if not self._is_allowed_file(file_name):
                logger.warning("Skipping disallowed file: %s", file_name)
                continue

            file_path = self._dir / file_name

            # パストラバーサル検出
            try:
                resolved = file_path.resolve()
                dir_resolved = self._dir.resolve()
                if not str(resolved).startswith(str(dir_resolved)):
                    logger.warning("Path traversal detected: %s", file_name)
                    continue
            except (OSError, ValueError):
                logger.warning("Invalid path: %s", file_name)
                continue

            if action == "rewrite":
                section_header = update.get("section", "")
                success = self._rewrite_section(file_path, section_header, new_content)
            elif action == "append":
                success = self._append(file_path, new_content)
            else:
                logger.warning("Unknown action: %s", action)
                continue

            if success:
                applied += 1
                if reason:
                    reasons.append(reason)

        # Git commit if any updates were applied
        if applied > 0:
            commit_msg = "session-end: " + "; ".join(reasons) if reasons else "session-end: update"
            self._git_commit(commit_msg)

        return applied

    def _is_allowed_file(self, file_name: str) -> bool:
        """ファイル名が許可リストに含まれるか."""
        if file_name in _ALLOWED_FILES:
            return True
        for prefix in _ALLOWED_PREFIXES:
            if file_name.startswith(prefix) and file_name.endswith(".md"):
                return True
        return False

    def _rewrite_section(
        self, file_path: Path, section_header: str, new_content: str,
    ) -> bool:
        """マークダウンのセクションを書き換える.

        セクションが見つからない場合は append にフォールバックする。

        Algorithm:
        1. ファイルを読み込む
        2. section_header と完全一致する行を探す
        3. セクションレベル（# の数）を取得
        4. 次の同レベル以上のヘッダーまたは EOF までがセクション範囲
        5. ヘッダー + new_content で置換
        6. ファイルに書き戻す
        """
        if not file_path.exists():
            logger.warning("File not found for rewrite: %s", file_path)
            return False

        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # セクションヘッダーの行を探す
        header_line_idx = None
        for i, line in enumerate(lines):
            if line.strip() == section_header.strip():
                header_line_idx = i
                break

        if header_line_idx is None:
            # セクションが見つからない → append にフォールバック
            logger.info(
                "Section '%s' not found in %s, falling back to append",
                section_header, file_path.name,
            )
            return self._append_section(file_path, section_header, new_content)

        # セクションレベルを取得
        level = self._header_level(section_header)

        # セクション終了位置を探す
        end_line_idx = len(lines)
        for i in range(header_line_idx + 1, len(lines)):
            line_stripped = lines[i].strip()
            if line_stripped.startswith("#"):
                line_level = self._header_level(line_stripped)
                if line_level <= level:
                    end_line_idx = i
                    break

        # 新しいセクション内容を構築
        new_section_lines = [section_header, "", new_content, ""]

        # 前後を結合
        before = lines[:header_line_idx]
        after = lines[end_line_idx:]

        new_lines = before + new_section_lines + after

        file_path.write_text("\n".join(new_lines), encoding="utf-8")
        return True

    def _append(self, file_path: Path, content: str) -> bool:
        """ファイル末尾に追記する. ファイルが存在しなければ新規作成."""
        try:
            # 親ディレクトリが存在しなければ作成
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.exists():
                existing = file_path.read_text(encoding="utf-8")
                # 末尾に改行がなければ追加
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                new_text = existing + "\n" + content + "\n"
            else:
                new_text = content + "\n"

            file_path.write_text(new_text, encoding="utf-8")
            return True
        except OSError:
            logger.warning("Failed to append to %s", file_path, exc_info=True)
            return False

    def _append_section(
        self, file_path: Path, section_header: str, content: str,
    ) -> bool:
        """セクションヘッダー付きで末尾に追記する."""
        full_content = f"{section_header}\n\n{content}"
        return self._append(file_path, full_content)

    def _git_commit(self, message: str) -> None:
        """git add + commit を実行する."""
        try:
            subprocess.run(
                ["git", "add", "."],
                cwd=str(self._dir),
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(self._dir),
                check=True,
                capture_output=True,
            )
            logger.info("Git commit: %s", message)
        except (subprocess.SubprocessError, OSError):
            logger.warning("Git commit failed", exc_info=True)

    @staticmethod
    def _header_level(header: str) -> int:
        """マークダウンヘッダーのレベル（# の数）を返す."""
        stripped = header.strip()
        level = 0
        for ch in stripped:
            if ch == "#":
                level += 1
            else:
                break
        return level
