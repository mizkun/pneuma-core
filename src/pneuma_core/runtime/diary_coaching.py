"""DiaryCoaching: ユーザー日記を読んでコーチング的な応答を補助する."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


# JST (Asia/Tokyo, UTC+9)
JST = timezone(timedelta(hours=9))


class DiaryCoaching:
    """ユーザー日記を読んでコーチング的な応答を補助する."""

    TRIGGER_KEYWORDS = ["日記書いた", "日記読んで", "日記見て", "日記書いたよ"]

    def __init__(self, diary_dir: Path) -> None:
        self._diary_dir = diary_dir

    def should_trigger(self, user_input: str) -> bool:
        """ユーザー発言にトリガーキーワードが含まれるか判定."""
        return any(kw in user_input for kw in self.TRIGGER_KEYWORDS)

    def get_diary_content(self, date_str: str | None = None) -> str | None:
        """日記の内容を取得する.

        date_str が None なら今日の日記を探し、なければ最新の日記を返す。
        """
        if date_str:
            path = self._diary_dir / f"{date_str}.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
            return None

        # 今日の日記を探す
        today = datetime.now(JST).strftime("%Y-%m-%d")
        today_path = self._diary_dir / f"{today}.md"
        if today_path.exists():
            return today_path.read_text(encoding="utf-8")

        # なければ最新の日記を探す
        if self._diary_dir.exists():
            diary_files = sorted(self._diary_dir.glob("*.md"), reverse=True)
            if diary_files:
                return diary_files[0].read_text(encoding="utf-8")

        return None

    def build_coaching_context(self, user_input: str) -> str | None:
        """トリガーされた場合、日記内容をコーチング用コンテキストとして返す.

        RuntimeEngine が LLM に渡すメッセージに追加する。
        """
        if not self.should_trigger(user_input):
            return None

        content = self.get_diary_content()
        if content is None:
            return None

        return (
            f"\n\n[ユーザーの日記]\n{content}\n\n"
            f"[コーチングガイド]\n"
            f"この日記を読んで、以下の方針で応答してください：\n"
            f"- アドバイスではなく「問いかけ」「視点の提示」をする\n"
            f"- 日記の内容を踏まえて、感じたことや気づいたことを自然に伝える\n"
            f"- 押し付けがましくならない、軽いトーンで\n"
            f"- キャラクターの口調を維持する"
        )
