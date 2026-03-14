"""DiaryWriter: キャラクターの日記を自動生成する."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest

logger = logging.getLogger(__name__)

# JST (Asia/Tokyo, UTC+9)
JST = timezone(timedelta(hours=9))


class DiaryWriter:
    """キャラクターの日記を自動生成する."""

    TRIGGER_KEYWORDS = ["おやすみ", "おはよう", "おやすみなさい", "おはようございます"]

    def __init__(
        self,
        llm: LLMAdapter,
        logs_dir: Path,
        diary_dir: Path,
        character_name: str,
        character_profile: str,
        model: str = "claude-opus-4-6",
    ) -> None:
        self._llm = llm
        self._logs_dir = logs_dir
        self._diary_dir = diary_dir
        self._character_name = character_name
        self._character_profile = character_profile
        self._model = model
        self._generated_dates: set[str] = set()  # 1日1回制限

    def _get_effective_date(self) -> str:
        """実効日付を返す。0:00-3:59 JST は前日として扱う."""
        now = datetime.now(JST)
        if now.hour < 4:
            now = now - timedelta(days=1)
        return now.strftime("%Y-%m-%d")

    def should_trigger(self, user_input: str) -> bool:
        """ユーザー発言にトリガーキーワードが含まれるか判定."""
        return any(kw in user_input for kw in self.TRIGGER_KEYWORDS)

    async def maybe_generate(self, user_input: str) -> None:
        """トリガー条件を満たせば日記を生成する."""
        if not self.should_trigger(user_input):
            return
        today = self._get_effective_date()
        if today in self._generated_dates:
            return  # 1日1回制限
        await self._generate_diary(today)
        self._generated_dates.add(today)

    async def _generate_diary(self, date_str: str) -> None:
        """日記を生成して diary_dir/YYYY-MM-DD.md に保存."""
        # 1. その日の会話ログを読む
        log_path = self._logs_dir / f"{date_str}.md"
        log_content = ""
        if log_path.exists():
            log_content = log_path.read_text(encoding="utf-8")

        # 2. LLM に日記を書かせる
        system_prompt = (
            f"あなたは「{self._character_name}」です。"
            f"以下のキャラクター設定に基づいて、今日の日記を書いてください。\n\n"
            f"{self._character_profile}\n\n"
            f"## 日記の書き方\n"
            f"- 一人称で書く（キャラクターとして）\n"
            f"- その日あったこと、感じたこと、考えていることを自由に書く\n"
            f"- キャラクターの口調・性格を反映する\n"
            f"- 自然な日記として読めるように（箇条書きではなく文章で）\n"
            f"- 長さは200〜400字程度\n"
            f"- マークダウンのヘッダーや装飾は不要。本文のみ"
        )

        user_message = (
            f"今日は{date_str}です。今日の会話ログを元に日記を書いてください。\n\n"
            f"## 今日の会話ログ\n"
            f"{log_content if log_content else '（今日は会話がありませんでした）'}"
        )

        response = await self._llm.generate(
            LLMRequest(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                model=self._model,
            )
        )

        # 3. ファイルに保存
        self._diary_dir.mkdir(parents=True, exist_ok=True)
        diary_path = self._diary_dir / f"{date_str}.md"
        diary_path.write_text(response.content, encoding="utf-8")
