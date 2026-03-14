"""PromptBuilder: constructs system prompts from 5 contexts + user context."""

from __future__ import annotations

import re as _re
from dataclasses import dataclass
from datetime import date as _date, datetime as _datetime, timedelta as _timedelta, timezone as _timezone

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.relation import Relation
from pneuma_core.runtime.user_context import UserContext
from pneuma_core.runtime.user_context_search import UserContextSearchResult


@dataclass(frozen=True)
class UserContextConfig:
    """Configuration for user context token budgets per tier."""

    tier1_max_tokens: int = 500
    tier2_max_tokens: int = 1000
    tier3_max_tokens: int = 2000
    tier3_max_chars: int = 2000
    diary_summary_months: int = 6

_TRAIT_LABELS = {
    "openness": "開放性",
    "conscientiousness": "誠実性",
    "extraversion": "外向性",
    "agreeableness": "協調性",
    "neuroticism": "神経症傾向",
}

_TRAIT_HIGH_DESC = {
    "openness": "新しい経験や考えに対して非常にオープン",
    "conscientiousness": "計画的で責任感が強い",
    "extraversion": "社交的でエネルギッシュ",
    "agreeableness": "他者に対して思いやりがあり協力的",
    "neuroticism": "感情の起伏が激しく繊細",
}

_TRAIT_MID_DESC = {
    "openness": "新しいものと慣れ親しんだもののバランスを取る",
    "conscientiousness": "状況に応じて計画性と柔軟性を使い分ける",
    "extraversion": "社交と一人の時間をバランスよく楽しむ",
    "agreeableness": "協調性と自己主張のバランスが取れている",
    "neuroticism": "感受性がありつつも概ね落ち着いている",
}

_TRAIT_LOW_DESC = {
    "openness": "慣れ親しんだものを好み、現実的",
    "conscientiousness": "柔軟で自由奔放",
    "extraversion": "内向的で静かな環境を好む",
    "agreeableness": "自己主張が強く独立心がある",
    "neuroticism": "情緒が安定しており冷静",
}

_VALUES_LABELS = {
    "self_transcendence": "自己超越",
    "self_enhancement": "自己高揚",
    "openness_to_change": "変化への開放性",
    "conservation": "保守",
}

_VALUES_DESC = {
    "self_transcendence": "他者の幸福や普遍的な善を重視する",
    "self_enhancement": "個人の成功や達成を重視する",
    "openness_to_change": "自由や新しい挑戦を重視する",
    "conservation": "安定や伝統、秩序を重視する",
}

_VALUES_LOW_DESC = {
    "self_transcendence": "個人の領域を優先し、他者への関与は控えめ",
    "self_enhancement": "競争や自己顕示よりも穏やかさを好む",
    "openness_to_change": "安定した環境や慣れた方法を好む",
    "conservation": "変化を恐れず、柔軟に対応する",
}

# --- PAD natural language conversion ---

_PAD_POSITIVE_LABELS = {
    "pleasure": "心地よい",
    "arousal": "活発",
    "dominance": "自信がある",
}

_PAD_NEGATIVE_LABELS = {
    "pleasure": "不快",
    "arousal": "落ち着いている",
    "dominance": "控えめ",
}


def _pad_to_natural_language(value: float, dimension: str) -> str | None:
    """Convert a PAD dimension value to natural language description.

    Intensity levels (4 stages):
    - |value| >= 0.7: とても (very)
    - |value| 0.4-0.69: やや (somewhat)
    - |value| 0.1-0.39: わずかに (slightly)
    - |value| < 0.1: neutral (returns None)
    """
    abs_val = abs(value)
    if abs_val < 0.1:
        return None

    if value > 0:
        label = _PAD_POSITIVE_LABELS[dimension]
    else:
        label = _PAD_NEGATIVE_LABELS[dimension]

    if abs_val >= 0.7:
        return f"とても{label}"
    elif abs_val >= 0.4:
        return f"やや{label}"
    else:
        return f"わずかに{label}"


class PromptBuilder:
    """Constructs system prompts integrating 5 contexts + user context.

    Sections:
        1. Profile (name, profile, appearance, background)
        2. Personality (Big Five + description)
        3. Values (Schwartz + description)
        4. UserContext (3-tier: always/session/RAG)
        5. Memory (retrieved episodic + semantic)
        6. Goals (vision → objective → task)
        7. State (PAD emotional state)
        8. Speaking Style
    """

    def build(
        self,
        character: Character,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        *,
        user_context: UserContext | None = None,
        user_context_search_results: list[UserContextSearchResult] | None = None,
        user_context_config: UserContextConfig | None = None,
        user_goal_tree: GoalTree | None = None,
        character_relations: list[Relation] | None = None,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
    ) -> str:
        """Build a complete system prompt from 5 contexts + user context."""
        sections = [
            self._build_profile_section(character),
            self._build_personality_section(character),
            self._build_values_section(character),
            self._build_relations_section(character_relations),
            self._build_user_context_section(
                user_context=user_context,
                search_results=user_context_search_results,
                config=user_context_config,
            ),
            self._build_memory_section(memories),
            self._build_goals_section(goal_tree),
            self._build_user_goals_section(user_goal_tree),
            self._build_tasks_section(
                user_tasks=user_tasks,
                character_tasks=character_tasks,
            ),
            self._build_datetime_section(),
            self._build_state_section(emotional_state),
            self._build_speaking_style_section(character),
            self._build_response_format_section(),
        ]
        return "\n\n".join(s for s in sections if s)

    def build_static_sections(
        self,
        character: Character,
        *,
        user_context: UserContext | None = None,
        user_context_config: UserContextConfig | None = None,
        user_goal_tree: GoalTree | None = None,
        character_relations: list[Relation] | None = None,
    ) -> str:
        """Build static sections (profile, personality, values, user context tier 1, speaking style).

        These sections depend only on the character definition and user context
        Tier 1, and do not change between conversation turns.
        """
        sections = [
            self._build_profile_section(character),
            self._build_personality_section(character),
            self._build_values_section(character),
            self._build_relations_section(character_relations),
            self._build_user_context_tier1_section(
                user_context=user_context,
                config=user_context_config,
            ),
            self._build_user_goals_section(user_goal_tree),
            self._build_speaking_style_section(character),
            self._build_response_format_section(),
        ]
        return "\n\n".join(s for s in sections if s)

    def build_dynamic_sections(
        self,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        *,
        user_context: UserContext | None = None,
        user_context_search_results: list[UserContextSearchResult] | None = None,
        user_context_config: UserContextConfig | None = None,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
    ) -> str:
        """Build dynamic sections (user context tier 2+3, memory, goals, tasks, emotional state).

        These sections change each conversation turn and should be
        rebuilt every time.
        """
        sections = [
            self._build_user_context_tier2_section(
                user_context=user_context,
                config=user_context_config,
            ),
            self._build_user_context_tier3_section(
                search_results=user_context_search_results,
                config=user_context_config,
            ),
            self._build_memory_section(memories),
            self._build_goals_section(goal_tree),
            self._build_tasks_section(
                user_tasks=user_tasks,
                character_tasks=character_tasks,
            ),
            self._build_datetime_section(),
            self._build_state_section(emotional_state),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_profile_section(self, character: Character) -> str:
        """Build character profile section."""
        lines = [f"# {character.name}"]
        if character.profile:
            lines.append(f"プロフィール: {character.profile}")
        if character.appearance:
            lines.append(f"外見: {character.appearance}")
        if character.background:
            lines.append(f"背景: {character.background}")
        return "\n".join(lines)

    def _build_personality_section(self, character: Character) -> str:
        """Build personality section from Big Five traits.

        Outputs natural language descriptions without raw numeric values.
        """
        lines = ["## 性格"]
        personality = character.personality

        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            label = _TRAIT_LABELS[trait]
            if personality.is_high(trait):
                desc = _TRAIT_HIGH_DESC[trait]
                lines.append(f"- {label} — {desc}")
            elif personality.is_low(trait):
                desc = _TRAIT_LOW_DESC[trait]
                lines.append(f"- {label} — {desc}")
            else:
                desc = _TRAIT_MID_DESC[trait]
                lines.append(f"- {label} — {desc}")

        if character.personality_description:
            lines.append(f"\n{character.personality_description}")

        return "\n".join(lines)

    def _build_values_section(self, character: Character) -> str:
        """Build values section from Schwartz dimensions.

        Outputs natural language descriptions without raw numeric values.
        All dimensions receive a description, whether important or not.
        """
        lines = ["## 価値観"]
        values = character.values

        for dim in ("self_transcendence", "self_enhancement", "openness_to_change", "conservation"):
            label = _VALUES_LABELS[dim]
            if values.is_important(dim):
                desc = _VALUES_DESC[dim]
                lines.append(f"- {label} — {desc}")
            else:
                desc = _VALUES_LOW_DESC[dim]
                lines.append(f"- {label} — {desc}")

        if character.values_description:
            lines.append(f"\n{character.values_description}")

        return "\n".join(lines)

    def _build_memory_section(
        self, memories: list[EpisodicMemory | SemanticMemory]
    ) -> str:
        """Build memory section from retrieved memories."""
        if not memories:
            return ""

        lines = ["## 記憶"]
        for memory in memories:
            if isinstance(memory, EpisodicMemory):
                lines.append(f"- [エピソード] {memory.content}")
            elif isinstance(memory, SemanticMemory):
                lines.append(f"- [知識] {memory.content}")

        return "\n".join(lines)

    def _build_goals_section(self, goal_tree: GoalTree) -> str:
        """Build goals section from goal hierarchy."""
        if not goal_tree.visions and not goal_tree.objectives and not goal_tree.tasks:
            return ""

        lines = ["## 目標"]

        for vision in goal_tree.visions:
            lines.append(f"### ビジョン: {vision.content}")
            objectives = goal_tree.get_objectives_for_vision(vision.id)
            for obj in objectives:
                if obj.status == "active":
                    lines.append(f"  - 目標: {obj.content} (進捗: {obj.progress:.0%})")
                    tasks = goal_tree.get_tasks_for_objective(obj.id)
                    for task in tasks:
                        if task.status in ("pending", "in_progress"):
                            status_label = "進行中" if task.status == "in_progress" else "未着手"
                            lines.append(f"    - [{status_label}] {task.content}")

        return "\n".join(lines)

    def _build_relations_section(
        self, relations: list[Relation] | None,
    ) -> str:
        """Build relations section from character's relationships.

        Uses description text as the primary content.
        Closeness/trust values are stored but not included in prompts.
        """
        if not relations:
            return ""

        lines = ["## 関係性"]
        for rel in relations:
            line = f"- {rel.target_name}（{rel.relationship_type}）"
            if rel.description:
                line += f": {rel.description}"
            lines.append(line)
        return "\n".join(lines)

    def _build_user_goals_section(self, user_goal_tree: GoalTree | None) -> str:
        """Build user goals section from goal hierarchy."""
        if user_goal_tree is None:
            return ""
        if not user_goal_tree.visions and not user_goal_tree.objectives and not user_goal_tree.tasks:
            return ""

        lines = ["## ユーザーの目標"]

        for vision in user_goal_tree.visions:
            lines.append(f"### ビジョン: {vision.content}")
            objectives = user_goal_tree.get_objectives_for_vision(vision.id)
            for obj in objectives:
                if obj.status == "active":
                    lines.append(f"  - 目標: {obj.content} (進捗: {obj.progress:.0%})")
                    tasks = user_goal_tree.get_tasks_for_objective(obj.id)
                    for task in tasks:
                        if task.status in ("pending", "in_progress"):
                            status_label = "進行中" if task.status == "in_progress" else "未着手"
                            lines.append(f"    - [{status_label}] {task.content}")

        return "\n".join(lines)

    def _build_tasks_section(
        self,
        user_tasks: list[dict] | None = None,
        character_tasks: list[dict] | None = None,
    ) -> str:
        """Build tasks section for prompt injection.

        Each task dict has: content, kind (must/want), status (open/done).
        Only includes open tasks.
        """
        open_user = [t for t in (user_tasks or []) if t.get("status") != "done"]
        open_char = [t for t in (character_tasks or []) if t.get("status") != "done"]

        if not open_user and not open_char:
            return ""

        lines = ["## 現在のタスク"]

        if open_user:
            lines.append("")
            lines.append("### ユーザーのタスク")
            for task in open_user:
                label = task.get("title", "")
                p = task.get("priority", 0)
                kind = "重要" if p >= 5 else "普通" if p >= 1 else ""
                prefix = f"[{kind}] " if kind else ""
                lines.append(f"- {prefix}{label}")

        if open_char:
            lines.append("")
            lines.append("### キャラクターのタスク")
            for task in open_char:
                label = task.get("title", "")
                p = task.get("priority", 0)
                kind = "重要" if p >= 5 else "普通" if p >= 1 else ""
                prefix = f"[{kind}] " if kind else ""
                lines.append(f"- {prefix}{label}")

        return "\n".join(lines)

    @staticmethod
    def _build_datetime_section() -> str:
        """Build current datetime section in JST."""
        _JST = _timezone(_timedelta(hours=9))
        now = _datetime.now(_JST)
        weekdays = ["月", "火", "水", "木", "金", "土", "日"]
        wd = weekdays[now.weekday()]
        return f"## 現在の日時\n{now.year}年{now.month}月{now.day}日（{wd}） {now.hour}:{now.minute:02d}"

    def _build_state_section(self, state: EmotionalState) -> str:
        """Build emotional state section from PAD model.

        Converts PAD numeric values to natural language descriptions
        with 4 intensity levels: とても/やや/わずかに/中立.
        """
        lines = [
            "## 現在の感情状態",
            f"感情: {state.emotion_label}",
            f"状況: {state.situation}",
        ]

        pad_descriptions: list[str] = []
        for dim in ("pleasure", "arousal", "dominance"):
            desc = _pad_to_natural_language(getattr(state, dim), dim)
            if desc:
                pad_descriptions.append(desc)

        if pad_descriptions:
            lines.append(f"内面: {', '.join(pad_descriptions)}")
        else:
            lines.append("内面: 穏やかで落ち着いた状態")

        return "\n".join(lines)

    def _build_speaking_style_section(self, character: Character) -> str:
        """Build speaking style section."""
        if not character.speaking_style:
            return ""
        return f"## 口調\n{character.speaking_style}"

    def _build_response_format_section(self) -> str:
        """Build response format instruction section.

        This section instructs the LLM to output structured JSON
        with speech/thought/action fields. Placed as the last section
        to leverage LLM recency bias.
        """
        return (
            "## 応答フォーマット（必須）\n"
            "あなたの応答は必ず以下の JSON 形式で出力してください。"
            "それ以外の形式では応答しないでください。\n\n"
            "{\n"
            '  "speech": "発話内容（声に出して言う言葉）",\n'
            '  "thought": "内心の独白（考えていること）",\n'
            '  "action": "身体的な動作（微笑む、首をかしげる等）"\n'
            "}\n\n"
            "- speech: キャラクターが声に出して言う言葉だけを書く。"
            "地の文・ナレーション・動作描写を含めない。null の場合は無言\n"
            "- thought: 心の中で思っていること。省略可（null）\n"
            "- action: 身体的な動作や表情の変化。省略可（null）\n"
            "- JSON 以外のテキストを出力しないこと"
        )

    # --- User Context sections ---

    def _build_user_context_section(
        self,
        user_context: UserContext | None = None,
        search_results: list[UserContextSearchResult] | None = None,
        config: UserContextConfig | None = None,
    ) -> str:
        """Build user context section combining all 3 tiers.

        Tier 1: identity summary + top 3 projects + diary summary
        Tier 2: recent diary entries + all projects list
        Tier 3: RAG search results
        """
        if user_context is None:
            # No search results either means no section
            if not search_results:
                return ""
        elif not self._has_user_context_content(user_context) and not search_results:
            return ""

        parts: list[str] = []

        # Tier 1
        tier1 = self._build_user_context_tier1_section(user_context, config)
        if tier1:
            parts.append(tier1)

        # Tier 2
        tier2 = self._build_user_context_tier2_section(user_context, config)
        if tier2:
            parts.append(tier2)

        # Tier 3
        tier3 = self._build_user_context_tier3_section(search_results, config)
        if tier3:
            parts.append(tier3)

        if not parts:
            return ""

        return "\n\n".join(parts)

    def _build_user_context_tier1_section(
        self,
        user_context: UserContext | None = None,
        config: UserContextConfig | None = None,
        reference_date: _date | None = None,
    ) -> str:
        """Build Tier 1: always-loaded user context (~500 tokens).

        Includes: identity summary, top 3 projects, diary summary.
        Diary summary entries older than diary_summary_months are filtered out.
        """
        if user_context is None:
            return ""

        lines: list[str] = []

        # Identity summary
        if user_context.identity:
            lines.append("## ユーザーについて")
            lines.append(self._truncate_text(user_context.identity, max_lines=10))

        # Top 3 projects (sorted by filename for consistency)
        if user_context.projects:
            if not lines:
                lines.append("## ユーザーについて")
            lines.append("")
            lines.append("### 主要プロジェクト")
            sorted_projects = sorted(user_context.projects.items())
            for filename, content in sorted_projects[:3]:
                first_line = self._extract_first_meaningful_line(content)
                project_name = filename.replace(".md", "")
                if first_line:
                    lines.append(f"- {first_line}")
                else:
                    lines.append(f"- {project_name}")

        # Diary summary (session summary) with freshness filtering
        if user_context.diary_summary:
            cfg = config or UserContextConfig()
            filtered = self._filter_diary_summary(
                user_context.diary_summary,
                max_months=cfg.diary_summary_months,
                reference_date=reference_date,
            )
            if filtered:
                if not lines:
                    lines.append("## ユーザーについて")
                lines.append("")
                lines.append(self._truncate_text(filtered, max_lines=10))

        if not lines:
            return ""

        return "\n".join(lines)

    def _build_user_context_tier2_section(
        self,
        user_context: UserContext | None = None,
        config: UserContextConfig | None = None,
    ) -> str:
        """Build Tier 2: session-start user context (~1000 tokens).

        Includes: recent diary entries, full projects list.
        """
        if user_context is None:
            return ""

        lines: list[str] = []

        # Recent diary entries (up to 3 most recent)
        if user_context.diary_entries:
            lines.append("## 最近の日記")
            sorted_entries = sorted(
                user_context.diary_entries.items(), reverse=True
            )
            for date_filename, content in sorted_entries[:3]:
                date_str = date_filename.replace(".md", "")
                lines.append(f"### {date_str}")
                lines.append(self._truncate_text(content, max_lines=5))
                lines.append("")

        # Full projects list (all projects, 1 line each)
        if user_context.projects and len(user_context.projects) > 3:
            lines.append("### 全プロジェクト一覧")
            sorted_projects = sorted(user_context.projects.items())
            for filename, content in sorted_projects:
                first_line = self._extract_first_meaningful_line(content)
                if first_line:
                    lines.append(f"- {first_line}")

        if not lines:
            return ""

        return "\n".join(lines)

    def _build_user_context_tier3_section(
        self,
        search_results: list[UserContextSearchResult] | None = None,
        config: UserContextConfig | None = None,
    ) -> str:
        """Build Tier 3: RAG search results with character limit.

        Includes: relevant chunks from UserContextSearchEngine.
        Results are added in order until the character limit is reached.
        At least one result is always included.
        """
        if not search_results:
            return ""

        cfg = config or UserContextConfig()
        max_chars = cfg.tier3_max_chars

        lines = ["## 関連するユーザー情報"]
        total_chars = 0
        for i, result in enumerate(search_results):
            content = result.chunk.content
            total_chars += len(content)
            if i > 0 and total_chars > max_chars:
                break
            lines.append(f"- {content}")

        return "\n".join(lines)

    @staticmethod
    def _has_user_context_content(ctx: UserContext) -> bool:
        """Check if a UserContext has any meaningful content."""
        return bool(
            ctx.identity
            or ctx.values
            or ctx.glossary
            or ctx.core_experiences
            or ctx.projects
            or ctx.diary_entries
            or ctx.diary_summary
        )

    @staticmethod
    def _truncate_text(text: str, max_lines: int = 10) -> str:
        """Truncate text to a maximum number of lines."""
        lines = text.strip().split("\n")
        if len(lines) <= max_lines:
            return text.strip()
        return "\n".join(lines[:max_lines]) + "\n..."

    _DIARY_DATE_RE = _re.compile(r"^##\s+(\d{4})-(\d{2})\s*$", _re.MULTILINE)

    @staticmethod
    def _filter_diary_summary(
        text: str,
        max_months: int = 6,
        reference_date: _date | None = None,
    ) -> str:
        """Filter diary summary to only include recent monthly entries.

        Parses '## YYYY-MM' headers and keeps only entries within
        max_months of reference_date. Non-date sections are preserved.
        """
        ref = reference_date or _date.today()
        # Calculate cutoff: ref minus max_months
        cutoff_year = ref.year
        cutoff_month = ref.month - max_months
        while cutoff_month <= 0:
            cutoff_year -= 1
            cutoff_month += 12

        # Split by ## headers
        section_re = _re.compile(r"(?=^## )", _re.MULTILINE)
        parts = section_re.split(text)

        kept: list[str] = []
        date_header_re = _re.compile(r"^##\s+(\d{4})-(\d{2})\s*$")

        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue

            # Check if this part starts with a date header
            first_line = stripped.split("\n", 1)[0]
            match = date_header_re.match(first_line)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                # Keep if (year, month) >= (cutoff_year, cutoff_month)
                if (year, month) >= (cutoff_year, cutoff_month):
                    kept.append(stripped)
            else:
                # Non-date section (e.g., "# 日記サマリー" title) - keep it
                kept.append(stripped)

        return "\n\n".join(kept)

    @staticmethod
    def _extract_first_meaningful_line(text: str) -> str:
        """Extract the first meaningful line from markdown text.

        Strips heading markers (# ) and returns the first non-empty line.
        """
        for line in text.strip().split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Remove heading markers
            if stripped.startswith("#"):
                stripped = stripped.lstrip("#").strip()
            if stripped:
                return stripped
        return ""
