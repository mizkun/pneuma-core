"""ContextAssembler: 3-stage context injection (Issue #115).

Classifies context into 3 stages:
  Stage 1 (Always): Identity layer - profile, personality, values,
                     speaking style, response format, emotion state, user context tier1
  Stage 2 (Relevant): State layer - relations (filtered), goals, memories,
                       user context search results
  Stage 3 (On-Demand): Artifacts layer - future: tool use (OUT OF SCOPE)

This module provides a clean abstraction for context assembly without
modifying existing PromptBuilder or engine code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pneuma_core.models.character import Character
from pneuma_core.models.emotion import EmotionalState
from pneuma_core.models.goals import GoalTree
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.relation import Relation
from pneuma_core.runtime.prompt_builder import PromptBuilder
from pneuma_core.runtime.user_context import UserContext
from pneuma_core.runtime.user_context_search import UserContextSearchResult


class ContextStage(Enum):
    """3-stage context injection levels."""

    ALWAYS = 1  # Identity layer - always injected
    RELEVANT = 2  # State layer - dynamically selected by relevance
    ON_DEMAND = 3  # Artifacts layer - future: tool use


@dataclass
class AssembledContext:
    """Result of context assembly with stage annotations.

    always: Stage 1 sections (identity, always present)
    relevant: Stage 2 sections (dynamically selected by relevance)
    Stage 3 (on_demand) is out of scope for this issue.
    """

    always: dict[str, str] = field(default_factory=dict)
    relevant: dict[str, str] = field(default_factory=dict)


class ContextAssembler:
    """Assembles context from multiple sources with stage classification.

    Uses PromptBuilder's section-building methods internally to generate
    text for each section, then classifies them into stages.
    """

    def __init__(self) -> None:
        self._builder = PromptBuilder()

    def assemble(
        self,
        character: Character,
        emotional_state: EmotionalState,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        *,
        relations: list[Relation] | None = None,
        conversation_partner_id: str | None = None,
        user_context: UserContext | None = None,
        user_context_search_results: list[UserContextSearchResult] | None = None,
    ) -> AssembledContext:
        """Assemble context into stage-classified sections.

        Args:
            character: The character identity.
            emotional_state: Current PAD emotional state.
            goal_tree: Character's goal hierarchy.
            memories: Retrieved memories (episodic + semantic).
            relations: Character's relationships (optional).
            conversation_partner_id: ID of the conversation partner for
                relation filtering. When provided, only the partner's
                relation is included. When None, all relations are included.
            user_context: User context data (optional).
            user_context_search_results: RAG search results (optional).

        Returns:
            AssembledContext with always and relevant dicts.
        """
        always: dict[str, str] = {}
        relevant: dict[str, str] = {}

        # --- Stage 1: Always (Identity) ---
        self._add_always_sections(always, character, emotional_state, user_context)

        # --- Stage 2: Relevant (State) ---
        self._add_relevant_sections(
            relevant,
            goal_tree=goal_tree,
            memories=memories,
            relations=relations,
            conversation_partner_id=conversation_partner_id,
            user_context_search_results=user_context_search_results,
        )

        return AssembledContext(always=always, relevant=relevant)

    @staticmethod
    def _set_if_nonempty(target: dict[str, str], key: str, value: str) -> None:
        """Add a section to the target dict only if the value is non-empty."""
        if value:
            target[key] = value

    def _add_always_sections(
        self,
        always: dict[str, str],
        character: Character,
        emotional_state: EmotionalState,
        user_context: UserContext | None,
    ) -> None:
        """Add Stage 1 (Always) sections."""
        b = self._builder
        s = self._set_if_nonempty

        s(always, "profile", b._build_profile_section(character))
        s(always, "personality", b._build_personality_section(character))
        s(always, "values", b._build_values_section(character))
        s(always, "speaking_style", b._build_speaking_style_section(character))
        s(always, "response_format", b._build_response_format_section())
        s(always, "emotion", b._build_state_section(emotional_state))

        # User context tier 1 (always loaded)
        if user_context is not None:
            s(
                always,
                "user_context",
                b._build_user_context_tier1_section(user_context=user_context),
            )

    def _add_relevant_sections(
        self,
        relevant: dict[str, str],
        *,
        goal_tree: GoalTree,
        memories: list[EpisodicMemory | SemanticMemory],
        relations: list[Relation] | None,
        conversation_partner_id: str | None,
        user_context_search_results: list[UserContextSearchResult] | None,
    ) -> None:
        """Add Stage 2 (Relevant) sections."""
        b = self._builder
        s = self._set_if_nonempty

        # Relations (filtered by conversation partner)
        filtered_relations = self._filter_relations(
            relations, conversation_partner_id
        )
        if filtered_relations:
            s(relevant, "relations", b._build_relations_section(filtered_relations))

        s(relevant, "goals", b._build_goals_section(goal_tree))
        s(relevant, "memory", b._build_memory_section(memories))

        # User context search results (RAG)
        if user_context_search_results:
            s(
                relevant,
                "user_context_search",
                b._build_user_context_tier3_section(
                    search_results=user_context_search_results,
                ),
            )

    @staticmethod
    def _filter_relations(
        relations: list[Relation] | None,
        conversation_partner_id: str | None,
    ) -> list[Relation] | None:
        """Filter relations to only include the conversation partner.

        When conversation_partner_id is provided, returns only the matching
        relation. When None, returns all relations (backward compatibility).
        """
        if relations is None:
            return None

        if not relations:
            return None

        if conversation_partner_id is None:
            return relations

        filtered = [r for r in relations if r.target_id == conversation_partner_id]
        return filtered if filtered else None
