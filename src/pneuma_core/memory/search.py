"""Memory search engine with hybrid scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from pneuma_core.memory.similarity import cosine_similarity
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality


@dataclass(frozen=True)
class SearchConfig:
    """Tunable parameters for memory search."""

    alpha_base: float = 0.5
    beta_base: float = 0.3
    beta_neuroticism_factor: float = 0.2
    gamma_base: float = 0.2
    openness_diversity_factor: float = 0.5
    neuroticism_negativity_factor: float = 0.15
    lambda_decay: float = 0.01
    top_k: int = 10


class MemorySearchEngine:
    """Hybrid scoring memory search engine.

    score(memory, query) = α × similarity + β × importance + γ × recency
                           + negativity_bias (episodic only)

    where:
        α = alpha_base (constant)
        β = beta_base + beta_neuroticism_factor × neuroticism
        γ = gamma_base (constant, no personality influence)
        recency = exp(-λ × days_elapsed)  (episodic)
        recency = 1.0  (semantic, no decay)
        negativity_bias = neuroticism_negativity_factor × neuroticism × |valence|
                          (only for episodic memories with negative emotional_valence)

    Openness affects search diversity (effective top_k), not scoring weights.
    """

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()

    def compute_weights(
        self, personality: Personality
    ) -> tuple[float, float, float]:
        """Compute scoring weights based on personality."""
        alpha = self.config.alpha_base
        beta = (
            self.config.beta_base
            + self.config.beta_neuroticism_factor * personality.neuroticism
        )
        gamma = self.config.gamma_base
        return alpha, beta, gamma

    def compute_effective_top_k(self, personality: Personality) -> int:
        """Compute effective top_k based on openness.

        Higher openness = wider search (more diverse memories).
        effective_top_k = base_top_k + floor(openness_diversity_factor × openness × base_top_k)
        """
        base = self.config.top_k
        extra = math.floor(
            self.config.openness_diversity_factor * personality.openness * base
        )
        return base + extra

    def compute_recency(
        self, timestamp: datetime, now: datetime
    ) -> float:
        """Compute recency decay: exp(-λ × days_elapsed)."""
        delta = now - timestamp
        days_elapsed = max(delta.total_seconds() / 86400.0, 0.0)
        return math.exp(-self.config.lambda_decay * days_elapsed)

    def score_memory(
        self,
        memory: EpisodicMemory | SemanticMemory,
        query_embedding: list[float],
        personality: Personality,
        now: datetime,
    ) -> float:
        """Score a single memory against a query."""
        if memory.embedding is None:
            return 0.0

        alpha, beta, gamma = self.compute_weights(personality)

        similarity = cosine_similarity(memory.embedding, query_embedding)

        if isinstance(memory, EpisodicMemory):
            importance = memory.importance
            recency = self.compute_recency(memory.timestamp, now)
        else:
            # SemanticMemory: use confidence as importance, recency=1.0
            importance = memory.confidence
            recency = 1.0

        score = alpha * similarity + beta * importance + gamma * recency

        # Negativity bias: high neuroticism boosts negative emotional memories
        if (
            isinstance(memory, EpisodicMemory)
            and memory.emotional_valence < 0
        ):
            score += (
                self.config.neuroticism_negativity_factor
                * personality.neuroticism
                * abs(memory.emotional_valence)
            )

        return score

    def search(
        self,
        memories: list[EpisodicMemory | SemanticMemory],
        query_embedding: list[float],
        personality: Personality,
        now: datetime,
    ) -> list[tuple[EpisodicMemory | SemanticMemory, float]]:
        """Search memories and return top-k results sorted by score."""
        effective_top_k = self.compute_effective_top_k(personality)

        scored = []
        for memory in memories:
            score = self.score_memory(memory, query_embedding, personality, now)
            if score > 0.0:
                scored.append((memory, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:effective_top_k]
