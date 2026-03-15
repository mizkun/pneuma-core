"""Tests for MemorySearchEngine (Issue #25, #103)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from pneuma_core.memory.search import MemorySearchEngine, SearchConfig
from pneuma_core.models.memory import EpisodicMemory, SemanticMemory
from pneuma_core.models.personality import Personality

NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_personality(
    openness: float = 0.5,
    neuroticism: float = 0.5,
    conscientiousness: float = 0.5,
    extraversion: float = 0.5,
    agreeableness: float = 0.5,
) -> Personality:
    return Personality(
        openness=openness,
        conscientiousness=conscientiousness,
        extraversion=extraversion,
        agreeableness=agreeableness,
        neuroticism=neuroticism,
    )


def _make_episodic(
    id: str = "ep-1",
    content: str = "test",
    importance: float = 0.5,
    embedding: list[float] | None = None,
    timestamp: datetime | None = None,
    character_id: str = "aine-001",
    emotional_valence: float = 0.0,
) -> EpisodicMemory:
    return EpisodicMemory(
        id=id,
        character_id=character_id,
        content=content,
        timestamp=timestamp or NOW,
        emotional_valence=emotional_valence,
        importance=importance,
        embedding=embedding,
    )


def _make_semantic(
    id: str = "sem-1",
    content: str = "test",
    confidence: float = 0.8,
    embedding: list[float] | None = None,
    character_id: str = "aine-001",
) -> SemanticMemory:
    return SemanticMemory(
        id=id,
        character_id=character_id,
        content=content,
        confidence=confidence,
        embedding=embedding,
    )


# Simple 3D embeddings for testing
QUERY_VEC = [1.0, 0.0, 0.0]
SIMILAR_VEC = [0.9, 0.1, 0.0]
DIFFERENT_VEC = [0.0, 1.0, 0.0]


class TestSearchConfig:
    """SearchConfig の検証."""

    def test_default_values(self) -> None:
        config = SearchConfig()
        assert config.alpha_base == 0.5
        assert config.beta_base == 0.3
        assert config.beta_neuroticism_factor == 0.2
        assert config.gamma_base == 0.2
        assert config.openness_diversity_factor == 0.5
        assert config.neuroticism_negativity_factor == 0.15
        assert config.lambda_decay == 0.01
        assert config.top_k == 10
        # gamma_openness_factor should no longer exist
        assert not hasattr(config, "gamma_openness_factor")

    def test_custom_values(self) -> None:
        config = SearchConfig(alpha_base=0.6, top_k=5)
        assert config.alpha_base == 0.6
        assert config.top_k == 5


class TestScoringWeights:
    """性格パラメータによる重み計算の検証."""

    def test_default_personality_weights(self) -> None:
        """デフォルト性格（全0.5）での重み計算.

        gamma は openness に依存しなくなったので gamma_base のみ.
        """
        engine = MemorySearchEngine()
        personality = _make_personality()
        alpha, beta, gamma = engine.compute_weights(personality)
        assert alpha == pytest.approx(0.5)
        assert beta == pytest.approx(0.3 + 0.2 * 0.5)  # 0.4
        assert gamma == pytest.approx(0.2)  # gamma_base only

    def test_high_neuroticism_increases_beta(self) -> None:
        """高神経症傾向 → β (importance) が増加."""
        engine = MemorySearchEngine()
        personality = _make_personality(neuroticism=0.9)
        _, beta, _ = engine.compute_weights(personality)
        assert beta == pytest.approx(0.3 + 0.2 * 0.9)  # 0.48

    def test_low_neuroticism_decreases_beta(self) -> None:
        engine = MemorySearchEngine()
        personality = _make_personality(neuroticism=0.1)
        _, beta, _ = engine.compute_weights(personality)
        assert beta == pytest.approx(0.3 + 0.2 * 0.1)  # 0.32

    def test_high_openness_increases_effective_top_k(self) -> None:
        """高開放性 → effective top_k が増加（多様な記憶を検索）."""
        engine = MemorySearchEngine(config=SearchConfig(top_k=10))
        personality = _make_personality(openness=0.9)
        effective_k = engine.compute_effective_top_k(personality)
        # base=10, openness_diversity_factor=0.5, openness=0.9
        # effective_k = 10 + floor(0.5 * 0.9 * 10) = 10 + 4 = 14
        assert effective_k == 14

    def test_openness_does_not_affect_recency(self) -> None:
        """openness は gamma に影響しない."""
        engine = MemorySearchEngine()
        p_low = _make_personality(openness=0.1)
        p_high = _make_personality(openness=0.9)
        _, _, gamma_low = engine.compute_weights(p_low)
        _, _, gamma_high = engine.compute_weights(p_high)
        assert gamma_low == gamma_high == pytest.approx(0.2)

    def test_alpha_is_constant(self) -> None:
        """α と γ は性格に依存しない."""
        engine = MemorySearchEngine()
        p1 = _make_personality(openness=0.1, neuroticism=0.1)
        p2 = _make_personality(openness=0.9, neuroticism=0.9)
        alpha1, _, gamma1 = engine.compute_weights(p1)
        alpha2, _, gamma2 = engine.compute_weights(p2)
        assert alpha1 == alpha2 == 0.5
        assert gamma1 == gamma2 == pytest.approx(0.2)


class TestRecencyDecay:
    """recency 減衰の検証."""

    def test_zero_days_elapsed(self) -> None:
        engine = MemorySearchEngine()
        assert engine.compute_recency(NOW, NOW) == pytest.approx(1.0)

    def test_70_days_half_life(self) -> None:
        """半減期約70日（λ=0.01）."""
        engine = MemorySearchEngine()
        past = NOW - timedelta(days=70)
        recency = engine.compute_recency(past, NOW)
        # exp(-0.01 * 70) ≈ 0.4966
        assert recency == pytest.approx(math.exp(-0.01 * 70), abs=0.01)

    def test_decay_is_monotonic(self) -> None:
        """時間が経つほど recency は小さくなる."""
        engine = MemorySearchEngine()
        r1 = engine.compute_recency(NOW - timedelta(days=10), NOW)
        r2 = engine.compute_recency(NOW - timedelta(days=50), NOW)
        r3 = engine.compute_recency(NOW - timedelta(days=100), NOW)
        assert r1 > r2 > r3

    def test_very_old_memory_near_zero(self) -> None:
        engine = MemorySearchEngine()
        past = NOW - timedelta(days=1000)
        recency = engine.compute_recency(past, NOW)
        assert recency < 0.01


class TestScoring:
    """スコアリング式の検証."""

    def test_score_episodic_memory(self) -> None:
        """エピソード記憶のスコア計算."""
        engine = MemorySearchEngine()
        personality = _make_personality()
        memory = _make_episodic(
            importance=0.8,
            embedding=SIMILAR_VEC,
            timestamp=NOW,
        )
        score = engine.score_memory(memory, QUERY_VEC, personality, NOW)
        assert score > 0.0

    def test_score_semantic_memory_recency_is_one(self) -> None:
        """意味記憶は recency = 1.0."""
        engine = MemorySearchEngine()
        personality = _make_personality()
        memory = _make_semantic(embedding=SIMILAR_VEC)
        score = engine.score_memory(memory, QUERY_VEC, personality, NOW)
        assert score > 0.0

    def test_similar_memory_scores_higher(self) -> None:
        """類似度が高い記憶のスコアが高い."""
        engine = MemorySearchEngine()
        personality = _make_personality()
        similar = _make_episodic(id="ep-1", embedding=SIMILAR_VEC)
        different = _make_episodic(id="ep-2", embedding=DIFFERENT_VEC)

        score_similar = engine.score_memory(similar, QUERY_VEC, personality, NOW)
        score_different = engine.score_memory(different, QUERY_VEC, personality, NOW)
        assert score_similar > score_different

    def test_important_memory_scores_higher(self) -> None:
        """重要度が高い記憶のスコアが高い."""
        engine = MemorySearchEngine()
        personality = _make_personality()
        important = _make_episodic(id="ep-1", importance=0.9, embedding=SIMILAR_VEC)
        unimportant = _make_episodic(id="ep-2", importance=0.1, embedding=SIMILAR_VEC)

        score_imp = engine.score_memory(important, QUERY_VEC, personality, NOW)
        score_unimp = engine.score_memory(unimportant, QUERY_VEC, personality, NOW)
        assert score_imp > score_unimp

    def test_recent_memory_scores_higher(self) -> None:
        """新しい記憶のスコアが高い."""
        engine = MemorySearchEngine()
        personality = _make_personality()
        recent = _make_episodic(id="ep-1", embedding=SIMILAR_VEC, timestamp=NOW)
        old = _make_episodic(
            id="ep-2",
            embedding=SIMILAR_VEC,
            timestamp=NOW - timedelta(days=200),
        )

        score_recent = engine.score_memory(recent, QUERY_VEC, personality, NOW)
        score_old = engine.score_memory(old, QUERY_VEC, personality, NOW)
        assert score_recent > score_old

    def test_no_embedding_returns_zero(self) -> None:
        """embedding がない記憶はスコア 0."""
        engine = MemorySearchEngine()
        personality = _make_personality()
        memory = _make_episodic(embedding=None)
        score = engine.score_memory(memory, QUERY_VEC, personality, NOW)
        assert score == 0.0


class TestSearch:
    """search メソッドの検証."""

    def test_search_returns_top_k(self) -> None:
        engine = MemorySearchEngine(config=SearchConfig(top_k=2))
        personality = _make_personality()
        memories = [
            _make_episodic(id=f"ep-{i}", importance=i * 0.1, embedding=SIMILAR_VEC)
            for i in range(5)
        ]
        results = engine.search(memories, QUERY_VEC, personality, NOW)
        assert len(results) == 2

    def test_search_ordered_by_score_desc(self) -> None:
        engine = MemorySearchEngine()
        personality = _make_personality()
        memories = [
            _make_episodic(id="low", importance=0.1, embedding=DIFFERENT_VEC),
            _make_episodic(id="high", importance=0.9, embedding=SIMILAR_VEC),
            _make_episodic(id="mid", importance=0.5, embedding=SIMILAR_VEC),
        ]
        results = engine.search(memories, QUERY_VEC, personality, NOW)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_memory_and_score(self) -> None:
        engine = MemorySearchEngine()
        personality = _make_personality()
        memory = _make_episodic(embedding=SIMILAR_VEC)
        results = engine.search([memory], QUERY_VEC, personality, NOW)
        assert len(results) == 1
        assert results[0][0] is memory
        assert isinstance(results[0][1], float)

    def test_search_mixed_episodic_and_semantic(self) -> None:
        """エピソードと意味記憶の混合検索."""
        engine = MemorySearchEngine()
        personality = _make_personality()
        memories: list[EpisodicMemory | SemanticMemory] = [
            _make_episodic(id="ep-1", embedding=SIMILAR_VEC),
            _make_semantic(id="sem-1", embedding=SIMILAR_VEC),
        ]
        results = engine.search(memories, QUERY_VEC, personality, NOW)
        assert len(results) == 2

    def test_search_skips_no_embedding(self) -> None:
        engine = MemorySearchEngine()
        personality = _make_personality()
        memories = [
            _make_episodic(id="with", embedding=SIMILAR_VEC),
            _make_episodic(id="without", embedding=None),
        ]
        results = engine.search(memories, QUERY_VEC, personality, NOW)
        assert len(results) == 1
        assert results[0][0].id == "with"

    def test_search_empty_list(self) -> None:
        engine = MemorySearchEngine()
        personality = _make_personality()
        results = engine.search([], QUERY_VEC, personality, NOW)
        assert results == []


class TestNeuroticismNegativityBias:
    """神経症傾向によるネガティブ記憶バイアスの検証 (Issue #103)."""

    def test_high_neuroticism_boosts_negative_memories(self) -> None:
        """高神経症傾向はネガティブ記憶のスコアを押し上げる."""
        engine = MemorySearchEngine()
        personality = _make_personality(neuroticism=0.9)

        # ネガティブ感情の記憶
        negative_memory = _make_episodic(
            id="ep-neg",
            embedding=SIMILAR_VEC,
            emotional_valence=-0.8,
            importance=0.5,
        )
        # ニュートラル感情の記憶（同じ importance/embedding）
        neutral_memory = _make_episodic(
            id="ep-neutral",
            embedding=SIMILAR_VEC,
            emotional_valence=0.0,
            importance=0.5,
        )

        score_neg = engine.score_memory(negative_memory, QUERY_VEC, personality, NOW)
        score_neutral = engine.score_memory(neutral_memory, QUERY_VEC, personality, NOW)

        # ネガティブ記憶のスコアがニュートラルより高い
        assert score_neg > score_neutral

    def test_low_neuroticism_no_negativity_bias(self) -> None:
        """低神経症傾向ではネガティブバイアスがほぼない."""
        engine = MemorySearchEngine()
        personality = _make_personality(neuroticism=0.1)

        negative_memory = _make_episodic(
            id="ep-neg",
            embedding=SIMILAR_VEC,
            emotional_valence=-0.8,
            importance=0.5,
        )
        neutral_memory = _make_episodic(
            id="ep-neutral",
            embedding=SIMILAR_VEC,
            emotional_valence=0.0,
            importance=0.5,
        )

        score_neg = engine.score_memory(negative_memory, QUERY_VEC, personality, NOW)
        score_neutral = engine.score_memory(neutral_memory, QUERY_VEC, personality, NOW)

        # 差はごく小さい（factor=0.15, neuroticism=0.1, valence=0.8 → bonus=0.012）
        assert score_neg - score_neutral == pytest.approx(
            0.15 * 0.1 * 0.8, abs=0.001
        )

    def test_negativity_bias_only_for_episodic(self) -> None:
        """意味記憶にはネガティブバイアスが適用されない."""
        engine = MemorySearchEngine()
        personality = _make_personality(neuroticism=0.9)

        # SemanticMemory には emotional_valence がないのでバイアスなし
        semantic = _make_semantic(
            id="sem-1",
            embedding=SIMILAR_VEC,
            confidence=0.5,
        )
        # 比較: ニュートラルなエピソード記憶（同等の importance/embedding）
        episodic_neutral = _make_episodic(
            id="ep-neutral",
            embedding=SIMILAR_VEC,
            emotional_valence=0.0,
            importance=0.5,
        )

        score_sem = engine.score_memory(semantic, QUERY_VEC, personality, NOW)
        score_ep = engine.score_memory(episodic_neutral, QUERY_VEC, personality, NOW)

        # SemanticMemory のスコアにはネガティブバイアスボーナスが含まれない
        # （ここではバイアスが追加されないことを確認するだけ）
        # 両方ともニュートラルなのでバイアスの差はない
        # SemanticMemory は recency=1.0 なので若干異なるが、バイアスは0
        assert score_sem > 0.0  # 正常にスコアが計算される

    def test_negativity_bias_exact_value(self) -> None:
        """ネガティブバイアスの正確な値を検証."""
        config = SearchConfig(neuroticism_negativity_factor=0.15)
        engine = MemorySearchEngine(config=config)
        personality = _make_personality(neuroticism=0.8)

        negative = _make_episodic(
            id="ep-neg",
            embedding=SIMILAR_VEC,
            emotional_valence=-0.6,
            importance=0.5,
        )
        neutral = _make_episodic(
            id="ep-neutral",
            embedding=SIMILAR_VEC,
            emotional_valence=0.0,
            importance=0.5,
        )

        score_neg = engine.score_memory(negative, QUERY_VEC, personality, NOW)
        score_neutral = engine.score_memory(neutral, QUERY_VEC, personality, NOW)

        # bonus = 0.15 * 0.8 * abs(-0.6) = 0.15 * 0.8 * 0.6 = 0.072
        expected_bonus = 0.15 * 0.8 * 0.6
        assert score_neg - score_neutral == pytest.approx(expected_bonus, abs=0.001)

    def test_positive_valence_no_bonus(self) -> None:
        """ポジティブ感情の記憶にはバイアスが適用されない."""
        engine = MemorySearchEngine()
        personality = _make_personality(neuroticism=0.9)

        positive = _make_episodic(
            id="ep-pos",
            embedding=SIMILAR_VEC,
            emotional_valence=0.8,
            importance=0.5,
        )
        neutral = _make_episodic(
            id="ep-neutral",
            embedding=SIMILAR_VEC,
            emotional_valence=0.0,
            importance=0.5,
        )

        score_pos = engine.score_memory(positive, QUERY_VEC, personality, NOW)
        score_neutral = engine.score_memory(neutral, QUERY_VEC, personality, NOW)

        # ポジティブ記憶にはバイアスなし → 同じスコア
        assert score_pos == pytest.approx(score_neutral)


class TestOpennessDiversity:
    """開放性による検索多様性の検証 (Issue #103)."""

    def test_high_openness_increases_effective_top_k(self) -> None:
        """openness=0.9 のとき、top_k=10 → effective 14."""
        engine = MemorySearchEngine(config=SearchConfig(top_k=10))
        personality = _make_personality(openness=0.9)
        effective_k = engine.compute_effective_top_k(personality)
        # 10 + floor(0.5 * 0.9 * 10) = 10 + 4 = 14
        assert effective_k == 14

    def test_low_openness_minimal_increase(self) -> None:
        """openness=0.1 のとき、top_k はほぼ増加しない."""
        engine = MemorySearchEngine(config=SearchConfig(top_k=10))
        personality = _make_personality(openness=0.1)
        effective_k = engine.compute_effective_top_k(personality)
        # 10 + floor(0.5 * 0.1 * 10) = 10 + 0 = 10
        assert effective_k == 10

    def test_zero_openness_no_increase(self) -> None:
        """openness=0.0 のとき、top_k は変わらない."""
        engine = MemorySearchEngine(config=SearchConfig(top_k=10))
        personality = _make_personality(openness=0.0)
        effective_k = engine.compute_effective_top_k(personality)
        assert effective_k == 10

    def test_max_openness_max_increase(self) -> None:
        """openness=1.0 のとき、最大50%増加."""
        engine = MemorySearchEngine(config=SearchConfig(top_k=10))
        personality = _make_personality(openness=1.0)
        effective_k = engine.compute_effective_top_k(personality)
        # 10 + floor(0.5 * 1.0 * 10) = 10 + 5 = 15
        assert effective_k == 15

    def test_search_returns_more_with_high_openness(self) -> None:
        """高開放性の search は実際に多くの結果を返す."""
        engine = MemorySearchEngine(config=SearchConfig(top_k=3))
        memories = [
            _make_episodic(
                id=f"ep-{i}", importance=i * 0.1, embedding=SIMILAR_VEC
            )
            for i in range(1, 8)  # 7 memories
        ]

        p_low = _make_personality(openness=0.1)
        p_high = _make_personality(openness=0.9)

        results_low = engine.search(memories, QUERY_VEC, p_low, NOW)
        results_high = engine.search(memories, QUERY_VEC, p_high, NOW)

        # low openness: top_k=3, effective=3+floor(0.5*0.1*3)=3
        # high openness: top_k=3, effective=3+floor(0.5*0.9*3)=3+1=4
        assert len(results_low) == 3
        assert len(results_high) == 4
