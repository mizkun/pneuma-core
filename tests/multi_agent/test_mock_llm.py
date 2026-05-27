"""Tests for MockLLMAdapter (used when ANTHROPIC_API_KEY is missing)."""

from __future__ import annotations

import json

import pytest

from pneuma_core.llm.adapter import LLMRequest
from pneuma_core.multi_agent.mock_llm import MockLLMAdapter


@pytest.mark.asyncio
async def test_returns_structured_response() -> None:
    """AC: MockLLM は structured (speech/thought/action) JSON を返す."""
    adapter = MockLLMAdapter(seed=42)
    response = await adapter.generate(
        LLMRequest(
            system_prompt="You are A. extraversion=0.9",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    # JSON parseable
    parsed = json.loads(response.content)
    assert "speech" in parsed
    assert isinstance(parsed["speech"], str)
    assert len(parsed["speech"]) > 0


@pytest.mark.asyncio
async def test_emotion_estimation_json() -> None:
    """AC: emotion estimation 形式のリクエストには PAD JSON を返す."""
    adapter = MockLLMAdapter(seed=1)
    response = await adapter.generate(
        LLMRequest(
            system_prompt=(
                "あなたは会話の感情分析エキスパートです。"
                "pleasure arousal dominance を JSON で返してください。"
            ),
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    parsed = json.loads(response.content)
    assert "pleasure" in parsed
    assert -1.0 <= float(parsed["pleasure"]) <= 1.0
    assert -1.0 <= float(parsed["arousal"]) <= 1.0
    assert -1.0 <= float(parsed["dominance"]) <= 1.0


@pytest.mark.asyncio
async def test_session_end_json() -> None:
    """AC: SessionEndPipeline 形式のリクエストには空の analysis JSON を返す."""
    adapter = MockLLMAdapter(seed=1)
    response = await adapter.generate(
        LLMRequest(
            system_prompt="記憶管理アシスタント episodic_memories",
            messages=[{"role": "user", "content": "..."}],
        )
    )
    parsed = json.loads(response.content)
    # SessionEndPipeline expects these keys
    assert "episodic_memories" in parsed
    assert "semantic_updates" in parsed
    assert "relationship_changes" in parsed


@pytest.mark.asyncio
async def test_different_speakers_vary_response() -> None:
    """AC: 性格に応じて応答プールから多様な発話を返す."""
    adapter = MockLLMAdapter(seed=1)
    seen: set[str] = set()
    for i in range(20):
        response = await adapter.generate(
            LLMRequest(
                system_prompt=f"あなたは A です。",
                messages=[{"role": "user", "content": f"turn {i}"}],
            )
        )
        parsed = json.loads(response.content)
        seen.add(parsed["speech"])
    # At least 3 distinct responses out of 20 calls
    assert len(seen) >= 3
