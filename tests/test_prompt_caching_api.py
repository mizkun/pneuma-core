"""Tests for Anthropic Prompt Caching API support (Issue #117).

Tests cover:
1. LLMRequest supports static/dynamic system prompt sections
2. ClaudeAdapter sends cache_control on static sections
3. RuntimeEngine properly splits prompt sections for API caching
4. Backward compatibility with plain system_prompt
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import TextBlock

from pneuma_core.llm.adapter import LLMRequest, LLMResponse


# --- LLMRequest: system_prompt_sections support ---


class TestLLMRequestPromptSections:
    """LLMRequest がstatic/dynamicセクション分割をサポートすること."""

    def test_backward_compatible_with_plain_system_prompt(self) -> None:
        """従来の system_prompt だけの使い方が引き続き動作する."""
        req = LLMRequest(
            system_prompt="You are a helpful AI.",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.system_prompt == "You are a helpful AI."
        assert req.system_prompt_cached is None
        assert req.system_prompt_dynamic is None

    def test_supports_cached_and_dynamic_sections(self) -> None:
        """static(cached) と dynamic のセクション分割を持てる."""
        req = LLMRequest(
            system_prompt="full prompt",
            messages=[],
            system_prompt_cached="static section",
            system_prompt_dynamic="dynamic section",
        )
        assert req.system_prompt_cached == "static section"
        assert req.system_prompt_dynamic == "dynamic section"

    def test_sections_are_optional(self) -> None:
        """セクション分割はオプショナル."""
        req = LLMRequest(system_prompt="test", messages=[])
        assert req.system_prompt_cached is None
        assert req.system_prompt_dynamic is None

    def test_frozen_dataclass(self) -> None:
        """frozen dataclass のまま."""
        req = LLMRequest(
            system_prompt="test",
            messages=[],
            system_prompt_cached="static",
            system_prompt_dynamic="dynamic",
        )
        with pytest.raises(AttributeError):
            req.system_prompt_cached = "changed"  # type: ignore[misc]


# --- ClaudeAdapter: cache_control support ---


class TestClaudeAdapterPromptCaching:
    """ClaudeAdapter がAnthropic API のcache_controlを適切に送信すること."""

    @pytest.fixture
    def adapter(self):
        from pneuma_core.llm.claude import ClaudeAdapter
        return ClaudeAdapter(api_key="test-key")

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        resp = MagicMock()
        resp.content = [TextBlock(type="text", text="response")]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 20
        return resp

    @pytest.mark.asyncio
    async def test_plain_system_prompt_sent_as_string(
        self, adapter, mock_response: MagicMock
    ) -> None:
        """セクション分割なしの場合、system はそのまま文字列で送信."""
        with patch.object(
            adapter._client.messages, "create",
            new_callable=AsyncMock, return_value=mock_response,
        ) as mock_create:
            req = LLMRequest(
                system_prompt="You are Aine.",
                messages=[{"role": "user", "content": "hello"}],
            )
            await adapter.generate(req)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["system"] == "You are Aine."

    @pytest.mark.asyncio
    async def test_sections_sent_as_structured_blocks(
        self, adapter, mock_response: MagicMock
    ) -> None:
        """セクション分割ありの場合、system はcache_control付きブロックリストで送信."""
        with patch.object(
            adapter._client.messages, "create",
            new_callable=AsyncMock, return_value=mock_response,
        ) as mock_create:
            req = LLMRequest(
                system_prompt="full prompt fallback",
                messages=[{"role": "user", "content": "hello"}],
                system_prompt_cached="# Character\nProfile info",
                system_prompt_dynamic="## Emotion\nHappy",
            )
            await adapter.generate(req)

            call_kwargs = mock_create.call_args.kwargs
            system_param = call_kwargs["system"]

            # system は list であること
            assert isinstance(system_param, list)
            assert len(system_param) == 2

            # 静的セクション: cache_control あり
            static_block = system_param[0]
            assert static_block["type"] == "text"
            assert static_block["text"] == "# Character\nProfile info"
            assert static_block["cache_control"] == {"type": "ephemeral"}

            # 動的セクション: cache_control なし
            dynamic_block = system_param[1]
            assert dynamic_block["type"] == "text"
            assert dynamic_block["text"] == "## Emotion\nHappy"
            assert "cache_control" not in dynamic_block

    @pytest.mark.asyncio
    async def test_cached_only_without_dynamic(
        self, adapter, mock_response: MagicMock
    ) -> None:
        """dynamic が空文字でも正しくブロック構造で送信."""
        with patch.object(
            adapter._client.messages, "create",
            new_callable=AsyncMock, return_value=mock_response,
        ) as mock_create:
            req = LLMRequest(
                system_prompt="full",
                messages=[],
                system_prompt_cached="static content",
                system_prompt_dynamic="",
            )
            await adapter.generate(req)

            call_kwargs = mock_create.call_args.kwargs
            system_param = call_kwargs["system"]

            # 空の dynamic セクションはスキップ、静的のみ
            assert isinstance(system_param, list)
            assert len(system_param) == 1
            assert system_param[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_response_includes_cache_usage(
        self, adapter,
    ) -> None:
        """cache_creation_input_tokens / cache_read_input_tokens が usage に含まれる."""
        resp = MagicMock()
        resp.content = [TextBlock(type="text", text="response")]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 100
        resp.usage.output_tokens = 20
        resp.usage.cache_creation_input_tokens = 1500
        resp.usage.cache_read_input_tokens = 1500

        with patch.object(
            adapter._client.messages, "create",
            new_callable=AsyncMock, return_value=resp,
        ):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "hi"}],
                system_prompt_cached="static",
                system_prompt_dynamic="dynamic",
            )
            result = await adapter.generate(req)

            assert result.usage["cache_creation_input_tokens"] == 1500
            assert result.usage["cache_read_input_tokens"] == 1500


# --- RuntimeEngine integration ---


class TestRuntimeEnginePromptCachingIntegration:
    """RuntimeEngine が PromptCache 使用時にセクション分割した CachedPrompt を扱えること."""

    @pytest.fixture
    def _setup(self):
        """RuntimeEngine のセットアップ用fixture."""
        from pneuma_core.models.character import Character
        from pneuma_core.models.emotion import EmotionalState
        from pneuma_core.models.goals import GoalTree
        from pneuma_core.models.personality import Personality
        from pneuma_core.models.values import Values
        from pneuma_core.runtime.prompt_cache import PromptCache

        character = Character(
            id="test-001",
            name="TestChar",
            personality=Personality(
                openness=0.8, conscientiousness=0.6,
                extraversion=0.7, agreeableness=0.9, neuroticism=0.3,
            ),
            values=Values(
                self_transcendence=0.8, self_enhancement=0.3,
                openness_to_change=0.7, conservation=0.4,
            ),
            profile="Test profile",
            speaking_style="Test style",
        )
        emotion = EmotionalState(
            pleasure=0.5, arousal=0.3, dominance=0.1,
            emotion_label="neutral", situation="test",
        )
        return {
            "character": character,
            "emotion": emotion,
            "goals": GoalTree(),
            "cache": PromptCache(),
        }

    def test_prompt_cache_produces_separate_sections(self, _setup) -> None:
        """PromptCache.build() が static/dynamic を分離した CachedPrompt を返す."""
        cached = _setup["cache"].build(
            character=_setup["character"],
            emotional_state=_setup["emotion"],
            goal_tree=_setup["goals"],
            memories=[],
        )
        # 静的: profile, personality, values, speaking style
        assert "TestChar" in cached.static_section
        assert "Test profile" in cached.static_section
        # 動的: emotion
        assert "neutral" in cached.dynamic_section

    def test_build_system_prompt_returns_cached_prompt_when_cache_enabled(
        self, _setup
    ) -> None:
        """PromptCache 使用時に CachedPrompt が取得できる."""
        from pneuma_core.runtime.prompt_cache import CachedPrompt

        cached = _setup["cache"].build(
            character=_setup["character"],
            emotional_state=_setup["emotion"],
            goal_tree=_setup["goals"],
            memories=[],
        )
        assert isinstance(cached, CachedPrompt)
        assert len(cached.static_section) > 0
        assert len(cached.dynamic_section) > 0


# --- Pipeline test: PromptCache -> LLMRequest -> ClaudeAdapter API call ---


class TestPromptCachingPipeline:
    """パイプラインテスト: PromptCache -> LLMRequest -> ClaudeAdapter."""

    @pytest.mark.asyncio
    async def test_end_to_end_caching_pipeline(self) -> None:
        """PromptCache で分割 -> LLMRequest に渡す -> ClaudeAdapter が cache_control 付きで送信."""
        from pneuma_core.llm.claude import ClaudeAdapter
        from pneuma_core.models.character import Character
        from pneuma_core.models.emotion import EmotionalState
        from pneuma_core.models.goals import GoalTree
        from pneuma_core.models.personality import Personality
        from pneuma_core.models.values import Values
        from pneuma_core.runtime.prompt_cache import PromptCache

        # 1. Build prompt sections
        character = Character(
            id="pipeline-001",
            name="PipelineChar",
            personality=Personality(
                openness=0.8, conscientiousness=0.6,
                extraversion=0.7, agreeableness=0.9, neuroticism=0.3,
            ),
            values=Values(
                self_transcendence=0.8, self_enhancement=0.3,
                openness_to_change=0.7, conservation=0.4,
            ),
            profile="Pipeline test profile",
            speaking_style="Pipeline test style",
        )
        emotion = EmotionalState(
            pleasure=0.3, arousal=0.2, dominance=0.0,
            emotion_label="calm", situation="testing",
        )

        cache = PromptCache()
        cached = cache.build(
            character=character,
            emotional_state=emotion,
            goal_tree=GoalTree(),
            memories=[],
        )

        # 2. Create LLMRequest with sections
        req = LLMRequest(
            system_prompt=cached.full_prompt,
            messages=[{"role": "user", "content": "hello"}],
            system_prompt_cached=cached.static_section,
            system_prompt_dynamic=cached.dynamic_section,
        )

        # 3. Send through ClaudeAdapter
        adapter = ClaudeAdapter(api_key="test-key")
        mock_resp = MagicMock()
        mock_resp.content = [TextBlock(type="text", text="response")]
        mock_resp.model = "claude-sonnet-4-20250514"
        mock_resp.usage.input_tokens = 50
        mock_resp.usage.output_tokens = 10
        mock_resp.usage.cache_creation_input_tokens = 0
        mock_resp.usage.cache_read_input_tokens = 0

        with patch.object(
            adapter._client.messages, "create",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_create:
            await adapter.generate(req)

            call_kwargs = mock_create.call_args.kwargs
            system_param = call_kwargs["system"]

            # Verify structured blocks
            assert isinstance(system_param, list)
            assert len(system_param) >= 1

            # Static block has cache_control
            static_block = system_param[0]
            assert "PipelineChar" in static_block["text"]
            assert "Pipeline test profile" in static_block["text"]
            assert static_block["cache_control"] == {"type": "ephemeral"}

            # Dynamic block exists and has emotion
            if len(system_param) > 1:
                dynamic_block = system_param[1]
                assert "calm" in dynamic_block["text"]
                assert "cache_control" not in dynamic_block
