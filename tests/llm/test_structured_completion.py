"""Tests for ClaudeAdapter.structured_completion (Issue #12 / AC-1).

invariant: structured-output-via-tool-use — 内部状態評価系の LLM コールは
Anthropic Tool Use API で JSON Schema を強制する。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pneuma_core.llm.claude import ClaudeAdapter
from pneuma_core.llm.validation import StructuredResponseError


@pytest.fixture
def adapter() -> ClaudeAdapter:
    return ClaudeAdapter(api_key="test-key")


_SCHEMA = {
    "type": "object",
    "properties": {
        "pleasure": {"type": "number"},
        "arousal": {"type": "number"},
        "dominance": {"type": "number"},
        "emotion_label": {"type": "string"},
        "situation": {"type": "string"},
    },
    "required": ["pleasure", "arousal", "dominance", "emotion_label", "situation"],
}


def _mock_tool_use_response(payload: dict) -> MagicMock:
    """Return a mocked anthropic response containing a tool_use block."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "report_emotion"
    tool_use_block.input = payload

    resp = MagicMock()
    resp.content = [tool_use_block]
    resp.model = "claude-haiku-4-20250514"
    resp.usage.input_tokens = 12
    resp.usage.output_tokens = 8
    return resp


class TestStructuredCompletion:
    """AC-1: structured_completion uses Tool Use API and returns dict."""

    @pytest.mark.asyncio
    async def test_returns_parsed_dict_from_tool_use_block(
        self, adapter: ClaudeAdapter
    ) -> None:
        """ToolUseBlock.input がそのまま dict として返ること."""
        payload = {
            "pleasure": 0.6,
            "arousal": 0.4,
            "dominance": 0.2,
            "emotion_label": "happy",
            "situation": "雑談中",
        }
        resp = _mock_tool_use_response(payload)

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            result = await adapter.structured_completion(
                system_prompt="あなたは感情分析エキスパートです。",
                messages=[{"role": "user", "content": "hi"}],
                tool_name="report_emotion",
                tool_description="Report current emotional state",
                schema=_SCHEMA,
            )

        assert result == payload

    @pytest.mark.asyncio
    async def test_passes_tools_and_tool_choice_params(
        self, adapter: ClaudeAdapter
    ) -> None:
        """Anthropic SDK に tools / tool_choice が正しく渡ること."""
        resp = _mock_tool_use_response({
            "pleasure": 0.1, "arousal": 0.0, "dominance": 0.0,
            "emotion_label": "neutral", "situation": "",
        })

        mock_create = AsyncMock(return_value=resp)
        with patch.object(adapter._client.messages, "create", mock_create):
            await adapter.structured_completion(
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                tool_name="report_emotion",
                tool_description="Report emotion",
                schema=_SCHEMA,
            )

        assert mock_create.called
        kwargs = mock_create.call_args.kwargs
        assert "tools" in kwargs
        assert isinstance(kwargs["tools"], list) and len(kwargs["tools"]) == 1
        tool_def = kwargs["tools"][0]
        assert tool_def["name"] == "report_emotion"
        assert tool_def["description"] == "Report emotion"
        assert tool_def["input_schema"] == _SCHEMA

        assert kwargs.get("tool_choice") == {"type": "tool", "name": "report_emotion"}

    @pytest.mark.asyncio
    async def test_passes_system_and_messages(self, adapter: ClaudeAdapter) -> None:
        """system / messages が SDK 呼び出しに含まれること."""
        resp = _mock_tool_use_response({
            "pleasure": 0.0, "arousal": 0.0, "dominance": 0.0,
            "emotion_label": "neutral", "situation": "",
        })
        mock_create = AsyncMock(return_value=resp)
        with patch.object(adapter._client.messages, "create", mock_create):
            await adapter.structured_completion(
                system_prompt="SYSTEM",
                messages=[{"role": "user", "content": "MSG"}],
                tool_name="report_emotion",
                tool_description="d",
                schema=_SCHEMA,
            )

        kwargs = mock_create.call_args.kwargs
        assert kwargs["system"] == "SYSTEM" or any(
            b.get("text") == "SYSTEM" for b in kwargs["system"]
        )
        assert kwargs["messages"] == [{"role": "user", "content": "MSG"}]

    @pytest.mark.asyncio
    async def test_uses_supplied_model(self, adapter: ClaudeAdapter) -> None:
        """model 引数が SDK に渡ること."""
        resp = _mock_tool_use_response({
            "pleasure": 0.0, "arousal": 0.0, "dominance": 0.0,
            "emotion_label": "neutral", "situation": "",
        })
        mock_create = AsyncMock(return_value=resp)
        with patch.object(adapter._client.messages, "create", mock_create):
            await adapter.structured_completion(
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                tool_name="report_emotion",
                tool_description="d",
                schema=_SCHEMA,
                model="claude-haiku-4-20250514",
            )

        assert mock_create.call_args.kwargs["model"] == "claude-haiku-4-20250514"

    @pytest.mark.asyncio
    async def test_raises_when_no_tool_use_block(
        self, adapter: ClaudeAdapter
    ) -> None:
        """ToolUseBlock が無い応答は StructuredResponseError."""
        from anthropic.types import TextBlock

        resp = MagicMock()
        resp.content = [TextBlock(type="text", text="just text")]
        resp.model = "claude-haiku-4-20250514"
        resp.usage.input_tokens = 1
        resp.usage.output_tokens = 1
        resp.stop_reason = "end_turn"

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            with pytest.raises(StructuredResponseError):
                await adapter.structured_completion(
                    system_prompt="sys",
                    messages=[{"role": "user", "content": "hi"}],
                    tool_name="t",
                    tool_description="d",
                    schema=_SCHEMA,
                )

    @pytest.mark.asyncio
    async def test_raises_when_tool_use_input_not_dict(
        self, adapter: ClaudeAdapter
    ) -> None:
        """ToolUseBlock.input が dict でない場合 StructuredResponseError."""
        tool_use_block = MagicMock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "t"
        tool_use_block.input = "not-a-dict"

        resp = MagicMock()
        resp.content = [tool_use_block]
        resp.model = "claude-haiku-4-20250514"
        resp.usage.input_tokens = 1
        resp.usage.output_tokens = 1

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            with pytest.raises(StructuredResponseError):
                await adapter.structured_completion(
                    system_prompt="sys",
                    messages=[{"role": "user", "content": "hi"}],
                    tool_name="t",
                    tool_description="d",
                    schema=_SCHEMA,
                )

    @pytest.mark.asyncio
    async def test_picks_matching_tool_use_block(
        self, adapter: ClaudeAdapter
    ) -> None:
        """複数ブロックがあっても名前一致 ToolUseBlock を選ぶ."""
        from anthropic.types import TextBlock

        wrong = MagicMock()
        wrong.type = "tool_use"
        wrong.name = "other_tool"
        wrong.input = {"x": 1}

        right = MagicMock()
        right.type = "tool_use"
        right.name = "report_emotion"
        right.input = {
            "pleasure": 0.3, "arousal": 0.1, "dominance": 0.0,
            "emotion_label": "neutral", "situation": "ok",
        }

        resp = MagicMock()
        resp.content = [TextBlock(type="text", text="..."), wrong, right]
        resp.model = "claude-haiku-4-20250514"
        resp.usage.input_tokens = 1
        resp.usage.output_tokens = 1

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            result = await adapter.structured_completion(
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                tool_name="report_emotion",
                tool_description="d",
                schema=_SCHEMA,
            )

        assert result == right.input
