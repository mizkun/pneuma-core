"""Tests for ClaudeAdapter (Issue #23, #47)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic import APIStatusError, BadRequestError, InternalServerError, RateLimitError
from anthropic.types import TextBlock

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest, LLMResponse, ModelConfig
from pneuma_core.llm.claude import ClaudeAdapter


def _make_api_status_error(
    status_code: int, message: str = "error"
) -> APIStatusError:
    """APIStatusError のテスト用ヘルパー."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=status_code, request=request)
    if status_code == 429:
        return RateLimitError(message=message, response=response, body=None)
    elif status_code == 500:
        return InternalServerError(message=message, response=response, body=None)
    else:
        return APIStatusError(message=message, response=response, body=None)


class TestClaudeAdapterProtocol:
    """Protocol 準拠の検証."""

    def test_satisfies_llm_adapter_protocol(self) -> None:
        adapter = ClaudeAdapter(api_key="test-key")
        assert isinstance(adapter, LLMAdapter)


class TestClaudeAdapterInit:
    """初期化の検証."""

    def test_create_with_api_key(self) -> None:
        adapter = ClaudeAdapter(api_key="test-key")
        assert adapter is not None

    def test_create_with_default_model(self) -> None:
        adapter = ClaudeAdapter(api_key="test-key")
        assert adapter.default_model is not None
        assert len(adapter.default_model) > 0

    def test_create_with_custom_default_model(self) -> None:
        adapter = ClaudeAdapter(
            api_key="test-key",
            default_model="claude-haiku-4-5-20251001",
        )
        assert adapter.default_model == "claude-haiku-4-5-20251001"

    def test_create_from_env_variable(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            adapter = ClaudeAdapter.from_env()
            assert adapter is not None

    def test_create_from_env_raises_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ClaudeAdapter.from_env()


class TestClaudeAdapterGenerate:
    """generate メソッドの検証（API モック）."""

    @pytest.fixture
    def adapter(self) -> ClaudeAdapter:
        return ClaudeAdapter(api_key="test-key")

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """Anthropic SDK のレスポンスをモック."""
        resp = MagicMock()
        resp.content = [TextBlock(type="text", text="こんにちは！")]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 15
        resp.usage.output_tokens = 8
        return resp

    @pytest.mark.asyncio
    async def test_generate_basic(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        with patch.object(
            adapter._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
        ):
            req = LLMRequest(
                system_prompt="You are Aine.",
                messages=[{"role": "user", "content": "こんにちは"}],
            )
            resp = await adapter.generate(req)

            assert isinstance(resp, LLMResponse)
            assert resp.content == "こんにちは！"
            assert resp.model == "claude-sonnet-4-20250514"
            assert resp.usage["input_tokens"] == 15
            assert resp.usage["output_tokens"] == 8

    @pytest.mark.asyncio
    async def test_generate_uses_request_model(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        with patch.object(
            adapter._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
        ) as mock_create:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
                model="claude-haiku-4-5-20251001",
            )
            await adapter.generate(req)

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_generate_uses_default_model_when_none(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        with patch.object(
            adapter._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
        ) as mock_create:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
                model=None,
            )
            await adapter.generate(req)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["model"] == adapter.default_model

    @pytest.mark.asyncio
    async def test_generate_passes_temperature(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        with patch.object(
            adapter._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
        ) as mock_create:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.3,
            )
            await adapter.generate(req)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_generate_passes_max_tokens(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        with patch.object(
            adapter._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
        ) as mock_create:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=2048,
            )
            await adapter.generate(req)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_generate_passes_system_prompt(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        with patch.object(
            adapter._client.messages, "create", new_callable=AsyncMock, return_value=mock_response
        ) as mock_create:
            req = LLMRequest(
                system_prompt="You are Aine, an AI character.",
                messages=[{"role": "user", "content": "test"}],
            )
            await adapter.generate(req)

            call_kwargs = mock_create.call_args
            assert call_kwargs.kwargs["system"] == "You are Aine, an AI character."


    @pytest.mark.asyncio
    async def test_generate_empty_content_returns_empty_string(
        self, adapter: ClaudeAdapter
    ) -> None:
        """content が空の場合、空文字列を返す."""
        empty_resp = MagicMock()
        empty_resp.content = []
        empty_resp.model = "claude-sonnet-4-20250514"
        empty_resp.usage.input_tokens = 5
        empty_resp.usage.output_tokens = 0

        with patch.object(
            adapter._client.messages, "create", new_callable=AsyncMock, return_value=empty_resp
        ):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            resp = await adapter.generate(req)
            assert resp.content == ""


class TestClaudeAdapterErrorHandling:
    """エラーハンドリングの検証."""

    @pytest.fixture
    def adapter(self) -> ClaudeAdapter:
        return ClaudeAdapter(api_key="test-key")

    @pytest.mark.asyncio
    async def test_api_error_raises(self, adapter: ClaudeAdapter) -> None:
        """API エラー（非リトライ対象）が適切に伝播すること."""
        from anthropic import APIError

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=APIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            ),
        ):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            with pytest.raises(APIError):
                await adapter.generate(req)

    @pytest.mark.asyncio
    async def test_rate_limit_error_raises_after_retries(
        self, adapter: ClaudeAdapter
    ) -> None:
        """レート制限エラーがリトライ上限後に伝播すること."""
        error = _make_api_status_error(429, "Rate limited")

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ), patch("pneuma_core.llm.claude.asyncio.sleep", new_callable=AsyncMock):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            with pytest.raises(RateLimitError):
                await adapter.generate(req)


class TestClaudeAdapterModelConfig:
    """ModelConfig との連携テスト."""

    def test_model_config_for_conversation(self) -> None:
        config = ModelConfig(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=1024,
        )
        adapter = ClaudeAdapter(api_key="test-key", default_model=config.model)
        assert adapter.default_model == "claude-sonnet-4-20250514"

    def test_model_config_for_lightweight(self) -> None:
        config = ModelConfig(
            model="claude-haiku-4-5-20251001",
            temperature=0.3,
            max_tokens=512,
        )
        req = LLMRequest(
            system_prompt="test",
            messages=[],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        assert req.model == "claude-haiku-4-5-20251001"
        assert req.temperature == 0.3
        assert req.max_tokens == 512


class TestClaudeAdapterRetry:
    """リトライ機構の検証（Issue #47）."""

    @pytest.fixture
    def adapter(self) -> ClaudeAdapter:
        return ClaudeAdapter(api_key="test-key")

    @pytest.fixture
    def mock_response(self) -> MagicMock:
        """成功レスポンスのモック."""
        resp = MagicMock()
        resp.content = [TextBlock(type="text", text="成功")]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5
        return resp

    @pytest.mark.asyncio
    async def test_retry_on_429_rate_limit(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        """429 Rate Limit エラーでリトライし、成功すること."""
        error = _make_api_status_error(429, "Rate limited")

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=[error, mock_response],
        ), patch("pneuma_core.llm.claude.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            resp = await adapter.generate(req)

            assert resp.content == "成功"
            assert adapter._client.messages.create.call_count == 2
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_500_server_error(
        self, adapter: ClaudeAdapter, mock_response: MagicMock
    ) -> None:
        """5xx サーバーエラーでリトライし、成功すること."""
        error = _make_api_status_error(500, "Internal server error")

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=[error, mock_response],
        ), patch("pneuma_core.llm.claude.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            resp = await adapter.generate(req)

            assert resp.content == "成功"
            assert adapter._client.messages.create.call_count == 2
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(
        self, adapter: ClaudeAdapter
    ) -> None:
        """リトライ上限（3回）到達後にエラーが伝播すること."""
        error = _make_api_status_error(429, "Rate limited")

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ), patch("pneuma_core.llm.claude.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            with pytest.raises(RateLimitError):
                await adapter.generate(req)

            # 初回 + リトライ3回 = 合計4回呼ばれる
            assert adapter._client.messages.create.call_count == 4
            # sleep は3回（リトライの前に呼ばれる）
            assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_400_client_error(
        self, adapter: ClaudeAdapter
    ) -> None:
        """4xx エラー（429以外）はリトライせず即座にエラーが伝播すること."""
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(status_code=400, request=request)
        error = BadRequestError(message="Bad request", response=response, body=None)

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ), patch("pneuma_core.llm.claude.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            with pytest.raises(BadRequestError):
                await adapter.generate(req)

            # リトライなし: 1回のみ呼ばれる
            assert adapter._client.messages.create.call_count == 1
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(
        self, adapter: ClaudeAdapter
    ) -> None:
        """リトライ間隔が指数バックオフになっていること."""
        error = _make_api_status_error(500, "Server error")

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ), patch("pneuma_core.llm.claude.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            with pytest.raises(APIStatusError):
                await adapter.generate(req)

            # 指数バックオフ: 1秒, 2秒, 4秒
            sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
            assert sleep_calls == [1.0, 2.0, 4.0]


class TestClaudeAdapterTextBlockJoin:
    """複数 TextBlock の結合テスト（Issue #47）."""

    @pytest.fixture
    def adapter(self) -> ClaudeAdapter:
        return ClaudeAdapter(api_key="test-key")

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_are_joined(
        self, adapter: ClaudeAdapter
    ) -> None:
        """複数の TextBlock が正しく結合されること."""
        resp = MagicMock()
        resp.content = [
            TextBlock(type="text", text="こんにちは、"),
            TextBlock(type="text", text="世界！"),
        ]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 8

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            result = await adapter.generate(req)

            assert result.content == "こんにちは、世界！"

    @pytest.mark.asyncio
    async def test_three_text_blocks_are_joined(
        self, adapter: ClaudeAdapter
    ) -> None:
        """3つの TextBlock が正しく結合されること."""
        resp = MagicMock()
        resp.content = [
            TextBlock(type="text", text="A"),
            TextBlock(type="text", text="B"),
            TextBlock(type="text", text="C"),
        ]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 8

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            result = await adapter.generate(req)

            assert result.content == "ABC"

    @pytest.mark.asyncio
    async def test_single_text_block_unchanged(
        self, adapter: ClaudeAdapter
    ) -> None:
        """単一の TextBlock は従来通り動作すること."""
        resp = MagicMock()
        resp.content = [TextBlock(type="text", text="単一ブロック")]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            result = await adapter.generate(req)

            assert result.content == "単一ブロック"

    @pytest.mark.asyncio
    async def test_mixed_content_only_text_blocks_joined(
        self, adapter: ClaudeAdapter
    ) -> None:
        """TextBlock 以外のコンテンツ（ToolUse等）が混在する場合、TextBlock のみ結合すること."""
        tool_use_block = MagicMock()  # ToolUseBlock のモック
        tool_use_block.type = "tool_use"

        resp = MagicMock()
        resp.content = [
            TextBlock(type="text", text="前半"),
            tool_use_block,
            TextBlock(type="text", text="後半"),
        ]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 8

        with patch.object(
            adapter._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=resp,
        ):
            req = LLMRequest(
                system_prompt="test",
                messages=[{"role": "user", "content": "test"}],
            )
            result = await adapter.generate(req)

            assert result.content == "前半後半"
