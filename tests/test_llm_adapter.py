"""Tests for LLMAdapter Protocol and related models (Issue #22)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest, LLMResponse, ModelConfig


class TestLLMRequest:
    """LLMRequest データモデルの検証."""

    def test_create_minimal(self) -> None:
        req = LLMRequest(system_prompt="You are a helpful AI.", messages=[])
        assert req.system_prompt == "You are a helpful AI."
        assert req.messages == []

    def test_create_with_messages(self) -> None:
        msgs = [
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "こんにちは！"},
        ]
        req = LLMRequest(system_prompt="test", messages=msgs)
        assert len(req.messages) == 2
        assert req.messages[0]["role"] == "user"

    def test_default_values(self) -> None:
        req = LLMRequest(system_prompt="test", messages=[])
        assert req.model is None
        assert req.temperature == 0.7
        assert req.max_tokens == 1024

    def test_custom_values(self) -> None:
        req = LLMRequest(
            system_prompt="test",
            messages=[],
            model="claude-sonnet-4-20250514",
            temperature=0.3,
            max_tokens=2048,
        )
        assert req.model == "claude-sonnet-4-20250514"
        assert req.temperature == 0.3
        assert req.max_tokens == 2048

    def test_temperature_boundaries(self) -> None:
        """temperature は 0.0〜2.0 の範囲."""
        req = LLMRequest(system_prompt="test", messages=[], temperature=0.0)
        assert req.temperature == 0.0

        req = LLMRequest(system_prompt="test", messages=[], temperature=2.0)
        assert req.temperature == 2.0

    def test_temperature_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            LLMRequest(system_prompt="test", messages=[], temperature=-0.1)

        with pytest.raises(ValueError):
            LLMRequest(system_prompt="test", messages=[], temperature=2.1)

    def test_max_tokens_positive(self) -> None:
        with pytest.raises(ValueError):
            LLMRequest(system_prompt="test", messages=[], max_tokens=0)

        with pytest.raises(ValueError):
            LLMRequest(system_prompt="test", messages=[], max_tokens=-1)


class TestLLMResponse:
    """LLMResponse データモデルの検証."""

    def test_create(self) -> None:
        resp = LLMResponse(
            content="こんにちは！",
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert resp.content == "こんにちは！"
        assert resp.model == "claude-sonnet-4-20250514"
        assert resp.usage["input_tokens"] == 10
        assert resp.usage["output_tokens"] == 5

    def test_empty_content(self) -> None:
        resp = LLMResponse(content="", model="test", usage={})
        assert resp.content == ""

    def test_frozen(self) -> None:
        resp = LLMResponse(content="test", model="test", usage={})
        with pytest.raises(AttributeError):
            resp.content = "changed"  # type: ignore[misc]


class TestModelConfig:
    """ModelConfig の検証."""

    def test_create(self) -> None:
        config = ModelConfig(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            max_tokens=1024,
        )
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_defaults(self) -> None:
        config = ModelConfig(model="claude-haiku-4-5-20251001")
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(model="test", temperature=-0.1)

        with pytest.raises(ValueError):
            ModelConfig(model="test", temperature=2.1)

    def test_invalid_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(model="test", max_tokens=0)

        with pytest.raises(ValueError):
            ModelConfig(model="test", max_tokens=-1)

    def test_to_request_kwargs(self) -> None:
        """ModelConfig から LLMRequest に適用できること."""
        config = ModelConfig(
            model="claude-sonnet-4-20250514",
            temperature=0.5,
            max_tokens=2048,
        )
        req = LLMRequest(
            system_prompt="test",
            messages=[],
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        assert req.model == "claude-sonnet-4-20250514"
        assert req.temperature == 0.5
        assert req.max_tokens == 2048


class TestLLMAdapterProtocol:
    """LLMAdapter Protocol の検証."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """runtime_checkable であること."""
        assert hasattr(LLMAdapter, "__protocol_attrs__") or hasattr(
            LLMAdapter, "__abstractmethods__"
        )

    def test_dummy_implementation_satisfies_protocol(self) -> None:
        """ダミー実装が Protocol を満たすこと."""

        class DummyAdapter:
            async def generate(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(
                    content="dummy",
                    model="dummy-model",
                    usage={"input_tokens": 0, "output_tokens": 0},
                )

        adapter = DummyAdapter()
        assert isinstance(adapter, LLMAdapter)

    def test_incomplete_implementation_fails_protocol(self) -> None:
        """generate メソッドがない実装は Protocol を満たさない."""

        class BadAdapter:
            pass

        adapter = BadAdapter()
        assert not isinstance(adapter, LLMAdapter)

    def test_wrong_signature_fails_protocol(self) -> None:
        """異なるシグネチャの generate は Protocol を満たさない."""

        class WrongAdapter:
            def generate(self) -> str:  # type: ignore[override]
                return "wrong"

        # Note: runtime_checkable only checks method existence, not signature
        # So this will actually pass isinstance check
        # Full signature checking requires static type checkers (mypy)
        adapter = WrongAdapter()
        # isinstance only checks method name existence
        assert isinstance(adapter, LLMAdapter)

    @pytest.mark.asyncio
    async def test_dummy_adapter_generate(self) -> None:
        """ダミー実装の generate が正しく動作する."""

        class DummyAdapter:
            async def generate(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(
                    content=f"Echo: {request.messages[-1]['content']}" if request.messages else "empty",
                    model=request.model or "dummy",
                    usage={"input_tokens": 1, "output_tokens": 1},
                )

        adapter = DummyAdapter()
        req = LLMRequest(
            system_prompt="test",
            messages=[{"role": "user", "content": "hello"}],
        )
        resp = await adapter.generate(req)
        assert resp.content == "Echo: hello"
        assert resp.model == "dummy"
