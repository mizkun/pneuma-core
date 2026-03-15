"""Tests for EmbeddingService Protocol and OpenAI implementation (Issue #24)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pneuma_core.llm.embedding import EmbeddingService, OpenAIEmbeddingService

DIMENSION = 1536


class TestEmbeddingServiceProtocol:
    """EmbeddingService Protocol の検証."""

    def test_protocol_is_runtime_checkable(self) -> None:
        class DummyService:
            async def embed(self, text: str) -> list[float]:
                return [0.0] * DIMENSION

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * DIMENSION for _ in texts]

        service = DummyService()
        assert isinstance(service, EmbeddingService)

    def test_incomplete_implementation_fails(self) -> None:
        class BadService:
            pass

        service = BadService()
        assert not isinstance(service, EmbeddingService)


class TestOpenAIEmbeddingServiceInit:
    """初期化の検証."""

    def test_create_with_api_key(self) -> None:
        service = OpenAIEmbeddingService(api_key="test-key")
        assert service is not None

    def test_default_model(self) -> None:
        service = OpenAIEmbeddingService(api_key="test-key")
        assert service.model == "text-embedding-3-small"

    def test_custom_model(self) -> None:
        service = OpenAIEmbeddingService(
            api_key="test-key", model="text-embedding-3-large"
        )
        assert service.model == "text-embedding-3-large"

    def test_create_from_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            service = OpenAIEmbeddingService.from_env()
            assert service is not None

    def test_create_from_env_raises_without_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIEmbeddingService.from_env()

    def test_satisfies_protocol(self) -> None:
        service = OpenAIEmbeddingService(api_key="test-key")
        assert isinstance(service, EmbeddingService)


class TestOpenAIEmbeddingServiceEmbed:
    """embed メソッドの検証（API モック）."""

    @pytest.fixture
    def service(self) -> OpenAIEmbeddingService:
        return OpenAIEmbeddingService(api_key="test-key")

    @pytest.fixture
    def mock_embedding(self) -> list[float]:
        return [0.1 * (i % 10) for i in range(DIMENSION)]

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_floats(
        self, service: OpenAIEmbeddingService, mock_embedding: list[float]
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=mock_embedding)]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await service.embed("テスト文章")
            assert isinstance(result, list)
            assert len(result) == DIMENSION
            assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_passes_correct_params(
        self, service: OpenAIEmbeddingService, mock_embedding: list[float]
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=mock_embedding)]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_create:
            await service.embed("テスト文章")

            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input="テスト文章",
            )

    @pytest.mark.asyncio
    async def test_embed_dimension(
        self, service: OpenAIEmbeddingService, mock_embedding: list[float]
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=mock_embedding)]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await service.embed("テスト")
            assert len(result) == DIMENSION


class TestOpenAIEmbeddingServiceBatch:
    """embed_batch メソッドの検証."""

    @pytest.fixture
    def service(self) -> OpenAIEmbeddingService:
        return OpenAIEmbeddingService(api_key="test-key")

    @pytest.mark.asyncio
    async def test_batch_returns_multiple_embeddings(
        self, service: OpenAIEmbeddingService
    ) -> None:
        embeddings = [
            [0.1 * (i % 10) for i in range(DIMENSION)],
            [0.2 * (i % 10) for i in range(DIMENSION)],
            [0.3 * (i % 10) for i in range(DIMENSION)],
        ]
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=e) for e in embeddings]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            results = await service.embed_batch(["テスト1", "テスト2", "テスト3"])
            assert len(results) == 3
            assert all(len(r) == DIMENSION for r in results)

    @pytest.mark.asyncio
    async def test_batch_passes_correct_params(
        self, service: OpenAIEmbeddingService
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(embedding=[0.0] * DIMENSION),
            MagicMock(embedding=[0.0] * DIMENSION),
        ]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_create:
            await service.embed_batch(["text1", "text2"])

            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input=["text1", "text2"],
            )

    @pytest.mark.asyncio
    async def test_batch_empty_list(
        self, service: OpenAIEmbeddingService
    ) -> None:
        result = await service.embed_batch([])
        assert result == []


class TestOpenAIEmbeddingServiceEmptyStringHandling:
    """embed_batch での空文字列フィルタリングの検証."""

    @pytest.fixture
    def service(self) -> OpenAIEmbeddingService:
        return OpenAIEmbeddingService(api_key="test-key")

    @pytest.mark.asyncio
    async def test_batch_all_empty_strings_returns_zero_vectors(
        self, service: OpenAIEmbeddingService
    ) -> None:
        """全て空文字列の場合、API を呼ばずにゼロベクトルを返す."""
        result = await service.embed_batch(["", "", ""])
        assert len(result) == 3
        for vec in result:
            assert all(v == 0.0 for v in vec)

    @pytest.mark.asyncio
    async def test_batch_mixed_empty_and_nonempty(
        self, service: OpenAIEmbeddingService
    ) -> None:
        """空文字列と通常文字列が混在する場合、空文字列にはゼロベクトル、通常文字列には正常な embedding を返す."""
        mock_embedding = [0.1 * (i % 10) for i in range(DIMENSION)]
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=mock_embedding)]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_create:
            result = await service.embed_batch(["", "hello", ""])
            assert len(result) == 3
            # 空文字列 → ゼロベクトル
            assert all(v == 0.0 for v in result[0])
            assert all(v == 0.0 for v in result[2])
            # 通常文字列 → API からの embedding
            assert result[1] == mock_embedding
            # API には非空文字列のみが渡される
            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input=["hello"],
            )

    @pytest.mark.asyncio
    async def test_batch_whitespace_only_treated_as_empty(
        self, service: OpenAIEmbeddingService
    ) -> None:
        """ホワイトスペースのみの文字列も空として扱う."""
        result = await service.embed_batch(["   ", "\n\t", ""])
        assert len(result) == 3
        for vec in result:
            assert all(v == 0.0 for v in vec)

    @pytest.mark.asyncio
    async def test_batch_preserves_order_with_empty_strings(
        self, service: OpenAIEmbeddingService
    ) -> None:
        """空文字列が混在しても、元のインデックス順序が保持される."""
        emb_a = [1.0] * DIMENSION
        emb_b = [2.0] * DIMENSION
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(embedding=emb_a),
            MagicMock(embedding=emb_b),
        ]

        with patch.object(
            service._client.embeddings,
            "create",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_create:
            result = await service.embed_batch(["alpha", "", "beta"])
            assert len(result) == 3
            assert result[0] == emb_a  # "alpha"
            assert all(v == 0.0 for v in result[1])  # ""
            assert result[2] == emb_b  # "beta"
            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input=["alpha", "beta"],
            )
