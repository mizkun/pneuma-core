"""Tests for Protocol consolidation into protocols/ package (#147).

Verifies:
- All Protocols are importable from pneuma_core.protocols
- anthropic SDK is not imported anywhere in core/ except ClaudeAdapter
- LLMTimeoutError is defined and used instead of anthropic.APITimeoutError
- Back-compat re-exports work from old paths
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


# =============================================================================
# protocols/ パッケージの存在と Protocol のインポート
# =============================================================================


class TestProtocolsPackage:
    """protocols/ パッケージが存在し、全 Protocol が集約されていること."""

    def test_protocols_package_exists(self) -> None:
        """pneuma.core.protocols パッケージがインポート可能."""
        import pneuma_core.protocols

        assert pneuma_core.protocols is not None

    def test_llm_adapter_in_protocols(self) -> None:
        """LLMAdapter が protocols からインポート可能."""
        from pneuma_core.protocols import LLMAdapter

        assert LLMAdapter is not None

    def test_llm_request_in_protocols(self) -> None:
        """LLMRequest が protocols からインポート可能."""
        from pneuma_core.protocols import LLMRequest

        assert LLMRequest is not None

    def test_llm_response_in_protocols(self) -> None:
        """LLMResponse が protocols からインポート可能."""
        from pneuma_core.protocols import LLMResponse

        assert LLMResponse is not None

    def test_storage_backend_in_protocols(self) -> None:
        """StorageBackend が protocols からインポート可能."""
        from pneuma_core.protocols import StorageBackend

        assert StorageBackend is not None

    def test_memory_store_in_protocols(self) -> None:
        """MemoryStore が protocols からインポート可能."""
        from pneuma_core.protocols import MemoryStore

        assert MemoryStore is not None

    def test_embedding_service_in_protocols(self) -> None:
        """EmbeddingService が protocols からインポート可能."""
        from pneuma_core.protocols import EmbeddingService

        assert EmbeddingService is not None

    def test_tts_adapter_in_protocols(self) -> None:
        """TTSAdapter が protocols からインポート可能."""
        from pneuma_core.protocols import TTSAdapter

        assert TTSAdapter is not None

    def test_stt_service_in_protocols(self) -> None:
        """STTService が protocols からインポート可能."""
        from pneuma_core.protocols import STTService

        assert STTService is not None

    def test_task_backend_in_protocols(self) -> None:
        """TaskBackend が protocols からインポート可能."""
        from pneuma_core.protocols import TaskBackend

        assert TaskBackend is not None

    def test_middleware_in_protocols(self) -> None:
        """Middleware が protocols からインポート可能."""
        from pneuma_core.protocols import Middleware

        assert Middleware is not None


# =============================================================================
# Protocol 定義が protocols/ に実体として存在すること
# =============================================================================


class TestProtocolDefinitionFiles:
    """Protocol 定義ファイルが protocols/ ディレクトリに存在すること."""

    @pytest.fixture()
    def protocols_dir(self) -> Path:
        return (
            Path(__file__).parent.parent
            / "src"
            / "pneuma_core"
            / "protocols"
        )

    def test_llm_protocol_file(self, protocols_dir: Path) -> None:
        assert (protocols_dir / "llm.py").exists()

    def test_storage_protocol_file(self, protocols_dir: Path) -> None:
        assert (protocols_dir / "storage.py").exists()

    def test_memory_store_protocol_file(self, protocols_dir: Path) -> None:
        assert (protocols_dir / "memory_store.py").exists()

    def test_embedding_protocol_file(self, protocols_dir: Path) -> None:
        assert (protocols_dir / "embedding.py").exists()

    def test_voice_protocol_file(self, protocols_dir: Path) -> None:
        assert (protocols_dir / "voice.py").exists()

    def test_task_protocol_file(self, protocols_dir: Path) -> None:
        assert (protocols_dir / "task.py").exists()

    def test_middleware_protocol_file(self, protocols_dir: Path) -> None:
        assert (protocols_dir / "middleware.py").exists()


# =============================================================================
# LLMTimeoutError
# =============================================================================


class TestLLMTimeoutError:
    """LLMTimeoutError がコアの例外として定義されていること."""

    def test_importable(self) -> None:
        """LLMTimeoutError がインポート可能."""
        from pneuma_core.exceptions import LLMTimeoutError

        assert LLMTimeoutError is not None

    def test_is_exception(self) -> None:
        """LLMTimeoutError が Exception のサブクラス."""
        from pneuma_core.exceptions import LLMTimeoutError

        assert issubclass(LLMTimeoutError, Exception)

    def test_can_raise_and_catch(self) -> None:
        """raise / catch が正常に動作する."""
        from pneuma_core.exceptions import LLMTimeoutError

        with pytest.raises(LLMTimeoutError):
            raise LLMTimeoutError("timeout")


# =============================================================================
# anthropic 依存の排除
# =============================================================================


class TestNoAnthropicInCore:
    """core/ から anthropic SDK への直接依存が排除されていること."""

    def _get_core_python_files(self) -> list[Path]:
        """core/ 配下の全 .py ファイルを取得（claude.py を除く）."""
        core_dir = (
            Path(__file__).parent.parent / "src" / "pneuma_core"
        )
        files = list(core_dir.rglob("*.py"))
        # ClaudeAdapter は anthropic を使って良い
        return [f for f in files if f.name != "claude.py"]

    def test_no_anthropic_import_in_core(self) -> None:
        """core/ の Python ファイル（claude.py 除く）に anthropic import がない."""
        for py_file in self._get_core_python_files():
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("anthropic"), (
                            f"{py_file.name} has 'import {alias.name}'"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("anthropic"):
                        pytest.fail(
                            f"{py_file.name} has 'from {node.module} import ...'"
                        )


# =============================================================================
# 後方互換 re-export
# =============================================================================


class TestBackwardCompatReExports:
    """旧パスからの re-export が動作すること."""

    def test_llm_adapter_from_old_path(self) -> None:
        """旧パス pneuma.core.llm.adapter から LLMAdapter がインポート可能."""
        from pneuma_core.llm.adapter import LLMAdapter

        assert LLMAdapter is not None

    def test_storage_backend_from_old_path(self) -> None:
        """旧パスから StorageBackend がインポート可能."""
        from pneuma_core.storage.backend import StorageBackend

        assert StorageBackend is not None

    def test_memory_store_from_old_path(self) -> None:
        """旧パスから MemoryStore がインポート可能."""
        from pneuma_core.memory.store import MemoryStore

        assert MemoryStore is not None

    def test_embedding_from_old_path(self) -> None:
        """旧パスから EmbeddingService がインポート可能."""
        from pneuma_core.llm.embedding import EmbeddingService

        assert EmbeddingService is not None

    def test_voice_protocols_from_old_path(self) -> None:
        """旧パスから TTSAdapter, STTService がインポート可能."""
        from pneuma_core.voice.tts import TTSAdapter
        from pneuma_core.voice.stt import STTService

        assert TTSAdapter is not None
        assert STTService is not None

    def test_task_backend_from_old_path(self) -> None:
        """旧パスから TaskBackend がインポート可能."""
        from pneuma_core.task.backend import TaskBackend

        assert TaskBackend is not None

    def test_middleware_from_old_path(self) -> None:
        """旧パスから Middleware がインポート可能."""
        from pneuma_core.runtime.middleware import Middleware

        assert Middleware is not None


# =============================================================================
# パイプラインテスト: Protocol の同一性
# =============================================================================


class TestProtocolIdentity:
    """新旧パスの Protocol が同一オブジェクトであること."""

    def test_llm_adapter_identity(self) -> None:
        from pneuma_core.protocols import LLMAdapter as new
        from pneuma_core.llm.adapter import LLMAdapter as old

        assert new is old

    def test_storage_backend_identity(self) -> None:
        from pneuma_core.protocols import StorageBackend as new
        from pneuma_core.storage.backend import StorageBackend as old

        assert new is old

    def test_memory_store_identity(self) -> None:
        from pneuma_core.protocols import MemoryStore as new
        from pneuma_core.memory.store import MemoryStore as old

        assert new is old

    def test_embedding_service_identity(self) -> None:
        from pneuma_core.protocols import EmbeddingService as new
        from pneuma_core.llm.embedding import EmbeddingService as old

        assert new is old
