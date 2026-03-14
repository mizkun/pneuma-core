"""Consolidated Protocol definitions for pneuma.core."""

from pneuma_core.protocols.embedding import EmbeddingService
from pneuma_core.protocols.llm import LLMAdapter, LLMRequest, LLMResponse, ModelConfig
from pneuma_core.protocols.memory_store import MemoryStore
from pneuma_core.protocols.middleware import Middleware, PipelineContext
from pneuma_core.protocols.storage import StorageBackend
from pneuma_core.protocols.task import TaskBackend
from pneuma_core.protocols.voice import STTService, TTSAdapter

__all__ = [
    "EmbeddingService",
    "LLMAdapter",
    "LLMRequest",
    "LLMResponse",
    "MemoryStore",
    "Middleware",
    "ModelConfig",
    "PipelineContext",
    "STTService",
    "StorageBackend",
    "TTSAdapter",
    "TaskBackend",
]
