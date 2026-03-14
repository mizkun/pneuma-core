"""LLM adapters: LLMAdapter protocol, Claude, OpenAI, Ollama.

Protocol classes are always available. Concrete adapters (ClaudeAdapter,
OpenAIEmbeddingService) require optional dependencies (anthropic, openai).
"""

from pneuma_core.llm.adapter import LLMAdapter, LLMRequest, LLMResponse, ModelConfig

# Lazy imports for concrete adapters to avoid hard dependency on anthropic/openai


def __getattr__(name: str):
    if name == "ClaudeAdapter":
        from pneuma_core.llm.claude import ClaudeAdapter

        return ClaudeAdapter
    if name == "EmbeddingService":
        from pneuma_core.protocols.embedding import EmbeddingService

        return EmbeddingService
    if name == "OpenAIEmbeddingService":
        from pneuma_core.llm.embedding import OpenAIEmbeddingService

        return OpenAIEmbeddingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClaudeAdapter",
    "EmbeddingService",
    "LLMAdapter",
    "LLMRequest",
    "LLMResponse",
    "ModelConfig",
    "OpenAIEmbeddingService",
]
