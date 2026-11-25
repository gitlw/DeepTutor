"""Embedding adapter implementations and backend registry."""

from .base import (
    BaseEmbeddingAdapter,
    EmbeddingProviderError,
    EmbeddingRequest,
    EmbeddingResponse,
)
from .cohere import CohereEmbeddingAdapter
from .dashscope_native import DashScopeMultiModalEmbeddingAdapter
from .gemini import GeminiEmbeddingAdapter
from .jina import JinaEmbeddingAdapter
from .nvidia_nim import NvidiaNIMEmbeddingAdapter
from .ollama import OllamaEmbeddingAdapter
from .openai_compatible import OpenAICompatibleEmbeddingAdapter
from .openai_sdk import OpenAISDKEmbeddingAdapter
from .vertex import VertexPredictEmbeddingAdapter

ADAPTER_BACKENDS: dict[str, type[BaseEmbeddingAdapter]] = {
    "openai_compat": OpenAICompatibleEmbeddingAdapter,
    "openai_sdk": OpenAISDKEmbeddingAdapter,
    "cohere": CohereEmbeddingAdapter,
    "jina": JinaEmbeddingAdapter,
    "ollama": OllamaEmbeddingAdapter,
    "dashscope_native": DashScopeMultiModalEmbeddingAdapter,
    "gemini": GeminiEmbeddingAdapter,
    "nvidia_nim": NvidiaNIMEmbeddingAdapter,
    "vertex": VertexPredictEmbeddingAdapter,
}

__all__ = [
    "ADAPTER_BACKENDS",
    "BaseEmbeddingAdapter",
    "EmbeddingProviderError",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "OpenAICompatibleEmbeddingAdapter",
    "OpenAISDKEmbeddingAdapter",
    "DashScopeMultiModalEmbeddingAdapter",
    "GeminiEmbeddingAdapter",
    "NvidiaNIMEmbeddingAdapter",
    "VertexPredictEmbeddingAdapter",
    "JinaEmbeddingAdapter",
    "CohereEmbeddingAdapter",
    "OllamaEmbeddingAdapter",
]
