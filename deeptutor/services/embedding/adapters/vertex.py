"""Embeddings via a Vertex AI publisher ``:predict`` endpoint.

Reuses the OpenAI-compatible transport (retries, mTLS, auth header); only the
request and response shapes differ. Use the full endpoint URL, e.g.
``.../publishers/google/models/text-embedding-005:predict``.
"""

from __future__ import annotations

from typing import Any

from .base import EmbeddingRequest
from .openai_compatible import OpenAICompatibleEmbeddingAdapter


class VertexPredictEmbeddingAdapter(OpenAICompatibleEmbeddingAdapter):
    SUPPORTS_INPUT_TYPE = True
    MODELS_INFO = {
        "text-embedding-005": {"default": 768, "dimensions": [256, 768]},
        "gemini-embedding-001": {"default": 3072, "dimensions": [768, 1536, 3072]},
    }
    _TASK_TYPES = {"search_query": "RETRIEVAL_QUERY", "search_document": "RETRIEVAL_DOCUMENT"}

    def _build_payload(self, request: EmbeddingRequest) -> dict[str, Any]:
        if request.contents:
            raise ValueError("Vertex :predict embeddings here are text-only.")
        task = self._TASK_TYPES.get(request.input_type or "")
        payload: dict[str, Any] = {
            "instances": [
                {"content": text, **({"task_type": task} if task else {})}
                for text in request.texts
            ]
        }
        if dim := request.dimensions or self.dimensions:
            payload["parameters"] = {"outputDimensionality": dim}
        return payload

    @staticmethod
    def _extract_embeddings_from_response(data: Any) -> list[list[float]]:
        rows = data.get("predictions") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("Vertex embedding response is missing a `predictions` list.")
        return [row["embeddings"]["values"] for row in rows]

    def get_model_info(self) -> dict[str, Any]:
        return {**super().get_model_info(), "provider": "vertex"}
