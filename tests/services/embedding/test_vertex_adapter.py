from deeptutor.services.embedding.adapters import ADAPTER_BACKENDS
from deeptutor.services.embedding.adapters.base import EmbeddingRequest


def _adapter():
    return ADAPTER_BACKENDS["vertex"]({"model": "text-embedding-005", "dimensions": 256})


def test_payload_maps_task_type_and_dimension():
    req = EmbeddingRequest(texts=["a", "b"], model="text-embedding-005", input_type="search_query")
    assert _adapter()._build_payload(req) == {
        "instances": [
            {"content": "a", "task_type": "RETRIEVAL_QUERY"},
            {"content": "b", "task_type": "RETRIEVAL_QUERY"},
        ],
        "parameters": {"outputDimensionality": 256},
    }


def test_extracts_predictions():
    data = {"predictions": [{"embeddings": {"values": [0.1, 0.2]}}]}
    assert _adapter()._extract_embeddings_from_response(data) == [[0.1, 0.2]]
