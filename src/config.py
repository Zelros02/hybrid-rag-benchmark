"""Central configuration for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass

BEIR_URL_TEMPLATE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

DATASETS = ["nfcorpus", "scifact", "fiqa"]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP_RATIO = 0.1
DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.5

WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8080
WEAVIATE_GRPC_PORT = 50051

OLLAMA_MODEL = "llama3.2:3b"

DATASETS_DIR = "datasets"
RESULTS_DIR = "results"


@dataclass(frozen=True)
class IndexConfig:
    dataset: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO

    @property
    def collection_name(self) -> str:
        # Weaviate class names must start with uppercase letter.
        return f"{self.dataset.capitalize()}_c{self.chunk_size}"
