"""Weaviate connection, schema, and batch indexing."""
from __future__ import annotations

from typing import Iterable, Iterator, List

import numpy as np
import weaviate
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from weaviate.classes.config import Configure, DataType, Property, Tokenization

from src.chunking import Chunk, chunk_corpus
from src.config import (
    EMBEDDING_MODEL,
    IndexConfig,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
)


def get_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_HTTP_PORT,
        grpc_port=WEAVIATE_GRPC_PORT,
    )


def get_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def ensure_collection(client: weaviate.WeaviateClient, name: str, recreate: bool = False) -> None:
    exists = client.collections.exists(name)
    if exists and recreate:
        client.collections.delete(name)
        exists = False
    if exists:
        return
    client.collections.create(
        name=name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(
                name="doc_id",
                data_type=DataType.TEXT,
                skip_vectorization=True,
                tokenization=Tokenization.FIELD,
            ),
            Property(name="chunk_idx", data_type=DataType.INT, skip_vectorization=True),
        ],
    )


def _batched(iterable: Iterable[Chunk], batch_size: int) -> Iterator[List[Chunk]]:
    batch: List[Chunk] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_dataset(
    client: weaviate.WeaviateClient,
    corpus: dict,
    cfg: IndexConfig,
    embedder: SentenceTransformer,
    encode_batch_size: int = 64,
    insert_batch_size: int = 256,
    recreate: bool = False,
) -> int:
    """Chunk, encode, and insert a corpus into Weaviate. Returns chunk count."""
    ensure_collection(client, cfg.collection_name, recreate=recreate)
    collection = client.collections.get(cfg.collection_name)

    if not recreate:
        existing = collection.aggregate.over_all(total_count=True).total_count
        if existing and existing > 0:
            print(f"  [skip] {cfg.collection_name} already populated with {existing} objects")
            return existing

    tokenizer = embedder.tokenizer
    chunk_iter = chunk_corpus(corpus, tokenizer, cfg.chunk_size, cfg.overlap_ratio)

    total = 0
    pbar = tqdm(desc=f"index {cfg.collection_name}", unit="chunk")
    with collection.batch.dynamic() as batch:
        for chunk_batch in _batched(chunk_iter, encode_batch_size):
            texts = [c.text for c in chunk_batch]
            vectors = embedder.encode(
                texts,
                batch_size=encode_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for chunk, vec in zip(chunk_batch, vectors):
                batch.add_object(
                    properties={
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "chunk_idx": chunk.chunk_idx,
                    },
                    vector=np.asarray(vec, dtype=np.float32).tolist(),
                )
                total += 1
            pbar.update(len(chunk_batch))
    pbar.close()
    return total
