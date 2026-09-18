"""Retrieval primitives over Weaviate collections.

All three retrievers (BM25, dense, hybrid) return ranked *document* lists. Internally
the index stores chunks, so we retrieve K * `chunk_multiplier` chunks and dedupe by
`doc_id`, keeping the best chunk score per document. This is the standard practice
when evaluating chunk-level indices against document-level qrels.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

import numpy as np
import weaviate
from weaviate.classes.query import MetadataQuery
from weaviate.collections import Collection


class Retriever(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass
class RankedDoc:
    doc_id: str
    score: float
    best_chunk_text: str
    best_chunk_idx: int


def _dedupe_by_doc(items, k: int) -> List[RankedDoc]:
    """Keep the best-scoring chunk per doc_id, return the top-k unique docs."""
    seen: dict[str, RankedDoc] = {}
    for doc_id, score, text, chunk_idx in items:
        cur = seen.get(doc_id)
        if cur is None or score > cur.score:
            seen[doc_id] = RankedDoc(
                doc_id=doc_id,
                score=float(score),
                best_chunk_text=text,
                best_chunk_idx=int(chunk_idx),
            )
    ranked = sorted(seen.values(), key=lambda d: d.score, reverse=True)
    return ranked[:k]


def _materialize(result) -> list:
    """Convert a weaviate result object into (doc_id, score, text, chunk_idx) tuples."""
    out = []
    for obj in result.objects:
        meta = obj.metadata
        # BM25 and Hybrid both return `score`; near_vector returns `distance` (and `certainty` if cosine).
        if meta.score is not None:
            score = meta.score
        elif meta.certainty is not None:
            score = meta.certainty
        elif meta.distance is not None:
            # Convert distance to similarity (assuming cosine distance in [0,2])
            score = 1.0 - meta.distance
        else:
            score = 0.0
        props = obj.properties
        out.append((props["doc_id"], score, props.get("text", ""), props.get("chunk_idx", 0)))
    return out


def bm25_search(collection: Collection, query: str, k: int, chunk_multiplier: int = 8) -> List[RankedDoc]:
    res = collection.query.bm25(
        query=query,
        limit=k * chunk_multiplier,
        return_metadata=MetadataQuery(score=True),
    )
    return _dedupe_by_doc(_materialize(res), k)


def dense_search(
    collection: Collection, query_vector: np.ndarray, k: int, chunk_multiplier: int = 8
) -> List[RankedDoc]:
    res = collection.query.near_vector(
        near_vector=np.asarray(query_vector, dtype=np.float32).tolist(),
        limit=k * chunk_multiplier,
        return_metadata=MetadataQuery(distance=True, certainty=True),
    )
    return _dedupe_by_doc(_materialize(res), k)


def hybrid_search(
    collection: Collection,
    query: str,
    query_vector: np.ndarray,
    alpha: float,
    k: int,
    chunk_multiplier: int = 8,
) -> List[RankedDoc]:
    res = collection.query.hybrid(
        query=query,
        vector=np.asarray(query_vector, dtype=np.float32).tolist(),
        alpha=alpha,
        limit=k * chunk_multiplier,
        return_metadata=MetadataQuery(score=True),
    )
    return _dedupe_by_doc(_materialize(res), k)


def make_retriever(
    name: Retriever,
    embedder,
    alpha: float = 0.5,
) -> Callable[[Collection, str, int], List[RankedDoc]]:
    """Return a unified retriever callable `(collection, query, k) -> List[RankedDoc]`."""

    def _encode(q: str) -> np.ndarray:
        return embedder.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]

    if name == Retriever.BM25:
        return lambda col, q, k: bm25_search(col, q, k)
    if name == Retriever.DENSE:
        return lambda col, q, k: dense_search(col, _encode(q), k)
    if name == Retriever.HYBRID:
        return lambda col, q, k: hybrid_search(col, q, _encode(q), alpha, k)
    raise ValueError(f"Unknown retriever: {name}")
