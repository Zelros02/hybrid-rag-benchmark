"""Demo end-to-end: one query (qid 1012, SciFact radioiodine claim) through
BM25, dense, and hybrid retrieval, with Llama 3.2 3B generating a RAG answer
with citations for each retriever.

A single query traversing the full pipeline, for a quick end-to-end sanity
check. Equivalent to running:
    python -m scripts.rag_eval --dataset scifact --retriever bm25   --qids 1012
    python -m scripts.rag_eval --dataset scifact --retriever dense  --qids 1012
    python -m scripts.rag_eval --dataset scifact --retriever hybrid --qids 1012
but in a single process so the embedder and Weaviate client are loaded only once.

Run as:
    python -m scripts.demo
"""
from __future__ import annotations

import sys
import time

from src import data_loader
from src.config import IndexConfig, OLLAMA_MODEL
from src.indexing import get_client, get_embedder
from src.rag import generate
from src.retrieval import Retriever, make_retriever

DATASET = "scifact"
QID = "1012"  # Radioiodine treatment claim, produces multi-citation RAG answers.
K = 10
ALPHA = 0.5


def _print_retrieval(name: str, results, took_ms: float, relevant: list[str]) -> None:
    print(f"\n--- {name.upper()} (top-{K}, retrieval {took_ms:.0f} ms) ---")
    ids = [r.doc_id for r in results]
    print(f"  Retrieved doc_ids: {ids}")
    print(f"  Relevant doc_ids : {relevant}")
    hits = [i + 1 for i, did in enumerate(ids) if did in relevant]
    print(f"  Relevant hits at ranks: {hits if hits else 'none in top-K'}")


def main() -> int:
    print(f"=== Demo: {DATASET} / qid={QID} ===")

    _, queries, qrels = data_loader.load(DATASET)
    if QID not in queries:
        print(f"ERROR: qid {QID} not found in {DATASET}.")
        return 1
    query = queries[QID]
    relevant = sorted(d for d, g in qrels.get(QID, {}).items() if g > 0)
    print(f"\nQuery: {query}")
    print(f"Relevant per qrels: {relevant}")

    embedder = get_embedder()
    client = get_client()
    try:
        cfg = IndexConfig(dataset=DATASET)
        if not client.collections.exists(cfg.collection_name):
            print(f"ERROR: collection {cfg.collection_name} not found. Run scripts.index_corpus first.")
            return 1
        collection = client.collections.get(cfg.collection_name)

        for name in [Retriever.BM25, Retriever.DENSE, Retriever.HYBRID]:
            retrieve = make_retriever(name, embedder, alpha=ALPHA)
            t0 = time.time()
            passages = retrieve(collection, query, K)
            ret_ms = (time.time() - t0) * 1000
            _print_retrieval(name.value, passages, ret_ms, relevant)

            t0 = time.time()
            result = generate(query, passages, model=OLLAMA_MODEL)
            gen_s = time.time() - t0
            print(f"\n  RAG answer ({name.value}, Llama 3.2 3B, {gen_s:.1f}s generation):")
            print("  " + result.answer.replace("\n", "\n  "))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
