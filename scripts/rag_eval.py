"""Stage 3 — end-to-end RAG on a single dataset.

Pick a dataset (default scifact since it's small + interesting). Use the best retriever
(default hybrid) and Ollama to answer 10 test queries with citations. Save the prompt,
retrieved passages, and generated answer for manual review.

The script always picks the *same* 10 queries given the same dataset (deterministic
sampling sorted by query id) so manual annotation stays reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src import data_loader
from src.config import (
    DEFAULT_ALPHA,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    OLLAMA_MODEL,
    RESULTS_DIR,
    IndexConfig,
)
from src.indexing import get_client, get_embedder
from src.rag import generate
from src.retrieval import Retriever, make_retriever


def pick_queries(queries: dict, qrels: dict, n: int) -> list[tuple[str, str]]:
    """Deterministic: take the first n queries (sorted by id) that have qrels."""
    candidates = sorted(qid for qid in queries if qid in qrels and any(g > 0 for g in qrels[qid].values()))
    return [(qid, queries[qid]) for qid in candidates[:n]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="End-to-end RAG evaluation on 10 queries.")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--retriever", default="hybrid", choices=[r.value for r in Retriever])
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--qids", nargs="+", default=None,
                   help="Run only these specific query ids (overrides --n-queries).")
    p.add_argument("--model", default=OLLAMA_MODEL)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "rag_answers.json"))
    args = p.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading {args.dataset}…")
    _, queries, qrels = data_loader.load(args.dataset)
    if args.qids:
        missing = [q for q in args.qids if q not in queries]
        if missing:
            raise SystemExit(f"qids not in dataset: {missing}")
        picks = [(q, queries[q]) for q in args.qids]
    else:
        picks = pick_queries(queries, qrels, args.n_queries)
    print(f"Selected {len(picks)} queries.")

    embedder = get_embedder()
    client = get_client()
    cfg = IndexConfig(dataset=args.dataset, chunk_size=args.chunk_size)
    if not client.collections.exists(cfg.collection_name):
        client.close()
        raise SystemExit(f"Collection {cfg.collection_name} missing. Run scripts/index_corpus.py first.")
    collection = client.collections.get(cfg.collection_name)
    retrieve = make_retriever(Retriever(args.retriever), embedder, alpha=args.alpha)

    runs = []
    try:
        for qid, qtext in picks:
            print(f"\n--- Q{qid}: {qtext}")
            passages = retrieve(collection, qtext, args.k)
            result = generate(qtext, passages, model=args.model)
            relevant = sorted(d for d, g in qrels.get(qid, {}).items() if g > 0)
            retrieved_ids = [p.doc_id for p in passages]
            print(f"Retrieved doc_ids: {retrieved_ids}")
            print(f"Relevant doc_ids : {relevant}")
            print(f"Answer:\n{result.answer}")
            runs.append(
                {
                    "qid": qid,
                    "query": qtext,
                    "relevant_doc_ids": relevant,
                    "retrieved": [
                        {
                            "doc_id": p.doc_id,
                            "chunk_idx": p.best_chunk_idx,
                            "score": p.score,
                            "text": p.best_chunk_text,
                        }
                        for p in passages
                    ],
                    "answer": result.answer,
                    "prompt": result.prompt,
                }
            )
    finally:
        client.close()

    with open(args.out, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "runs": runs,
            },
            f,
            indent=2,
        )
    print(f"\nSaved RAG runs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
