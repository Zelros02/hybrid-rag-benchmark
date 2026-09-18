"""Stage 2 — compare BM25, dense, and hybrid retrievers across all 3 datasets.

Outputs a metrics table (printed + saved as CSV) with rows = retriever and columns
= (dataset × metric). Reports Recall@k, MRR@k and nDCG@k at k=10 by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import pandas as pd
from tqdm import tqdm

from src import data_loader, metrics
from src.config import (
    DATASETS,
    DEFAULT_ALPHA,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    RESULTS_DIR,
    IndexConfig,
)
from src.indexing import get_client, get_embedder
from src.retrieval import Retriever, make_retriever


def run_retriever_on_dataset(
    client,
    embedder,
    dataset: str,
    retriever_name: Retriever,
    chunk_size: int,
    k: int,
    alpha: float,
) -> Dict[str, float]:
    cfg = IndexConfig(dataset=dataset, chunk_size=chunk_size)
    if not client.collections.exists(cfg.collection_name):
        raise SystemExit(
            f"Collection {cfg.collection_name} missing. Run scripts/index_corpus.py first."
        )
    collection = client.collections.get(cfg.collection_name)

    _, queries, qrels = data_loader.load(dataset)
    retrieve = make_retriever(retriever_name, embedder, alpha=alpha)

    runs: Dict[str, list] = {}
    for qid, qtext in tqdm(queries.items(), desc=f"{dataset}/{retriever_name.value}", unit="q"):
        if qid not in qrels:
            continue
        ranked = retrieve(collection, qtext, k)
        runs[qid] = [r.doc_id for r in ranked]

    agg = metrics.aggregate(runs, qrels, k=k)
    return agg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare BM25 / dense / hybrid retrievers.")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Hybrid alpha (0=BM25, 1=dense)")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "retriever_comparison.csv"))
    args = p.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading embedder…")
    embedder = get_embedder()
    client = get_client()

    rows = []
    raw = {}
    try:
        for ds in args.datasets:
            for retr in [Retriever.BM25, Retriever.DENSE, Retriever.HYBRID]:
                t0 = time.time()
                agg = run_retriever_on_dataset(
                    client, embedder, ds, retr, args.chunk_size, args.k, args.alpha
                )
                elapsed = time.time() - t0
                rows.append(
                    {
                        "dataset": ds,
                        "retriever": retr.value,
                        "recall@k": agg["recall@k"],
                        "mrr@k": agg["mrr@k"],
                        "ndcg@k": agg["ndcg@k"],
                        "n_queries": agg["n_queries"],
                        "elapsed_s": elapsed,
                    }
                )
                raw[f"{ds}/{retr.value}"] = agg
                print(
                    f"{ds:9s} {retr.value:6s}  recall@{args.k}={agg['recall@k']:.4f} "
                    f"mrr@{args.k}={agg['mrr@k']:.4f} ndcg@{args.k}={agg['ndcg@k']:.4f} "
                    f"({agg['n_queries']} q, {elapsed:.1f}s)"
                )
    finally:
        client.close()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved per-row metrics to {args.out}")

    pivot = df.pivot(index="retriever", columns="dataset", values=["recall@k", "mrr@k", "ndcg@k"])
    pivot_out = args.out.replace(".csv", "_pivot.csv")
    pivot.to_csv(pivot_out)
    print(f"Saved pivoted table to {pivot_out}")
    print("\n=== Comparison table ===")
    print(pivot.round(4).to_string())

    raw_out = args.out.replace(".csv", ".json")
    with open(raw_out, "w") as f:
        json.dump(raw, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
