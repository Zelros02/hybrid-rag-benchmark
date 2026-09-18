"""Stage 4 — sweep chunk_size, k, and alpha on a single dataset.

For each hyperparameter, holds the others at their defaults and evaluates nDCG.
Produces three plots and one CSV with all results.

Chunk-size sweep requires re-indexing (each value gets its own Weaviate collection).
The k and alpha sweeps reuse the default-chunk-size collection.

With `--rag-qualitative`, also regenerates RAG answers at each sweep value on the
same 10 queries used by `scripts.rag_eval`, so you can compare the actual LLM
outputs as hyperparameters change (this requires Ollama to be running).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src import data_loader, metrics
from src.config import (
    DEFAULT_ALPHA,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_RATIO,
    DEFAULT_TOP_K,
    OLLAMA_MODEL,
    RESULTS_DIR,
    IndexConfig,
)
from src.indexing import get_client, get_embedder, index_dataset
from src.rag import generate
from src.retrieval import Retriever, make_retriever

# Reuse the deterministic query selection so sweeps stay comparable run to run.
from scripts.rag_eval import pick_queries


def evaluate_collection(client, embedder, collection_name, queries, qrels, retriever, k, alpha):
    collection = client.collections.get(collection_name)
    retrieve = make_retriever(retriever, embedder, alpha=alpha)
    runs = {}
    for qid, qtext in tqdm(queries.items(), desc=collection_name, unit="q", leave=False):
        if qid not in qrels:
            continue
        ranked = retrieve(collection, qtext, k)
        runs[qid] = [r.doc_id for r in ranked]
    return metrics.aggregate(runs, qrels, k=k)


def rag_on_picks(client, embedder, collection_name, picks, retriever, k, alpha, model):
    """Generate RAG answers for the fixed 10-query subset. Returns list of dicts."""
    import ollama  # noqa: F401  # imported lazily so the script works without ollama installed for retrieval-only runs

    collection = client.collections.get(collection_name)
    retrieve = make_retriever(retriever, embedder, alpha=alpha)
    outputs = []
    for qid, qtext in picks:
        passages = retrieve(collection, qtext, k)
        result = generate(qtext, passages, model=model)
        outputs.append(
            {
                "qid": qid,
                "query": qtext,
                "retrieved_doc_ids": [p.doc_id for p in passages],
                "answer": result.answer,
            }
        )
    return outputs


def sweep_chunk_size(
    client, embedder, dataset, corpus, queries, qrels, picks, values: List[int],
    retriever, k, alpha, rag_model: Optional[str],
):
    rows, rag_runs = [], []
    for cs in values:
        cfg = IndexConfig(dataset=dataset, chunk_size=cs, overlap_ratio=DEFAULT_OVERLAP_RATIO)
        if not client.collections.exists(cfg.collection_name):
            print(f"  [build] {cfg.collection_name} (chunk_size={cs})")
            index_dataset(client, corpus, cfg, embedder)
        agg = evaluate_collection(client, embedder, cfg.collection_name, queries, qrels, retriever, k, alpha)
        agg["metric_k"] = k
        rows.append({"param": "chunk_size", "value": cs, **agg})
        print(f"  chunk_size={cs}  ndcg@{k}={agg['ndcg@k']:.4f}")
        if rag_model:
            outputs = rag_on_picks(client, embedder, cfg.collection_name, picks, retriever, k, alpha, rag_model)
            rag_runs.append({"param": "chunk_size", "value": cs, "answers": outputs})
    return rows, rag_runs


def sweep_k(
    client, embedder, dataset, queries, qrels, picks, values: List[int],
    retriever, chunk_size, alpha, rag_model: Optional[str],
):
    cfg = IndexConfig(dataset=dataset, chunk_size=chunk_size)
    rows, rag_runs = [], []
    for kv in values:
        agg = evaluate_collection(client, embedder, cfg.collection_name, queries, qrels, retriever, kv, alpha)
        agg["metric_k"] = kv
        rows.append({"param": "k", "value": kv, **agg})
        print(f"  k={kv}  ndcg@{kv}={agg['ndcg@k']:.4f}")
        if rag_model:
            outputs = rag_on_picks(client, embedder, cfg.collection_name, picks, retriever, kv, alpha, rag_model)
            rag_runs.append({"param": "k", "value": kv, "answers": outputs})
    return rows, rag_runs


def sweep_alpha(
    client, embedder, dataset, queries, qrels, picks, values: List[float],
    chunk_size, k, rag_model: Optional[str],
):
    cfg = IndexConfig(dataset=dataset, chunk_size=chunk_size)
    rows, rag_runs = [], []
    for a in values:
        agg = evaluate_collection(client, embedder, cfg.collection_name, queries, qrels, Retriever.HYBRID, k, a)
        agg["metric_k"] = k
        rows.append({"param": "alpha", "value": a, **agg})
        print(f"  alpha={a}  ndcg@{k}={agg['ndcg@k']:.4f}")
        if rag_model:
            outputs = rag_on_picks(client, embedder, cfg.collection_name, picks, Retriever.HYBRID, k, a, rag_model)
            rag_runs.append({"param": "alpha", "value": a, "answers": outputs})
    return rows, rag_runs


def plot_sweep(df: pd.DataFrame, param: str, out_path: str, y_label: str):
    sub = df[df["param"] == param].sort_values("value")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sub["value"], sub["ndcg@k"], marker="o")
    ax.set_xlabel(param)
    ax.set_ylabel(y_label)
    ax.set_title(f"Sweep: {param}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Hyperparameter sweep on chunk_size, k, alpha.")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--retriever", default="hybrid", choices=[r.value for r in Retriever])
    p.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 512, 1024])
    p.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--default-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--default-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--default-alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--skip", nargs="*", default=[], choices=["chunk_size", "k", "alpha"])
    p.add_argument("--rag-qualitative", action="store_true",
                   help="Also run RAG generation on 10 fixed queries at each sweep value (slow; needs Ollama).")
    p.add_argument("--rag-model", default=OLLAMA_MODEL)
    p.add_argument("--n-queries", type=int, default=10)
    args = p.parse_args(argv)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    embedder = get_embedder()
    client = get_client()
    retriever = Retriever(args.retriever)
    rag_model = args.rag_model if args.rag_qualitative else None

    print(f"Loading {args.dataset}…")
    corpus, queries, qrels = data_loader.load(args.dataset)
    picks = pick_queries(queries, qrels, args.n_queries)

    all_rows = []
    all_rag = []
    try:
        if "chunk_size" not in args.skip:
            print("\n=== chunk_size sweep ===")
            r, rr = sweep_chunk_size(
                client, embedder, args.dataset, corpus, queries, qrels, picks,
                args.chunk_sizes, retriever, args.default_k, args.default_alpha, rag_model,
            )
            all_rows += r
            all_rag += rr
        if "k" not in args.skip:
            print("\n=== k sweep ===")
            r, rr = sweep_k(
                client, embedder, args.dataset, queries, qrels, picks,
                args.ks, retriever, args.default_chunk_size, args.default_alpha, rag_model,
            )
            all_rows += r
            all_rag += rr
        if "alpha" not in args.skip:
            print("\n=== alpha sweep (hybrid) ===")
            r, rr = sweep_alpha(
                client, embedder, args.dataset, queries, qrels, picks,
                args.alphas, args.default_chunk_size, args.default_k, rag_model,
            )
            all_rows += r
            all_rag += rr
    finally:
        client.close()

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep results to {csv_path}")

    label_for = {
        "chunk_size": f"nDCG@{args.default_k}",
        "k": "nDCG@k (varying)",
        "alpha": f"nDCG@{args.default_k}",
    }
    for param in ["chunk_size", "k", "alpha"]:
        if param in args.skip:
            continue
        out = os.path.join(RESULTS_DIR, f"sweep_{param}.png")
        plot_sweep(df, param, out, label_for[param])
        print(f"  plot -> {out}")

    if all_rag:
        rag_path = os.path.join(RESULTS_DIR, "sweep_rag_answers.json")
        with open(rag_path, "w") as f:
            json.dump({"config": vars(args), "sweeps": all_rag}, f, indent=2)
        print(f"  RAG qualitative outputs -> {rag_path}")

    print("\n=== Sweep table ===")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
