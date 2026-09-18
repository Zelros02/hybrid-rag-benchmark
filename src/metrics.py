"""IR metrics: Recall@k, MRR@k, nDCG@k.

All inputs are document-level: `retrieved` is a ranked list of `doc_id`s; `qrels`
is the per-query relevance dict `{doc_id: graded_relevance}`. A document with
relevance > 0 is treated as relevant for Recall and MRR; nDCG uses graded gains.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping


def recall_at_k(retrieved: List[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, g in qrels.items() if g > 0}
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def mrr_at_k(retrieved: List[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = {d for d, g in qrels.items() if g > 0}
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], qrels: Mapping[str, int], k: int) -> float:
    gains = [qrels.get(d, 0) for d in retrieved[:k]]
    dcg = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains, start=1) if g > 0)
    ideal_gains = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(rank + 1) for rank, g in enumerate(ideal_gains, start=1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def aggregate(
    per_query_runs: Mapping[str, List[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """Average each metric across queries that have at least one relevance judgement."""
    queries = [qid for qid in per_query_runs if qid in qrels and any(g > 0 for g in qrels[qid].values())]
    if not queries:
        return {"recall@k": 0.0, "mrr@k": 0.0, "ndcg@k": 0.0, "n_queries": 0}
    r = sum(recall_at_k(per_query_runs[qid], qrels[qid], k) for qid in queries) / len(queries)
    m = sum(mrr_at_k(per_query_runs[qid], qrels[qid], k) for qid in queries) / len(queries)
    n = sum(ndcg_at_k(per_query_runs[qid], qrels[qid], k) for qid in queries) / len(queries)
    return {"recall@k": r, "mrr@k": m, "ndcg@k": n, "n_queries": len(queries)}
