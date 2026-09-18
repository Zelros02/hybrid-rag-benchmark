"""Stage 1 — download BEIR datasets, chunk, encode, and index into Weaviate.

Run after `docker compose up -d weaviate`.

    python -m scripts.index_corpus                   # all 3 datasets, default chunk_size
    python -m scripts.index_corpus --recreate        # drop and rebuild collections
    python -m scripts.index_corpus --chunk-size 256  # custom chunk size
    python -m scripts.index_corpus --datasets nfcorpus scifact
"""
from __future__ import annotations

import argparse
import sys
import time

from src import data_loader
from src.config import DATASETS, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_RATIO, IndexConfig
from src.indexing import get_client, get_embedder, index_dataset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Index BEIR datasets into Weaviate.")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    p.add_argument("--recreate", action="store_true", help="Drop and rebuild target collections")
    args = p.parse_args(argv)

    print(f"Loading sentence-transformer embedder…")
    embedder = get_embedder()

    client = get_client()
    try:
        for ds in args.datasets:
            cfg = IndexConfig(dataset=ds, chunk_size=args.chunk_size, overlap_ratio=args.overlap_ratio)
            print(f"\n=== {ds} (chunk_size={cfg.chunk_size}, overlap={cfg.overlap_ratio}) ===")
            t0 = time.time()
            print("  loading BEIR data…")
            corpus, queries, qrels = data_loader.load(ds)
            print(f"  corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")
            n = index_dataset(client, corpus, cfg, embedder, recreate=args.recreate)
            print(f"  indexed {n} chunks into {cfg.collection_name} in {time.time()-t0:.1f}s")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
