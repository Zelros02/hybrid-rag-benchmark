"""BEIR dataset download and loading."""
from __future__ import annotations

import os
from typing import Dict, Tuple

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from src.config import BEIR_URL_TEMPLATE, DATASETS_DIR

Corpus = Dict[str, Dict[str, str]]
Queries = Dict[str, str]
Qrels = Dict[str, Dict[str, int]]


def download(dataset: str, out_dir: str = DATASETS_DIR) -> str:
    """Download (if needed) and unzip a BEIR dataset. Returns path to the data folder."""
    os.makedirs(out_dir, exist_ok=True)
    expected_path = os.path.join(out_dir, dataset)
    if os.path.isdir(expected_path) and os.listdir(expected_path):
        return expected_path
    return util.download_and_unzip(BEIR_URL_TEMPLATE.format(dataset), out_dir)


def load(dataset: str, split: str = "test") -> Tuple[Corpus, Queries, Qrels]:
    """Load corpus, queries, and qrels for a BEIR dataset."""
    data_path = download(dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels
