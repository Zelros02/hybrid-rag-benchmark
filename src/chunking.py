"""Token-based fixed-size chunking with overlap."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_idx: int
    text: str


def _compose_doc(title: str | None, text: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def chunk_text(
    doc_id: str,
    title: str | None,
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    overlap_ratio: float,
) -> List[Chunk]:
    """Split a document into token-windowed chunks with overlap.

    The tokenizer's plain `encode` (without special tokens) defines token boundaries;
    each chunk is then decoded back to a text span suitable for both BM25 and vector
    indexing.
    """
    body = _compose_doc(title, text)
    if not body:
        return []

    token_ids = tokenizer.encode(body, add_special_tokens=False)
    if not token_ids:
        return []

    stride = max(1, int(chunk_size * (1 - overlap_ratio)))
    chunks: List[Chunk] = []
    idx = 0
    start = 0
    n = len(token_ids)
    while start < n:
        end = min(start + chunk_size, n)
        window = token_ids[start:end]
        chunk_str = tokenizer.decode(window, skip_special_tokens=True).strip()
        if chunk_str:
            chunks.append(Chunk(doc_id=doc_id, chunk_idx=idx, text=chunk_str))
            idx += 1
        if end == n:
            break
        start += stride
    return chunks


def chunk_corpus(
    corpus: dict,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    overlap_ratio: float,
) -> Iterable[Chunk]:
    for doc_id, doc in corpus.items():
        yield from chunk_text(
            doc_id=doc_id,
            title=doc.get("title"),
            text=doc.get("text", ""),
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
        )
