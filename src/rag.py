"""RAG generation via Ollama with chunk-id citations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import ollama

from src.retrieval import RankedDoc

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using ONLY the "
    "information present in the supplied context passages. After each factual claim, cite "
    "the chunk ids you used in square brackets, e.g. [doc_id#chunk_idx]. If the context "
    "does not contain the answer, reply exactly: 'The provided context does not answer "
    "this question.' Do not invent facts."
)


def _format_context(passages: List[RankedDoc]) -> str:
    lines = []
    for p in passages:
        cid = f"{p.doc_id}#{p.best_chunk_idx}"
        lines.append(f"[{cid}] {p.best_chunk_text}")
    return "\n\n".join(lines)


def build_prompt(query: str, passages: List[RankedDoc]) -> str:
    ctx = _format_context(passages)
    return (
        f"Context passages:\n{ctx}\n\n"
        f"Question: {query}\n\n"
        "Answer with citations in [doc_id#chunk_idx] format."
    )


@dataclass
class RAGResult:
    query: str
    answer: str
    passages: List[RankedDoc]
    prompt: str


def generate(
    query: str,
    passages: List[RankedDoc],
    model: str,
    temperature: float = 0.2,
    num_predict: int = 512,
) -> RAGResult:
    prompt = build_prompt(query, passages)
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature, "num_predict": num_predict},
    )
    answer = resp["message"]["content"].strip()
    return RAGResult(query=query, answer=answer, passages=passages, prompt=prompt)
