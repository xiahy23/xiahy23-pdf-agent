#!/usr/bin/env python3
"""Local RAG retrieval ablation: structured parsing (MinerU) vs naive text extraction (PyMuPDF).

Motivation
----------
The science-knowledge platform's knowledge base is RAG-backed: a user question is
embedded, the most similar document chunks are retrieved, and only those chunks are
handed to the downstream LLM. Retrieval is therefore the gate on every downstream
answer -- if the relevant chunk is not retrieved, no amount of LLM quality recovers it.

This script quantifies how much the *parsing* stage upstream of RAG matters. It builds
two corpora from the SAME source PDFs and runs the SAME retrieval pipeline over both:

  * baseline : raw text dumped by PyMuPDF (``page.get_text()``) -- the typical
               "just extract the text" approach. Display equations collapse to
               unreadable token soup and tables lose all row/column structure.
  * ours     : MinerU-parsed Markdown, where equations are kept as LaTeX and tables
               as HTML/Markdown.

For each query we embed query + chunks with a local sentence-transformers model on the
GPU, rank chunks by cosine similarity, and check whether a *relevant* chunk appears in
the top-1 / top-3. Relevance is decided by the SAME normalized keyword test for both
corpora (see ``QA set`` below), so a hit is earned by retrieval quality, not assumed.

Metrics: Hit@1 and Hit@3 (a.k.a. recall@k for a single relevant target per query).

Usage
-----
    python3 scripts/rag_eval/evaluate_rag_retrieval.py \
        --qa scripts/rag_eval/qa_testset.json \
        --out outputs/rag_eval

Run with the MinerU venv (torch + CUDA + sentence-transformers):
    MinerU/.venv/bin/python scripts/rag_eval/evaluate_rag_retrieval.py ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Text loading
# --------------------------------------------------------------------------- #
def load_baseline_text(pdf_path: Path) -> str:
    """Naive extraction: concatenate PyMuPDF per-page text, mirroring the common
    'just pull the text out of the PDF' baseline that ignores layout and math."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(parts)


def load_mineru_markdown(md_path: Path) -> str:
    """Structured extraction: MinerU Markdown (LaTeX equations + HTML tables preserved)."""
    return md_path.read_text(encoding="utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, target_chars: int = 700, overlap_chars: int = 120) -> list[str]:
    """Paragraph-aware chunking with a character budget.

    Paragraphs (blank-line separated) are greedily packed up to ``target_chars``.
    A paragraph longer than the budget on its own (e.g. a wide HTML table) is kept
    whole rather than split mid-structure -- splitting a table row across chunks
    would defeat the very advantage we are trying to measure. A small character
    overlap is carried between consecutive chunks to avoid losing matches that
    straddle a boundary. The identical function is applied to BOTH corpora.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) >= target_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(para)
            continue
        if not buf:
            buf = para
        elif len(buf) + 1 + len(para) <= target_chars:
            buf = f"{buf}\n{para}"
        else:
            chunks.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail}\n{para}" if tail else para
    if buf:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------- #
# Relevance test (identical for both corpora)
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Lowercase and reduce to [a-z0-9] for a *formatting-agnostic* relevance test.

    The experiment measures retrieval quality, not markup, so before stripping we
    remove the structural scaffolding that differs purely by extraction style:
    HTML tags (``<td>``, ``<tr>`` ...) emitted by MinerU for tables, and LaTeX
    control words (``\\times``, ``\\mathcal`` ...) emitted for equations. Otherwise
    identical answer content would match in one corpus but not the other simply
    because of how the parser wrapped it. The SAME normalization is applied to
    chunks from both corpora, so neither side is advantaged by the test itself.
    """
    text = re.sub(r"<[^>]+>", " ", text)        # drop HTML tags (MinerU tables)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)     # drop LaTeX control words (\times, \mathcal...)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_relevant(chunk: str, answer_norm: list[str]) -> bool:
    norm = normalize(chunk)
    return any(key in norm for key in answer_norm)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
@dataclass
class Corpus:
    name: str
    method: str  # "baseline" | "ours"
    chunks: list[str]
    embeddings: Any = None  # np.ndarray [n_chunks, dim], L2-normalized


class Retriever:
    """Cosine-similarity retriever over L2-normalized sentence-transformer embeddings."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name

    def embed(self, texts: list[str]):
        return self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def build(self, name: str, method: str, chunks: list[str]) -> Corpus:
        embeddings = self.embed(chunks) if chunks else None
        return Corpus(name=name, method=method, chunks=chunks, embeddings=embeddings)

    def rank(self, query: str, corpus: Corpus, top_k: int) -> list[int]:
        import numpy as np

        if corpus.embeddings is None or len(corpus.chunks) == 0:
            return []
        q = self.embed([query])[0]
        sims = corpus.embeddings @ q  # cosine, both sides normalized
        k = min(top_k, len(corpus.chunks))
        return list(np.argsort(-sims)[:k])


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class MethodScore:
    method: str
    n: int = 0
    hit1: int = 0
    hit3: int = 0
    per_query: list[dict] = field(default_factory=list)

    @property
    def hit1_rate(self) -> float:
        return self.hit1 / self.n if self.n else 0.0

    @property
    def hit3_rate(self) -> float:
        return self.hit3 / self.n if self.n else 0.0


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def build_corpora(spec: dict, retriever: Retriever, target_chars: int, overlap: int) -> dict[str, dict[str, Corpus]]:
    """For every corpus in the QA spec, build a baseline and an 'ours' Corpus."""
    out: dict[str, dict[str, Corpus]] = {}
    for cname, paths in spec["corpora"].items():
        raw = load_baseline_text(resolve(paths["raw_pdf"]))
        md = load_mineru_markdown(resolve(paths["mineru_markdown"]))
        base_chunks = chunk_text(raw, target_chars, overlap)
        ours_chunks = chunk_text(md, target_chars, overlap)
        print(f"[corpus] {cname}: baseline={len(base_chunks)} chunks, ours={len(ours_chunks)} chunks")
        out[cname] = {
            "baseline": retriever.build(cname, "baseline", base_chunks),
            "ours": retriever.build(cname, "ours", ours_chunks),
        }
    return out


def evaluate(spec: dict, corpora: dict[str, dict[str, Corpus]], retriever: Retriever, top_k: int = 3) -> dict:
    scores = {"baseline": MethodScore("baseline"), "ours": MethodScore("ours")}
    for q in spec["queries"]:
        cname = q["corpus"]
        for method in ("baseline", "ours"):
            corpus = corpora[cname][method]
            ranked = retriever.rank(q["query"], corpus, top_k)
            rel_flags = [is_relevant(corpus.chunks[i], q["answer_norm"]) for i in ranked]
            hit1 = bool(rel_flags[:1] and rel_flags[0])
            hit3 = any(rel_flags[:3])
            s = scores[method]
            s.n += 1
            s.hit1 += int(hit1)
            s.hit3 += int(hit3)
            s.per_query.append({
                "id": q["id"], "corpus": cname, "category": q.get("category"),
                "hit1": hit1, "hit3": hit3,
                "top_rank_relevant": (rel_flags.index(True) + 1) if any(rel_flags) else None,
            })
    return scores


def print_report(spec: dict, scores: dict) -> None:
    b, o = scores["baseline"], scores["ours"]
    print("\n" + "=" * 64)
    print("RAG retrieval ablation — Hit@k over {} queries".format(b.n))
    print("model: {}".format(spec.get("embedding_model")))
    print("=" * 64)
    print(f"{'method':<12}{'Hit@1':>10}{'Hit@3':>10}")
    print("-" * 64)
    print(f"{'baseline':<12}{b.hit1_rate:>9.1%}{b.hit3_rate:>10.1%}")
    print(f"{'ours':<12}{o.hit1_rate:>9.1%}{o.hit3_rate:>10.1%}")
    print("-" * 64)
    print(f"{'Δ (ours-base)':<12}{o.hit1_rate - b.hit1_rate:>+9.1%}{o.hit3_rate - b.hit3_rate:>+10.1%}")
    print("=" * 64)
    print("\nPer-query (✓=relevant chunk retrieved):")
    print(f"{'query id':<26}{'cat':<14}{'base@3':>8}{'ours@3':>8}")
    for qb, qo in zip(b.per_query, o.per_query):
        bm = "✓" if qb["hit3"] else "✗"
        om = "✓" if qo["hit3"] else "✗"
        print(f"{qb['id']:<26}{str(qb['category']):<14}{bm:>8}{om:>8}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local RAG retrieval ablation (baseline vs MinerU).")
    ap.add_argument("--qa", default="scripts/rag_eval/qa_testset.json", help="QA test set JSON")
    ap.add_argument("--out", default="outputs/rag_eval", help="output directory for results JSON")
    ap.add_argument("--model", default=None, help="override embedding model (else QA set's embedding_model)")
    ap.add_argument("--target-chars", type=int, default=700, help="chunk size budget in characters")
    ap.add_argument("--overlap", type=int, default=120, help="chunk overlap in characters")
    ap.add_argument("--top-k", type=int, default=3, help="retrieval depth for Hit@k (>=3)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args(argv)

    spec = json.loads(resolve(args.qa).read_text(encoding="utf-8"))
    model_name = args.model or spec.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    retriever = Retriever(model_name, device=args.device)
    print(f"[init] model={model_name} device={retriever.device}")
    corpora = build_corpora(spec, retriever, args.target_chars, args.overlap)
    scores = evaluate(spec, corpora, retriever, top_k=max(3, args.top_k))
    print_report(spec, scores)

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "embedding_model": model_name,
        "device": retriever.device,
        "chunking": {"target_chars": args.target_chars, "overlap_chars": args.overlap},
        "n_queries": scores["baseline"].n,
        "summary": {
            m: {
                "hit1": scores[m].hit1, "hit3": scores[m].hit3, "n": scores[m].n,
                "hit1_rate": round(scores[m].hit1_rate, 4),
                "hit3_rate": round(scores[m].hit3_rate, 4),
            } for m in ("baseline", "ours")
        },
        "per_query": {m: scores[m].per_query for m in ("baseline", "ours")},
        "corpus_chunk_counts": {
            c: {m: len(corpora[c][m].chunks) for m in ("baseline", "ours")} for c in corpora
        },
    }
    out_path = out_dir / "rag_eval_result.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] results written to {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
