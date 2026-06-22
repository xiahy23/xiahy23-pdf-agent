#!/usr/bin/env python3
"""Unified three-layer RAG retrieval ablation: parsing quality -> downstream
retrievability, measured layer by layer on ONE document with ONE QA set and ONE
relevance test, so the numbers are comparable hop to hop.

  baseline : PyMuPDF page.get_text() — reads only the digital text layer, so the
             embedded scanned image pages are invisible to it.
  no_gate  : MinerU markdown — OCRs the image pages but joins words together.
  gated    : MinerU markdown after DocGate's GLM visual arbitration re-inserted
             the word boundaries on flagged blocks.

The relevance test is WORD-BOUNDARY preserving (single spaces kept), unlike the
deprecated MinerU-vs-PyMuPDF ablation which stripped all spaces. Keeping spaces
is what makes the second hop (no_gate -> gated) visible: a run-on passage no
longer spuriously matches a spaced answer phrase. The retrieval pipeline
(chunking, embedding, ranking) is reused verbatim from evaluate_rag_retrieval.py
so only the parsing layer differs between corpora.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_rag_retrieval import (  # noqa: E402
    Corpus, MethodScore, Retriever, chunk_text, load_baseline_text, resolve,
)

ROOT = Path(__file__).resolve().parents[2]
LAYERS = ("baseline", "no_gate", "gated")
LAYER_LABEL = {"baseline": "PyMuPDF", "no_gate": "MinerU", "gated": "MinerU+GLM"}


def normalize_ws(text: str) -> str:
    """Word-boundary preserving normalization: drop HTML tags and LaTeX control
    words, lowercase, map every non-alphanumeric run to a single space, and
    collapse whitespace. Spaces are KEPT (the one difference from the deprecated
    space-stripping normalize), so word boundaries survive the relevance test."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def is_relevant_ws(chunk: str, answers: list[str]) -> bool:
    norm = normalize_ws(chunk)
    return any(normalize_ws(a) in norm for a in answers)


def build(spec: dict, retriever: Retriever, target_chars: int, overlap: int):
    out: dict[str, dict[str, Corpus]] = {}
    for cname, paths in spec["corpora"].items():
        texts = {
            "baseline": load_baseline_text(resolve(paths["raw_pdf"])),
            "no_gate": resolve(paths["no_gate_markdown"]).read_text(encoding="utf-8", errors="ignore"),
            "gated": resolve(paths["gated_markdown"]).read_text(encoding="utf-8", errors="ignore"),
        }
        corpora = {}
        for layer in LAYERS:
            chunks = chunk_text(texts[layer], target_chars, overlap)
            corpora[layer] = retriever.build(cname, layer, chunks)
        print(f"[corpus] {cname}: " + ", ".join(f"{LAYER_LABEL[l]}={len(corpora[l].chunks)}ch" for l in LAYERS))
        out[cname] = corpora
    return out


def evaluate(spec: dict, corpora, retriever: Retriever, top_k: int = 3):
    scores = {l: MethodScore(l) for l in LAYERS}
    for q in spec["queries"]:
        cname = q["corpus"]
        for layer in LAYERS:
            corpus = corpora[cname][layer]
            ranked = retriever.rank(q["query"], corpus, top_k)
            rel = [is_relevant_ws(corpus.chunks[i], q["answer_norm"]) for i in ranked]
            s = scores[layer]
            s.n += 1
            s.hit1 += int(bool(rel[:1] and rel[0]))
            s.hit3 += int(any(rel[:3]))
            s.per_query.append({"id": q["id"], "category": q.get("category"),
                                "hit1": bool(rel[:1] and rel[0]), "hit3": any(rel[:3]),
                                "top_rank_relevant": (rel.index(True) + 1) if any(rel) else None})
    return scores


def report(spec: dict, scores: dict) -> None:
    print("\n" + "=" * 70)
    print(f"Three-layer RAG ablation — Hit@k over {scores['baseline'].n} queries")
    print(f"model: {spec.get('embedding_model')}   relevance: word_boundary")
    print("=" * 70)
    print(f"{'layer':<16}{'Hit@1':>10}{'Hit@3':>10}")
    print("-" * 70)
    for l in LAYERS:
        s = scores[l]
        print(f"{LAYER_LABEL[l]:<16}{s.hit1_rate:>9.1%}{s.hit3_rate:>10.1%}")
    print("-" * 70)
    b, n, g = scores["baseline"], scores["no_gate"], scores["gated"]
    print(f"{'Δ MinerU-PyMuPDF':<16}{n.hit1_rate-b.hit1_rate:>+9.1%}{n.hit3_rate-b.hit3_rate:>+10.1%}")
    print(f"{'Δ +GLM-MinerU':<16}{g.hit1_rate-n.hit1_rate:>+9.1%}{g.hit3_rate-n.hit3_rate:>+10.1%}")
    print("=" * 70)
    print(f"\n{'query id':<24}{'cat':<15}{'PyMu@3':>8}{'MinerU@3':>10}{'+GLM@3':>8}")
    pq = {l: scores[l].per_query for l in LAYERS}
    for i in range(len(pq["baseline"])):
        mark = lambda l: "✓" if pq[l][i]["hit3"] else "✗"  # noqa: E731
        a = pq["baseline"][i]
        print(f"{a['id']:<24}{str(a['category']):<15}{mark('baseline'):>8}{mark('no_gate'):>10}{mark('gated'):>8}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="scripts/rag_eval/qa_three_layer.json")
    ap.add_argument("--out", default="outputs/rag_eval_3layer")
    ap.add_argument("--target-chars", type=int, default=700)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    spec = json.loads(resolve(args.qa).read_text(encoding="utf-8"))
    model_name = spec.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    retriever = Retriever(model_name, device=args.device)
    print(f"[init] model={model_name} device={retriever.device}")
    corpora = build(spec, retriever, args.target_chars, args.overlap)
    scores = evaluate(spec, corpora, retriever, top_k=max(3, args.top_k))
    report(spec, scores)

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "embedding_model": model_name,
        "device": retriever.device,
        "relevance": "word_boundary",
        "chunking": {"target_chars": args.target_chars, "overlap_chars": args.overlap},
        "n_queries": scores["baseline"].n,
        "summary": {l: {"hit1": scores[l].hit1, "hit3": scores[l].hit3, "n": scores[l].n,
                        "hit1_rate": round(scores[l].hit1_rate, 4),
                        "hit3_rate": round(scores[l].hit3_rate, 4)} for l in LAYERS},
        "deltas": {
            "minerU_minus_pymupdf": {
                "hit1": round(scores["no_gate"].hit1_rate - scores["baseline"].hit1_rate, 4),
                "hit3": round(scores["no_gate"].hit3_rate - scores["baseline"].hit3_rate, 4)},
            "glm_minus_minerU": {
                "hit1": round(scores["gated"].hit1_rate - scores["no_gate"].hit1_rate, 4),
                "hit3": round(scores["gated"].hit3_rate - scores["no_gate"].hit3_rate, 4)},
        },
        "per_query": {l: scores[l].per_query for l in LAYERS},
        "corpus_chunk_counts": {c: {l: len(corpora[c][l].chunks) for l in LAYERS} for c in corpora},
    }
    out_path = out_dir / "rag_eval_3layer_result.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] results -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
