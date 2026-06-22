#!/usr/bin/env python3
"""GLM-arbitration ablation: does the online DocGate (GLM visual arbitration on
flagged OCR text) improve downstream RAG retrieval over ungated MinerU output?

Same source PDF, same retrieval pipeline, same relevance test — only the parsing
post-processing differs:

  * no_gate : MinerU raw markdown (scanned-page OCR with words joined together)
  * gated   : same markdown after DocGate re-inserted word boundaries on the
              blocks its intrinsic checks flagged and GLM-4.6V corrected

Because the relevance test strips spaces before matching, an answer that matches
in one corpus matches in both; any Hit@k gap is purely embedding/retrieval
quality on run-on vs. de-joined text. Reuses the ablation library so chunking,
normalization, and ranking are byte-identical to the MinerU-vs-PyMuPDF study.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_rag_retrieval import (  # noqa: E402
    Corpus, MethodScore, Retriever, chunk_text, is_relevant, resolve,
)

ROOT = Path(__file__).resolve().parents[2]


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def build(spec: dict, retriever: Retriever, target_chars: int, overlap: int):
    out: dict[str, dict[str, Corpus]] = {}
    for cname, paths in spec["corpora"].items():
        ng = chunk_text(load_markdown(resolve(paths["no_gate_markdown"])), target_chars, overlap)
        gt = chunk_text(load_markdown(resolve(paths["gated_markdown"])), target_chars, overlap)
        print(f"[corpus] {cname}: no_gate={len(ng)} chunks, gated={len(gt)} chunks")
        out[cname] = {
            "no_gate": retriever.build(cname, "no_gate", ng),
            "gated": retriever.build(cname, "gated", gt),
        }
    return out


def evaluate(spec: dict, corpora, retriever: Retriever, top_k: int = 3):
    scores = {"no_gate": MethodScore("no_gate"), "gated": MethodScore("gated")}
    for q in spec["queries"]:
        cname = q["corpus"]
        for method in ("no_gate", "gated"):
            corpus = corpora[cname][method]
            ranked = retriever.rank(q["query"], corpus, top_k)
            rel = [is_relevant(corpus.chunks[i], q["answer_norm"]) for i in ranked]
            s = scores[method]
            s.n += 1
            s.hit1 += int(bool(rel[:1] and rel[0]))
            s.hit3 += int(any(rel[:3]))
            s.per_query.append({"id": q["id"], "category": q.get("category"),
                                "hit1": bool(rel[:1] and rel[0]), "hit3": any(rel[:3]),
                                "top_rank_relevant": (rel.index(True) + 1) if any(rel) else None})
    return scores


def report(spec: dict, scores: dict) -> None:
    ng, gt = scores["no_gate"], scores["gated"]
    print("\n" + "=" * 64)
    print(f"GLM-arbitration ablation — Hit@k over {ng.n} queries")
    print(f"model: {spec.get('embedding_model')}")
    print("=" * 64)
    print(f"{'method':<14}{'Hit@1':>10}{'Hit@3':>10}")
    print("-" * 64)
    print(f"{'no_gate':<14}{ng.hit1_rate:>9.1%}{ng.hit3_rate:>10.1%}")
    print(f"{'gated (GLM)':<14}{gt.hit1_rate:>9.1%}{gt.hit3_rate:>10.1%}")
    print("-" * 64)
    print(f"{'Δ (gated-ng)':<14}{gt.hit1_rate - ng.hit1_rate:>+9.1%}{gt.hit3_rate - ng.hit3_rate:>+10.1%}")
    print("=" * 64)
    print(f"\n{'query id':<24}{'cat':<16}{'ng@3':>6}{'glm@3':>7}")
    for a, b in zip(ng.per_query, gt.per_query):
        print(f"{a['id']:<24}{str(a['category']):<16}{'✓' if a['hit3'] else '✗':>6}{'✓' if b['hit3'] else '✗':>7}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="scripts/rag_eval/qa_glm_ablation.json")
    ap.add_argument("--out", default="outputs/rag_eval_glm")
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
        "chunking": {"target_chars": args.target_chars, "overlap_chars": args.overlap},
        "n_queries": scores["no_gate"].n,
        "summary": {m: {"hit1": scores[m].hit1, "hit3": scores[m].hit3, "n": scores[m].n,
                        "hit1_rate": round(scores[m].hit1_rate, 4),
                        "hit3_rate": round(scores[m].hit3_rate, 4)} for m in ("no_gate", "gated")},
        "per_query": {m: scores[m].per_query for m in ("no_gate", "gated")},
        "corpus_chunk_counts": {c: {m: len(corpora[c][m].chunks) for m in ("no_gate", "gated")} for c in corpora},
    }
    out_path = out_dir / "rag_eval_glm_result.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] results -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
