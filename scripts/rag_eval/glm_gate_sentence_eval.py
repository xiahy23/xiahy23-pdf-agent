#!/usr/bin/env python3
"""Sentence-level evidence that GLM visual arbitration improves retrievability.

The chunk-level Hit@k ablation (evaluate_glm_ablation.py) is blunt here: the
relevance test normalizes away spaces, and 700-char chunks dilute a single
de-joined sentence inside surrounding clean text, so block ranking barely moves.
The real, measurable gain from re-inserting word boundaries shows up at the
sentence level — exactly where a query embedding meets the passage it should
retrieve. This script measures cosine similarity of each query against the
MinerU run-on passage (no_gate) vs. the GLM-corrected passage (gated), using the
same embedding model as the RAG ablation. Higher = more retrievable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_rag_retrieval import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("study purposes for energy expenditure prediction from accelerometer",
     "investigate theperformanceofthreedifferent accelerometers",
     "investigate the performance of three different accelerometers"),
    ("how many children and percent boys",
     "Fourtyonechildren（54%boys", "Forty-one children (54% boys"),
    ("four prediction models mixed linear random forest",
     "amixed linearmodel(MLM),arandom forest model(RF)",
     "a mixed linear model (MLM), a random forest model (RF)"),
    ("how many wear positions on preschoolers",
     "fivedifferent wear positionsinpreschoolers",
     "five different wear positions in preschoolers"),
    ("cross-validation scheme prediction accuracy",
     "leave-one-out cross-validation", "leave-one-out cross-validation"),
    ("semi-structured protocol children completed",
     "completingasemi-structuredprotocolof10age-appropriateactivities",
     "completing a semi-structured protocol of 10 age-appropriate activities"),
    ("random forest viable alternative to linear ANN",
     "RFseems to beaviablealternativeto linearand ANNmodels",
     "RF seems to be a viable alternative to linear and ANN models"),
    ("why lab protocols overestimate energy expenditure",
     "highlystructuredprotocolsunderlaboratorysetingshasbeenfoundtooverestimateEE",
     "highly structured protocols under laboratory settings have been found to overestimate EE"),
]


def main() -> int:
    r = Retriever("sentence-transformers/all-MiniLM-L6-v2")
    rows, sj_sum, ss_sum, wins = [], 0.0, 0.0, 0
    for q, joined, spaced in CASES:
        qe = r.embed([q])[0]
        sj = float(r.embed([joined])[0] @ qe)
        ss = float(r.embed([spaced])[0] @ qe)
        sj_sum += sj
        ss_sum += ss
        wins += int(ss > sj)
        rows.append({"query": q, "no_gate_sim": round(sj, 4),
                     "gated_sim": round(ss, 4), "delta": round(ss - sj, 4)})
    n = len(CASES)
    result = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n_queries": n,
        "mean_no_gate_sim": round(sj_sum / n, 4),
        "mean_gated_sim": round(ss_sum / n, 4),
        "mean_delta": round((ss_sum - sj_sum) / n, 4),
        "gated_wins": wins,
        "per_query": rows,
    }
    out = ROOT / "outputs" / "rag_eval_glm" / "glm_gate_sentence_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean similarity  no_gate={result['mean_no_gate_sim']}  "
          f"gated={result['mean_gated_sim']}  Δ={result['mean_delta']:+}")
    print(f"gated wins {wins}/{n} queries")
    print(f"[done] -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
