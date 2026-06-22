#!/usr/bin/env bash
# Reproduce the unified three-layer RAG ablation (PyMuPDF -> MinerU -> MinerU+GLM).
#
# Layer data:
#   baseline (PyMuPDF) : extracted live from the PDF by the evaluator
#   no_gate  (MinerU)  : existing mineru_paper.md
#   gated    (+GLM)    : regenerated here by DocGate so the run is self-contained
#
# Requires the GLM credentials in the environment (token read only from env, never
# written to disk). The MinerU venv supplies torch + sentence-transformers + fitz.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ANTHROPIC_AUTH_TOKEN:?set ANTHROPIC_AUTH_TOKEN (GLM api key) first}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://open.bigmodel.cn/api/anthropic}"
export GLM_ARBITER_MODEL="${GLM_ARBITER_MODEL:-glm-4.6v}"

PARSE_DIR="outputs/pdf_miner_agent/ab_minerupaper_pipeline/mineru_paper_pipeline_auto/mineru_paper/auto"

echo "[1/2] regenerating gated markdown via DocGate (GLM visual arbitration)"
PYTHONPATH="$PWD" python3 - "$PARSE_DIR" <<'PY'
import sys
from pathlib import Path
import pdf_miner_agent.docgate as d
parse_dir = Path(sys.argv[1])
rep = d.gate_and_rewrite(
    Path("data/raw_pdfs/mineru_paper.pdf"), parse_dir,
    enable_glm=True, glm_max_calls=15, text_gate="on",
)
print("  gate_counts:", rep["gate_counts"], "replacements:", rep.get("replacements_applied"))
print("  gated_markdown:", rep.get("gated_markdown"))
PY

echo "[2/2] running three-layer ablation"
MinerU/.venv/bin/python scripts/rag_eval/evaluate_three_layer.py
