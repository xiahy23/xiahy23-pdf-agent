#!/usr/bin/env bash
# Reproduce the unified three-layer RAG ablation (PyMuPDF -> MinerU -> MinerU+GLM).
#
# Layer data:
#   baseline (PyMuPDF) : extracted live from the PDF by the evaluator
#   no_gate  (MinerU)  : existing mineru_paper.md
#   gated    (+GLM)    : regenerated here through the WRAPPED REST API (POST /gate),
#                        i.e. the same Flask service whose usability is covered by
#                        tests/test_pdf_miner_api.py — not an in-process import. This
#                        makes the RAG ablation a genuine end-to-end API consumer.
#
# Requires the GLM credentials in the environment (token read only from env, never
# written to disk). The MinerU venv supplies torch + sentence-transformers + fitz.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ANTHROPIC_AUTH_TOKEN:?set ANTHROPIC_AUTH_TOKEN (GLM api key) first}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://open.bigmodel.cn/api/anthropic}"
export GLM_ARBITER_MODEL="${GLM_ARBITER_MODEL:-glm-4.6v}"

PARSE_DIR="outputs/pdf_miner_agent/ab_minerupaper_pipeline/mineru_paper_pipeline_auto/mineru_paper/auto"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8765}"
API_BASE="http://${API_HOST}:${API_PORT}"

echo "[1/3] starting wrapped REST API (Flask) on ${API_BASE}"
PYTHONPATH="$PWD" MinerU/.venv/bin/python -m pdf_miner_agent.api >/tmp/pdf_miner_api.log 2>&1 &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

# wait for /health (max ~20s)
for _ in $(seq 1 40); do
  curl -fs "${API_BASE}/health" >/dev/null 2>&1 && break
  sleep 0.5
done
curl -fs "${API_BASE}/health" >/dev/null || { echo "API failed to start; see below"; cat /tmp/pdf_miner_api.log; exit 1; }
echo "  /health OK: $(curl -fs ${API_BASE}/health)"

echo "[2/3] regenerating gated markdown via POST /gate (GLM arbitration through the API)"
GATE_RESP=$(curl -fs -X POST "${API_BASE}/gate" \
  -H 'content-type: application/json' \
  -d "{\"pdf_path\":\"data/raw_pdfs/mineru_paper.pdf\",\"parse_dir\":\"${PARSE_DIR}\",\"enable_glm\":true,\"glm_max_calls\":15,\"text_gate\":\"on\"}")
echo "${GATE_RESP}" | MinerU/.venv/bin/python -c "import sys,json; r=json.load(sys.stdin); assert not r.get('error'), r; assert r.get('gated_markdown'), 'no gated_markdown in API response'; print('  gate_counts:', r['gate_counts'], '| gated_markdown:', r['gated_markdown'])"

echo "[3/3] running three-layer ablation"
MinerU/.venv/bin/python scripts/rag_eval/evaluate_three_layer.py
