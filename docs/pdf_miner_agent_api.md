# PDF-Miner Agent API

This document defines the REST interface used by the assignment-2 PDF-Miner
Agent. The implementation is in `pdf_miner_agent/api.py` and uses Flask so it
can run in the current local environment without extra dependencies. It exposes
four endpoints: `GET /health`, `POST /parse`, `POST /parse_upload`, and
`POST /gate` (DocGate quality gate).

## Start Service

```bash
cd /home/robot/workspace/AI4S
python3 -m pdf_miner_agent.api
```

Default endpoint:

```text
http://127.0.0.1:8765
```

## GET `/health`

Health probe.

Response:

```json
{
  "status": "ok",
  "service": "pdf-miner-agent"
}
```

## POST `/parse`

Parse a local PDF path or reuse an existing benchmark parse directory.

Request:

```json
{
  "pdf_path": "/home/robot/workspace/AI4S/data/raw_pdfs/standard_two_column_attention.pdf",
  "backend": "pipeline",
  "method": "auto",
  "effort": null,
  "reuse_existing": true,
  "timeout_sec": 900,
  "force": false
}
```

Response fields:

| field | meaning |
| --- | --- |
| `input` | PDF path, sha256 and file size |
| `classification` | PDF type tags and OCR/text-layer probe |
| `execution` | MinerU command, return code, runtime and log path |
| `structured_output` | Markdown length, element counts and output examples |
| `quality_reference` | OmniDocBench quality metrics and GLM arbitration summary |
| `artifacts` | Markdown, JSON, layout PDF, span PDF, image and package paths |

Minimal curl example:

```bash
curl -s http://127.0.0.1:8765/parse \
  -H 'content-type: application/json' \
  -d '{
    "pdf_path": "/home/robot/workspace/AI4S/data/raw_pdfs/standard_two_column_attention.pdf",
    "reuse_existing": true
  }' | python3 -m json.tool
```

## POST `/parse_upload`

Upload a PDF using multipart form data.

```bash
curl -s http://127.0.0.1:8765/parse_upload \
  -F file=@/home/robot/workspace/AI4S/data/raw_pdfs/standard_two_column_attention.pdf \
  -F reuse_existing=true | python3 -m json.tool
```

## POST `/gate`

Run the DocGate quality gate over an existing MinerU parse directory and write a
`<stem>_gated.md` with adopted corrections applied. This exposes
`docgate.gate_and_rewrite` so downstream consumers (e.g. the three-layer RAG
ablation) can obtain gated markdown through the API rather than an in-process
import. The gate logic (intrinsic table/formula/text checks) lives in
`pdf_miner_agent/docgate.py`; the optional GLM visual arbitration lives in
`pdf_miner_agent/arbiter.py` and is only invoked when `enable_glm=true`.

Request:

```json
{
  "pdf_path": "data/raw_pdfs/mineru_paper.pdf",
  "parse_dir": "outputs/pdf_miner_agent/ab_minerupaper_pipeline/mineru_paper_pipeline_auto/mineru_paper/auto",
  "enable_glm": true,
  "glm_max_calls": 15,
  "text_gate": "on",
  "dry_run": false
}
```

| field | meaning |
| --- | --- |
| `pdf_path` | source PDF (required; 400 if missing) |
| `parse_dir` | existing MinerU parse directory to gate (required; 400 if missing) |
| `enable_glm` | whether to call GLM visual arbitration; falls back silently if unavailable |
| `glm_max_calls` | max GLM arbitration calls (cost cap) |
| `text_gate` | `on` / `log_only` — whether text corrections are adopted or only logged |
| `dry_run` | if true, only run intrinsic L1+L2 checks, never call GLM |

Response: a gate report containing `gate_counts`, `adoptions`, and the
`gated_markdown` output path. GLM credentials are read only from the
`ANTHROPIC_AUTH_TOKEN` environment variable and never written to disk.

```bash
curl -s -X POST http://127.0.0.1:8765/gate \
  -H 'content-type: application/json' \
  -d '{"pdf_path":"data/raw_pdfs/mineru_paper.pdf",
       "parse_dir":"outputs/pdf_miner_agent/ab_minerupaper_pipeline/mineru_paper_pipeline_auto/mineru_paper/auto",
       "enable_glm":true,"glm_max_calls":15,"text_gate":"on"}' | python3 -m json.tool
```

Errors: missing `pdf_path` or `parse_dir` → 400; internal exception → 500.

## SciPilot Integration

The script `scripts/run_pdf_agent_scipilot_pipeline.py` builds a local research
payload, uploads local artifacts to `https://scipilot.chat/discovery`, and
prefills the Discovery node editors for task parsing and literature review.

Dry run:

```bash
python3 scripts/run_pdf_agent_scipilot_pipeline.py \
  --pdf data/raw_pdfs/standard_two_column_attention.pdf \
  --reuse-existing \
  --dry-run
```

Real platform run for upload + local node prefill:

```bash
HEADLESS=1 python3 scripts/run_pdf_agent_scipilot_pipeline.py \
  --pdf data/raw_pdfs/standard_two_column_attention.pdf \
  --reuse-existing \
  --stages ''
```

Optional online stage run, if the platform-side model/API configuration is
available:

```bash
HEADLESS=0 python3 scripts/run_pdf_agent_scipilot_pipeline.py \
  --pdf data/raw_pdfs/standard_two_column_attention.pdf \
  --reuse-existing \
  --stages 任务解析,文献调研,假设构建,方法设计,实验分析,报告生成
```

If the Discovery UI exposes file inputs, the script uploads the package JSON,
summary Markdown, content JSON, benchmark reports, and the generated
`task_parsing_prefill.md` / `literature_review_prefill.md` files. It then opens
the corresponding Discovery editors (`taskMdEditor` and `litSummaryEditor`),
fills the site templates, clicks save, switches to the hypothesis-construction
tab, and saves screenshot/text/html/metadata evidence.
