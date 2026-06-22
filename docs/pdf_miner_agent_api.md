# PDF-Miner Agent API

This document defines the REST interface used by the assignment-2 PDF-Miner
Agent. The implementation is in `pdf_miner_agent/api.py` and uses Flask so it
can run in the current local environment without extra dependencies.

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
