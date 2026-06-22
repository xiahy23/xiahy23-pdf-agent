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
  "force": false,
  "enable_glm_gate": true,
  "glm_max_calls": 12,
  "text_gate": "log_only",
  "glm_dry_run": false
}
```

Request parameters (all optional except `pdf_path`):

| param | default | meaning |
| --- | --- | --- |
| `pdf_path` | — (required) | Local PDF path to parse |
| `backend` | `"pipeline"` | MinerU backend (`pipeline` / `hybrid-engine`) |
| `method` | `"auto"` | Parse method routing |
| `effort` | `null` | Optional effort hint |
| `start_page` / `end_page` | `null` | Optional page range |
| `timeout_sec` | `900` | MinerU subprocess timeout |
| `force` | `false` | Force re-parse even if output exists |
| `reuse_existing` | `false` | Reuse an existing parse dir (fast, no GPU) |
| `enable_glm_gate` | `true` | Run the online GLM visual-arbitration quality gate |
| `glm_max_calls` | `12` | Cap on GLM arbitration calls per document |
| `text_gate` | `"log_only"` | Text-kind adoption policy (`log_only` = record but do not auto-adopt) |
| `glm_dry_run` | `false` | Build the gate plan without calling the GLM API |
| `run_id` | `null` | Optional explicit run id |

Response fields:

| field | meaning |
| --- | --- |
| `input` | PDF path, name, sha256 and file size in bytes |
| `classification` | PDF type tags and OCR/text-layer probe |
| `config` | The effective `PDFMinerConfig` used for this run |
| `execution` | MinerU command, return code, runtime and log path |
| `structured_output` | Markdown length, element counts and output examples |
| `online_quality` | Live DocGate report for **this** document: intrinsic checks + GLM arbitration outcomes, gate counts and document-quality summary (or `{"enabled": false}` if the gate did not run) |
| `offline_benchmark_reference` | System-level OmniDocBench / GLM arbitration summary from a fixed eval run, kept for provenance only — explicitly **NOT** this document's score |
| `artifacts` | Markdown, content/middle/model JSON, layout PDF, span PDF and image paths |

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

Parameter parsing mirrors `POST /parse`; missing `file` returns HTTP 400 and a
parse error returns HTTP 500. The response package is identical in shape to
`POST /parse`.

## POST `/gate`

Run the DocGate quality gate over an **existing** MinerU parse directory and
write a `<stem>_gated.md` with adopted corrections applied. This exposes
`docgate.gate_and_rewrite` so downstream consumers can obtain gated markdown
through the API rather than an in-process import — the three-layer RAG ablation
(`scripts/run_three_layer_ablation.sh`) uses this endpoint to produce its
`MinerU+GLM` layer, making the ablation a genuine end-to-end API consumer.

Request:

```json
{
  "pdf_path": "data/raw_pdfs/mineru_paper.pdf",
  "parse_dir": "outputs/.../mineru_paper/auto",
  "enable_glm": true,
  "glm_max_calls": 12,
  "text_gate": "log_only",
  "dry_run": false
}
```

| param | default | meaning |
| --- | --- | --- |
| `pdf_path` | — (required) | Source PDF (regions are cropped from it for visual arbitration) |
| `parse_dir` | — (required) | Existing MinerU parse dir (must contain `*_content_list.json` and `*.md`) |
| `enable_glm` | `true` | Run GLM visual arbitration; if false, only intrinsic L1/L2 flagging |
| `glm_max_calls` | `12` | Cap on GLM calls |
| `text_gate` | `"log_only"` | Text adoption policy (`log_only` records but does not auto-adopt; `on` adopts when all four text guardrails pass) |
| `dry_run` | `false` | Flag elements but make no GLM call (deterministic, offline) |

Missing `pdf_path` or `parse_dir` returns HTTP 400; an error returns HTTP 500.
The response is the DocGate report (`gate_counts`, `adoptions`, and a
`gated_markdown` path when corrections were written).

```bash
curl -s -X POST http://127.0.0.1:8765/gate \
  -H 'content-type: application/json' \
  -d '{"pdf_path":"data/raw_pdfs/mineru_paper.pdf","parse_dir":"outputs/.../auto","dry_run":true}' \
  | python3 -m json.tool
```
