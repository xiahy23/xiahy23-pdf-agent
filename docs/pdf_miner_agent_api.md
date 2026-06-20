# PDF-Miner Agent API

This document defines the REST interface used by the assignment-2 PDF-Miner
Agent. The implementation is in `pdf_miner_agent/api.py` and uses Flask so it
can run in the current local environment without extra dependencies.

## Start Service

```bash
cd <repo-root>
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
  "pdf_path": "data/raw_pdfs/standard_two_column_attention.pdf",
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
    "pdf_path": "data/raw_pdfs/standard_two_column_attention.pdf",
    "reuse_existing": true
  }' | python3 -m json.tool
```

## POST `/parse_upload`

Upload a PDF using multipart form data.

```bash
curl -s http://127.0.0.1:8765/parse_upload \
  -F file=@data/raw_pdfs/standard_two_column_attention.pdf \
  -F reuse_existing=true | python3 -m json.tool
```
