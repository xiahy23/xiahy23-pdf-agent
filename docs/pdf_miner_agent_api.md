# PDF-Miner Agent REST API

基于 Flask 的 REST 接口，把 MinerU 的 PDF 解析能力封装为 HTTP 服务。实现见 `pdf_miner_agent/api.py`。

- **Base URL**：`http://127.0.0.1:8765`
- **Content-Type**：`/parse` 为 `application/json`；`/parse_upload` 为 `multipart/form-data`
- **认证**：无（本地服务）
- **统一返回**：成功 `200` + package JSON；参数缺失 `400`；解析异常 `500`。错误体统一为 `{"error": "<信息>"}`

## 启动服务

```bash
python3 -m pip install -r requirements.txt
python3 -m pdf_miner_agent.api          # 监听 0.0.0.0:8765
```

---

## 端点一览

| 方法 | 路径 | 作用 | 请求体 |
|---|---|---|---|
| `GET`  | `/health` | 健康检查 | 无 |
| `POST` | `/parse` | 解析服务器本地路径上的 PDF | JSON |
| `POST` | `/parse_upload` | 上传 PDF 文件并解析 | multipart |

---

## GET `/health`

存活探针，无参数。

**请求**

```bash
curl -s http://127.0.0.1:8765/health
```

**响应 `200`**

```json
{
  "status": "ok",
  "service": "pdf-miner-agent"
}
```

---

## POST `/parse`

解析服务器可访问的本地 PDF 路径，返回结构化 package。

### 请求参数（JSON body）

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `pdf_path` | string | **是** | — | PDF 在服务器上的路径（绝对或相对仓库根） |
| `backend` | string | 否 | `"pipeline"` | MinerU 后端：`pipeline` 或 `hybrid-engine` |
| `method` | string | 否 | `"auto"` | 解析模式：`auto` / `ocr` / `txt`（`auto` 下扫描件自动转 `ocr`） |
| `effort` | string\|null | 否 | `null` | 力度档位（如 `medium`/`high`），传给 MinerU |
| `start_page` | int\|null | 否 | `null` | 起始页（从 0 计） |
| `end_page` | int\|null | 否 | `null` | 结束页 |
| `timeout_sec` | int | 否 | `900` | MinerU 子进程超时（秒） |
| `force` | bool | 否 | `false` | 强制重解析，覆盖已有产物 |
| `reuse_existing` | bool | 否 | `false` | 复用 `outputs/mineru_benchmark/` 中已有的解析目录（不实跑 MinerU，秒级返回） |
| `run_id` | string\|null | 否 | 当前时间戳 | 产物归档子目录名 |

### 请求示例

```bash
curl -s -X POST http://127.0.0.1:8765/parse \
  -H 'Content-Type: application/json' \
  -d '{
    "pdf_path": "data/raw_pdfs/standard_two_column_attention.pdf",
    "backend": "pipeline",
    "method": "auto",
    "reuse_existing": true
  }' | python3 -m json.tool
```

### 响应 `200`（真实样例，长字段已截断）

```jsonc
{
  "run_id": "doc_sample",
  "created_at": "2026-06-20T09:00:00",
  "agent": {
    "name": "PDF-Miner Agent",
    "version": "0.2",
    "backend": "MinerU + rule postprocess + OmniDocBench scorer + GLM arbitration"
  },
  "input": {
    "pdf_name": "standard_two_column_attention.pdf",
    "pdf_path": "data/raw_pdfs/standard_two_column_attention.pdf",
    "sha256": "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
    "bytes": 2215244
  },
  "classification": {
    "pages": 15,
    "has_text_layer": true,
    "tags": ["standard_two_column"],
    "recommended_method": "auto"
  },
  "config": {
    "backend": "pipeline",
    "method": "auto",
    "method_resolved": "auto",
    "effort": null,
    "timeout_sec": 900,
    "force": false
  },
  "execution": {
    "returncode": 0,
    "seconds": 0.0,
    "timeout": false,
    "cmd": ["reuse_existing", "outputs/mineru_benchmark/.../auto"],
    "log": "outputs/pdf_miner_agent/doc_sample/logs/standard_two_column_attention_pipeline_auto.log"
  },
  "structured_output": {
    "has_markdown": true,
    "has_content_json": true,
    "markdown_chars": 19190,
    "image_count": 7,
    "content_summary": {
      "content_items": 80,
      "content_types": {
        "text": 62, "equation": 4, "table": 1, "image": 2,
        "page_number": 5, "page_footnote": 4, "footer": 1, "aside_text": 1
      },
      "examples": [
        { "type": "text", "text": "Attention Is All You Need" }
      ]
    },
    "markdown_excerpt": "Provided proper attribution is provided, Google hereby grants ..."
  },
  "quality_reference": {
    "omnidocbench_quality": {
      "text_accuracy": 0.7552,
      "formula_edit": 0.4038,
      "formula_cdm": 0.6978,
      "table_teds": 0.9871,
      "reading_order_accuracy": 0.5812,
      "overall_proxy": 0.8133
    },
    "glm_arbitration": { "status_counts": { "ok": 12 } }
  },
  "artifacts": {
    "parse_dir": "outputs/mineru_benchmark/.../auto",
    "markdown": ".../standard_two_column_attention.md",
    "content_json": ".../standard_two_column_attention_content_list.json",
    "middle_json": ".../*_middle.json",
    "model_json": ".../*_model.json",
    "layout_pdf": ".../*_layout.pdf",
    "span_pdf": ".../*_span.pdf",
    "images": [".../images/xxx.jpg"],
    "package_json": "outputs/pdf_miner_agent/doc_sample/..._package.json",
    "summary_markdown": "outputs/pdf_miner_agent/doc_sample/..._summary.md"
  }
}
```

> 注：`reuse_existing=true` 时 `execution.seconds` 接近 0、`cmd[0]` 为 `reuse_existing`；实跑 MinerU 时 `cmd` 为完整 mineru 命令行、`seconds` 为真实耗时。`quality_reference` 仅在本地存在对应 OmniDocBench/GLM 产物时有值，否则为空对象。

### 响应字段说明

| 顶层字段 | 含义 |
|---|---|
| `input` | PDF 文件名、路径、SHA256、字节数 |
| `classification` | 页数、文字层探测、类型标签（如 `formula_dense`/`table_complex`/`scanned_or_rasterized`）、推荐解析模式 |
| `config` | 实际使用的解析配置，`method_resolved` 为路由后的最终模式 |
| `execution` | MinerU 命令、返回码、耗时（秒）、是否超时、日志路径 |
| `structured_output` | Markdown 字符数、是否产出 Markdown/content JSON、图片数、内容类型计数与示例片段 |
| `quality_reference` | OmniDocBench 质量指标（text/OCR accuracy、formula edit/CDM、table TEDS、reading order）与 GLM 仲裁摘要 |
| `artifacts` | Markdown、content/middle/model JSON、layout/span PDF、图片、package JSON、summary Markdown 等路径 |

### 错误响应

| 状态码 | 触发条件 | 响应体 |
|---|---|---|
| `400` | 缺少 `pdf_path` | `{"error": "missing pdf_path"}` |
| `500` | 文件不存在 / 解析异常 | `{"error": "FileNotFoundError: .../__nope__.pdf"}` |

```bash
# 缺参数 -> 400
curl -s -X POST http://127.0.0.1:8765/parse -H 'Content-Type: application/json' -d '{}'
# {"error": "missing pdf_path"}

# 文件不存在 -> 500
curl -s -X POST http://127.0.0.1:8765/parse -H 'Content-Type: application/json' \
  -d '{"pdf_path": "data/raw_pdfs/__nope__.pdf", "reuse_existing": true}'
# {"error": "FileNotFoundError: .../__nope__.pdf"}
```

---

## POST `/parse_upload`

直接上传 PDF 文件后解析，适合客户端没有服务器本地路径的场景。

### 请求参数（multipart/form-data）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | file | **是** | 待解析的 PDF 文件 |
| `backend` / `method` / `effort` / `start_page` / `end_page` / `timeout_sec` / `force` / `reuse_existing` / `run_id` | form 字段 | 否 | 与 `/parse` 同名，语义一致（布尔用 `1`/`true`，数值用字符串） |

### 请求示例

```bash
curl -s -X POST http://127.0.0.1:8765/parse_upload \
  -F file=@data/raw_pdfs/standard_two_column_attention.pdf \
  -F reuse_existing=true \
  -F backend=pipeline | python3 -m json.tool
```

### 响应

成功 `200`，结构与 `/parse` 完全一致（上传文件先写入临时路径再解析）。

### 错误响应

| 状态码 | 触发条件 | 响应体 |
|---|---|---|
| `400` | 缺少 `file` 字段 | `` {"error": "missing multipart file field `file`"} `` |
| `500` | 解析异常 | `{"error": "<异常类型>: <信息>"}` |

```bash
# 不带文件 -> 400
curl -s -X POST http://127.0.0.1:8765/parse_upload -F reuse_existing=true
# {"error": "missing multipart file field `file`"}
```

---

## 调用方式对照

同一套解析能力有三种调用界面（详见仓库 README）：

| 方式 | 入口 | 适用场景 |
|---|---|---|
| Python 类 | `from pdf_miner_agent import PDFMinerAgent` | 进程内直接调用 |
| CLI | `python3 -m pdf_miner_agent.cli <pdf>` | 命令行 / 脚本批处理 |
| REST API | 本文档 | 跨进程 / 跨语言 / 服务化 |

## 可用性测试

`tests/test_pdf_miner_api.py` 用 `pytest` 覆盖三个端点的正常与异常路径（8 个用例）：

```bash
python3 -m pytest tests/test_pdf_miner_api.py -v
```
