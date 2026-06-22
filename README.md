# xiahy23-pdf-agent

PDF-Miner Agent —— AI4S 大作业二中「PDF 清洗与抽取 Agent」的**封装层与可用性测试**。

把基于 MinerU 的 PDF 智能解析能力封装为三层互不耦合的调用界面（Python 类 / CLI / REST API），并以 `pytest` 覆盖其正常与异常路径。

> 本仓库**只包含 Agent 封装层与测试代码**，不含 MinerU 本体、模型权重、解析产物或样例 PDF（这些体积巨大且非源码）。实际解析能力由本机安装的 MinerU 提供，见下文「运行前提」。

## 三层封装

| 层 | 文件 | 调用方式 |
|---|---|---|
| 核心类 | `pdf_miner_agent/agent.py` | `from pdf_miner_agent import PDFMinerAgent, PDFMinerConfig` |
| 命令行 | `pdf_miner_agent/cli.py` | `python3 -m pdf_miner_agent.cli <pdf> [--backend ...]` |
| REST API | `pdf_miner_agent/api.py` | Flask 服务，暴露 3 个 HTTP 端点 |

Agent 接收 PDF 路径（或 multipart 上传文件），调用 MinerU 解析，并把结果整理成一个统一的结构化 JSON package：`input`（文件名/路径/sha256/字节数）、`classification`（类型探测与推荐解析模式）、`config`（本次实际生效的解析配置）、`execution`（命令/返回码/耗时/日志）、`structured_output`（Markdown 长度、内容类型计数、示例）、`online_quality`（针对**本文档**的在线 DocGate 报告：内在自检 + GLM 仲裁结果与门控计数）、`offline_benchmark_reference`（固定评测run 的系统级 OmniDocBench / GLM 摘要，仅作溯源，**并非本文档得分**）、`artifacts`（Markdown / content JSON / layout PDF / package JSON 等路径）。

## REST 端点

| 方法 + 路径 | 作用 |
|---|---|
| `GET /health` | 健康检查，返回 `{"status":"ok","service":"pdf-miner-agent"}` |
| `POST /parse` | JSON body 传 `pdf_path`，解析并返回 package |
| `POST /parse_upload` | multipart 上传 PDF 文件，解析并返回 package |

`POST /parse` / `POST /parse_upload` 均做参数校验：缺失必填参数返回 HTTP 400，解析异常返回 HTTP 500（附错误类型与信息），正常返回 HTTP 200 与完整 package JSON。完整字段说明见 [`docs/pdf_miner_agent_api.md`](docs/pdf_miner_agent_api.md)。

## 运行前提

| 依赖 | 说明 |
|---|---|
| Python | 3.9+ |
| Flask / pytest | `pip install -r requirements.txt` |
| **MinerU** | 实际解析引擎，需单独安装（见 [MinerU 官方仓库](https://github.com/opendatalab/MinerU)）。本仓库不含其本体与 CUDA/torch 依赖。 |

`pdf_miner_agent/agent.py` 顶部的 `MINERU_BIN` / `OUT_ROOT` 等路径常量默认相对仓库根目录（`MinerU/.venv/bin/mineru`、`outputs/pdf_miner_agent/` 等）。在本独立仓库中这些路径默认不存在——**导入包与启动 API 不受影响**，仅在真正调用 `parse_pdf` 实跑 MinerU 时才需要这些资源就位。请按自己的环境调整这些常量，或把 MinerU 安装到对应位置。

## 启动 API 并验证

```bash
python3 -m pip install -r requirements.txt

# 启动服务（默认 0.0.0.0:8765）
python3 -m pdf_miner_agent.api

# 健康检查
curl -s http://127.0.0.1:8765/health

# 解析（reuse_existing 复用已有 MinerU 产物，秒级返回，不占 GPU）
curl -s -X POST http://127.0.0.1:8765/parse \
  -H "Content-Type: application/json" \
  -d '{"pdf_path":"data/raw_pdfs/standard_two_column_attention.pdf","reuse_existing":true}'
```

## 可用性测试（pytest）

`tests/test_pdf_miner_api.py` 共 8 个用例，覆盖核心类契约与三个端点的正常/异常路径（400/500）：

```bash
python3 -m pytest tests/test_pdf_miner_api.py -v
```

测试设计为**不依赖 GPU、不触发 MinerU 实推理、不连接任何外部平台**：用例统一以 `reuse_existing=True` 复用 `outputs/mineru_benchmark/` 中 returncode 为 0 的既有解析产物。

> ⚠️ 这些基准产物体积大，已被 `.gitignore` 排除、不入库。若测试机上不存在样例 PDF（`data/raw_pdfs/standard_two_column_attention.pdf`），整个测试模块会**优雅 skip**（而非失败）——这是 `pytest.mark.skipif` 的预期行为。要真正跑通测试，请先准备样例 PDF 并用 `python3 -m pdf_miner_agent.cli <pdf>` 生成一次解析产物，或去掉 `reuse_existing` 让其实跑 MinerU。

## 主要文件

```
pdf_miner_agent/
  agent.py      # 核心类 PDFMinerAgent / PDFMinerConfig
  cli.py        # 命令行入口
  api.py        # Flask REST 服务
  __init__.py
tests/
  test_pdf_miner_api.py   # 8 个 pytest 可用性用例
docs/
  pdf_miner_agent_api.md  # REST 接口文档
requirements.txt
```
