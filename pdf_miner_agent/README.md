# PDF-Miner Agent

基于 MinerU 的 PDF 高质量解析 Agent：把「解析 → 结构化输出 → 质量门控 → 服务化」封装为可调用接口（核心类 / CLI / REST API 三层），并可作为下游 RAG 实验的真实 API 消费者。

- **解析后端**：MinerU `pipeline`（轻量、~0.63s/页）与 `hybrid-engine`（VLM 混合、精度更高、~1.75s/页），按 PDF 类型路由。
- **质量门控（DocGate）**：在线、无需 GT。表格须解析为合法网格、公式须可编译、文本须通过四道护栏（编辑距离 / 缺陷分下降 / 不引入乱码 / 长度比），拿不准一律保留 MinerU 原输出。GLM 视觉仲裁不可用时静默回退，解析永不阻断。

---

## 1. 环境与依赖

### 1.1 Python 环境

本仓库复用 MinerU 自带的虚拟环境（含 torch + CUDA + sentence-transformers + PyMuPDF），统一用它作为解释器：

```bash
# 仓库根目录
cd /home/robot/workspace/AI4S

# MinerU venv（解析、门控、RAG 评测都用它）
MinerU/.venv/bin/python --version
```

> 所有命令里的 `python` 都用 `MinerU/.venv/bin/python`，不要用系统 python，否则缺 torch / mineru / sentence-transformers。

### 1.2 关键依赖

- `mineru`（已随 `MinerU/` 子目录安装到该 venv）
- `flask`（REST API）
- `sentence-transformers`、`torch`（RAG 检索）
- `PyMuPDF`(`fitz`)（PyMuPDF 基线 & PDF 探测）

---

## 2. 模型权重（ckpt）放在哪

解析所需的本地权重统一放在 **HuggingFace 缓存目录** `~/.cache/huggingface/hub/`，首次使用会自动下载，之后离线加载。当前已就位三套：

| 模型 | 缓存目录 | 用途 |
| --- | --- | --- |
| MinerU2.5-Pro-2605-1.2B | `~/.cache/huggingface/hub/models--opendatalab--MinerU2.5-Pro-2605-1.2B` | 解析主模型 |
| PDF-Extract-Kit-1.0 | `~/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0` | 布局 / 公式 / 表格检测 |
| all-MiniLM-L6-v2 | `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` | RAG 句向量检索 |

> **国内无法访问 HuggingFace 时**，切换镜像源后再首次运行即可自动下载：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> # 或改用 ModelScope 源：export MINERU_MODEL_SOURCE=modelscope
> ```
> 已有缓存则无需联网。验证缓存是否就位：
> ```bash
> ls ~/.cache/huggingface/hub | grep -E "MinerU2.5|PDF-Extract-Kit|all-MiniLM"
> ```

---

## 3. API Key 写在哪（仅质量门控的 GLM 仲裁需要）

GLM 视觉仲裁调用智谱云端 `glm-4.6v`，**凭证只从环境变量读取，绝不写入磁盘**。解析与检索本身不需要任何 key——只有当你要启用 GLM 门控（`enable_glm`/`enable_glm_gate=true`）时才需要：

```bash
export ANTHROPIC_AUTH_TOKEN="你的智谱GLM-API-Key"        # 必填（启用 GLM 时）
export ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic"   # 默认值，可不设
export GLM_ARBITER_MODEL="glm-4.6v"                       # 默认值，可不设
```

> 不设 token 时，门控会自动跳过 GLM、回退保留 MinerU 原文，流程不报错。

---

## 4. 三层调用方式

| 层 | 文件 | 调用方式 | 场景 |
| --- | --- | --- | --- |
| 核心类 | `agent.py` | `PDFMinerAgent().parse_pdf(...)` | 进程内集成 |
| 命令行 | `cli.py` | `python -m pdf_miner_agent.cli` | 批处理 / 调试 |
| REST API | `api.py` | Flask 四端点（默认 `0.0.0.0:8765`） | 服务化 / 远程 |

### 4.1 CLI：raw PDF → 结构化产物（最简单）

```bash
cd /home/robot/workspace/AI4S
PYTHONPATH="$PWD" MinerU/.venv/bin/python -m pdf_miner_agent.cli \
  data/raw_pdfs/mineru_paper.pdf \
  --backend pipeline --method auto
# 复杂扫描件 / 表格可用 --backend hybrid-engine --effort high
```
输出：完整 package JSON 打到 stdout；解析产物落在 `outputs/pdf_miner_agent/<run_id>/...`。

### 4.2 REST API 四端点

```text
GET  /health        # 健康检查 → 200
POST /parse         # JSON 传 pdf_path → 完整解析（raw PDF→markdown/JSON，含可选 GLM 门控）
POST /parse_upload  # multipart 上传 PDF → 完整解析
POST /gate          # 对已有解析目录单独跑质量门控 → 写 <stem>_gated.md
# 缺参 400 / 异常 500 / 正常 200
```

> `/parse` 与 `/parse_upload` 是**从 raw PDF 到结构化产物的完整 pipeline**（内部调 MinerU 解析 + 可选 GLM 门控）；`/gate` 只把门控这一层单独暴露，供已有解析产物的下游复用。

---

## 5. 完整跑通全流程

### 5.1 启动 REST 服务

```bash
cd /home/robot/workspace/AI4S
export ANTHROPIC_AUTH_TOKEN="你的智谱GLM-API-Key"   # 启用门控时
PYTHONPATH="$PWD" MinerU/.venv/bin/python -m pdf_miner_agent.api
# 监听 0.0.0.0:8765，日志见终端
```

另开一个终端：

```bash
# 健康检查
curl -s http://127.0.0.1:8765/health

# A) 完整 pipeline：raw PDF → 解析产物（force 强制重解析，不走复用）
curl -s -X POST http://127.0.0.1:8765/parse \
  -H 'content-type: application/json' \
  -d '{"pdf_path":"data/raw_pdfs/mineru_paper.pdf","backend":"pipeline","enable_glm_gate":true,"text_gate":"on","force":true}'

# B) 上传文件解析
curl -s -X POST http://127.0.0.1:8765/parse_upload -F 'file=@data/raw_pdfs/mineru_paper.pdf'

# C) 仅对已有解析目录跑门控
curl -s -X POST http://127.0.0.1:8765/gate \
  -H 'content-type: application/json' \
  -d '{"pdf_path":"data/raw_pdfs/mineru_paper.pdf",
       "parse_dir":"outputs/pdf_miner_agent/ab_minerupaper_pipeline/mineru_paper_pipeline_auto/mineru_paper/auto",
       "enable_glm":true,"glm_max_calls":15,"text_gate":"on"}'
```

### 5.2 一键复现 RAG 三层消融（PyMuPDF → MinerU → MinerU+GLM）

```bash
cd /home/robot/workspace/AI4S
export ANTHROPIC_AUTH_TOKEN="你的智谱GLM-API-Key"
bash scripts/run_three_layer_ablation.sh
```
该脚本自动：起 Flask API → 等 `/health` → `POST /gate` 生成门控语料 → 跑三层 Hit@k 评测。
> ① PyMuPDF 与 ② MinerU 两层**纯本地可复现**；③ MinerU+GLM 层需自备 GLM token（无 token 则退化为保留 MinerU 原文）。

---

## 6. 产物在哪看

| 看什么 | 路径 |
| --- | --- |
| 解析产物根目录 | `outputs/pdf_miner_agent/<run_id>/` |
| 门控前 markdown（②层语料） | `<parse_dir>/mineru_paper.md` |
| 门控后 markdown（③层语料） | `<parse_dir>/mineru_paper_gated.md` |
| GLM 看图裁剪的图块 | `<parse_dir>/docgate_crops/` |
| MinerU 中间件（解析痕迹证据） | `<parse_dir>/*_middle.json`、`*_model.json`、`*_layout.pdf`、`*_span.pdf`、`images/` |
| RAG 三层结果 | 终端输出 + `outputs/rag_eval/` 下结果 JSON |
| API 服务日志（脚本模式） | `/tmp/pdf_miner_api.log` |

> `<parse_dir>` 指某次解析的 `.../<pdf_stem>/auto`（或 `ocr`）目录。

---

## 7. 测试

```bash
cd /home/robot/workspace/AI4S
PYTHONPATH="$PWD" MinerU/.venv/bin/python -m pytest tests/test_pdf_miner_api.py -v       # API 可用性（health/parse/gate 契约）
PYTHONPATH="$PWD" MinerU/.venv/bin/python -m pytest tests/test_docgate_quality.py -v     # 门控规则
PYTHONPATH="$PWD" MinerU/.venv/bin/python -m pytest tests/test_three_layer_eval.py -v    # 三层评测口径
```

---

## 8. 模块一览

| 文件 | 职责 |
| --- | --- |
| `agent.py` | 核心类 `PDFMinerAgent`：PDF 类型探测 / 后端路由 / 调 MinerU / 组装结构化 package |
| `cli.py` | 命令行入口 |
| `api.py` | Flask REST 四端点 |
| `docgate.py` | 在线质量门控：按 kind 分档采纳，写 `<stem>_gated.md` |
| `arbiter.py` | GLM-4.6V 视觉仲裁（云端，凭证仅读环境变量） |
| `quality.py` | 规则清洗 / 缺陷打分 / 护栏判定 |
