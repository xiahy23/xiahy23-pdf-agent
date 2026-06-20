#!/usr/bin/env python3
"""可用性测试：PDF-Miner Agent 的 API 封装（核心类 + Flask REST 端点）。

设计原则：
- 全程使用 reuse_existing=True，复用 outputs/mineru_benchmark 里已有的解析产物，
  因此不依赖 GPU、不调用 MinerU 实推理、不挂任何外部平台，CI 内秒级可跑。
- 既覆盖正常路径（产出契约字段齐全），也覆盖错误路径（400/500）。
运行：python3 -m pytest tests/test_pdf_miner_api.py -v
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pdf_miner_agent import PDFMinerAgent, PDFMinerConfig
from pdf_miner_agent.api import app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = ROOT / "data" / "raw_pdfs" / "standard_two_column_attention.pdf"

# 该样例必须在 mineru_benchmark 里有 returncode==0 的 pipeline 解析产物，
# 否则 reuse_existing 找不到目录、测试失去意义——这里先行断言其存在。
pytestmark = pytest.mark.skipif(
    not SAMPLE_PDF.exists(),
    reason=f"缺少样例 PDF：{SAMPLE_PDF}",
)


@pytest.fixture(scope="module")
def client():
    app.config.update(TESTING=True)
    return app.test_client()


# ---------------------------------------------------------------------------
# 1) 核心封装类 PDFMinerAgent
# ---------------------------------------------------------------------------
def test_agent_parse_reuse_existing_contract():
    """核心类可调用，且返回的 package 满足结构化输出契约。"""
    pkg = PDFMinerAgent().parse_pdf(
        SAMPLE_PDF,
        PDFMinerConfig(backend="pipeline", method="auto"),
        reuse_existing=True,
        run_id="pytest_agent",
    )
    assert pkg["execution"]["returncode"] == 0
    # 契约字段：输入、分类、结构化输出、产物路径
    assert pkg["input"]["pdf_name"] == SAMPLE_PDF.name
    assert len(pkg["input"]["sha256"]) == 64
    assert pkg["classification"]["tags"]
    so = pkg["structured_output"]
    assert so["has_markdown"] is True
    assert so["markdown_chars"] > 0
    assert so["content_summary"]["content_items"] > 0
    # 落盘产物
    assert pkg["artifacts"]["markdown"]
    assert pkg["artifacts"]["content_json"]
    assert pkg["artifacts"]["package_json"]
    assert (ROOT / pkg["artifacts"]["package_json"]).exists()


def test_agent_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        PDFMinerAgent().parse_pdf(
            ROOT / "data" / "raw_pdfs" / "__does_not_exist__.pdf",
            PDFMinerConfig(),
            reuse_existing=True,
        )


# ---------------------------------------------------------------------------
# 2) HTTP 层：/health
# ---------------------------------------------------------------------------
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["service"] == "pdf-miner-agent"


# ---------------------------------------------------------------------------
# 3) HTTP 层：/parse （JSON body）
# ---------------------------------------------------------------------------
def test_parse_json_ok(client):
    resp = client.post(
        "/parse",
        json={
            "pdf_path": str(SAMPLE_PDF.relative_to(ROOT)),
            "reuse_existing": True,
            "run_id": "pytest_http_parse",
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["execution"]["returncode"] == 0
    assert body["structured_output"]["has_markdown"] is True
    assert body["structured_output"]["content_summary"]["content_items"] > 0


def test_parse_missing_path_returns_400(client):
    resp = client.post("/parse", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing pdf_path"


def test_parse_bad_file_returns_500(client):
    resp = client.post(
        "/parse",
        json={"pdf_path": "data/raw_pdfs/__nope__.pdf", "reuse_existing": True},
    )
    assert resp.status_code == 500
    assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# 4) HTTP 层：/parse_upload （multipart 上传）
# ---------------------------------------------------------------------------
def test_parse_upload_ok(client):
    data = {
        "reuse_existing": "1",
        "run_id": "pytest_http_upload",
        "file": (io.BytesIO(SAMPLE_PDF.read_bytes()), SAMPLE_PDF.name),
    }
    resp = client.post("/parse_upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert resp.get_json()["structured_output"]["has_markdown"] is True


def test_parse_upload_missing_file_returns_400(client):
    resp = client.post("/parse_upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "error" in resp.get_json()
