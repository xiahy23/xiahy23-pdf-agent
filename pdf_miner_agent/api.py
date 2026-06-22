#!/usr/bin/env python3
"""Flask REST API for PDF-Miner Agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from .agent import PDFMinerAgent, PDFMinerConfig


app = Flask(__name__)
agent = PDFMinerAgent()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "pdf-miner-agent"})


@app.route("/parse", methods=["POST"])
def parse_pdf():
    payload = request.get_json(silent=True) or {}
    pdf_path = payload.get("pdf_path")
    if not pdf_path:
        return jsonify({"error": "missing pdf_path"}), 400
    try:
        package = agent.parse_pdf(
            Path(pdf_path),
            config=PDFMinerConfig(
                backend=payload.get("backend", "pipeline"),
                method=payload.get("method", "auto"),
                effort=payload.get("effort"),
                start_page=payload.get("start_page"),
                end_page=payload.get("end_page"),
                timeout_sec=int(payload.get("timeout_sec", 900)),
                force=bool(payload.get("force", False)),
                enable_glm_gate=bool(payload.get("enable_glm_gate", True)),
                glm_max_calls=int(payload.get("glm_max_calls", 12)),
                text_gate=payload.get("text_gate", "log_only"),
                glm_dry_run=bool(payload.get("glm_dry_run", False)),
            ),
            reuse_existing=bool(payload.get("reuse_existing", False)),
            run_id=payload.get("run_id"),
        )
        return jsonify(package)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/parse_upload", methods=["POST"])
def parse_upload():
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "missing multipart file field `file`"}), 400
    suffix = Path(uploaded.filename or "input.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(prefix="pdf_miner_upload_", suffix=suffix, delete=False) as tmp:
        uploaded.save(tmp.name)
        tmp_path = Path(tmp.name)
    try:
        package = agent.parse_pdf(
            tmp_path,
            config=PDFMinerConfig(
                backend=request.form.get("backend", "pipeline"),
                method=request.form.get("method", "auto"),
                effort=request.form.get("effort") or None,
                start_page=int(request.form["start_page"]) if request.form.get("start_page") else None,
                end_page=int(request.form["end_page"]) if request.form.get("end_page") else None,
                timeout_sec=int(request.form.get("timeout_sec", 900)),
                force=request.form.get("force", "0") in {"1", "true", "True"},
                enable_glm_gate=request.form.get("enable_glm_gate", "1") in {"1", "true", "True"},
                glm_max_calls=int(request.form.get("glm_max_calls", 12)),
                text_gate=request.form.get("text_gate", "log_only"),
                glm_dry_run=request.form.get("glm_dry_run", "0") in {"1", "true", "True"},
            ),
            reuse_existing=request.form.get("reuse_existing", "0") in {"1", "true", "True"},
            run_id=request.form.get("run_id") or None,
        )
        return jsonify(package)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765, debug=False)
