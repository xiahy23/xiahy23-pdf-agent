#!/usr/bin/env python3
"""GLM visual arbitration — shared by the offline OmniDocBench experiment and the
online DocGate. Prompt building, the Anthropic-compatible API call, response
parsing, and deterministic rule postprocessing live here so both paths use one
implementation. Credentials are read only from environment variables.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def rule_postprocess(kind: str, value: str) -> str:
    text = value or ""
    if kind == "table":
        text = re.sub(r"\s(rowspan|colspan)=['\"]?1['\"]?", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r">\s+<", "><", text)
        text = re.sub(r"\s+>", ">", text)
        text = re.sub(r"<(td|th)\s+>", r"<\1>", text)
        return text.strip()
    if kind == "formula":
        text = text.strip()
        text = re.sub(r"^\s*\\\[\s*", "", text)
        text = re.sub(r"\s*\\\]\s*$", "", text)
        text = re.sub(r"^\s*\$\$\s*", "", text)
        text = re.sub(r"\s*\$\$\s*$", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def build_prompt(kind: str, pred: str, rule_value: str | None = None) -> str:
    target = {"formula": "LaTeX formula", "table": "HTML table"}.get(kind, "plain OCR text")
    rule_block = f"\nrule_postprocessed_prediction:\n{rule_value}\n" if rule_value and rule_value != pred else ""
    return (
        f"You are a document recognition arbitration model. The image crop contains one {target} region.\n"
        "Given the current MinerU prediction, return only a compact JSON object with keys: "
        "`kind`, `corrected`, `rationale`.\n"
        "Do not include markdown fences. Keep `corrected` as the final recognized content only.\n\n"
        f"kind: {kind}\n"
        f"current_prediction:\n{pred or ''}\n"
        f"{rule_block}"
    )


def parse_corrected(response: str) -> str:
    """Pull the `corrected` field out of a (possibly doubly-encoded) GLM response."""
    try:
        data = json.loads(response)
        content = data.get("content", [])
        text = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    except Exception:
        text = response
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            corrected: Any = obj.get("corrected", "")
            for _ in range(3):
                if not isinstance(corrected, str) or not corrected.strip().startswith("{"):
                    break
                try:
                    nested = json.loads(corrected.strip())
                except Exception:
                    break
                if not isinstance(nested, dict) or "corrected" not in nested:
                    break
                corrected = nested.get("corrected", "")
            return str(corrected)
        except Exception:
            m = re.search(r'"corrected"\s*:\s*"(.*?)"\s*,\s*"rationale"', match.group(0), flags=re.S)
            if m:
                return m.group(1)
    return text.strip()


def render_pdf_region(pdf_path: Path, page_idx: int, bbox: list[float], out_path: Path,
                      dpi: int = 200, pad: int = 8) -> bool:
    """Rasterize one page of the *user's* PDF with pdftoppm and crop the bbox.

    MinerU bboxes are in PDF point space; pdftoppm renders at `dpi`, so we scale
    by dpi/72. Returns False (never raises) so the caller can fall back to MinerU.
    """
    try:
        from PIL import Image

        out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = out_path.with_suffix("")
        proc = subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page_idx + 1), "-l", str(page_idx + 1),
             str(pdf_path), str(stem)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120, check=False,
        )
        if proc.returncode != 0:
            return False
        rendered = sorted(stem.parent.glob(f"{stem.name}*.png"))
        if not rendered:
            return False
        page_png = rendered[0]
        img = Image.open(page_png).convert("RGB")
        s = dpi / 72.0
        x0, y0, x1, y1 = (v * s for v in bbox)
        left = max(0, int(x0) - pad)
        top = max(0, int(y0) - pad)
        right = min(img.width, int(x1) + pad)
        bottom = min(img.height, int(y1) + pad)
        if right <= left or bottom <= top:
            return False
        img.crop((left, top, right, bottom)).save(out_path)
        if page_png != out_path:
            page_png.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def call_glm(kind: str, pred: str, crop_path: Path, rule_value: str | None = None,
             model: str | None = None, max_tokens: int = 2000) -> dict[str, Any]:
    """Call the Anthropic-compatible GLM endpoint with the crop. Never raises."""
    token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic").rstrip("/")
    model = model or os.getenv("GLM_ARBITER_MODEL", "glm-4.5v")
    if not token:
        return {"status": "skipped", "reason": "ANTHROPIC_AUTH_TOKEN is not set"}
    if not crop_path.exists():
        return {"status": "skipped", "reason": "crop missing"}
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(kind, pred, rule_value)},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(crop_path.read_bytes()).decode("ascii")}},
            ],
        }],
    }
    started = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"x-api-key": token, "authorization": f"Bearer {token}",
                 "anthropic-version": "2023-06-01", "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return {"status": "ok", "http_status": resp.status, "model": model,
                    "seconds": round(time.perf_counter() - started, 3),
                    "response": resp.read().decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as exc:
        return {"status": "failed", "http_status": exc.code, "model": model,
                "seconds": round(time.perf_counter() - started, 3),
                "response": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"status": "failed", "model": model,
                "seconds": round(time.perf_counter() - started, 3), "response": repr(exc)}
