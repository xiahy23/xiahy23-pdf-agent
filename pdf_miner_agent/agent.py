#!/usr/bin/env python3
"""Reusable PDF-Miner Agent core.

The agent wraps MinerU outputs into a platform-friendly JSON/Markdown package.
It can either run MinerU on a new PDF or reuse an existing parse directory from
the benchmark artifacts. The latter keeps SciPilot integration tests fast while
still exercising the same output contract.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MINERU_BIN = ROOT / "MinerU" / ".venv" / "bin" / "mineru"
OUT_ROOT = ROOT / "outputs" / "pdf_miner_agent"
QUALITY_RESULTS = ROOT / "outputs" / "omnidocbench_quality" / "20260613_142721" / "quality_results.json"
GLM_SUMMARY = ROOT / "outputs" / "omnidocbench_quality" / "20260613_142721" / "glm_arbitration" / "glm_arbitration_summary.json"


@dataclass
class PDFMinerConfig:
    backend: str = "pipeline"
    method: str = "auto"
    effort: str | None = None
    start_page: int | None = None
    end_page: int | None = None
    timeout_sec: int = 900
    force: bool = False


def rel(path: Path | None) -> str | None:
    if not path:
        return None
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run_capture(cmd: list[str], cwd: Path = ROOT, timeout: int | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=proc_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "output": proc.stdout,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "seconds": round(time.perf_counter() - started, 3),
            "output": exc.stdout or "",
            "timeout": True,
        }


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_info(pdf_path: Path) -> dict[str, Any]:
    proc = run_capture(["pdfinfo", str(pdf_path)])
    info: dict[str, Any] = {"pdfinfo_returncode": proc["returncode"]}
    for line in proc["output"].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        info[key] = value.strip()
    try:
        info["pages"] = int(str(info.get("pages", "0")).split()[0])
    except Exception:
        info["pages"] = None
    return info


def pdf_text_probe(pdf_path: Path, max_pages: int = 2) -> dict[str, Any]:
    proc = run_capture(["pdftotext", "-f", "1", "-l", str(max_pages), str(pdf_path), "-"])
    text = proc["output"] if proc["returncode"] == 0 else ""
    chars = len(text.strip())
    return {
        "text_probe_returncode": proc["returncode"],
        "text_chars_first_pages": chars,
        "has_text_layer": chars >= 80,
        "probe_excerpt": re.sub(r"\s+", " ", text.strip())[:500],
    }


def classify_pdf(pdf_path: Path) -> dict[str, Any]:
    info = pdf_info(pdf_path)
    probe = pdf_text_probe(pdf_path)
    name = pdf_path.name.lower()
    tags: list[str] = []
    if not probe["has_text_layer"]:
        tags.append("scanned_or_rasterized")
    if "formula" in name or "pinn" in name or "fno" in name:
        tags.append("formula_dense")
    if "table" in name or "pdebench" in name:
        tags.append("table_complex")
    if "two_column" in name or "attention" in name:
        tags.append("standard_two_column")
    if "generated_problem" in name or "low_quality" in name:
        tags.append("problem_pdf")
    if not tags:
        tags.append("general_academic_pdf")
    method = "ocr" if "scanned_or_rasterized" in tags else "auto"
    return {**info, **probe, "tags": tags, "recommended_method": method}


def find_parse_dir(case_out: Path, pdf_stem: str, method: str, backend: str, effort: str | None) -> Path | None:
    if backend == "hybrid-engine":
        expected = case_out / pdf_stem / "hybrid_auto"
    else:
        expected = case_out / pdf_stem / method
    if expected.exists():
        return expected
    matches = sorted((case_out / pdf_stem).glob("*")) if (case_out / pdf_stem).exists() else []
    return matches[0] if matches else None


def load_json(path: Path | None) -> Any:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_content_types(content: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                counter[str(item.get("type", "unknown"))] += 1
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict):
                        counter[str(child.get("type", "unknown"))] += 1
    return dict(counter)


def summarize_content(content: Any) -> dict[str, Any]:
    counts = count_content_types(content)
    examples: list[dict[str, Any]] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "unknown"))
            if item_type in {"text", "equation", "table", "image"} and len(examples) < 10:
                text = str(item.get("text") or item.get("latex") or item.get("html") or item.get("caption") or "")
                examples.append({"type": item_type, "text": re.sub(r"\s+", " ", text)[:280]})
    return {"content_types": counts, "content_items": sum(counts.values()), "examples": examples}


def latest_benchmark_parse(pdf_name: str, backend: str = "pipeline", effort: str | None = None) -> Path | None:
    bench_root = ROOT / "outputs" / "mineru_benchmark"
    if not bench_root.exists():
        return None
    target_stem = Path(pdf_name).stem
    for run_dir in sorted(bench_root.glob("*"), reverse=True):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        data = load_json(results_path)
        for row in data.get("results", []) if isinstance(data, dict) else []:
            if row.get("filename") != pdf_name or row.get("returncode") != 0:
                continue
            if row.get("backend") != backend:
                continue
            if effort and row.get("effort") != effort:
                continue
            parse_dir = ROOT / row.get("parse_dir", "")
            if parse_dir.exists() and parse_dir.name in {"auto", "ocr", "hybrid_auto"}:
                return parse_dir
        fallback = sorted(run_dir.glob(f"**/{target_stem}.md"))
        if fallback:
            return fallback[0].parent
    return None


class PDFMinerAgent:
    def __init__(self, out_root: Path = OUT_ROOT):
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)

    def parse_pdf(
        self,
        pdf_path: Path,
        config: PDFMinerConfig | None = None,
        reuse_existing: bool = False,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        config = config or PDFMinerConfig()
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        classification = classify_pdf(pdf_path)
        method = config.method
        if method == "auto" and classification["recommended_method"] == "ocr":
            method = "ocr"

        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = f"{pdf_path.stem}_{config.backend}_{config.effort or method}"
        case_out = self.out_root / run_id / slug
        log_dir = self.out_root / run_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if config.force and case_out.exists():
            shutil.rmtree(case_out)
        case_out.mkdir(parents=True, exist_ok=True)

        parse_dir = latest_benchmark_parse(pdf_path.name, config.backend, config.effort) if reuse_existing else None
        result: dict[str, Any]
        if parse_dir:
            result = {
                "cmd": ["reuse_existing", str(parse_dir)],
                "returncode": 0,
                "seconds": 0.0,
                "output": "reused existing benchmark parse directory",
                "timeout": False,
            }
        else:
            cmd = [
                str(MINERU_BIN),
                "-p",
                str(pdf_path),
                "-o",
                str(case_out),
                "-b",
                config.backend,
                "-m",
                method,
            ]
            if config.effort:
                cmd.extend(["--effort", config.effort])
            if config.start_page is not None:
                cmd.extend(["-s", str(config.start_page)])
            if config.end_page is not None:
                cmd.extend(["-e", str(config.end_page)])
            result = run_capture(cmd, timeout=config.timeout_sec, env={"VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1")})
            parse_dir = find_parse_dir(case_out, pdf_path.stem, method, config.backend, config.effort)

        log_path = log_dir / f"{slug}.log"
        log_path.write_text(result["output"], encoding="utf-8", errors="ignore")
        package = self._build_package(
            pdf_path=pdf_path,
            parse_dir=parse_dir,
            command_result=result,
            classification=classification,
            config={**config.__dict__, "method_resolved": method},
            run_id=run_id,
            log_path=log_path,
        )
        package_path = self.out_root / run_id / f"{slug}_package.json"
        package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        package["artifacts"]["package_json"] = rel(package_path)
        package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = self.write_markdown_summary(package, self.out_root / run_id / f"{slug}_summary.md")
        package["artifacts"]["summary_markdown"] = rel(md_path)
        package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        return package

    def _build_package(
        self,
        *,
        pdf_path: Path,
        parse_dir: Path | None,
        command_result: dict[str, Any],
        classification: dict[str, Any],
        config: dict[str, Any],
        run_id: str,
        log_path: Path,
    ) -> dict[str, Any]:
        md_file = next(iter(sorted(parse_dir.glob("*.md"))), None) if parse_dir else None
        content_file = next(iter(sorted(parse_dir.glob("*_content_list.json"))), None) if parse_dir else None
        middle_file = next(iter(sorted(parse_dir.glob("*_middle.json"))), None) if parse_dir else None
        model_file = next(iter(sorted(parse_dir.glob("*_model.json"))), None) if parse_dir else None
        layout_pdf = next(iter(sorted(parse_dir.glob("*_layout.pdf"))), None) if parse_dir else None
        span_pdf = next(iter(sorted(parse_dir.glob("*_span.pdf"))), None) if parse_dir else None
        image_files = sorted((parse_dir / "images").glob("*")) if parse_dir and (parse_dir / "images").exists() else []

        markdown = md_file.read_text(encoding="utf-8", errors="ignore") if md_file else ""
        content = load_json(content_file)
        content_summary = summarize_content(content)
        quality = load_json(QUALITY_RESULTS) or {}
        glm = load_json(GLM_SUMMARY) or {}

        return {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "agent": {
                "name": "PDF-Miner Agent",
                "version": "0.2",
                "backend": "MinerU + rule postprocess + OmniDocBench scorer + GLM arbitration",
            },
            "input": {
                "pdf_path": rel(pdf_path),
                "pdf_name": pdf_path.name,
                "sha256": file_sha256(pdf_path),
                "bytes": pdf_path.stat().st_size,
            },
            "classification": classification,
            "config": config,
            "execution": {
                "returncode": command_result["returncode"],
                "seconds": command_result["seconds"],
                "timeout": command_result["timeout"],
                "cmd": command_result["cmd"],
                "log": rel(log_path),
            },
            "structured_output": {
                "markdown_chars": len(markdown),
                "markdown_excerpt": markdown[:1800],
                "content_summary": content_summary,
                "has_markdown": bool(md_file),
                "has_content_json": bool(content_file),
                "image_count": len(image_files),
            },
            "quality_reference": {
                "omnidocbench_quality": (quality.get("quality") or [{}])[0] if isinstance(quality, dict) else {},
                "glm_arbitration": {
                    "status_counts": glm.get("status_counts"),
                    "by_kind": glm.get("by_kind"),
                    "overall": glm.get("overall"),
                } if isinstance(glm, dict) else {},
            },
            "artifacts": {
                "parse_dir": rel(parse_dir),
                "markdown": rel(md_file),
                "content_json": rel(content_file),
                "middle_json": rel(middle_file),
                "model_json": rel(model_file),
                "layout_pdf": rel(layout_pdf),
                "span_pdf": rel(span_pdf),
                "images": [rel(path) for path in image_files[:20]],
            },
        }

    def write_markdown_summary(self, package: dict[str, Any], path: Path) -> Path:
        q = package.get("quality_reference", {}).get("omnidocbench_quality", {})
        content = package.get("structured_output", {}).get("content_summary", {})
        counts = content.get("content_types", {})
        lines = [
            "# PDF-Miner Agent 调研包",
            "",
            f"- PDF: `{package['input']['pdf_name']}`",
            f"- 分类: `{', '.join(package['classification'].get('tags', []))}`",
            f"- MinerU returncode: `{package['execution']['returncode']}`",
            f"- 解析耗时: `{package['execution']['seconds']:.2f}s`",
            f"- Markdown 字符数: `{package['structured_output']['markdown_chars']}`",
            f"- 内容元素统计: `{counts}`",
            "",
            "## 质量基准",
            "",
            f"- OmniDocBench text acc: `{q.get('text_accuracy')}`",
            f"- OCR stress acc: `{q.get('ocr_stress_accuracy')}`",
            f"- formula edit acc: `{q.get('formula_accuracy')}`",
            f"- formula CDM: `{q.get('formula_cdm')}`",
            f"- table TEDS: `{q.get('table_teds')}`",
            f"- reading order acc: `{q.get('reading_order_accuracy')}`",
            "",
            "## 结构化输出摘录",
            "",
            "```markdown",
            package.get("structured_output", {}).get("markdown_excerpt", ""),
            "```",
            "",
            "## 关键产物",
            "",
        ]
        for key, value in package.get("artifacts", {}).items():
            if isinstance(value, list):
                lines.append(f"- {key}: {len(value)} files")
            elif value:
                lines.append(f"- {key}: `{value}`")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
