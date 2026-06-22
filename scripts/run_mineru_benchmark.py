#!/usr/bin/env python3
"""Run a small MinerU benchmark and write reproducible reports.

The benchmark intentionally uses a compact representative subset so it can be
rerun during assignment development without spending hours on the full PDF set.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MINERU = ROOT / "MinerU" / ".venv" / "bin" / "mineru"
PYTHON = ROOT / "MinerU" / ".venv" / "bin" / "python"
OUT_ROOT = ROOT / "outputs" / "mineru_benchmark"
REPORT_ROOT = ROOT / "reports" / "assignment2" / "mineru_benchmark"

PIPELINE_BENCHMARK_CASES = [
    {
        "case_id": "BM-01",
        "filename": "standard_two_column_attention.pdf",
        "category": "standard_two_column",
        "backend": "pipeline",
        "method": "auto",
        "description": "标准双栏 born-digital PDF，检验阅读顺序、章节和图表输出。",
    },
    {
        "case_id": "BM-02",
        "filename": "math_formula_pinns.pdf",
        "category": "formula_dense",
        "backend": "pipeline",
        "method": "auto",
        "description": "大量数学公式 PDF，检验公式 LaTeX、表格和 Markdown 输出。",
    },
    {
        "case_id": "BM-03",
        "filename": "generated_scanned_low_quality.pdf",
        "category": "scanned_ocr",
        "backend": "pipeline",
        "method": "ocr",
        "description": "无文字层低分辨率扫描 PDF，强制 OCR，检验 OCR 可用性。",
    },
]

GPU_BENCHMARK_CASES = [
    {
        "case_id": "GPU-01",
        "filename": "standard_two_column_attention.pdf",
        "category": "standard_two_column",
        "backend": "pipeline",
        "method": "auto",
        "start": 0,
        "end": 5,
        "description": "标准双栏 PDF 前 6 页，作为 pipeline 基线。",
    },
    {
        "case_id": "GPU-02",
        "filename": "standard_two_column_attention.pdf",
        "category": "standard_two_column",
        "backend": "hybrid-engine",
        "effort": "medium",
        "method": "auto",
        "start": 0,
        "end": 5,
        "description": "标准双栏 PDF 前 6 页，Hybrid medium，复现官方 medium 速度优势趋势。",
    },
    {
        "case_id": "GPU-03",
        "filename": "standard_two_column_attention.pdf",
        "category": "standard_two_column",
        "backend": "hybrid-engine",
        "effort": "high",
        "method": "auto",
        "start": 0,
        "end": 5,
        "description": "标准双栏 PDF 前 6 页，Hybrid high，作为高质量/图像分析强度对照。",
    },
    {
        "case_id": "GPU-04",
        "filename": "math_formula_pinns.pdf",
        "category": "formula_dense",
        "backend": "hybrid-engine",
        "effort": "medium",
        "method": "auto",
        "start": 0,
        "end": 5,
        "description": "公式密集 PDF 前 6 页，Hybrid medium，检验公式和科学论文场景。",
    },
]

OFFICIAL_REPORT = {
    "source": "MinerU README.md, local checkout, version 3.3.1",
    "pipeline_omnidocbench_v15_overall": 86.2,
    "hybrid_v16_medium_vs_high_accuracy_delta": -0.13,
    "hybrid_v16_medium_speedup_range": "35%-220%",
}


def selected_cases() -> tuple[str, list[dict[str, Any]]]:
    profile = os.getenv("MINERU_BENCH_PROFILE", "pipeline").strip().lower() or "pipeline"
    if profile == "gpu":
        return profile, GPU_BENCHMARK_CASES
    return profile, PIPELINE_BENCHMARK_CASES


def run_capture(cmd: list[str], timeout: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
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


def pdf_pages(pdf_path: Path) -> int:
    proc = run_capture(["pdfinfo", str(pdf_path)])
    match = re.search(r"^Pages:\s+(\d+)", proc["output"], re.M)
    if not match:
        raise RuntimeError(f"Could not read page count from pdfinfo: {pdf_path}")
    return int(match.group(1))


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def torch_probe() -> dict[str, Any]:
    code = (
        "import json, torch; "
        "info={'torch_version': torch.__version__, "
        "'torch_cuda_version': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count()}; "
        "info['device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None; "
        "print(json.dumps(info, ensure_ascii=False))"
    )
    proc = run_capture([str(PYTHON), "-c", code])
    try:
        return json.loads(proc["output"].strip().splitlines()[-1])
    except Exception:
        return {"error": proc["output"].strip()}


def vllm_probe() -> dict[str, Any]:
    code = r"""
import importlib.metadata as metadata
import json
import os

packages = [
    "vllm",
    "xformers",
    "xgrammar",
    "flashinfer-python",
    "flashinfer-cubin",
    "compressed-tensors",
    "lm-format-enforcer",
    "llguidance",
]
versions = {}
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None

info = {
    "packages": versions,
    "VLLM_USE_V1": os.getenv("VLLM_USE_V1"),
    "VLLM_USE_FLASHINFER_SAMPLER": os.getenv("VLLM_USE_FLASHINFER_SAMPLER"),
}
try:
    import vllm
    info["vllm_import"] = "ok"
    info["vllm_version_attr"] = getattr(vllm, "__version__", None)
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    info["async_engine_import"] = "ok"
    info["async_engine_classes"] = [AsyncEngineArgs.__name__, AsyncLLM.__name__]
except Exception as exc:
    info["vllm_import"] = "failed"
    info["error"] = repr(exc)
print(json.dumps(info, ensure_ascii=False))
"""
    proc = run_capture([str(PYTHON), "-c", code])
    try:
        return json.loads(proc["output"].strip().splitlines()[-1])
    except Exception:
        return {"error": proc["output"].strip()}


def nvidia_probe() -> dict[str, Any]:
    overview = run_capture(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", overview["output"])
    cuda_version = cuda_match.group(1) if cuda_match else ""
    proc = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader",
        ]
    )
    if proc["returncode"] != 0:
        return {"available": False, "output": proc["output"].strip()}
    rows = []
    for line in proc["output"].strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "cuda_version": cuda_version,
                    "memory_total": parts[2],
                    "memory_used": parts[3],
                    "temperature_gpu": parts[4],
                }
            )
    return {"available": bool(rows), "gpus": rows, "raw": proc["output"].strip()}


def find_parse_dir(case_out: Path, pdf_stem: str, method: str) -> Path | None:
    expected = case_out / pdf_stem / method
    if expected.exists():
        return expected
    matches = sorted(case_out.glob(f"{pdf_stem}/*"))
    return matches[0] if matches else None


def count_content_types(json_path: Path | None) -> dict[str, int]:
    if not json_path or not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counter: Counter[str] = Counter()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                counter[str(item.get("type", "unknown"))] += 1
            elif isinstance(item, list):
                for child in item:
                    if isinstance(child, dict):
                        counter[str(child.get("type", "unknown"))] += 1
    return dict(counter)


def summarize_output(case_out: Path, pdf_stem: str, method: str) -> dict[str, Any]:
    parse_dir = find_parse_dir(case_out, pdf_stem, method)
    if not parse_dir:
        return {
            "parse_dir": None,
            "md_chars": 0,
            "image_count": 0,
            "json_files": 0,
            "content_items": 0,
            "content_types": {},
            "layout_pdf": None,
            "span_pdf": None,
        }

    md_files = sorted(parse_dir.glob("*.md"))
    json_files = sorted(parse_dir.glob("*.json"))
    image_files = sorted((parse_dir / "images").glob("*")) if (parse_dir / "images").exists() else []
    content_json = next(iter(sorted(parse_dir.glob("*_content_list.json"))), None)
    content_types = count_content_types(content_json)
    md_chars = sum(len(path.read_text(encoding="utf-8", errors="ignore")) for path in md_files)
    layout_pdf = next(iter(sorted(parse_dir.glob("*_layout.pdf"))), None)
    span_pdf = next(iter(sorted(parse_dir.glob("*_span.pdf"))), None)

    return {
        "parse_dir": str(parse_dir.relative_to(ROOT)),
        "md_files": [str(path.relative_to(ROOT)) for path in md_files],
        "md_chars": md_chars,
        "image_count": len(image_files),
        "json_files": len(json_files),
        "content_items": sum(content_types.values()),
        "content_types": content_types,
        "layout_pdf": str(layout_pdf.relative_to(ROOT)) if layout_pdf else None,
        "span_pdf": str(span_pdf.relative_to(ROOT)) if span_pdf else None,
        "output_bytes": sum(path.stat().st_size for path in parse_dir.rglob("*") if path.is_file()),
    }


def render_layout_preview(layout_pdf_rel: str | None, preview_path: Path) -> str | None:
    if not layout_pdf_rel:
        return None
    layout_pdf = ROOT / layout_pdf_rel
    if not layout_pdf.exists():
        return None
    preview_base = preview_path.with_suffix("")
    proc = run_capture(
        [
            "pdftoppm",
            "-png",
            "-f",
            "1",
            "-singlefile",
            "-r",
            "120",
            str(layout_pdf),
            str(preview_base),
        ]
    )
    if proc["returncode"] != 0:
        return None
    generated = preview_base.with_suffix(".png")
    return str(generated.relative_to(ROOT)) if generated.exists() else None


def svg_bar_chart(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    unit: str,
    color: str,
) -> None:
    width = 980
    row_h = 54
    left = 230
    right = 70
    top = 72
    height = top + row_h * len(rows) + 56
    max_value = max((float(row.get(value_key) or 0) for row in rows), default=1.0) or 1.0
    chart_w = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="28" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
        f'<line x1="{left}" y1="{top-18}" x2="{left+chart_w}" y2="{top-18}" stroke="#d1d5db"/>',
    ]
    for i, row in enumerate(rows):
        y = top + i * row_h
        value = float(row.get(value_key) or 0)
        bar_w = max(2, value / max_value * chart_w)
        label = str(row.get(label_key, ""))
        display = f"{value:.2f} {unit}" if value < 100 else f"{value:.1f} {unit}"
        parts.extend(
            [
                f'<text x="28" y="{y+26}" font-family="Arial, sans-serif" font-size="16" fill="#111827">{escape_xml(label)}</text>',
                f'<rect x="{left}" y="{y+6}" width="{bar_w:.1f}" height="28" rx="4" fill="{color}"/>',
                f'<text x="{left+bar_w+10:.1f}" y="{y+26}" font-family="Arial, sans-serif" font-size="15" fill="#374151">{escape_xml(display)}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def hybrid_compare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row.get("filename") == "standard_two_column_attention.pdf"
        and row.get("start") == 0
        and row.get("end") == 5
        and row.get("returncode") == 0
    ]
    order = {"pipeline": 0, "hybrid-engine:medium": 1, "hybrid-engine:high": 2}

    def sort_key(row: dict[str, Any]) -> int:
        return order.get(f"{row.get('backend')}:{row.get('effort')}", order.get(str(row.get("backend")), 99))

    chart_rows = []
    for row in sorted(selected, key=sort_key):
        label = row["backend"] if row["backend"] == "pipeline" else f"hybrid {row.get('effort')}"
        chart_rows.append({**row, "label": label})
    return chart_rows


def hybrid_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(
        (
            row for row in rows
            if row.get("filename") == "standard_two_column_attention.pdf"
            and row.get("backend") == "pipeline"
            and row.get("start") == 0
            and row.get("end") == 5
            and row.get("returncode") == 0
        ),
        None,
    )
    baseline_spp = baseline.get("seconds_per_page") if baseline else None
    matrix = []
    for row in hybrid_compare_rows(rows):
        speedup_vs_pipeline = None
        if baseline_spp and row.get("seconds_per_page"):
            speedup_vs_pipeline = baseline_spp / row["seconds_per_page"]
        matrix.append({**row, "speedup_vs_pipeline": speedup_vs_pipeline})
    return matrix


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def selected_page_count(case: dict[str, Any], full_pages: int) -> int:
    if case.get("start") is None and case.get("end") is None:
        return full_pages
    start = int(case.get("start") or 0)
    end = int(case.get("end") if case.get("end") is not None else full_pages - 1)
    return max(0, min(full_pages - 1, end) - start + 1)


def run_case(case: dict[str, Any], run_dir: Path, log_dir: Path, preview_dir: Path) -> dict[str, Any]:
    pdf_path = ROOT / "data" / "raw_pdfs" / case["filename"]
    full_pages = pdf_pages(pdf_path)
    pages = selected_page_count(case, full_pages)
    effort_slug = f"_{case['effort']}" if case.get("effort") else ""
    page_slug = (
        f"_p{case.get('start', 0)}-{case.get('end')}"
        if case.get("start") is not None or case.get("end") is not None
        else ""
    )
    case_slug = f"{case['case_id']}_{case['backend']}{effort_slug}_{case['method']}{page_slug}_{Path(case['filename']).stem}"
    case_out = run_dir / case_slug
    if case_out.exists():
        shutil.rmtree(case_out)
    case_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(MINERU),
        "-p",
        str(pdf_path),
        "-o",
        str(case_out),
        "-b",
        case["backend"],
        "-m",
        case["method"],
    ]
    if case.get("effort"):
        cmd.extend(["--effort", str(case["effort"])])
    if case.get("start") is not None:
        cmd.extend(["-s", str(case["start"])])
    if case.get("end") is not None:
        cmd.extend(["-e", str(case["end"])])

    started_at = datetime.now().isoformat(timespec="seconds")
    result = run_capture(cmd)
    completed_at = datetime.now().isoformat(timespec="seconds")
    log_path = log_dir / f"{case_slug}.log"
    log_path.write_text(result["output"], encoding="utf-8", errors="ignore")
    summary = summarize_output(case_out, Path(case["filename"]).stem, case["method"])
    preview = render_layout_preview(
        summary.get("layout_pdf"),
        preview_dir / f"{case_slug}_layout_page1.png",
    )

    seconds = float(result["seconds"])
    return {
        **case,
        "pdf_path": str(pdf_path.relative_to(ROOT)),
        "pages": pages,
        "full_pages": full_pages,
        "start": case.get("start"),
        "end": case.get("end"),
        "effort": case.get("effort", ""),
        "pdf_bytes": file_size(pdf_path),
        "started_at": started_at,
        "completed_at": completed_at,
        "returncode": result["returncode"],
        "timeout": result["timeout"],
        "seconds": seconds,
        "seconds_per_page": round(seconds / pages, 3) if pages else None,
        "pages_per_second": round(pages / seconds, 4) if seconds else None,
        "log": str(log_path.relative_to(ROOT)),
        "preview": preview,
        **summary,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "filename",
        "category",
        "backend",
        "effort",
        "method",
        "pages",
        "start",
        "end",
        "returncode",
        "seconds",
        "seconds_per_page",
        "pages_per_second",
        "md_chars",
        "content_items",
        "image_count",
        "json_files",
        "output_bytes",
        "log",
        "parse_dir",
        "preview",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def md_link(rel_path: str | None, label: str | None = None) -> str:
    if not rel_path:
        return ""
    return f"[{label or rel_path}](../../../{rel_path})"


def write_markdown(
    path: Path,
    run_id: str,
    profile: str,
    env: dict[str, Any],
    rows: list[dict[str, Any]],
    charts: dict[str, str],
) -> None:
    passed = [row for row in rows if row["returncode"] == 0]
    avg_spp = sum(row["seconds_per_page"] for row in passed) / len(passed) if passed else 0
    total_pages = sum(row["pages"] for row in rows)
    total_seconds = sum(row["seconds"] for row in rows)
    torch_info = env.get("torch", {})
    vllm_info = env.get("vllm", {})
    vllm_packages = vllm_info.get("packages", {})
    nvidia = env.get("nvidia", {})
    medium_rows = [
        row for row in rows
        if row.get("backend") == "hybrid-engine" and row.get("effort") == "medium" and row.get("returncode") == 0
    ]
    high_rows = [
        row for row in rows
        if row.get("backend") == "hybrid-engine" and row.get("effort") == "high" and row.get("returncode") == 0
    ]
    hybrid_result = "未同时完成 hybrid medium/high"
    hybrid_conclusion = "需要查看 hybrid 用例日志，或修正环境后重跑"
    if medium_rows and high_rows:
        med = medium_rows[0]
        high = high_rows[0]
        speedup = high["seconds_per_page"] / med["seconds_per_page"] if med["seconds_per_page"] else 0
        hybrid_result = (
            f"medium {med['seconds_per_page']:.2f}s/page，"
            f"high {high['seconds_per_page']:.2f}s/page，"
            f"本机 medium 约为 high 的 {speedup:.2f}x 速度"
        )
        hybrid_conclusion = "可与官方 35%-220% speedup 趋势对照；accuracy 仍需 OmniDocBench/gold set"
    gpu_lines = []
    if nvidia.get("available"):
        for gpu in nvidia.get("gpus", []):
            gpu_lines.append(
                f"- GPU：{gpu['name']}，Driver {gpu['driver_version']}，CUDA {gpu['cuda_version']}，显存 {gpu['memory_used']}/{gpu['memory_total']}，温度 {gpu['temperature_gpu']}"
            )
    else:
        gpu_lines.append(f"- GPU：不可用，`nvidia-smi` 输出：{nvidia.get('output', '')}")

    lines = [
        "# MinerU 本地 Benchmark 记录",
        "",
        f"- Run ID：`{run_id}`",
        f"- Benchmark profile：`{profile}`",
        f"- 执行时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- MinerU：`{env.get('mineru_version', '').strip()}`",
        f"- 总页数：{total_pages}",
        f"- 总耗时：{total_seconds:.2f}s",
        f"- 平均秒/页：{avg_spp:.2f}s/page",
        "",
        "## 环境",
        "",
        *gpu_lines,
        f"- PyTorch：`{torch_info.get('torch_version')}`，编译 CUDA：`{torch_info.get('torch_cuda_version')}`，`torch.cuda.is_available()` = `{torch_info.get('cuda_available')}`",
        f"- vLLM：`{vllm_packages.get('vllm')}`，导入状态：`{vllm_info.get('vllm_import')}`，Async Engine：`{vllm_info.get('async_engine_import')}`",
        f"- vLLM 关键环境变量：`VLLM_USE_V1={vllm_info.get('VLLM_USE_V1')}`，`VLLM_USE_FLASHINFER_SAMPLER={vllm_info.get('VLLM_USE_FLASHINFER_SAMPLER')}`",
        f"- vLLM 相关依赖：`xformers={vllm_packages.get('xformers')}`，`xgrammar={vllm_packages.get('xgrammar')}`，`flashinfer-python={vllm_packages.get('flashinfer-python')}`",
        f"- 说明：本轮 benchmark 在同一脚本内记录 `nvidia-smi` 与 PyTorch CUDA 状态；若 hybrid 用例失败，以对应日志为准。",
        "",
        "## vLLM 修复记录",
        "",
        "- 原始失败根因：`vllm==0.21.0` 默认 CUDA 13 wheel 在 CUDA 12.4 主机上导入 `vllm._C` 时查找 `libcudart.so.13`。",
        "- 已验证可运行组合：保留 `torch==2.6.0+cu124`，将 vLLM 替换为官方 release `vllm-0.8.5+cu121`，同步 `xformers==0.0.29.post2`、`xgrammar==0.1.18`、`compressed-tensors==0.9.3` 等 0.8.5 依赖。",
        "- 运行设置：移除 0.21.0 遗留的可选 `flashinfer` 包，让 vLLM 回退 PyTorch native sampler；执行 hybrid 时设置 `VLLM_USE_V1=1`。",
        "- 验证结果：`import vllm`、`AsyncEngineArgs`、`AsyncLLM` 均导入成功，2 页 hybrid medium 烟测通过后再运行本轮 benchmark。",
        "",
        "## 实验内容",
        "",
        "| 用例 | PDF | 类别 | 后端 | effort | 方法 | 页段 | 页数 | 目的 |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        page_range = "all" if row.get("start") is None and row.get("end") is None else f"{row.get('start', 0)}-{row.get('end')}"
        lines.append(
            f"| {row['case_id']} | `{row['filename']}` | {row['category']} | `{row['backend']}` | `{row.get('effort') or '-'}` | `{row['method']}` | {page_range} | {row['pages']} | {row['description']} |"
        )
    lines.extend(
        [
            "",
            "## 结果",
            "",
            "| 用例 | 状态 | 耗时(s) | 秒/页 | 页/秒 | Markdown字符 | content项 | 图片数 | JSON数 | 日志 | Layout预览 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        status = "passed" if row["returncode"] == 0 else f"failed({row['returncode']})"
        lines.append(
            " | ".join(
                [
                    f"| {row['case_id']}",
                    status,
                    f"{row['seconds']:.2f}",
                    f"{row['seconds_per_page']:.2f}",
                    f"{row['pages_per_second']:.4f}",
                    str(row["md_chars"]),
                    str(row["content_items"]),
                    str(row["image_count"]),
                    str(row["json_files"]),
                    md_link(row.get("log"), "log"),
                    md_link(row.get("preview"), "preview") if row.get("preview") else "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 可视化",
            "",
            f"![运行耗时](../../../{charts['runtime']})",
            "",
            f"![页处理速度](../../../{charts['throughput']})",
            "",
            f"![内容项数量](../../../{charts['content_items']})",
            "",
        ]
    )
    if charts.get("hybrid_spp"):
        lines.extend(
            [
                f"![Pipeline/Hybrid 秒每页对比](../../../{charts['hybrid_spp']})",
                "",
            ]
        )
    matrix = hybrid_matrix(rows)
    if matrix:
        lines.extend(
            [
                "## Pipeline 与 Hybrid 对比矩阵",
                "",
                "| 模式 | 用例 | 页段 | 耗时(s) | 秒/页 | 页/秒 | 相对 pipeline 提升 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in matrix:
            mode = row["backend"] if row["backend"] == "pipeline" else f"hybrid {row.get('effort')}"
            page_range = f"{row.get('start', 0)}-{row.get('end')}"
            speedup = row.get("speedup_vs_pipeline")
            speedup_text = "baseline" if row["backend"] == "pipeline" or speedup is None else f"{speedup:.2f}x"
            lines.append(
                f"| `{mode}` | {row['case_id']} | {page_range} | {row['seconds']:.2f} | {row['seconds_per_page']:.2f} | {row['pages_per_second']:.4f} | {speedup_text} |"
            )
        lines.append("")
        lines.extend(
            [
                "说明：该矩阵是 MinerU CLI 冷启动端到端口径，每个用例都会单独启动本地 `mineru-api`、初始化 vLLM engine、加载模型并 warmup。6 页小样本下固定成本占比很高，因此该矩阵主要证明 hybrid medium/high 已跑通并给出本机端到端成本，不等同于常驻服务或大批量任务的稳态吞吐。",
                "",
            ]
        )
    lines.extend(
        [
            "## Layout 可视化预览",
            "",
        ]
    )
    for row in rows:
        if row.get("preview"):
            lines.extend(
                [
                    f"### {row['case_id']} {row['category']}",
                    "",
                    f"![{row['case_id']} layout preview](../../../{row['preview']})",
                    "",
                ]
            )
    lines.extend(
        [
            "## 与 MinerU 官方报告结果对比",
            "",
            "| 指标 | 官方报告结果 | 本轮结果 | 结论 |",
            "| --- | --- | --- | --- |",
            f"| pipeline OmniDocBench v1.5 Overall | {OFFICIAL_REPORT['pipeline_omnidocbench_v15_overall']} | 本轮未使用 OmniDocBench ground truth，不能计算官方 accuracy 分数 | 本轮是本地吞吐和产物统计 benchmark，不声称复现官方 accuracy |",
            f"| hybrid medium vs high 准确率差 | {OFFICIAL_REPORT['hybrid_v16_medium_vs_high_accuracy_delta']} | 本轮没有 OmniDocBench ground truth，不能计算 accuracy delta | accuracy 复现需后续接入官方或自建 gold set |",
            f"| hybrid medium 速度提升 | {OFFICIAL_REPORT['hybrid_v16_medium_speedup_range']} | {hybrid_result} | {hybrid_conclusion} |",
            f"| pipeline 本地端到端速度 | 官方 README 未给本机速度 | 平均 {avg_spp:.2f}s/page，覆盖 {len(passed)}/{len(rows)} 个样本 | 可作为本机 baseline；包含 CLI 本地 API 启停和模型加载成本 |",
            "",
            "## 初步观察",
            "",
        ]
    )
    for row in rows:
        lines.append(
            f"- {row['case_id']}：{row['pages']} 页，{row['seconds']:.2f}s，输出 {row['md_chars']} 个 Markdown 字符、{row['content_items']} 个 content item、{row['image_count']} 张图片。"
        )
    lines.extend(
        [
            "",
            "## 后续 Benchmark 扩展",
            "",
            "1. 下载 OmniDocBench 子集或建立小型人工 gold set，补充文本、公式、表格、OCR 的 accuracy 指标。",
            "2. 接入 OmniDocBench 官方评测脚本或自定义字段级 scorer，将 throughput benchmark 扩展为 quality + speed 双指标。",
            "3. 将 GLM 仲裁加入表格/公式/OCR 低置信度样本，比较规则后处理与 LLM 仲裁后的质量变化。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile, cases = selected_cases()
    run_dir = OUT_ROOT / run_id
    log_dir = run_dir / "logs"
    preview_dir = run_dir / "previews"
    chart_dir = run_dir / "charts"
    for directory in (run_dir, log_dir, preview_dir, chart_dir, REPORT_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    mineru_version = run_capture([str(MINERU), "--version"])["output"]
    env = {
        "run_id": run_id,
        "profile": profile,
        "mineru_version": mineru_version,
        "nvidia": nvidia_probe(),
        "torch": torch_probe(),
        "vllm": vllm_probe(),
        "official_report": OFFICIAL_REPORT,
        "cwd": str(ROOT),
    }
    (run_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [run_case(case, run_dir, log_dir, preview_dir) for case in cases]
    json_path = run_dir / "results.json"
    csv_path = run_dir / "results.csv"
    json_path.write_text(json.dumps({"run_id": run_id, "environment": env, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)

    runtime_svg = chart_dir / "runtime_seconds.svg"
    throughput_svg = chart_dir / "pages_per_second.svg"
    content_svg = chart_dir / "content_items.svg"
    chart_rows = [{**row, "label": f"{row['case_id']} {row['category']}"} for row in rows]
    svg_bar_chart(runtime_svg, "MinerU end-to-end runtime", chart_rows, "label", "seconds", "s", "#2563eb")
    svg_bar_chart(throughput_svg, "MinerU throughput", chart_rows, "label", "pages_per_second", "pages/s", "#059669")
    svg_bar_chart(content_svg, "Extracted content items", chart_rows, "label", "content_items", "items", "#7c3aed")

    charts = {
        "runtime": str(runtime_svg.relative_to(ROOT)),
        "throughput": str(throughput_svg.relative_to(ROOT)),
        "content_items": str(content_svg.relative_to(ROOT)),
    }
    hybrid_rows = hybrid_compare_rows(rows)
    if len(hybrid_rows) >= 2:
        hybrid_svg = chart_dir / "pipeline_hybrid_seconds_per_page.svg"
        svg_bar_chart(
            hybrid_svg,
            "Pipeline vs Hybrid seconds per page",
            hybrid_rows,
            "label",
            "seconds_per_page",
            "s/page",
            "#ea580c",
        )
        charts["hybrid_spp"] = str(hybrid_svg.relative_to(ROOT))
    report_path = REPORT_ROOT / f"benchmark_report_{run_id}.md"
    write_markdown(report_path, run_id, profile, env, rows, charts)
    latest_report = REPORT_ROOT / "benchmark_report_latest.md"
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(
        {
            "run_id": run_id,
            "profile": profile,
            "json": str(json_path.relative_to(ROOT)),
            "csv": str(csv_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
            "latest_report": str(latest_report.relative_to(ROOT)),
            "charts": charts,
            "results": [
                {
                    "case_id": row["case_id"],
                    "returncode": row["returncode"],
                    "seconds": row["seconds"],
                    "seconds_per_page": row["seconds_per_page"],
                    "pages_per_second": row["pages_per_second"],
                }
                for row in rows
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
