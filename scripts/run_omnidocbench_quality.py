#!/usr/bin/env python3
"""Run a compact OmniDocBench quality benchmark for MinerU outputs.

The script builds a representative page subset, runs MinerU on OmniDocBench
page images, exports page-level Markdown predictions in the official naming
format, runs the official OmniDocBench end-to-end scorer, and writes a compact
quality+speed report.
"""

from __future__ import annotations

import argparse
import base64
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
OMNI = ROOT / "OmniDocBench"
DATASET = OMNI / "Dataset" / "OmniDocBench.json"
IMAGES = OMNI / "Dataset" / "images"
MINERU = ROOT / "MinerU" / ".venv" / "bin" / "mineru"
OUT_ROOT = ROOT / "outputs" / "omnidocbench_quality"
REPORT_ROOT = ROOT / "reports" / "assignment2" / "omnidocbench_quality"
LOCAL_TEXBIN = ROOT / ".local_tools" / "tinytex" / "texlive" / "bin" / "x86_64-linux"
LOCAL_IMBIN = ROOT / ".local_tools" / "cdm-tools" / "bin"
OMNI_PYTHON = OMNI / ".venv" / "bin" / "python"

DEFAULT_INDICES = [39, 266, 595, 954, 110, 257, 704, 1537]


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


def local_cdm_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path_parts = []
    if LOCAL_TEXBIN.exists():
        path_parts.append(str(LOCAL_TEXBIN))
        env["CDM_TEXLIVE_BIN"] = str(LOCAL_TEXBIN)
        env["OMNIDOCBENCH_TEXLIVE_BIN"] = str(LOCAL_TEXBIN)
    if LOCAL_IMBIN.exists():
        path_parts.append(str(LOCAL_IMBIN))
    if path_parts:
        path_parts.append(os.environ.get("PATH", ""))
        env["PATH"] = ":".join(part for part in path_parts if part)
    env.setdefault("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", "/tmp/mplconfig"))
    env.setdefault("UV_CACHE_DIR", os.environ.get("UV_CACHE_DIR", "/tmp/uv-cache"))
    return env


def load_dataset() -> list[dict[str, Any]]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def item_counts(sample: dict[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in sample.get("layout_dets", []):
        if not item.get("ignore"):
            counter[str(item.get("category_type", ""))] += 1
    return dict(counter)


def select_subset(dataset: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    subset = []
    for idx in indices:
        sample = dict(dataset[idx])
        sample["_source_index"] = idx
        sample["_element_counts"] = item_counts(sample)
        subset.append(sample)
    return subset


def method_slug(backend: str, method: str, effort: str | None) -> str:
    if backend == "pipeline":
        return method
    if backend == "hybrid-engine":
        return "hybrid_auto"
    return method


def find_parse_dir(case_out: Path, image_stem: str, backend: str, method: str, effort: str | None) -> Path | None:
    expected = case_out / image_stem / method_slug(backend, method, effort)
    if expected.exists():
        return expected
    matches = sorted((case_out / image_stem).glob("*")) if (case_out / image_stem).exists() else []
    return matches[0] if matches else None


def find_markdown(parse_dir: Path | None) -> Path | None:
    if not parse_dir:
        return None
    files = sorted(parse_dir.glob("*.md"))
    return files[0] if files else None


def copy_prediction_md(md_path: Path | None, image_name: str, pred_dir: Path) -> Path:
    out_path = pred_dir / f"{Path(image_name).stem}.md"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if md_path and md_path.exists():
        shutil.copy2(md_path, out_path)
    else:
        out_path.write_text("", encoding="utf-8")
    return out_path


def content_stats(parse_dir: Path | None) -> dict[str, Any]:
    if not parse_dir:
        return {"md_chars": 0, "content_items": 0, "image_count": 0, "json_files": 0}
    md_chars = sum(len(path.read_text(encoding="utf-8", errors="ignore")) for path in parse_dir.glob("*.md"))
    image_count = len(list((parse_dir / "images").glob("*"))) if (parse_dir / "images").exists() else 0
    json_files = list(parse_dir.glob("*.json"))
    content_items = 0
    content_json = next(iter(sorted(parse_dir.glob("*_content_list.json"))), None)
    if content_json and content_json.exists():
        try:
            data = json.loads(content_json.read_text(encoding="utf-8"))
            content_items = len(data) if isinstance(data, list) else 0
        except Exception:
            content_items = 0
    return {"md_chars": md_chars, "content_items": content_items, "image_count": image_count, "json_files": len(json_files)}


def run_mineru_subset(
    subset: list[dict[str, Any]],
    run_dir: Path,
    backend: str,
    method: str,
    effort: str | None,
    force: bool,
) -> list[dict[str, Any]]:
    pred_dir = run_dir / "pred_md" / backend_label(backend, effort)
    raw_dir = run_dir / "mineru_raw" / backend_label(backend, effort)
    log_dir = run_dir / "logs" / backend_label(backend, effort)
    pred_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in subset:
        image_name = Path(sample["page_info"]["image_path"]).name
        image_path = IMAGES / image_name
        image_stem = image_path.stem
        case_slug = image_stem
        case_out = raw_dir / case_slug
        pred_path = pred_dir / f"{image_stem}.md"
        log_path = log_dir / f"{case_slug}.log"

        if force and case_out.exists():
            shutil.rmtree(case_out)
        case_out.mkdir(parents=True, exist_ok=True)

        if not force and pred_path.exists() and log_path.exists():
            result = {"returncode": 0, "seconds": 0.0, "output": "reused existing prediction\n", "timeout": False}
        else:
            cmd = [
                str(MINERU),
                "-p",
                str(image_path),
                "-o",
                str(case_out),
                "-b",
                backend,
                "-m",
                method,
            ]
            if effort:
                cmd.extend(["--effort", effort])
            result = run_capture(cmd, timeout=900, env={"VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1")})
            log_path.write_text(result["output"], encoding="utf-8", errors="ignore")

        parse_dir = find_parse_dir(case_out, image_stem, backend, method, effort)
        md_path = find_markdown(parse_dir)
        copied = copy_prediction_md(md_path, image_name, pred_dir)
        stats = content_stats(parse_dir)
        rows.append(
            {
                "source_index": sample["_source_index"],
                "image_name": image_name,
                "data_source": sample["page_info"].get("page_attribute", {}).get("data_source", ""),
                "language": sample["page_info"].get("page_attribute", {}).get("language", ""),
                "layout": sample["page_info"].get("page_attribute", {}).get("layout", ""),
                "special_issue": sample["page_info"].get("page_attribute", {}).get("special_issue", []),
                "counts": sample["_element_counts"],
                "backend": backend,
                "effort": effort or "",
                "returncode": result["returncode"],
                "seconds": result["seconds"],
                "pred_md": str(copied.relative_to(ROOT)),
                "parse_dir": str(parse_dir.relative_to(ROOT)) if parse_dir else "",
                "log": str(log_path.relative_to(ROOT)),
                **stats,
            }
        )
    return rows


def backend_label(backend: str, effort: str | None) -> str:
    if backend == "hybrid-engine":
        return f"hybrid_{effort or 'medium'}"
    return backend


def write_subset_gt(subset: list[dict[str, Any]], run_dir: Path) -> Path:
    gt_path = run_dir / "gt_subset.json"
    clean_subset = []
    for sample in subset:
        copied = {k: v for k, v in sample.items() if not k.startswith("_")}
        clean_subset.append(copied)
    gt_path.write_text(json.dumps(clean_subset, ensure_ascii=False, indent=2), encoding="utf-8")
    return gt_path


def write_eval_config(run_dir: Path, gt_path: Path, pred_dir: Path, label: str, include_cdm: bool = False) -> Path:
    metrics_formula = "[Edit_dist, CDM]" if include_cdm else "[Edit_dist]"
    cdm_cfg = "\n      cdm_workers: 1" if include_cdm else ""
    text = f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: {metrics_formula}{cdm_cfg}
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: 2
      timeout_sec: 60
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {gt_path.resolve()}
    prediction:
      data_path: {pred_dir.resolve()}
    match_method: quick_match
    match_workers: 2
    quick_match_truncated_timeout_sec: 120
    match_timeout_sec: 180
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""
    config_path = run_dir / f"eval_{label}.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def run_official_eval(config_path: Path, label: str, run_dir: Path) -> dict[str, Any]:
    result_dir = run_dir / "official_eval_result" / label
    result_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "CDM_SAVE_VIS": "0",
        "OMNIDOCBENCH_TEDS_TIMEOUT_SEC": "60",
        "OMNIDOCBENCH_TIMEOUT_INPUT_DIR": str((run_dir / "timeout_inputs").resolve()),
    }
    env.update(local_cdm_env())
    if OMNI_PYTHON.exists():
        cmd = [str(OMNI_PYTHON), "pdf_validation.py", "--config", str(config_path.resolve())]
    else:
        cmd = ["uv", "run", "--project", str(OMNI), "python", "pdf_validation.py", "--config", str(config_path.resolve())]
    proc = run_capture(cmd, cwd=OMNI, timeout=1200, env=env)
    log_path = result_dir / "eval.log"
    log_path.write_text(proc["output"], encoding="utf-8", errors="ignore")

    pred_base = Path(json.loads(json.dumps(str(config_path))))  # stable no-op to keep this pure stdlib
    save_prefix = Path(read_config_prediction_path(config_path)).name + "_quick_match"
    result_src = OMNI / "result"
    copied = []
    for path in result_src.glob(f"{save_prefix}*"):
        if path.is_file():
            dst = result_dir / path.name
            shutil.copy2(path, dst)
            copied.append(str(dst.relative_to(ROOT)))

    metric_path = result_dir / f"{save_prefix}_metric_result.json"
    summary_path = result_dir / f"{save_prefix}_run_summary.json"
    return {
        "label": label,
        "returncode": proc["returncode"],
        "seconds": proc["seconds"],
        "config": str(config_path.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "metric_result": str(metric_path.relative_to(ROOT)) if metric_path.exists() else "",
        "run_summary": str(summary_path.relative_to(ROOT)) if summary_path.exists() else "",
        "copied_files": copied,
    }


def read_config_prediction_path(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"prediction:\s*\n\s*data_path:\s*(.+)", text)
    if not match:
        return ""
    return match.group(1).strip()


def load_metric_result(eval_result: dict[str, Any]) -> dict[str, Any]:
    rel = eval_result.get("metric_result")
    if not rel:
        return {}
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_quality_scores(metric_result: dict[str, Any]) -> dict[str, Any]:
    def get(path: list[str], default: Any = None) -> Any:
        node: Any = metric_result
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    text_edit = get(["text_block", "all", "Edit_dist", "ALL_page_avg"])
    formula_edit = get(["display_formula", "all", "Edit_dist", "ALL_page_avg"])
    formula_cdm = get(["display_formula", "page", "CDM", "ALL"])
    if formula_cdm is None:
        formula_cdm = get(["display_formula", "all", "CDM", "all"])
    table_teds = get(["table", "all", "TEDS", "all"])
    table_edit = get(["table", "all", "Edit_dist", "ALL_page_avg"])
    reading_order_edit = get(["reading_order", "all", "Edit_dist", "ALL_page_avg"])
    quality = {
        "text_edit": text_edit,
        "text_accuracy": (1 - text_edit) if isinstance(text_edit, (int, float)) else None,
        "formula_edit": formula_edit,
        "formula_accuracy": (1 - formula_edit) if isinstance(formula_edit, (int, float)) else None,
        "formula_cdm": formula_cdm,
        "table_teds": table_teds,
        "table_edit": table_edit,
        "table_accuracy": table_teds if isinstance(table_teds, (int, float)) else None,
        "reading_order_edit": reading_order_edit,
        "reading_order_accuracy": (1 - reading_order_edit) if isinstance(reading_order_edit, (int, float)) else None,
    }
    formula_for_overall = quality.get("formula_cdm")
    if not isinstance(formula_for_overall, (int, float)):
        formula_for_overall = quality.get("formula_accuracy")
    vals = [quality.get("text_accuracy"), formula_for_overall, quality.get("table_accuracy")]
    vals = [v for v in vals if isinstance(v, (int, float))]
    quality["overall_proxy"] = sum(vals) / len(vals) if vals else None
    return quality


def extract_page_breakdowns(metric_result: dict[str, Any]) -> dict[str, dict[str, float]]:
    breakdowns: dict[str, dict[str, float]] = {}
    mapping = {
        "text_ocr_edit_by_page_attr": ["text_block", "page", "Edit_dist"],
        "formula_edit_by_page_attr": ["display_formula", "page", "Edit_dist"],
        "formula_cdm_by_page_attr": ["display_formula", "page", "CDM"],
        "table_teds_by_page_attr": ["table", "page", "TEDS"],
        "table_edit_by_page_attr": ["table", "page", "Edit_dist"],
        "reading_order_edit_by_page_attr": ["reading_order", "page", "Edit_dist"],
    }
    for name, path in mapping.items():
        node: Any = metric_result
        for part in path:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            breakdowns[name] = {str(k): float(v) for k, v in node.items() if isinstance(v, (int, float))}
    return breakdowns


def load_per_page_edit(eval_result: dict[str, Any], element: str) -> dict[str, float]:
    result_dir = ROOT / Path(eval_result["log"]).parent
    prefix = Path(read_config_prediction_path(ROOT / eval_result["config"])).name + "_quick_match"
    path = result_dir / f"{prefix}_{element}_per_page_edit.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in data.items() if isinstance(value, (int, float))}


def extract_ocr_subset_quality(subset: list[dict[str, Any]], eval_result: dict[str, Any]) -> dict[str, Any]:
    per_page = load_per_page_edit(eval_result, "text_block")
    stress_images: list[str] = []
    for sample in subset:
        attrs = sample["page_info"].get("page_attribute", {})
        issues = set(attrs.get("special_issue", []) or [])
        source = attrs.get("data_source", "")
        image_name = Path(sample["page_info"]["image_path"]).name
        if source in {"note", "historical_document"} or {"fuzzy_content", "geometric_deformation"} & issues:
            stress_images.append(image_name)
    values = [per_page[name] for name in stress_images if name in per_page]
    edit = sum(values) / len(values) if values else None
    return {
        "ocr_stress_pages": stress_images,
        "ocr_stress_page_count": len(values),
        "ocr_stress_text_edit": edit,
        "ocr_stress_accuracy": (1 - edit) if isinstance(edit, (int, float)) else None,
    }


def svg_bar_chart(path: Path, title: str, rows: list[dict[str, Any]], key: str, unit: str) -> None:
    width = 920
    row_h = 48
    left = 260
    top = 68
    right = 80
    height = top + row_h * len(rows) + 48
    max_v = max((float(row.get(key) or 0) for row in rows), default=1.0) or 1.0
    chart_w = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="28" y="38" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        value = float(row.get(key) or 0)
        bar_w = max(2, value / max_v * chart_w)
        label = str(row.get("label") or row.get("metric") or "")
        parts.extend([
            f'<text x="28" y="{y+24}" font-family="Arial, sans-serif" font-size="15" fill="#111827">{escape_xml(label)}</text>',
            f'<rect x="{left}" y="{y+6}" width="{bar_w:.1f}" height="26" rx="4" fill="#2563eb"/>',
            f'<text x="{left+bar_w+10:.1f}" y="{y+24}" font-family="Arial, sans-serif" font-size="14" fill="#374151">{value:.3f} {escape_xml(unit)}</text>',
        ])
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def collect_low_confidence(metric_result: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    specs = [
        ("text_block", "text", "edit", True),
        ("display_formula", "formula", "edit", True),
        ("table", "table", "edit", True),
    ]
    for element, kind, score_field, higher_bad in specs:
        result_items = metric_result.get(element, {})
        if not isinstance(result_items, dict):
            continue
        samples = result_items.get("all_samples") or result_items.get("samples")
        if not samples:
            samples = []
        if not samples and "all" in result_items:
            # Official result samples are normally saved in *_result.json, so this
            # fallback is intentionally conservative.
            continue
        for item in samples:
            score = item.get(score_field)
            if isinstance(score, (int, float)):
                candidates.append({"kind": kind, "score": score, **item})
    return sorted(candidates, key=lambda row: row.get("score", 0), reverse=True)[:limit]


def load_official_result_samples(eval_result: dict[str, Any], label: str) -> dict[str, list[dict[str, Any]]]:
    result_dir = ROOT / Path(eval_result["log"]).parent
    prefix = Path(read_config_prediction_path(ROOT / eval_result["config"])).name + "_quick_match"
    loaded = {}
    for element in ["text_block", "display_formula", "table", "reading_order"]:
        path = result_dir / f"{prefix}_{element}_result.json"
        if path.exists():
            loaded[element] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("kind", "")),
        str(row.get("img_id", "")),
        json.dumps(row.get("gt_idx", ""), ensure_ascii=False, sort_keys=True),
        json.dumps(row.get("pred_idx", ""), ensure_ascii=False, sort_keys=True),
    )


def write_glm_candidates(eval_result: dict[str, Any], label: str, out_dir: Path, limit: int = 12) -> Path:
    samples_by_element = load_official_result_samples(eval_result, label)
    buckets: dict[str, list[dict[str, Any]]] = {"ocr_text": [], "formula": [], "table": []}
    for element, kind in [("text_block", "ocr_text"), ("display_formula", "formula"), ("table", "table")]:
        for item in samples_by_element.get(element, []):
            edit = item.get("edit")
            if isinstance(edit, (int, float)) and edit >= 0.35:
                buckets[kind].append({
                    "kind": kind,
                    "element": element,
                    "score": edit,
                    "img_id": item.get("img_id"),
                    "gt_idx": item.get("gt_idx"),
                    "pred_idx": item.get("pred_idx"),
                    "gt": item.get("gt"),
                    "pred": item.get("pred"),
                    "gt_attribute": item.get("gt_attribute"),
                    "gt_category_type": item.get("gt_category_type"),
                    "pred_category_type": item.get("pred_category_type"),
                })
            elif element == "table":
                teds = item.get("TEDS")
                if isinstance(teds, (int, float)) and teds <= 0.65:
                    buckets[kind].append({
                        "kind": kind,
                        "element": element,
                        "score": 1 - teds,
                        "img_id": item.get("img_id"),
                        "gt_idx": item.get("gt_idx"),
                        "pred_idx": item.get("pred_idx"),
                        "gt": item.get("gt"),
                        "pred": item.get("pred"),
                        "gt_attribute": item.get("gt_attribute"),
                        "gt_category_type": item.get("gt_category_type"),
                        "pred_category_type": item.get("pred_category_type"),
                    })
    quotas = {"ocr_text": 4, "formula": 4, "table": 4}
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[str, str, str, str]] = set()
    for kind in ["ocr_text", "formula", "table"]:
        rows = sorted(buckets[kind], key=lambda row: row.get("score", 0), reverse=True)
        for row in rows[: quotas[kind]]:
            ident = row_identity(row)
            if ident not in selected_ids:
                selected.append(row)
                selected_ids.add(ident)
    remaining = sorted(
        [row for rows in buckets.values() for row in rows if row_identity(row) not in selected_ids],
        key=lambda row: row.get("score", 0),
        reverse=True,
    )
    candidates = (selected + remaining)[:limit]
    for idx, row in enumerate(candidates):
        row["candidate_rank"] = idx
    out_path = out_dir / f"glm_candidates_{label}.json"
    out_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def avg(values: list[float]) -> float | None:
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def summarize_glm_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(rows),
        "status_counts": dict(Counter(str(row.get("api", {}).get("status", "unknown")) for row in rows)),
        "by_kind": {},
    }
    for kind in sorted({str(row.get("kind")) for row in rows}):
        subset = [row for row in rows if str(row.get("kind")) == kind]
        before_avg = avg([row["before_edit"] for row in subset if isinstance(row.get("before_edit"), (int, float))])
        rule_avg = avg([row["rule_edit"] for row in subset if isinstance(row.get("rule_edit"), (int, float))])
        after_avg = avg([row["after_edit"] for row in subset if isinstance(row.get("after_edit"), (int, float))])
        summary["by_kind"][kind] = {
            "count": len(subset),
            "api_ok": sum(1 for row in subset if row.get("api", {}).get("status") == "ok"),
            "before_edit_avg": before_avg,
            "rule_edit_avg": rule_avg,
            "after_edit_avg": after_avg,
            "rule_delta_avg": (before_avg - rule_avg) if before_avg is not None and rule_avg is not None else None,
            "llm_delta_avg": (before_avg - after_avg) if before_avg is not None and after_avg is not None else None,
        }
    before_all = avg([row["before_edit"] for row in rows if isinstance(row.get("before_edit"), (int, float))])
    rule_all = avg([row["rule_edit"] for row in rows if isinstance(row.get("rule_edit"), (int, float))])
    after_all = avg([row["after_edit"] for row in rows if isinstance(row.get("after_edit"), (int, float))])
    summary["overall"] = {
        "before_edit_avg": before_all,
        "rule_edit_avg": rule_all,
        "after_edit_avg": after_all,
        "rule_delta_avg": (before_all - rule_all) if before_all is not None and rule_all is not None else None,
        "llm_delta_avg": (before_all - after_all) if before_all is not None and after_all is not None else None,
    }
    return summary


def load_glm_artifacts(run_dir: Path) -> dict[str, Any] | None:
    out_dir = run_dir / "glm_arbitration"
    result_path = out_dir / "glm_arbitration_results.json"
    if not result_path.exists():
        return None
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    summary_path = out_dir / "glm_arbitration_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = summarize_glm_rows(rows)
    artifacts = {
        "results": str(result_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)) if summary_path.exists() else "",
        "report": str((out_dir / "glm_arbitration_report.md").relative_to(ROOT)) if (out_dir / "glm_arbitration_report.md").exists() else "",
        "chart": str((out_dir / "glm_arbitration_before_after.svg").relative_to(ROOT)) if (out_dir / "glm_arbitration_before_after.svg").exists() else "",
        "rows": rows,
        "summary_data": summary,
    }
    return artifacts


def write_markdown_report(
    path: Path,
    run_id: str,
    subset: list[dict[str, Any]],
    mineru_rows: list[dict[str, Any]],
    eval_results: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    breakdowns: dict[str, dict[str, float]],
    charts: dict[str, str],
    glm_candidate_paths: list[Path],
    glm_artifacts: dict[str, Any] | None = None,
) -> None:
    has_formula_cdm = any(isinstance(row.get("formula_cdm"), (int, float)) for row in quality_rows)
    cdm_note = (
        "- CDM: enabled with local user-space TeX Live + ImageMagick/Ghostscript under `.local_tools`; visual dumps are disabled for benchmark speed."
        if has_formula_cdm
        else "- CDM: not enabled in this run; formula accuracy falls back to normalized edit distance."
    )
    lines = [
        "# OmniDocBench Quality + Speed Benchmark",
        "",
        f"- Run ID: `{run_id}`",
        f"- Dataset: `{DATASET.relative_to(ROOT)}`",
        f"- Pages: {len(subset)}",
        "- Official scorer: OmniDocBench end2end `quick_match`",
        "- Metrics: text OCR Edit distance, display formula Edit distance/CDM, table TEDS/Edit distance, reading order Edit distance.",
        cdm_note,
        "",
        "## Subset",
        "",
        "| idx | image | source | language | layout | text | formula | table | note |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for sample in subset:
        counts = sample["_element_counts"]
        attrs = sample["page_info"].get("page_attribute", {})
        note = ",".join(attrs.get("special_issue", []) or [])
        lines.append(
            f"| {sample['_source_index']} | `{Path(sample['page_info']['image_path']).name}` | {attrs.get('data_source','')} | {attrs.get('language','')} | {attrs.get('layout','')} | {counts.get('text_block',0)} | {counts.get('equation_isolated',0)} | {counts.get('table',0)} | {note} |"
        )

    lines.extend([
        "",
        "## Speed",
        "",
        "| backend | effort | pages | passed | total(s) | avg s/page | md chars | content items |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in mineru_rows:
        grouped.setdefault((row["backend"], row.get("effort", "")), []).append(row)
    for (backend, effort), rows in grouped.items():
        total_s = sum(float(row["seconds"]) for row in rows)
        passed = sum(1 for row in rows if row["returncode"] == 0)
        lines.append(
            f"| `{backend}` | `{effort or '-'}` | {len(rows)} | {passed} | {total_s:.2f} | {total_s/max(1,len(rows)):.2f} | {sum(row['md_chars'] for row in rows)} | {sum(row['content_items'] for row in rows)} |"
        )

    lines.extend([
        "",
        "## Accuracy",
        "",
        "| label | text acc | OCR stress acc | formula edit acc | formula CDM | table TEDS | table edit | reading order acc | overall proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in quality_rows:
        lines.append(
            "| {label} | {text_accuracy:.3f} | {ocr_stress_accuracy:.3f} | {formula_accuracy:.3f} | {formula_cdm:.3f} | {table_teds:.3f} | {table_edit:.3f} | {reading_order_accuracy:.3f} | {overall_proxy:.3f} |".format(
                label=row["label"],
                text_accuracy=fmt_float(row.get("text_accuracy")),
                ocr_stress_accuracy=fmt_float(row.get("ocr_stress_accuracy")),
                formula_accuracy=fmt_float(row.get("formula_accuracy")),
                formula_cdm=fmt_float(row.get("formula_cdm")),
                table_teds=fmt_float(row.get("table_teds")),
                table_edit=fmt_float(row.get("table_edit")),
                reading_order_accuracy=fmt_float(row.get("reading_order_accuracy")),
                overall_proxy=fmt_float(row.get("overall_proxy")),
            )
        )
    lines.extend([
        "",
        "| label | text edit | OCR stress edit | formula edit | table edit | reading order edit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in quality_rows:
        lines.append(
            "| {label} | {text_edit:.3f} | {ocr_stress_text_edit:.3f} | {formula_edit:.3f} | {table_edit:.3f} | {reading_order_edit:.3f} |".format(
                label=row["label"],
                text_edit=fmt_float(row.get("text_edit")),
                ocr_stress_text_edit=fmt_float(row.get("ocr_stress_text_edit")),
                formula_edit=fmt_float(row.get("formula_edit")),
                table_edit=fmt_float(row.get("table_edit")),
                reading_order_edit=fmt_float(row.get("reading_order_edit")),
            )
        )
    for row in quality_rows:
        if row.get("ocr_stress_pages"):
            pages = ", ".join(f"`{name}`" for name in row["ocr_stress_pages"])
            lines.append(f"- OCR stress subset ({row.get('ocr_stress_page_count', 0)} pages): {pages}")

    lines.extend([
        "",
        "## Attribute-Level Accuracy",
        "",
        "这些分组来自 OmniDocBench 官方 `page` 维度输出，可用于补充 OCR/text、公式、表格在不同页面类型上的表现。",
        "",
    ])
    for title, key in [
        ("Text/OCR Edit Distance", "text_ocr_edit_by_page_attr"),
        ("Formula Edit Distance", "formula_edit_by_page_attr"),
        ("Formula CDM", "formula_cdm_by_page_attr"),
        ("Table TEDS", "table_teds_by_page_attr"),
        ("Reading Order Edit Distance", "reading_order_edit_by_page_attr"),
    ]:
        values = breakdowns.get(key, {})
        if not values:
            continue
        lines.extend([
            f"### {title}",
            "",
            "| group | score |",
            "| --- | ---: |",
        ])
        for group, score in sorted(values.items()):
            lines.append(f"| `{group}` | {score:.3f} |")
        lines.append("")

    lines.extend(["", "## Visualizations", ""])
    for label, rel in charts.items():
        lines.extend([f"![{label}](../../../{rel})", ""])

    lines.extend([
        "## Official Eval Artifacts",
        "",
        "| label | returncode | seconds | metric result | log |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    for result in eval_results:
        lines.append(
            f"| {result['label']} | {result['returncode']} | {result['seconds']:.2f} | {md_link(result.get('metric_result'), 'metric')} | {md_link(result.get('log'), 'log')} |"
        )

    lines.extend([
        "",
        "## GLM Arbitration",
        "",
        "Low-confidence candidates have been exported for formula/table/OCR arbitration. The arbitration runner reads `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL` from the environment only; no API token is written to disk.",
        "",
    ])
    for candidate_path in glm_candidate_paths:
        lines.append(f"- Candidates: {md_link(str(candidate_path.relative_to(ROOT)), candidate_path.name)}")
    if glm_artifacts:
        if glm_artifacts.get("results"):
            lines.append(f"- Results: {md_link(glm_artifacts.get('results'), 'glm_arbitration_results.json')}")
        if glm_artifacts.get("report"):
            lines.append(f"- Field report: {md_link(glm_artifacts.get('report'), 'glm_arbitration_report.md')}")
        if glm_artifacts.get("chart"):
            lines.extend(["", f"![glm arbitration](../../../{glm_artifacts.get('chart')})", ""])
        summary = glm_artifacts.get("summary_data") or {}
        lines.extend([
            "",
            "| kind | n | api ok | MinerU edit | rule edit | GLM edit | rule delta | GLM delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for kind, stats in (summary.get("by_kind") or {}).items():
            lines.append(
                f"| `{kind}` | {stats.get('count', 0)} | {stats.get('api_ok', 0)} | {fmt_metric(stats.get('before_edit_avg'))} | {fmt_metric(stats.get('rule_edit_avg'))} | {fmt_metric(stats.get('after_edit_avg'))} | {fmt_metric(stats.get('rule_delta_avg'))} | {fmt_metric(stats.get('llm_delta_avg'))} |"
            )
        status_counts = summary.get("status_counts", {})
        if status_counts:
            lines.append(f"- API status counts: `{status_counts}`")
        if not any((stats.get("api_ok") or 0) for stats in (summary.get("by_kind") or {}).values()):
            lines.append("- GLM edit is empty because no API call returned a usable correction in this run.")
        lines.extend([
            "- This arbitration experiment uses ground-truth boxes only for cropping recognition regions; it measures recognition correction potential, not layout detection improvement.",
        ])
        by_kind = summary.get("by_kind") or {}
        table_stats = by_kind.get("table") or {}
        formula_stats = by_kind.get("formula") or {}
        ocr_stats = by_kind.get("ocr_text") or {}
        if table_stats:
            lines.append(
                f"- Table arbitration shows a strong format-noise reduction: MinerU field edit {fmt_metric(table_stats.get('before_edit_avg'))} -> rule {fmt_metric(table_stats.get('rule_edit_avg'))} -> GLM {fmt_metric(table_stats.get('after_edit_avg'))}."
            )
        if formula_stats:
            lines.append(
                f"- Formula arbitration is not yet reliable on these hard mismatched samples: MinerU field edit {fmt_metric(formula_stats.get('before_edit_avg'))} -> GLM {fmt_metric(formula_stats.get('after_edit_avg'))}."
            )
        if ocr_stats:
            lines.append(
                f"- OCR arbitration did not improve the historical-document crops in this run: MinerU field edit {fmt_metric(ocr_stats.get('before_edit_avg'))} -> GLM {fmt_metric(ocr_stats.get('after_edit_avg'))}; individual examples are kept in the field report."
            )
    elif not os.getenv("ANTHROPIC_AUTH_TOKEN"):
        lines.append("- API arbitration status: not run or skipped because `ANTHROPIC_AUTH_TOKEN` is not set in the process environment.")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Text/OCR accuracy is reported as `1 - normalized edit distance` from the official text block matcher. Pages from notes, historical documents, colorful PPT/PDF and exam papers are included to stress OCR behavior.",
        "- OCR stress accuracy is a separate text-block score over note/historical/fuzzy/deformed pages in this compact subset.",
        "- Formula edit accuracy is `1 - normalized edit distance`; formula CDM is the rendered Character Detection Matching score from local TeX Live + ImageMagick.",
        "- Table accuracy uses official TEDS where table HTML can be extracted from Markdown; table edit distance is also recorded.",
        "- The subset is intentionally compact so it can be rerun locally while keeping enough coverage across text, formula, table and OCR-heavy pages.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt_float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else float("nan")


def fmt_metric(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "-"


def md_link(rel_path: str | None, label: str) -> str:
    if not rel_path:
        return ""
    return f"[{label}](../../../{rel_path})"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_index", "image_name", "data_source", "language", "layout", "backend", "effort",
        "returncode", "seconds", "md_chars", "content_items", "image_count", "json_files",
        "pred_md", "parse_dir", "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", default=",".join(str(i) for i in DEFAULT_INDICES))
    parser.add_argument("--backend", choices=["pipeline", "hybrid-engine"], default="pipeline")
    parser.add_argument("--effort", choices=["medium", "high"], default="medium")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-cdm", action="store_true")
    parser.add_argument("--summarize-existing", type=Path)
    args = parser.parse_args()

    if args.summarize_existing:
        run_dir = args.summarize_existing.resolve()
        run_id = run_dir.name
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUT_ROOT / run_id
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(item.strip()) for item in args.indices.split(",") if item.strip()]
    dataset = load_dataset()
    subset = select_subset(dataset, indices)
    effort = args.effort if args.backend == "hybrid-engine" else None
    label = backend_label(args.backend, effort)

    if args.summarize_existing:
        results_path = run_dir / "quality_results.json"
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        mineru_rows = previous["mineru_rows"]
        gt_path = run_dir / "gt_subset.json"
        pred_dir = run_dir / "pred_md" / label
        if args.include_cdm:
            config_path = write_eval_config(run_dir, gt_path, pred_dir, label, include_cdm=True)
            eval_result = run_official_eval(config_path, label, run_dir)
        else:
            eval_result = previous["eval_results"][0]
    else:
        gt_path = write_subset_gt(subset, run_dir)
        mineru_rows = run_mineru_subset(subset, run_dir, args.backend, "auto", effort, args.force)
        pred_dir = run_dir / "pred_md" / label
        config_path = write_eval_config(run_dir, gt_path, pred_dir, label, include_cdm=args.include_cdm)
        eval_result = run_official_eval(config_path, label, run_dir)

    metric_result = load_metric_result(eval_result)
    quality = {"label": label, **extract_quality_scores(metric_result)}
    quality.update(extract_ocr_subset_quality(subset, eval_result))
    breakdowns = extract_page_breakdowns(metric_result)

    csv_path = run_dir / "mineru_speed.csv"
    write_csv(csv_path, mineru_rows)
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    speed_rows = [{"label": Path(row["image_name"]).stem[:36], "seconds": row["seconds"]} for row in mineru_rows]
    speed_chart = charts_dir / "page_runtime_seconds.svg"
    svg_bar_chart(speed_chart, "MinerU page runtime", speed_rows, "seconds", "s")
    metric_rows = [
        {"label": "text acc", "score": quality.get("text_accuracy")},
        {"label": "OCR stress acc", "score": quality.get("ocr_stress_accuracy")},
        {"label": "formula edit acc", "score": quality.get("formula_accuracy")},
        {"label": "formula CDM", "score": quality.get("formula_cdm")},
        {"label": "table TEDS", "score": quality.get("table_teds")},
        {"label": "reading order acc", "score": quality.get("reading_order_accuracy")},
        {"label": "overall proxy", "score": quality.get("overall_proxy")},
    ]
    quality_chart = charts_dir / "quality_scores.svg"
    svg_bar_chart(quality_chart, "OmniDocBench quality scores", metric_rows, "score", "")

    glm_candidate_path = write_glm_candidates(eval_result, label, run_dir)
    glm_artifacts = load_glm_artifacts(run_dir)
    results = {
        "run_id": run_id,
        "indices": indices,
        "subset_gt": str(gt_path.relative_to(ROOT)),
        "mineru_rows": mineru_rows,
        "eval_results": [eval_result],
        "quality": [quality],
        "breakdowns": breakdowns,
        "csv": str(csv_path.relative_to(ROOT)),
        "glm_candidates": [str(glm_candidate_path.relative_to(ROOT))],
        "glm_arbitration": {
            key: value
            for key, value in (glm_artifacts or {}).items()
            if key not in {"rows"}
        },
        "charts": {
            "speed": str(speed_chart.relative_to(ROOT)),
            "quality": str(quality_chart.relative_to(ROOT)),
        },
    }
    results_path = run_dir / "quality_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = REPORT_ROOT / f"quality_report_{run_id}.md"
    charts = {"runtime": str(speed_chart.relative_to(ROOT)), "quality": str(quality_chart.relative_to(ROOT))}
    write_markdown_report(report_path, run_id, subset, mineru_rows, [eval_result], [quality], breakdowns, charts, [glm_candidate_path], glm_artifacts)
    latest = REPORT_ROOT / "quality_report_latest.md"
    latest.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({
        "run_id": run_id,
        "report": str(report_path.relative_to(ROOT)),
        "latest_report": str(latest.relative_to(ROOT)),
        "results": str(results_path.relative_to(ROOT)),
        "quality": quality,
        "eval_returncode": eval_result["returncode"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
