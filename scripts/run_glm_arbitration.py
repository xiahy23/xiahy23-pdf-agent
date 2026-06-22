#!/usr/bin/env python3
"""Run GLM arbitration for low-confidence OmniDocBench recognition samples.

The script reads candidates emitted by run_omnidocbench_quality.py, crops the
ground-truth region from OmniDocBench page images, and optionally calls a
Claude/Anthropic-compatible API. API credentials are read only from environment
variables and are never written to disk.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import os
import re
import time
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "OmniDocBench" / "Dataset" / "OmniDocBench.json"
IMAGES = ROOT / "OmniDocBench" / "Dataset" / "images"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_dataset_by_image() -> dict[str, dict[str, Any]]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    return {Path(item["page_info"]["image_path"]).name: item for item in data}


def find_gt_item(sample: dict[str, Any], gt_idx: Any) -> dict[str, Any] | None:
    ids = gt_idx if isinstance(gt_idx, list) else [gt_idx]
    ids = [item for item in ids if item != ""]
    if not ids:
        return None
    target = ids[0]
    candidates = [item for item in sample.get("layout_dets", []) if not item.get("ignore")]
    if isinstance(target, int) and 0 <= target < len(candidates):
        return candidates[target]
    target_s = str(target)
    for item in candidates:
        if str(item.get("anno_id")) == target_s:
            return item
    return None


def poly_bbox(poly: list[float], pad: int = 8) -> tuple[int, int, int, int]:
    xs = [float(poly[i]) for i in range(0, len(poly), 2)]
    ys = [float(poly[i]) for i in range(1, len(poly), 2)]
    return int(min(xs) - pad), int(min(ys) - pad), int(max(xs) + pad), int(max(ys) + pad)


def crop_image(image_path: Path, bbox: tuple[int, int, int, int], out_path: Path) -> None:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    left, top, right, bottom = bbox
    left = max(0, left)
    top = max(0, top)
    right = min(img.width, right)
    bottom = min(img.height, bottom)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.crop((left, top, right, bottom)).save(out_path)


def image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


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
        text = re.sub(r"\\left\s+", r"\\left", text)
        text = re.sub(r"\\right\s+", r"\\right", text)
        text = re.sub(r"\s*([{}_^=+\-(),:;])\s*", r"\1", text)
        return text.strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def build_prompt(candidate: dict[str, Any]) -> str:
    kind = candidate.get("kind")
    if kind == "formula":
        target = "LaTeX formula"
    elif kind == "table":
        target = "HTML table"
    else:
        target = "plain OCR text"
    rule_value = candidate.get("rule_postprocessed") or ""
    rule_block = f"\nrule_postprocessed_prediction:\n{rule_value}\n" if rule_value and rule_value != (candidate.get("pred") or "") else ""
    return (
        f"You are a document recognition arbitration model. The image crop contains one {target} region.\n"
        "Given the current MinerU prediction, return only a compact JSON object with keys: "
        "`kind`, `corrected`, `rationale`.\n"
        "Do not include markdown fences. Keep `corrected` as the final recognized content only.\n\n"
        f"kind: {kind}\n"
        f"current_prediction:\n{candidate.get('pred') or ''}\n"
        f"{rule_block}"
    )


def call_anthropic_glm(candidate: dict[str, Any], crop_path: Path, model: str, max_tokens: int) -> dict[str, Any]:
    token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic").rstrip("/")
    if not token:
        return {"status": "skipped", "reason": "ANTHROPIC_AUTH_TOKEN is not set"}
    if not crop_path.exists():
        return {"status": "skipped", "reason": f"crop does not exist: {crop_path}"}

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt(candidate)},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data_url(crop_path).split(",", 1)[1]}},
                ],
            }
        ],
    }
    started = time.perf_counter()
    request = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "x-api-key": token,
            "authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "status": "ok",
                "http_status": response.status,
                "model": model,
                "seconds": round(time.perf_counter() - started, 3),
                "response": body,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "status": "failed",
            "http_status": exc.code,
            "model": model,
            "seconds": round(time.perf_counter() - started, 3),
            "response": body,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "model": model,
            "seconds": round(time.perf_counter() - started, 3),
            "response": repr(exc),
        }


def parse_corrected(response: str) -> str:
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
                if not isinstance(corrected, str):
                    break
                stripped = corrected.strip()
                if not stripped.startswith("{"):
                    break
                try:
                    nested = json.loads(stripped)
                except Exception:
                    nested_match = re.search(r'"corrected"\s*:\s*"(.*?)"\s*,\s*"rationale"', stripped, flags=re.S)
                    if nested_match:
                        corrected = nested_match.group(1)
                        break
                    break
                if not isinstance(nested, dict) or "corrected" not in nested:
                    break
                corrected = nested.get("corrected", "")
            return str(corrected)
        except Exception:
            corrected_match = re.search(r'"corrected"\s*:\s*"(.*?)"\s*,\s*"rationale"', match.group(0), flags=re.S)
            if corrected_match:
                return corrected_match.group(1)
    return text.strip()


def norm_edit(a: str, b: str) -> float:
    left = a or ""
    right = b or ""
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, c1 in enumerate(left, 1):
        current = [i]
        for j, c2 in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if c1 == c2 else 1),
                )
            )
        previous = current
    return previous[-1] / max(1, len(a or ""), len(b or ""))


def avg(values: list[float]) -> float | None:
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(rows),
        "status_counts": dict(Counter(str(row.get("api", {}).get("status", "unknown")) for row in rows)),
        "by_kind": {},
    }
    for kind in sorted({str(row.get("kind")) for row in rows}):
        subset = [row for row in rows if str(row.get("kind")) == kind]
        before = [row["before_edit"] for row in subset if isinstance(row.get("before_edit"), (int, float))]
        rule = [row["rule_edit"] for row in subset if isinstance(row.get("rule_edit"), (int, float))]
        after = [row["after_edit"] for row in subset if isinstance(row.get("after_edit"), (int, float))]
        summary["by_kind"][kind] = {
            "count": len(subset),
            "api_ok": sum(1 for row in subset if row.get("api", {}).get("status") == "ok"),
            "before_edit_avg": avg(before),
            "rule_edit_avg": avg(rule),
            "after_edit_avg": avg(after),
            "rule_delta_avg": (avg(before) - avg(rule)) if avg(before) is not None and avg(rule) is not None else None,
            "llm_delta_avg": (avg(before) - avg(after)) if avg(before) is not None and avg(after) is not None else None,
            "rule_improved": sum(
                1
                for row in subset
                if isinstance(row.get("rule_edit"), (int, float))
                and isinstance(row.get("before_edit"), (int, float))
                and row["rule_edit"] < row["before_edit"]
            ),
            "llm_improved": sum(
                1
                for row in subset
                if isinstance(row.get("after_edit"), (int, float))
                and isinstance(row.get("before_edit"), (int, float))
                and row["after_edit"] < row["before_edit"]
            ),
        }
    all_before = [row["before_edit"] for row in rows if isinstance(row.get("before_edit"), (int, float))]
    all_rule = [row["rule_edit"] for row in rows if isinstance(row.get("rule_edit"), (int, float))]
    all_after = [row["after_edit"] for row in rows if isinstance(row.get("after_edit"), (int, float))]
    summary["overall"] = {
        "before_edit_avg": avg(all_before),
        "rule_edit_avg": avg(all_rule),
        "after_edit_avg": avg(all_after),
        "rule_delta_avg": (avg(all_before) - avg(all_rule)) if avg(all_before) is not None and avg(all_rule) is not None else None,
        "llm_delta_avg": (avg(all_before) - avg(all_after)) if avg(all_before) is not None and avg(all_after) is not None else None,
    }
    return summary


def fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_svg(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for kind, stats in summary.get("by_kind", {}).items():
        rows.append((kind, "before", stats.get("before_edit_avg")))
        rows.append((kind, "rule", stats.get("rule_edit_avg")))
        rows.append((kind, "glm", stats.get("after_edit_avg")))
    rows = [row for row in rows if isinstance(row[2], (int, float))]
    width = 920
    row_h = 34
    left = 180
    top = 64
    height = top + row_h * len(rows) + 42
    colors = {"before": "#6b7280", "rule": "#2563eb", "glm": "#059669"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="38" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">GLM arbitration field edit distance</text>',
    ]
    for idx, (kind, label, value) in enumerate(rows):
        y = top + idx * row_h
        bar_w = max(2, float(value) * 620)
        parts.extend(
            [
                f'<text x="28" y="{y+21}" font-family="Arial, sans-serif" font-size="14" fill="#111827">{escape_xml(kind)} / {label}</text>',
                f'<rect x="{left}" y="{y+7}" width="{bar_w:.1f}" height="18" rx="3" fill="{colors.get(label, "#6b7280")}"/>',
                f'<text x="{left+bar_w+8:.1f}" y="{y+21}" font-family="Arial, sans-serif" font-size="13" fill="#374151">{float(value):.3f}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def clip(value: Any, length: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[: length - 1] + "…" if len(text) > length else text


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], svg_path: Path) -> None:
    lines = [
        "# GLM Arbitration Field-Level Report",
        "",
        f"- Candidates: {summary.get('count', 0)}",
        f"- API status: `{summary.get('status_counts', {})}`",
        "- Baseline is MinerU output after official matching; rule is deterministic postprocess; GLM is the Claude-format visual arbitration output.",
        "",
        "## Summary",
        "",
        "| kind | n | api ok | before edit | rule edit | GLM edit | rule delta | GLM delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, stats in summary.get("by_kind", {}).items():
        lines.append(
            f"| `{kind}` | {stats.get('count', 0)} | {stats.get('api_ok', 0)} | {fmt(stats.get('before_edit_avg'))} | {fmt(stats.get('rule_edit_avg'))} | {fmt(stats.get('after_edit_avg'))} | {fmt(stats.get('rule_delta_avg'))} | {fmt(stats.get('llm_delta_avg'))} |"
        )
    lines.extend(
        [
            "",
            "## Visualization",
            "",
            f"![glm arbitration]({svg_path.name})",
            "",
            "## Examples",
            "",
            "| # | kind | score | before | rule | GLM | gt | pred/corrected |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        corrected = row.get("corrected") or row.get("rule_corrected") or row.get("pred") or ""
        lines.append(
            f"| {row.get('index')} | `{row.get('kind')}` | {fmt(row.get('score'))} | {fmt(row.get('before_edit'))} | {fmt(row.get('rule_edit'))} | {fmt(row.get('after_edit'))} | {clip(row.get('gt'))} | {clip(corrected)} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--model", default=os.getenv("GLM_ARBITER_MODEL", "glm-4.5v"))
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-api-results", type=Path, help="Reuse saved glm_arbitration_results.json API payloads and recompute metrics.")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))[: args.limit]
    out_dir = (args.out_dir or args.candidates.parent / "glm_arbitration").resolve()
    crop_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_by_image()
    rows = []
    reuse_rows = []
    if args.reuse_api_results:
        reuse_rows = json.loads(args.reuse_api_results.read_text(encoding="utf-8"))

    for idx, candidate in enumerate(candidates):
        img_name = str(candidate.get("img_id") or candidate.get("image_name") or "")
        sample = dataset.get(img_name)
        gt_item = find_gt_item(sample, candidate.get("gt_idx")) if sample else None
        crop_path = crop_dir / f"{idx:02d}_{Path(img_name).stem}.png"
        if gt_item and gt_item.get("poly"):
            crop_image(IMAGES / img_name, poly_bbox(gt_item["poly"]), crop_path)
        kind = str(candidate.get("kind") or "")
        rule_corrected = rule_postprocess(kind, str(candidate.get("pred") or ""))
        call_candidate = dict(candidate)
        call_candidate["rule_postprocessed"] = rule_corrected
        if idx < len(reuse_rows):
            api_result = reuse_rows[idx].get("api", {})
        else:
            api_result = {"status": "dry_run"} if args.dry_run else call_anthropic_glm(call_candidate, crop_path, args.model, args.max_tokens)
        corrected = parse_corrected(api_result.get("response", "")) if api_result.get("status") == "ok" else ""
        before_edit = norm_edit(str(candidate.get("gt") or ""), str(candidate.get("pred") or ""))
        rule_edit = norm_edit(str(candidate.get("gt") or ""), rule_corrected)
        after_edit = norm_edit(str(candidate.get("gt") or ""), corrected) if corrected else None
        rows.append({
            "index": idx,
            "kind": kind,
            "img_id": img_name,
            "gt_idx": candidate.get("gt_idx"),
            "score": candidate.get("score"),
            "crop": display_path(crop_path) if crop_path.exists() else "",
            "gt": candidate.get("gt"),
            "pred": candidate.get("pred"),
            "before_edit": before_edit,
            "rule_corrected": rule_corrected,
            "rule_edit": rule_edit,
            "corrected": corrected,
            "after_edit": after_edit,
            "api": api_result,
        })

    out_path = out_dir / "glm_arbitration_results.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize_rows(rows)
    summary_path = out_dir / "glm_arbitration_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    svg_path = out_dir / "glm_arbitration_before_after.svg"
    write_svg(svg_path, summary)
    report_path = out_dir / "glm_arbitration_report.md"
    write_markdown(report_path, rows, summary, svg_path)
    print(json.dumps({
        "results": display_path(out_path),
        "summary": display_path(summary_path),
        "report": display_path(report_path),
        "chart": display_path(svg_path),
        "count": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
