#!/usr/bin/env python3
"""DocGate — the online, per-document quality gate (Layers 1-4 wired together).

Flow for one parsed PDF:
  L1  read middle.json -> per-element intrinsic checks -> document quality score
  L2  flag only low-confidence / failing elements
  L3  crop each flagged region from the USER's PDF, call GLM visual arbitration
  L4  adopt a correction only if it passes the intrinsic check it had failed
      (kind-aware: tables auto, formulas strict, text log-only by default)

Failures are swallowed and fall back to MinerU; parsing is never blocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import arbiter, quality


def _iter_elements(middle: dict[str, Any]):
    """Yield (page_idx, kind, content, score, bbox) for gateable elements."""
    for page in middle.get("pdf_info", []):
        page_idx = page.get("page_idx", 0)
        for block in page.get("preproc_blocks", []):
            btype = block.get("type")
            if btype == "interline_equation":
                content = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        content += span.get("content", "")
                yield page_idx, "formula", content, block.get("score"), block.get("bbox")
            elif btype == "table":
                html = ""
                for sub in block.get("blocks", []):
                    for line in sub.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("type") == "table":
                                html += span.get("html") or span.get("content") or ""
                yield page_idx, "table", html, block.get("score"), block.get("bbox")
            elif btype in {"text", "abstract", "ref_text"}:
                content = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        content += span.get("content", "")
                yield page_idx, "text", content, block.get("score"), block.get("bbox")


def run_docgate(
    pdf_path: Path,
    parse_dir: Path | None,
    *,
    enable_glm: bool = True,
    glm_max_calls: int = 12,
    text_gate: str = "log_only",
    crop_dir: Path | None = None,
    model: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Return a per-document online quality report. Pure-Python L1 always runs;
    L3 GLM calls are bounded by glm_max_calls and skipped if no token / dry_run."""
    report: dict[str, Any] = {
        "enabled": True,
        "document_quality": {},
        "by_kind": {},
        "flagged": [],
        "gate_counts": {"flagged": 0, "glm_called": 0, "adopted": 0, "rejected": 0},
        "adoptions": [],
    }
    middle_file = next(iter(sorted(parse_dir.glob("*_middle.json"))), None) if parse_dir else None
    if not middle_file:
        report["enabled"] = False
        report["reason"] = "no middle.json"
        return report
    middle = json.loads(middle_file.read_text(encoding="utf-8", errors="ignore"))

    # ---- Layer 1+2: intrinsic checks + flagging ----
    elements = list(_iter_elements(middle))
    by_kind: dict[str, dict[str, Any]] = {}
    flagged: list[dict[str, Any]] = []
    for idx, (page_idx, kind, content, score, bbox) in enumerate(elements):
        stat = by_kind.setdefault(kind, {"count": 0, "flagged": 0, "defect_sum": 0.0})
        is_flagged, reason, defect = quality.element_defect(kind, content, score)
        stat["count"] += 1
        stat["defect_sum"] += defect
        if is_flagged:
            stat["flagged"] += 1
            flagged.append({"index": idx, "page_idx": page_idx, "kind": kind,
                            "reason": reason, "score": score, "bbox": bbox,
                            "content": content})
    for kind, stat in by_kind.items():
        stat["mean_defect"] = round(stat["defect_sum"] / max(stat["count"], 1), 4)
        stat.pop("defect_sum", None)
    total = sum(s["count"] for s in by_kind.values())
    total_flagged = sum(s["flagged"] for s in by_kind.values())
    report["by_kind"] = by_kind
    report["gate_counts"]["flagged"] = total_flagged
    report["document_quality"] = {
        "n_elements": total,
        "n_flagged": total_flagged,
        "clean_ratio": round(1 - total_flagged / max(total, 1), 4),
        "mean_defect": round(sum(s["mean_defect"] * s["count"] for s in by_kind.values()) / max(total, 1), 4),
    }
    report["flagged"] = [{k: v for k, v in f.items() if k != "content"} for f in flagged]

    # ---- Layer 3+4: GLM arbitration on flagged regions, bounded ----
    if not enable_glm or dry_run:
        report["glm"] = "disabled" if not enable_glm else "dry_run"
        return report
    crop_dir = crop_dir or (parse_dir / "docgate_crops")
    budget = glm_max_calls
    for f in flagged:
        if budget <= 0:
            break
        if not f.get("bbox"):
            continue
        budget -= 1
        report["gate_counts"]["glm_called"] += 1
        crop = crop_dir / f"{f['index']:03d}_{f['kind']}.png"
        ok = arbiter.render_pdf_region(pdf_path, f["page_idx"], f["bbox"], crop)
        rule_value = arbiter.rule_postprocess(f["kind"], f["content"])
        api = arbiter.call_glm(f["kind"], f["content"], crop, rule_value=rule_value, model=model) if ok else {"status": "skipped", "reason": "crop failed"}
        corrected = arbiter.parse_corrected(api.get("response", "")) if api.get("status") == "ok" else ""
        adopted, decision = quality.adopt(f["kind"], f["content"], corrected, text_gate=text_gate) if corrected else (False, "no_correction")
        if adopted:
            report["gate_counts"]["adopted"] += 1
        else:
            report["gate_counts"]["rejected"] += 1
        report["adoptions"].append({
            "index": f["index"], "kind": f["kind"], "reason_flagged": f["reason"],
            "api_status": api.get("status"), "adopted": adopted, "decision": decision,
            "pred": (f["content"] or "")[:200], "corrected": (corrected or "")[:200],
        })
    return report


def _content_list_elements(content_list: list[dict]):
    """Yield (idx, kind, md_content, bbox, page_idx) from a MinerU content_list.
    md_content is the markdown-form string actually embedded in the .md, so an
    adopted correction can be written back by string replacement."""
    for idx, item in enumerate(content_list):
        t = item.get("type")
        if t == "table":
            yield idx, "table", item.get("table_body", "") or "", item.get("bbox"), item.get("page_idx", 0)
        elif t == "equation":
            yield idx, "formula", item.get("text", "") or "", item.get("bbox"), item.get("page_idx", 0)
        elif t == "text":
            yield idx, "text", item.get("text", "") or "", item.get("bbox"), item.get("page_idx", 0)


def gate_and_rewrite(
    pdf_path: Path,
    parse_dir: Path,
    *,
    enable_glm: bool = True,
    glm_max_calls: int = 12,
    text_gate: str = "log_only",
    model: str | None = None,
    dry_run: bool = False,
    write_gated_md: bool = True,
) -> dict[str, Any]:
    """content_list-driven gate that ALSO writes a `<stem>_gated.md` with adopted
    corrections applied (markdown-aligned). Intrinsic checks need no MinerU score.
    Returns a report with a `gated_markdown` path when corrections were written."""
    report: dict[str, Any] = {
        "enabled": True, "mode": "content_list",
        "gate_counts": {"flagged": 0, "glm_called": 0, "adopted": 0, "rejected": 0},
        "adoptions": [], "gated_markdown": None,
    }
    cl_file = next(iter(sorted(parse_dir.glob("*_content_list.json"))), None)
    md_file = next(iter(sorted(parse_dir.glob("*.md"))), None)
    if not cl_file or not md_file:
        report["enabled"] = False
        report["reason"] = "no content_list/md"
        return report
    content_list = json.loads(cl_file.read_text(encoding="utf-8", errors="ignore"))
    md_text = md_file.read_text(encoding="utf-8", errors="ignore")

    # L1+L2: intrinsic flagging (score unavailable here -> intrinsic-only)
    flagged = []
    for idx, kind, content, bbox, page_idx in _content_list_elements(content_list):
        if not content.strip():
            continue
        is_flagged, reason, _ = quality.element_defect(kind, content, None)
        if is_flagged:
            flagged.append((idx, kind, content, bbox, page_idx, reason))
    report["gate_counts"]["flagged"] = len(flagged)
    if not enable_glm or dry_run:
        report["glm"] = "disabled" if not enable_glm else "dry_run"
        return report

    crop_dir = parse_dir / "docgate_crops"
    replacements: list[tuple[str, str]] = []
    budget = glm_max_calls
    for idx, kind, content, bbox, page_idx, reason in flagged:
        if budget <= 0:
            break
        if not bbox:
            continue
        budget -= 1
        report["gate_counts"]["glm_called"] += 1
        crop = crop_dir / f"{idx:03d}_{kind}.png"
        ok = arbiter.render_pdf_region(pdf_path, page_idx, bbox, crop)
        rule_value = arbiter.rule_postprocess(kind, content)
        api = arbiter.call_glm(kind, content, crop, rule_value=rule_value, model=model) if ok else {"status": "skipped"}
        corrected = arbiter.parse_corrected(api.get("response", "")) if api.get("status") == "ok" else ""
        adopted, decision = quality.adopt(kind, content, corrected, text_gate=text_gate) if corrected else (False, "no_correction")
        if adopted:
            report["gate_counts"]["adopted"] += 1
            md_form = corrected if kind == "table" else (f"$$\n{corrected}\n$$" if kind == "formula" else corrected)
            if content in md_text:
                replacements.append((content, md_form))
        else:
            report["gate_counts"]["rejected"] += 1
        report["adoptions"].append({
            "index": idx, "kind": kind, "reason_flagged": reason,
            "api_status": api.get("status"), "adopted": adopted, "decision": decision,
            "pred": content[:200], "corrected": (corrected or "")[:200],
        })

    if write_gated_md:
        gated = md_text
        for old, new in replacements:
            gated = gated.replace(old, new, 1)
        gated_path = md_file.with_name(md_file.stem + "_gated.md")
        gated_path.write_text(gated, encoding="utf-8")
        report["gated_markdown"] = str(gated_path)
        report["replacements_applied"] = len(replacements)
    return report
