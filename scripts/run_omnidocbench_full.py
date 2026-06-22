#!/usr/bin/env python3
"""Full-scale OmniDocBench reproduction for MinerU (all 1651 pages).

Unlike ``run_omnidocbench_quality.py`` (which launches one MinerU subprocess
per image — fine for an 8-page compact subset, but it reloads the model on
every call), this orchestrator runs MinerU **once per backend** over the whole
image directory so the model is loaded a single time. It is resumable: only
images that still lack a prediction are staged for inference, and the official
scorer can be re-run independently.

Stages
------
1. Batched inference: symlink the missing images into a staging dir, run
   ``mineru -p <stage_dir> -o <raw_dir> -b <backend>`` once, then collect every
   ``<stem>/<method>/<stem>.md`` into ``pred_md/<label>/<stem>.md``.
2. Official eval: write an end2end config pointing at the FULL OmniDocBench.json
   and the collected prediction dir, then run OmniDocBench ``pdf_validation.py``.
3. Report: parse the official metric JSON and emit a full-scale markdown report
   with per-attribute breakdowns and (when multiple backends are run) a
   side-by-side comparison.

Examples
--------
    # both backends, CDM enabled (the assignment's full reproduction)
    python scripts/run_omnidocbench_full.py \
        --backends pipeline hybrid-engine --include-cdm

    # re-score an existing inference run without re-running MinerU
    python scripts/run_omnidocbench_full.py --backends pipeline \
        --run-dir outputs/omnidocbench_full/20260614_xxxx --skip-inference
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Reuse the official-eval + report helpers from the compact script so the
# scoring/parsing logic stays identical across the two benchmarks.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_omnidocbench_quality as q  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OMNI = ROOT / "OmniDocBench"
DATASET = OMNI / "Dataset" / "OmniDocBench.json"
IMAGES = OMNI / "Dataset" / "images"
MINERU = ROOT / "MinerU" / ".venv" / "bin" / "mineru"
OUT_ROOT = ROOT / "outputs" / "omnidocbench_full"
REPORT_ROOT = ROOT / "reports" / "assignment2" / "omnidocbench_full"
OMNI_PYTHON = OMNI / ".venv" / "bin" / "python"


def backend_label(backend: str, effort: str | None) -> str:
    if backend == "hybrid-engine":
        return f"hybrid_{effort or 'medium'}"
    return backend


def method_subdir(backend: str) -> str:
    """The per-image output subdirectory MinerU writes under <stem>/."""
    if backend == "hybrid-engine":
        return "hybrid_auto"
    return "auto"


def load_gt_index() -> dict[str, dict[str, Any]]:
    """Map image basename -> GT sample (for per-page attribute lookups)."""
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for sample in data:
        name = Path(sample["page_info"]["image_path"]).name
        index[name] = sample
    return index


def all_image_names() -> list[str]:
    return sorted(p.name for p in IMAGES.iterdir() if p.is_file())


# --- MinerU output-stem normalization (vendored, must match MinerU exactly) ---
# MinerU truncates task stems to MAX_TASK_STEM_BYTES UTF-8 bytes and
# disambiguates collisions with a trailing ``_N`` (see
# MinerU/mineru/cli/common.py:uniquify_task_stems). We replicate it verbatim so
# we can map each GT image name to the directory MinerU actually wrote.
MAX_TASK_STEM_BYTES = 200


def _truncate_utf8(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    truncated = encoded[:max_bytes]
    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError as exc:
            truncated = truncated[: exc.start]
    return ""


def _stem_candidate(stem: str, suffix: str, max_bytes: int = MAX_TASK_STEM_BYTES) -> str:
    if len(f"{stem}{suffix}".encode("utf-8")) <= max_bytes:
        return f"{stem}{suffix}"
    suffix_bytes = len(suffix.encode("utf-8"))
    if suffix_bytes >= max_bytes:
        return _truncate_utf8(suffix, max_bytes)
    return f"{_truncate_utf8(stem, max_bytes - suffix_bytes)}{suffix}"


def uniquify_stems(stems: list[str]) -> dict[str, str]:
    """Replicate MinerU's task-stem assignment for a sorted input batch.

    Returns a mapping original_stem -> effective output-dir stem. MinerU sorts
    its directory glob, so callers must pass stems in sorted filename order.
    """
    normalized = [_truncate_utf8(s, MAX_TASK_STEM_BYTES) for s in stems]
    raw_keys = {s.casefold() for s in normalized}
    occurrence: dict[str, int] = {}
    assigned: set[str] = set()
    mapping: dict[str, str] = {}
    for original, norm in zip(stems, normalized):
        base = norm or original
        key = base.casefold()
        seen = occurrence.get(key, 0)
        occurrence[key] = seen + 1
        if seen == 0 and key not in assigned:
            effective = base
        else:
            suffix_n = seen + 1
            while True:
                cand = _stem_candidate(base, f"_{suffix_n}")
                if cand.casefold() not in raw_keys and cand.casefold() not in assigned:
                    effective = cand
                    break
                suffix_n += 1
        assigned.add(effective.casefold())
        mapping[original] = effective
    return mapping


def collect_predictions(raw_dir: Path, pred_dir: Path, backend: str,
                        image_names: list[str]) -> dict[str, Any]:
    """Copy every produced <stem>.md into pred_md/<label>/<stem>.md.

    Returns counts of done / empty / missing so the caller can report coverage.
    Missing predictions are written as empty .md files (the official scorer
    expects one prediction file per GT page).
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    sub = method_subdir(backend)
    done = empty = missing = 0
    missing_names: list[str] = []
    # MinerU truncates long stems and appends _N for collisions. Replicate its
    # exact assignment over the sorted basenames (MinerU sorts its dir glob) so
    # each GT image maps to the directory MinerU actually wrote.
    ordered = sorted(image_names)
    stem_map = uniquify_stems([Path(n).stem for n in ordered])
    for name in image_names:
        stem = Path(name).stem
        out = pred_dir / f"{stem}.md"
        # Prefer the exact stem dir; fall back to MinerU's normalized stem.
        md_src = raw_dir / stem / sub / f"{stem}.md"
        if not md_src.exists():
            eff = stem_map.get(stem, stem)
            cand = raw_dir / eff / sub / f"{eff}.md"
            if cand.exists():
                md_src = cand
        if md_src.exists():
            text = md_src.read_text(encoding="utf-8", errors="ignore")
            out.write_text(text, encoding="utf-8")
            if text.strip():
                done += 1
            else:
                empty += 1
        else:
            out.write_text("", encoding="utf-8")
            missing += 1
            missing_names.append(name)
    return {
        "predicted": done,
        "empty": empty,
        "missing": missing,
        "missing_names": missing_names,
    }


def pending_images(raw_dir: Path, backend: str, image_names: list[str]) -> list[str]:
    """Images that do not yet have a MinerU markdown output.

    Accounts for MinerU's stem truncation/renaming so already-processed pages
    with long filenames are not falsely re-staged.
    """
    sub = method_subdir(backend)
    stem_map = uniquify_stems([Path(n).stem for n in sorted(image_names)])
    pending = []
    for name in image_names:
        stem = Path(name).stem
        eff = stem_map.get(stem, stem)
        if not (raw_dir / stem / sub / f"{stem}.md").exists() and \
           not (raw_dir / eff / sub / f"{eff}.md").exists():
            pending.append(name)
    return pending


def stage_pending(stage_dir: Path, pending: list[str]) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name in pending:
        link = stage_dir / name
        try:
            link.symlink_to((IMAGES / name).resolve())
        except OSError:
            shutil.copy2(IMAGES / name, link)


def run_inference(run_dir: Path, backend: str, effort: str | None,
                  image_names: list[str], force: bool) -> dict[str, Any]:
    label = backend_label(backend, effort)
    raw_dir = run_dir / "mineru_raw" / label
    stage_dir = run_dir / "stage" / label
    pred_dir = run_dir / "pred_md" / label
    log_dir = run_dir / "logs" / label
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for name in image_names:
            stem_dir = raw_dir / Path(name).stem
            if stem_dir.exists():
                shutil.rmtree(stem_dir)

    pending = pending_images(raw_dir, backend, image_names)
    inference = {
        "backend": backend,
        "effort": effort or "",
        "label": label,
        "total_pages": len(image_names),
        "pending": len(pending),
        "seconds": 0.0,
        "returncode": 0,
        "batches": [],
    }

    if pending:
        stage_pending(stage_dir, pending)
        cmd = [str(MINERU), "-p", str(stage_dir), "-o", str(raw_dir),
               "-b", backend, "-m", "auto"]
        if effort:
            cmd.extend(["--effort", effort])
        started = time.perf_counter()
        result = q.run_capture(
            cmd, timeout=None,
            env={"VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1")},
        )
        inference["seconds"] = round(time.perf_counter() - started, 2)
        inference["returncode"] = result["returncode"]
        log_path = log_dir / "inference.log"
        log_path.write_text(result["output"], encoding="utf-8", errors="ignore")
        inference["log"] = str(log_path.relative_to(ROOT))
        # MinerU may flatten or nest under the stage dir name; re-collect handles both.
        shutil.rmtree(stage_dir, ignore_errors=True)
    else:
        inference["log"] = ""

    coverage = collect_predictions(raw_dir, pred_dir, backend, image_names)
    inference.update(coverage)
    inference["pred_dir"] = str(pred_dir.relative_to(ROOT))
    if coverage["missing_names"]:
        miss_path = log_dir / "missing_predictions.json"
        miss_path.write_text(
            json.dumps(coverage["missing_names"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    inference.pop("missing_names", None)
    return inference


def write_full_eval_config(run_dir: Path, pred_dir: Path, label: str,
                           include_cdm: bool, workers: int) -> Path:
    """End2end config over the FULL OmniDocBench.json with scaled workers."""
    if include_cdm:
        formula_block = (
            "      metric: [Edit_dist, CDM]\n"
            f"      cdm_workers: {workers}\n"
        )
    else:
        formula_block = "      metric: [Edit_dist]\n"
    text = f"""end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
{formula_block}    table:
      metric: [TEDS, Edit_dist]
      teds_workers: {workers}
      timeout_sec: 120
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: {DATASET.resolve()}
    prediction:
      data_path: {pred_dir.resolve()}
    match_method: quick_match
    match_workers: {workers}
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
"""
    config_path = run_dir / f"eval_{label}.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def run_official_eval(config_path: Path, label: str, run_dir: Path,
                      timeout: int) -> dict[str, Any]:
    """Run OmniDocBench pdf_validation.py and copy result files locally.

    Mirrors run_omnidocbench_quality.run_official_eval but with a configurable
    timeout (full-scale CDM/TEDS can take far longer than the compact subset).
    """
    result_dir = run_dir / "official_eval_result" / label
    result_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "CDM_SAVE_VIS": "0",
        "OMNIDOCBENCH_TEDS_TIMEOUT_SEC": "120",
        "OMNIDOCBENCH_TIMEOUT_INPUT_DIR": str((run_dir / "timeout_inputs").resolve()),
    }
    env.update(q.local_cdm_env())
    if OMNI_PYTHON.exists():
        cmd = [str(OMNI_PYTHON), "pdf_validation.py", "--config", str(config_path.resolve())]
    else:
        cmd = ["uv", "run", "--project", str(OMNI), "python", "pdf_validation.py",
               "--config", str(config_path.resolve())]
    proc = q.run_capture(cmd, cwd=OMNI, timeout=timeout, env=env)
    log_path = result_dir / "eval.log"
    log_path.write_text(proc["output"], encoding="utf-8", errors="ignore")

    save_prefix = Path(q.read_config_prediction_path(config_path)).name + "_quick_match"
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
        "timeout": proc["timeout"],
        "config": str(config_path.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        "metric_result": str(metric_path.relative_to(ROOT)) if metric_path.exists() else "",
        "run_summary": str(summary_path.relative_to(ROOT)) if summary_path.exists() else "",
        "copied_files": copied,
    }


def coverage_by_source(inference: dict[str, Any], gt_index: dict[str, dict[str, Any]],
                       run_dir: Path) -> dict[str, dict[str, int]]:
    """Per data_source predicted/empty counts, read back from pred_md."""
    label = inference["label"]
    pred_dir = run_dir / "pred_md" / label
    by_src: dict[str, dict[str, int]] = {}
    for name, sample in gt_index.items():
        src = sample["page_info"].get("page_attribute", {}).get("data_source", "")
        bucket = by_src.setdefault(src, {"total": 0, "predicted": 0, "empty": 0})
        bucket["total"] += 1
        md = pred_dir / f"{Path(name).stem}.md"
        if md.exists() and md.read_text(encoding="utf-8", errors="ignore").strip():
            bucket["predicted"] += 1
        else:
            bucket["empty"] += 1
    return dict(sorted(by_src.items()))


def build_quality(eval_result: dict[str, Any]) -> dict[str, Any]:
    metric_result = q.load_metric_result(eval_result)
    quality = {"label": eval_result["label"], **q.extract_quality_scores(metric_result)}
    breakdowns = q.extract_page_breakdowns(metric_result)
    return {"quality": quality, "breakdowns": breakdowns}


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "-"


def write_report(path: Path, run_id: str, total_pages: int, include_cdm: bool,
                 per_backend: list[dict[str, Any]]) -> None:
    lines = [
        "# OmniDocBench 全量复现报告 (MinerU)",
        "",
        f"- Run ID: `{run_id}`",
        f"- Dataset: `{DATASET.relative_to(ROOT)}` (全量 {total_pages} 页)",
        "- Official scorer: OmniDocBench end2end `quick_match`",
        "- Metrics: text OCR Edit distance, display formula Edit distance"
        + ("/CDM" if include_cdm else "")
        + ", table TEDS/Edit distance, reading order Edit distance.",
        f"- CDM: {'启用 (本地 TeX Live + ImageMagick/Ghostscript)' if include_cdm else '未启用，公式精度退化为归一化编辑距离'}",
        "",
        "## 推理覆盖率与速度",
        "",
        "| backend | effort | pages | predicted | empty | missing | 推理总时长(s) | 平均 s/page |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in per_backend:
        inf = entry["inference"]
        secs = inf.get("seconds", 0.0) or 0.0
        avg = secs / max(1, inf.get("total_pages", 1))
        lines.append(
            f"| `{inf['backend']}` | `{inf.get('effort') or '-'}` | {inf['total_pages']} | "
            f"{inf.get('predicted', 0)} | {inf.get('empty', 0)} | {inf.get('missing', 0)} | "
            f"{secs:.1f} | {avg:.2f} |"
        )

    lines.extend([
        "",
        "## 精度 (全量官方指标)",
        "",
        "| backend | text acc | formula edit acc | formula CDM | table TEDS | table edit acc | reading order acc | overall proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for entry in per_backend:
        qd = entry["quality"]
        lines.append(
            f"| `{qd['label']}` | {fmt(qd.get('text_accuracy'))} | {fmt(qd.get('formula_accuracy'))} | "
            f"{fmt(qd.get('formula_cdm'))} | {fmt(qd.get('table_teds'))} | "
            f"{fmt(qd.get('table_accuracy'))} | {fmt(qd.get('reading_order_accuracy'))} | "
            f"{fmt(qd.get('overall_proxy'))} |"
        )
    lines.extend([
        "",
        "| backend | text edit | formula edit | table edit | reading order edit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for entry in per_backend:
        qd = entry["quality"]
        lines.append(
            f"| `{qd['label']}` | {fmt(qd.get('text_edit'))} | {fmt(qd.get('formula_edit'))} | "
            f"{fmt(qd.get('table_edit'))} | {fmt(qd.get('reading_order_edit'))} |"
        )

    # Per-attribute breakdown tables, per backend.
    for entry in per_backend:
        label = entry["quality"]["label"]
        breakdowns = entry["breakdowns"]
        lines.extend(["", f"## 分维度精度: `{label}`", ""])
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
            lines.extend([f"### {title}", "", "| group | score |", "| --- | ---: |"])
            for group, score in sorted(values.items()):
                lines.append(f"| `{group}` | {score:.3f} |")
            lines.append("")

    # Per-source inference coverage (data completeness for the report).
    for entry in per_backend:
        cov = entry.get("coverage_by_source")
        if not cov:
            continue
        lines.extend([
            "", f"## 推理覆盖率 by data_source: `{entry['quality']['label']}`", "",
            "| data_source | total | predicted | empty |",
            "| --- | ---: | ---: | ---: |",
        ])
        for src, c in cov.items():
            lines.append(f"| `{src or '(none)'}` | {c['total']} | {c['predicted']} | {c['empty']} |")

    lines.extend([
        "",
        "## 官方评测产物",
        "",
        "| backend | returncode | seconds | metric result | log |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    for entry in per_backend:
        ev = entry["eval_result"]
        lines.append(
            f"| `{ev['label']}` | {ev['returncode']} | {ev['seconds']:.1f} | "
            f"{q.md_link(ev.get('metric_result'), 'metric')} | {q.md_link(ev.get('log'), 'log')} |"
        )
    lines.extend([
        "",
        "## 说明",
        "",
        "- 全量 1651 页通过单次批量 MinerU 进程完成推理（模型仅加载一次），相比逐页子进程显著降低了固定开销。",
        "- 文本/公式精度以 `1 - 归一化编辑距离` 报告；表格精度使用官方 TEDS。",
        "- 缺失或空预测页会写入空 Markdown，官方匹配器据此计入覆盖率。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["pipeline"],
                        choices=["pipeline", "hybrid-engine"])
    parser.add_argument("--effort", choices=["medium", "high"], default="medium",
                        help="effort for the hybrid-engine backend")
    parser.add_argument("--include-cdm", action="store_true",
                        help="enable formula CDM (slow on full scale)")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2),
                        help="match/TEDS/CDM worker count for the official scorer")
    parser.add_argument("--eval-timeout", type=int, default=36000,
                        help="seconds before the official scorer is killed")
    parser.add_argument("--limit", type=int, default=0,
                        help="only run the first N images (debug/smoke; 0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="re-run inference even if predictions exist")
    parser.add_argument("--skip-inference", action="store_true",
                        help="reuse an existing run-dir's predictions, only re-score")
    parser.add_argument("--skip-eval", action="store_true",
                        help="run inference only, skip the official scorer")
    parser.add_argument("--run-dir", type=Path,
                        help="reuse/resume this run directory")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    image_names = all_image_names()
    if args.limit:
        image_names = image_names[: args.limit]
    gt_index = load_gt_index()

    per_backend: list[dict[str, Any]] = []
    for backend in args.backends:
        effort = args.effort if backend == "hybrid-engine" else None
        label = backend_label(backend, effort)
        print(f"\n=== backend={backend} effort={effort} pages={len(image_names)} ===",
              flush=True)

        if args.skip_inference:
            pred_dir = run_dir / "pred_md" / label
            cov = collect_predictions(run_dir / "mineru_raw" / label, pred_dir,
                                      backend, image_names)
            inference = {"backend": backend, "effort": effort or "", "label": label,
                         "total_pages": len(image_names), "seconds": 0.0,
                         "returncode": 0, "pred_dir": str(pred_dir.relative_to(ROOT)),
                         **{k: v for k, v in cov.items() if k != "missing_names"}}
        else:
            inference = run_inference(run_dir, backend, effort, image_names, args.force)
        print(f"  inference: predicted={inference.get('predicted')} "
              f"empty={inference.get('empty')} missing={inference.get('missing')} "
              f"seconds={inference.get('seconds')}", flush=True)

        entry: dict[str, Any] = {"inference": inference}
        pred_dir = run_dir / "pred_md" / label

        if not args.skip_eval:
            config_path = write_full_eval_config(run_dir, pred_dir, label,
                                                  args.include_cdm, args.workers)
            eval_result = run_official_eval(config_path, label, run_dir, args.eval_timeout)
            print(f"  eval: returncode={eval_result['returncode']} "
                  f"seconds={eval_result['seconds']} timeout={eval_result.get('timeout')}",
                  flush=True)
            built = build_quality(eval_result)
            entry["eval_result"] = eval_result
            entry["quality"] = built["quality"]
            entry["breakdowns"] = built["breakdowns"]
            entry["coverage_by_source"] = coverage_by_source(inference, gt_index, run_dir)
        per_backend.append(entry)

    results = {
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "backends": args.backends,
        "include_cdm": args.include_cdm,
        "total_pages": len(image_names),
        "per_backend": per_backend,
    }
    (run_dir / "full_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_eval and any("quality" in e for e in per_backend):
        report_path = REPORT_ROOT / f"full_report_{run_id}.md"
        write_report(report_path, run_id, len(image_names), args.include_cdm,
                     [e for e in per_backend if "quality" in e])
        (REPORT_ROOT / "full_report_latest.md").write_text(
            report_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nreport: {report_path.relative_to(ROOT)}", flush=True)

    print(json.dumps({
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "backends": args.backends,
        "quality": {e["quality"]["label"]: e["quality"]
                    for e in per_backend if "quality" in e},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
