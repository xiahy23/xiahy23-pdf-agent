#!/usr/bin/env python3
"""Build a *medium-difficulty* GLM arbitration candidate set.

The original arbitration experiment (run_glm_arbitration.py via
run_omnidocbench_quality.py:write_glm_candidates) only kept the *hardest* tail
(edit >= 0.35 / TEDS <= 0.65). On that tail the failure root-cause sits on the
input side (mis-cropped regions, fully-failed OCR), so neither rules nor GLM can
recover and GLM merely ties the rule baseline.

This script samples the *medium* band (default 0.1 <= edit < 0.35): cases where
MinerU is mostly right but has local errors (wrong/missing characters) — exactly
where a vision model has room to beat deterministic rules. Output schema matches
exactly what run_glm_arbitration.py consumes (kind/img_id/gt_idx/gt/pred/...), so
the same metric and report table are reused; only the difficulty band changes.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_samples(result_dir: Path, element: str) -> list[dict[str, Any]]:
    matches = glob.glob(str(result_dir / f"*{element}_result.json"))
    if not matches:
        return []
    data = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    return [x for x in data if isinstance(x, dict)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "outputs/omnidocbench_quality/20260613_142721/official_eval_result/pipeline",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/omnidocbench_quality/20260613_142721/glm_candidates_medium.json")
    parser.add_argument("--low", type=float, default=0.10, help="inclusive lower edit bound")
    parser.add_argument("--high", type=float, default=0.35, help="exclusive upper edit bound")
    parser.add_argument("--per-kind", type=int, default=4, help="max samples per kind")
    args = parser.parse_args()

    specs = [("text_block", "ocr_text"), ("display_formula", "formula")]
    candidates: list[dict[str, Any]] = []
    for element, kind in specs:
        rows = []
        for item in load_samples(args.result_dir, element):
            edit = item.get("edit")
            if isinstance(edit, (int, float)) and args.low <= edit < args.high:
                rows.append({
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
        # Sample evenly across the band (low/mid/high within medium) so we do not
        # only pick the borderline-hard end. Sort by score then stride-sample.
        rows.sort(key=lambda r: r["score"])
        if len(rows) > args.per_kind:
            idxs = [round(i * (len(rows) - 1) / (args.per_kind - 1)) for i in range(args.per_kind)]
            rows = [rows[i] for i in sorted(set(idxs))]
        candidates.extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {}
    for c in candidates:
        summary.setdefault(c["kind"], []).append(round(c["score"], 3))
    print(json.dumps({"out": str(args.out.relative_to(ROOT)), "n": len(candidates), "scores_by_kind": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
