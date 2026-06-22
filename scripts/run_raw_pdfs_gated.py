#!/usr/bin/env python3
"""Run the gated PDF-Miner pipeline over every PDF in data/raw_pdfs and summarize
the per-document online quality gate. Reuses existing MinerU parses where present
so this exercises DocGate (L1 intrinsic + L3/4 GLM arbitration) without GPU."""

from __future__ import annotations

import json
from pathlib import Path

from pdf_miner_agent import PDFMinerAgent, PDFMinerConfig

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_pdfs"
RUN_ID = "raw_pdfs_gated"


def main() -> None:
    agent = PDFMinerAgent()
    rows = []
    for pdf in sorted(RAW.glob("*.pdf")):
        try:
            pkg = agent.parse_pdf(
                pdf,
                PDFMinerConfig(enable_glm_gate=True, glm_max_calls=8, text_gate="log_only"),
                reuse_existing=True,
                run_id=RUN_ID,
            )
        except Exception as exc:
            rows.append({"pdf": pdf.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL] {pdf.name}: {exc}")
            continue
        gate = pkg.get("online_quality", {})
        dq = gate.get("document_quality", {})
        gc = gate.get("gate_counts", {})
        rows.append({
            "pdf": pdf.name,
            "tags": pkg["classification"].get("tags"),
            "returncode": pkg["execution"]["returncode"],
            "n_elements": dq.get("n_elements"),
            "n_flagged": dq.get("n_flagged"),
            "clean_ratio": dq.get("clean_ratio"),
            "mean_defect": dq.get("mean_defect"),
            "gate_counts": gc,
        })
        print(f"[OK] {pdf.name:42s} clean={dq.get('clean_ratio')} "
              f"flagged={dq.get('n_flagged')}/{dq.get('n_elements')} gate={gc}")
    out = ROOT / "outputs" / "pdf_miner_agent" / RUN_ID / "gate_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] summary -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
