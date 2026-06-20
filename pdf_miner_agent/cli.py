#!/usr/bin/env python3
"""CLI entrypoint for PDF-Miner Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import PDFMinerAgent, PDFMinerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--backend", default="pipeline", choices=["pipeline", "hybrid-engine"])
    parser.add_argument("--method", default="auto", choices=["auto", "ocr", "txt"])
    parser.add_argument("--effort", choices=["medium", "high"])
    parser.add_argument("--start-page", type=int)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()

    package = PDFMinerAgent().parse_pdf(
        args.pdf,
        PDFMinerConfig(
            backend=args.backend,
            method=args.method,
            effort=args.effort,
            start_page=args.start_page,
            end_page=args.end_page,
            timeout_sec=args.timeout_sec,
            force=args.force,
        ),
        reuse_existing=args.reuse_existing,
        run_id=args.run_id,
    )
    print(json.dumps(package, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
