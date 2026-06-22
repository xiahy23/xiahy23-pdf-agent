#!/usr/bin/env python3
"""Unit tests for the three-layer ablation's word-boundary relevance test.

The whole point of the unified experiment is that relevance KEEPS word
boundaries (single spaces), unlike the deprecated space-stripping version. These
tests pin that behavior so the second hop (MinerU -> +GLM) can't silently
regress to space-stripping, which would erase the gate's measurable benefit.
No GPU or network needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rag_eval"))
from evaluate_three_layer import normalize_ws, is_relevant_ws  # noqa: E402


def test_normalize_keeps_single_spaces():
    assert normalize_ws("The  Random   Forest!!") == "the random forest"
    # HTML + LaTeX scaffolding dropped, words preserved
    assert normalize_ws(r"<td>a</td> \times b") == "a b"


def test_run_on_text_does_not_match_spaced_answer():
    # the crux: a joined OCR passage must NOT match a spaced answer phrase,
    # otherwise the gate's word-boundary fix would be invisible.
    joined = "investigate theperformanceofthreedifferent accelerometers"
    spaced = "investigate the performance of three different accelerometers"
    ans = ["performance of three different accelerometers"]
    assert not is_relevant_ws(joined, ans)
    assert is_relevant_ws(spaced, ans)


def test_relevant_matches_within_larger_text():
    chunk = "we used a random forest model and a mixed linear model here"
    assert is_relevant_ws(chunk, ["random forest model"])
    assert is_relevant_ws(chunk, ["mixed linear model"])
    assert not is_relevant_ws(chunk, ["support vector machine"])


def test_answer_phrase_is_normalized_too():
    # answer phrases pass through the same normalization, so punctuation/case
    # in the QA file doesn't break matching.
    chunk = "forty one children 54 boys mean age 4 8 years"
    assert is_relevant_ws(chunk, ["Forty-one children"])
    assert is_relevant_ws(chunk, ["41 children"]) is False  # digits differ from spelled-out
