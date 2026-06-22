#!/usr/bin/env python3
"""Unit tests for the online quality gate (Layers 1/4) — no GPU, no network.

These pin the intrinsic validators and the asymmetric adoption gate so the
'burden of proof on GLM' behavior can't silently regress.
"""

from __future__ import annotations

from pdf_miner_agent import quality as q


# --- intrinsic validators (Layer 1) ---
def test_formula_compiles_balanced():
    assert q.formula_compiles("E = mc^2")
    assert q.formula_compiles(r"\frac{a}{b}")
    assert not q.formula_compiles(r"\frac{a}{b")        # unbalanced brace
    assert not q.formula_compiles(r"\left( x")          # unbalanced \left/\right
    assert not q.formula_compiles("")                     # empty


def test_table_valid_consistent_columns():
    good = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    bad = "<table><tr><td>a</td><td>b</td><td>c</td></tr><tr><td>1</td></tr></table>"
    assert q.table_is_valid(good)
    assert not q.table_is_valid(bad)
    assert not q.table_is_valid("<p>not a table</p>")


def test_table_valid_with_rowspan_carryover():
    # rowspan=2 in row1 carries a column into row2, so row2's single <td> is fine.
    html = ('<table><tr><td rowspan="2">x</td><td>a</td></tr>'
            '<tr><td>b</td></tr></table>')
    assert q.table_is_valid(html)


def test_text_defect_flags_garble():
    assert q.text_defect_score("normal readable sentence here") < 0.1
    assert q.text_defect_score("���\x00bad") > 0.3


# --- adoption gate (Layer 4) ---
def test_accept_table_only_when_valid():
    bad = "<table><tr><td>a</td><td>b</td><td>c</td></tr><tr><td>1</td></tr></table>"
    fixed = "<table><tr><td>a</td><td>b</td><td>c</td></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>"
    ok, reason = q.accept_table(bad, fixed)
    assert ok and reason == "table_now_valid"
    ok2, _ = q.accept_table(bad, bad)
    assert not ok2


def test_accept_formula_conservative():
    ok, _ = q.accept_formula(r"\frac{a}{b", r"\frac{a}{b}")  # broken -> fixed
    assert ok
    ok2, reason = q.accept_formula("E=mc^2", "E=mc^3")        # both compile -> keep
    assert not ok2 and reason == "guard_keep_mineru"


def test_text_gate_log_only_never_auto_adopts():
    # a garbled pred cleaned to readable text is a real improvement, yet the
    # default log_only policy still refuses to auto-adopt it.
    pred = "th\x00e c\x00at sat"
    corrected = "the cat sat"
    ok, reason = q.accept_text(pred, corrected, text_gate="log_only")
    assert not ok and reason == "would_accept_but_log_only"
    ok2, _ = q.accept_text(pred, corrected, text_gate="off")
    assert ok2


def test_text_hallucination_guard():
    ok, reason = q.accept_text("short", "a completely different long rewrite", text_gate="off")
    assert not ok and reason == "edit_distance_too_large"


def test_adopt_dispatch_by_kind():
    fixed = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    bad = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td></tr></table>"
    assert q.adopt("table", bad, fixed)[0] is True
    assert q.adopt("formula", "E=mc^2", "E=mc^2")[0] is False
    assert q.adopt("text", "x", "x", text_gate="log_only")[0] is False
