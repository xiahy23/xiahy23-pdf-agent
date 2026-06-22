#!/usr/bin/env python3
"""Online, ground-truth-free quality checks and adoption gate (Layers 1/2/4).

These functions never need OmniDocBench ground truth: they inspect MinerU's own
``middle.json`` (per-element confidence + bbox + content) and run intrinsic
validators — LaTeX compilability, HTML table well-formedness, text garble/
truncation — so a per-document quality score and a defensible accept/reject
decision can be produced for *any* user PDF in real time.
"""

from __future__ import annotations

import re
from typing import Any

# Confidence below which an element is flagged for arbitration (Layer 2 trigger).
SCORE_FLAG_THRESHOLD = 0.80
# Text adoption (Layer 4) tunables.
TEXT_EDIT_HALLUCINATION_CAP = 0.40   # > this fraction changed => GLM rewrote, reject
TEXT_DEFECT_IMPROVE_MARGIN = 0.05    # defect must drop by at least this much
TEXT_LEN_RATIO_MIN = 0.70
TEXT_LEN_RATIO_MAX = 1.50

_MOJIBAKE = re.compile(r"[�\x00-\x08\x0b\x0c\x0e-\x1f]")
_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def norm_edit(a: str, b: str) -> float:
    """Levenshtein distance normalized by the longer length, in [0, 1]."""
    left, right = a or "", b or ""
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    if len(left) < len(right):
        left, right = right, left
    prev = list(range(len(right) + 1))
    for i, c1 in enumerate(left, 1):
        cur = [i]
        for j, c2 in enumerate(right, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if c1 == c2 else 1)))
        prev = cur
    return prev[-1] / max(len(a or ""), len(b or ""), 1)


# --------------------------------------------------------------------------- #
# Intrinsic validators (Layer 1) — no ground truth required
# --------------------------------------------------------------------------- #
def run_on_score(text: str) -> float:
    """Detect word-joining (missing spaces) — the dominant failure mode of OCR on
    scanned/rasterized pages. MinerU emits no garbled characters here, so the
    garble/mixing checks miss it entirely; this catches it via two signals:
      1) the longest unbroken alphabetic run (real English words rarely exceed
         ~18 chars), and
      2) an abnormally low space ratio for prose (English averages ~0.15-0.18
         spaces per character; heavy joining pushes it far lower).
    Returns a score in [0, 1]; higher = more likely run-on text."""
    t = (text or "").strip()
    if len(t) < 25:
        return 0.0  # too short to judge reliably (headings, labels)
    longest = 0
    cur = 0
    for ch in t:
        if ch.isalpha():
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    run_signal = 1.0 if longest >= 30 else (0.5 if longest >= 22 else 0.0)
    alpha = sum(ch.isalpha() for ch in t)
    space_ratio = t.count(" ") / max(len(t), 1)
    # only treat low space ratio as a defect when the text is alphabetic prose
    space_signal = 1.0 if (alpha / max(len(t), 1) > 0.6 and space_ratio < 0.08) else 0.0
    return max(run_signal, space_signal)


def text_defect_score(text: str) -> float:
    """Heuristic defect score in [0, 1]; higher = worse. Combines garble ratio,
    truncation, script-mixing, and word-joining — the failure modes OCR text
    actually shows."""
    t = text or ""
    if not t.strip():
        return 1.0
    n = len(t)
    garble = len(_MOJIBAKE.findall(t)) / n
    # script mixing: isolated CJK inside a mostly-latin run (or vice versa) hints
    # at recognition slips; measured as the minority-script share when both exist.
    cjk = len(_CJK.findall(t))
    lat = len(_LATIN.findall(t))
    mix = (min(cjk, lat) / max(cjk + lat, 1)) if (cjk and lat) else 0.0
    # truncation: ends mid-word / on a dangling hyphen.
    stripped = t.rstrip()
    trunc = 1.0 if stripped.endswith("-") or (stripped and stripped[-1].isalnum() and len(stripped.split()[-1]) <= 2 and " " in stripped) else 0.0
    run_on = run_on_score(t)
    return min(1.0, 0.6 * garble + 0.25 * mix + 0.15 * trunc + 0.5 * run_on)


def introduces_new_mojibake(pred: str, corrected: str) -> bool:
    return len(_MOJIBAKE.findall(corrected or "")) > len(_MOJIBAKE.findall(pred or ""))


def formula_compiles(latex: str) -> bool:
    """Lightweight LaTeX well-formedness proxy (no full TeX run): balanced braces
    and \\left/\\right, balanced $...$, and no empty body."""
    s = (latex or "").strip()
    if not s:
        return False
    if s.count("{") != s.count("}"):
        return False
    if s.count("$") % 2 != 0:
        return False
    if len(re.findall(r"\\left\b", s)) != len(re.findall(r"\\right\b", s)):
        return False
    if re.search(r"\\(begin|end)\{", s):
        envs = re.findall(r"\\begin\{([^}]*)\}", s)
        ends = re.findall(r"\\end\{([^}]*)\}", s)
        if sorted(envs) != sorted(ends):
            return False
    return True


def table_is_valid(html: str) -> bool:
    """A table is valid if it parses to a non-empty grid whose body rows share a
    consistent column count (counting colspans). Uses bs4 if present, else a
    regex fallback."""
    s = (html or "").strip()
    if "<tr" not in s.lower():
        return False
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(s, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return False
        # Account for rowspan carry-over: a cell with rowspan=k occupies a column
        # in the next (k-1) rows, so those rows have fewer explicit <td> but the
        # same logical width. Track columns carried into each row.
        widths = []
        carry = 0
        for tr in rows:
            cells = tr.find_all(["td", "th"])
            width = carry
            next_carry = 0
            for c in cells:
                cspan = int(c.get("colspan", 1) or 1)
                rspan = int(c.get("rowspan", 1) or 1)
                width += cspan
                if rspan > 1:
                    next_carry += cspan
            carry = next_carry
            if cells or width:
                widths.append(width)
        widths = [w for w in widths if w]
        if not widths:
            return False
        return max(widths) > 0 and (max(widths) - min(widths)) <= 1
    except Exception:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S | re.I)
        if not rows:
            return False
        widths = [len(re.findall(r"<t[dh][^>]*>", r, flags=re.I)) for r in rows]
        widths = [w for w in widths if w]
        return bool(widths) and (max(widths) - min(widths)) <= 1


# --------------------------------------------------------------------------- #
# Adoption gate (Layer 4) — asymmetric, burden of proof on GLM
# --------------------------------------------------------------------------- #
def accept_table(pred: str, corrected: str) -> tuple[bool, str]:
    if not corrected.strip():
        return False, "empty_correction"
    if table_is_valid(corrected) and not table_is_valid(pred):
        return True, "table_now_valid"
    if table_is_valid(corrected):
        return True, "table_valid"
    return False, "still_invalid"


def accept_formula(pred: str, corrected: str) -> tuple[bool, str]:
    """Conservative: adopt only when the correction compiles and the original did
    not. Offline data shows GLM hurts formulas on average, so anything else stays."""
    if not corrected.strip():
        return False, "empty_correction"
    if formula_compiles(corrected) and not formula_compiles(pred):
        return True, "formula_now_compiles"
    return False, "guard_keep_mineru"


def accept_text(pred: str, corrected: str, text_gate: str = "log_only") -> tuple[bool, str]:
    """Define 'significant improvement' operationally (no GT available):
    1) hallucination guard: <=40% characters changed,
    2) flagged defect drops by >= margin,
    3) no new mojibake,
    4) length ratio within [0.7, 1.5].
    With text_gate='log_only' (default) the verdict is recorded but never auto-
    adopted — offline data shows 0/4 text wins, so we don't pretend otherwise."""
    if not corrected.strip():
        return False, "empty_correction"
    if norm_edit(pred, corrected) > TEXT_EDIT_HALLUCINATION_CAP:
        return False, "edit_distance_too_large"
    d_before, d_after = text_defect_score(pred), text_defect_score(corrected)
    if d_after > d_before - TEXT_DEFECT_IMPROVE_MARGIN:
        return False, "improvement_below_margin"
    if introduces_new_mojibake(pred, corrected):
        return False, "new_garbled_chars"
    ratio = len(corrected) / max(len(pred), 1)
    if not (TEXT_LEN_RATIO_MIN <= ratio <= TEXT_LEN_RATIO_MAX):
        return False, "length_drift"
    if text_gate == "log_only":
        return False, "would_accept_but_log_only"
    return True, "accepted"


def adopt(kind: str, pred: str, corrected: str, text_gate: str = "log_only") -> tuple[bool, str]:
    if kind == "table":
        return accept_table(pred, corrected)
    if kind == "formula":
        return accept_formula(pred, corrected)
    return accept_text(pred, corrected, text_gate=text_gate)


def element_defect(kind: str, content: str, score: float | None) -> tuple[bool, str, float]:
    """Layer 1+2: return (flagged, reason, defect_score) for one element."""
    sc = score if isinstance(score, (int, float)) else 1.0
    if kind == "formula":
        if not formula_compiles(content):
            return True, "formula_not_compilable", 1.0
        if sc < SCORE_FLAG_THRESHOLD:
            return True, "low_confidence", 1.0 - sc
        return False, "ok", 0.0
    if kind == "table":
        if not table_is_valid(content):
            return True, "table_invalid", 1.0
        if sc < SCORE_FLAG_THRESHOLD:
            return True, "low_confidence", 1.0 - sc
        return False, "ok", 0.0
    d = text_defect_score(content)
    if d >= 0.15:
        return True, "text_defect", d
    if sc < SCORE_FLAG_THRESHOLD:
        return True, "low_confidence", max(d, 1.0 - sc)
    return False, "ok", d
