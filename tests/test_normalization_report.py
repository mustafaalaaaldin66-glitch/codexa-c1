"""Tests for text/normalization_report.py (why raw text changes).

Run:
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_normalization_report.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from text.normalization_report import (
    CATEGORIES,
    ROOT,
    analyze,
    classify,
)
from text.normalize import normalize_text


def test_classify_detects_tatweel() -> None:
    assert classify("مـحـمـد") == ["tatweel"]


def test_classify_detects_diacritics() -> None:
    assert classify("مُحَمَّد") == ["diacritics"]


def test_classify_detects_whitespace() -> None:
    assert classify("أنا   أحب") == ["whitespace"]


def test_classify_detects_nfkc() -> None:
    assert "nfkc" in classify("①②③")
    assert "nfkc" in classify("ﬀ")


def test_classify_returns_empty_for_canonical_text() -> None:
    text = "أنا أحب البرمجة"
    assert normalize_text(text) == text
    assert classify(text) == []


def test_classify_reports_multiple_reasons() -> None:
    reasons = classify("مُحَمَّد   يحب ①")
    assert "diacritics" in reasons
    assert "whitespace" in reasons
    assert "nfkc" in reasons


def test_analyze_counts_skip_blank_lines() -> None:
    analysis = analyze(["   ", "", "\t"])
    assert analysis["samples"] == 0
    assert analysis["changed_rate"] == 0.0
    assert all(value == 0 for value in analysis["reason_counts"].values())


def test_analyze_changed_and_unchanged_split() -> None:
    samples = ["أنا أحب البرمجة", "مُحَمَّد يحب البرمجة", "أنا   أحب"]
    analysis = analyze(samples)
    assert analysis["samples"] == 3
    assert analysis["unchanged"] == 1
    assert analysis["changed_by_normalization"] == 2
    assert analysis["changed_rate"] == 2 / 3


def test_analyze_reason_keys_are_complete() -> None:
    analysis = analyze(["مـحـمـد"])
    assert set(analysis["reason_counts"]) == set(CATEGORIES)


def test_analyze_records_one_example_per_reason() -> None:
    analysis = analyze(["مـحـمـد", "مُحَمَّد"])
    assert "tatweel" in analysis["reason_examples"]
    assert "diacritics" in analysis["reason_examples"]


def test_cli_writes_valid_report(tmp_path: Path) -> None:
    out = tmp_path / "norm.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "text" / "normalization_report.py"),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["version"] == "normalization_report_v1"
    assert set(report["analysis"]["reason_counts"]) == set(CATEGORIES)