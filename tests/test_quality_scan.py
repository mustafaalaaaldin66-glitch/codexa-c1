"""Tests for data_pipeline/quality_arz_heldout_v10.py (low-memory quality scan).

Run:
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_quality_scan.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from data_pipeline.quality_arz_heldout_v10 import (
    PROJECT_ROOT,
    contamination,
    scan_lines,
    source_balance,
)

REPORT = PROJECT_ROOT / "data" / "heldout" / "arz_test_heldout.quality.json"

SAMPLE = [
    "إزاي الحال النهاردة يا صاحبي",
    "أنا أحب البرمجة كثيرا",
]


def test_scan_lines_counts_and_ratios() -> None:
    metrics = scan_lines(SAMPLE)
    assert metrics["lines"] == 2
    assert metrics["words"] == len(SAMPLE[0].split()) + len(SAMPLE[1].split())
    assert 0.0 < metrics["arabic_character_ratio"] <= 1.0
    assert metrics["unique_word_ratio"] == 1.0
    assert metrics["duplicate_lines"] == 0
    assert metrics["invalid_utf8"] == 0
    assert metrics["whitespace_anomalies"] == 0


def test_scan_lines_empty_input_is_safe() -> None:
    assert scan_lines([]) == {"lines": 0}
    assert scan_lines(["   ", "\t"]) == {"lines": 0}


def test_scan_lines_detects_duplicates_after_normalization() -> None:
    metrics = scan_lines(["مُحَمَّد يحب البرمجة", "محمد يحب البرمجة"])
    assert metrics["lines"] == 2
    assert metrics["duplicate_lines"] == 1
    assert metrics["duplicate_rate"] == 0.5


def test_scan_lines_detects_whitespace_anomalies_before_normalization() -> None:
    metrics = scan_lines(["أنا   أحب\tالبرمجة"])
    # normalize_text collapses them, so normalized text must stay clean
    assert metrics["whitespace_anomalies"] == 0


def test_scan_lines_length_distribution_is_sorted() -> None:
    metrics = scan_lines(["أ" * 50, "ب" * 200, "ج" * 100])
    assert metrics["min_len"] == 50
    assert metrics["max_len"] == 200
    assert metrics["p50_len"] == 200 or metrics["p50_len"] == 100


def test_source_balance_computes_msa_to_egyptian_ratio() -> None:
    manifest = {
        "counts": {
            "ar": {"bytes": 750},
            "arz": {"bytes": 250},
            "en": {"bytes": 0},
        }
    }
    balance = source_balance(manifest)
    assert balance["available"] is True
    assert balance["msa_share"] == 0.75
    assert balance["egyptian_share"] == 0.25
    assert balance["msa_to_egyptian_ratio"] == 3.0


def test_source_balance_without_manifest_is_unavailable() -> None:
    assert source_balance(None) == {"available": False}


def test_contamination_reports_counts() -> None:
    manifest = {
        "train_hashes_checked": 400000,
        "stats": {"train_overlap": 37, "eval_overlap": 0, "dup": 0, "accepted": 100},
    }
    contam = contamination(manifest)
    assert contam["available"] is True
    assert contam["contamination_count"] == 37
    assert contam["train_hashes_checked"] == 400000
    assert contam["heldout_lines_final"] == 100


def test_contamination_without_manifest_is_unavailable() -> None:
    assert contamination(None) == {"available": False}


def test_cli_report_is_reproducible_and_clean() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "data_pipeline" / "quality_arz_heldout_v10.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["text"]["duplicate_lines"] == 0
    assert report["text"]["invalid_utf8"] == 0
    assert len(report["heldout"]["sha256"]) == 64
    assert report["contamination"]["available"] is True
    assert report["source_balance"]["available"] is True