"""Tests for tokenizer/evaluate_heldout.py (independent held-out evaluator).

Run:
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_evaluate_heldout.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tokenizer.evaluate_heldout import (
    ROOT,
    measure,
    read_lines,
    sha256_file,
)

TOKENIZER_PATH = ROOT / "tokenizer" / "artifacts" / "bpe32k_v10" / "tokenizer.json"
HELDOUT_PATH = ROOT / "data" / "heldout" / "arz_test_heldout.txt"
HELDOUT_MANIFEST = ROOT / "data" / "heldout" / "arz_test_heldout.manifest.json"

require_tokenizer = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(), reason="v10 tokenizer artifact not present"
)

SAMPLE_TEXTS = [
    "إزاي الحال النهاردة يا صاحبي",
    "أنا أحب البرمجة كثيرا",
    "النهاردة الجو حر جدا في القاهرة",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@require_tokenizer
def test_measure_round_trips_clean_arabic(tok: Tokenizer) -> None:
    metrics = measure(tok, SAMPLE_TEXTS)
    assert metrics["samples"] == len(SAMPLE_TEXTS)
    assert metrics["roundtrip_failed"] == 0
    assert metrics["unk_rate"] == 0.0
    assert metrics["tokens_per_word"] > 0.0
    assert metrics["chars_per_token"] > 0.0


@require_tokenizer
def test_measure_worst_cases_are_sorted_descending(tok: Tokenizer) -> None:
    metrics = measure(tok, SAMPLE_TEXTS, worst_n=3)
    scores = [case["tokens_per_word"] for case in metrics["worst_cases"]]
    assert scores == sorted(scores, reverse=True)
    assert len(metrics["worst_cases"]) <= 3


@require_tokenizer
def test_measure_ignores_normalization_only_inputs(tok: Tokenizer) -> None:
    metrics = measure(tok, ["   ", "\t"])
    assert metrics["samples"] == 0
    assert metrics["tokens_per_word"] == 0.0
    assert metrics["chars_per_token"] == 0.0
    assert metrics["unk_rate"] == 0.0


@require_tokenizer
def test_measure_is_deterministic(tok: Tokenizer) -> None:
    assert measure(tok, SAMPLE_TEXTS) == measure(tok, SAMPLE_TEXTS)


def test_sha256_file_matches_manifest_for_heldout() -> None:
    if not (HELDOUT_PATH.exists() and HELDOUT_MANIFEST.exists()):
        pytest.skip("held-out artifacts not present (local-only)")
    manifest = json.loads(HELDOUT_MANIFEST.read_text(encoding="utf-8"))
    assert sha256_file(HELDOUT_PATH) == manifest["output"]["sha256"]
    assert len(read_lines(HELDOUT_PATH)) == manifest["output"]["lines"]


@require_tokenizer
def test_cli_writes_auditable_report(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tokenizer" / "evaluate_heldout.py"),
            "--heldout",
            str(HELDOUT_PATH),
            "--tokenizer",
            str(TOKENIZER_PATH),
            "--out",
            str(out),
            "--max-lines",
            "20",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["heldout"]["lines"] == 20
    assert report["tokenizer"]["vocab_size"] == 32_768
    assert report["metrics"]["roundtrip_failed"] == 0
    assert len(report["tokenizer"]["sha256"]) == 64
    assert len(report["heldout"]["sha256"]) == 64


def test_cli_fails_on_missing_heldout(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tokenizer" / "evaluate_heldout.py"),
            "--heldout",
            str(tmp_path / "nope.txt"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "not found" in result.stdout