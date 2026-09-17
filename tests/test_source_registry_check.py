"""Tests for data_pipeline/source_registry_check.py (license policy gate).

Run:
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_source_registry_check.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.source_registry_check import (
    REGISTRY,
    load_records,
    validate_records,
)

APPROVED = {
    "source_id": "demo_ok",
    "language": "ar",
    "role": "web",
    "license": "ODC-By-1.0",
    "provenance": "HuggingFace demo/dataset",
    "status": "approved",
}


def test_shipped_registry_is_clean() -> None:
    records, problems = load_records(REGISTRY)
    assert problems == [], problems
    assert records, "registry should not be empty"
    assert validate_records(records) == []


def test_approved_source_with_allowed_license_is_accepted() -> None:
    assert validate_records([dict(APPROVED)]) == []


def test_missing_field_is_rejected() -> None:
    record = dict(APPROVED)
    record.pop("license")
    problems = validate_records([record])
    assert any("missing required field 'license'" in p for p in problems), problems


def test_approved_with_unverified_provenance_is_rejected() -> None:
    record = dict(APPROVED, provenance="UNVERIFIED")
    problems = validate_records([record])
    assert any("unverified provenance" in p for p in problems), problems


def test_approved_with_non_commercial_license_is_rejected() -> None:
    record = dict(APPROVED, license="CC-BY-NC-4.0")
    problems = validate_records([record])
    assert any("forbidden license" in p for p in problems), problems


def test_approved_with_share_alike_license_needs_review() -> None:
    record = dict(APPROVED, license="CC-BY-SA-4.0")
    problems = validate_records([record])
    assert any("conditional license" in p for p in problems), problems


def test_pending_provenance_source_is_allowed_to_register() -> None:
    record = dict(APPROVED, status="pending_provenance", provenance="UNVERIFIED",
                  license="MIT")
    assert validate_records([record]) == []


def test_duplicate_source_id_is_rejected() -> None:
    problems = validate_records([dict(APPROVED), dict(APPROVED)])
    assert any("duplicate source_id" in p for p in problems), problems


def test_invalid_status_is_rejected() -> None:
    record = dict(APPROVED, status="maybe")
    problems = validate_records([record])
    assert any("invalid status" in p for p in problems), problems


def test_load_records_reports_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text('{"source_id": "ok"}\nnot-json\n', encoding="utf-8")
    records, problems = load_records(path)
    assert len(records) == 1
    assert any("invalid JSON" in p for p in problems), problems


def test_load_records_reports_missing_file(tmp_path: Path) -> None:
    records, problems = load_records(tmp_path / "absent.jsonl")
    assert records == []
    assert any("registry not found" in p for p in problems), problems


def test_registry_lines_are_json_objects_with_ids() -> None:
    lines = [ln for ln in REGISTRY.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "registry must not be empty"
    for line in lines:
        record = json.loads(line)
        assert isinstance(record, dict)
        assert record.get("source_id")
        assert record.get("license")
        assert record.get("status")