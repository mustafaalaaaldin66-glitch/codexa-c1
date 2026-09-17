"""Tests for the P9 mix taxonomy + pilot record validator."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_pipeline.mix_plan_check import (  # noqa: E402
    load_jsonl,
    load_taxonomy,
    sha256_text,
    validate_pilot_records,
)

TAXONOMY = ROOT / "data_pipeline" / "mix_taxonomy.json"
PILOT = ROOT / "data" / "pilot" / "pilot_records.jsonl"
REGISTRY = ROOT / "data" / "source_registry.jsonl"


def _registry() -> dict[str, dict]:
    records, problems = load_jsonl(REGISTRY)
    assert problems == []
    return {r["source_id"]: r for r in records if r.get("source_id")}


def _real_inputs() -> tuple[dict, list[dict], dict[str, dict]]:
    taxonomy, tax_problems = load_taxonomy(TAXONOMY)
    assert tax_problems == []
    records, pilot_problems = load_jsonl(PILOT)
    assert pilot_problems == []
    return taxonomy, records, _registry()


def test_real_taxonomy_and_pilot_pass() -> None:
    taxonomy, records, registry = _real_inputs()
    assert validate_pilot_records(records, taxonomy, registry) == []


def _make_record(**overrides) -> dict:
    text = "sample pilot text for validation"
    record = {
        "record_id": "pilot-test-001",
        "domain": "general_web",
        "subdomain": "blogs",
        "language": "arz",
        "source_id": "fineweb2_arz_arab",
        "license": "ODC-By-1.0",
        "provenance": "HuggingFace HuggingFaceFW/fineweb-2",
        "text_sha256": sha256_text(text),
        "text_chars": len(text),
        "pilot": True,
        "intended_use": "pipeline_validation_only",
        "text": text,
    }
    record.update(overrides)
    return record


def test_missing_required_key_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    record = _make_record()
    del record["provenance"]
    problems = validate_pilot_records([record], taxonomy, registry)
    assert any("missing required field 'provenance'" in p for p in problems)


def test_duplicate_record_id_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(), _make_record()], taxonomy, registry
    )
    assert any("duplicate record_id" in p for p in problems)


def test_duplicate_text_sha256_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(record_id="a"), _make_record(record_id="b")],
        taxonomy,
        registry,
    )
    assert any("duplicate text_sha256" in p for p in problems)


def test_bad_sha_format_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(text_sha256="not-a-hash")], taxonomy, registry
    )
    assert any("64-char lowercase hex" in p for p in problems)


def test_text_hash_mismatch_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(text="changed text", text_chars=12)], taxonomy, registry
    )
    assert any("does not hash to text_sha256" in p for p in problems)


def test_text_chars_mismatch_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(text_chars=999)], taxonomy, registry
    )
    assert any("!= len(text)" in p for p in problems)


def test_non_pilot_record_refused() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(pilot=False)], taxonomy, registry
    )
    assert any("without 'pilot': true" in p for p in problems)


def test_wrong_intended_use_refused() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(intended_use="training")], taxonomy, registry
    )
    assert any("intended_use must be" in p for p in problems)


def test_unknown_source_id_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(source_id="ghost_source")], taxonomy, registry
    )
    assert any("not found in source registry" in p for p in problems)


def test_license_mismatch_with_registry_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(license="MIT")], taxonomy, registry
    )
    assert any("does not match registry license" in p for p in problems)


def test_unknown_domain_subdomain_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(domain="ghost", subdomain="nowhere")], taxonomy, registry
    )
    assert any("unknown domain/subdomain" in p for p in problems)


def test_language_not_allowed_for_subdomain_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(domain="egyptian_dialogue", subdomain="forum_threads", language="en")],
        taxonomy,
        registry,
    )
    assert any("not allowed for 'egyptian_dialogue/forum_threads'" in p for p in problems)


def test_unverified_provenance_rejected() -> None:
    taxonomy, _, registry = _real_inputs()
    problems = validate_pilot_records(
        [_make_record(provenance="UNVERIFIED")], taxonomy, registry
    )
    assert any("unverified provenance" in p for p in problems)


def test_taxonomy_weight_sum_enforced(tmp_path: Path) -> None:
    taxonomy, _, _ = _real_inputs()
    broken = json.loads(json.dumps(taxonomy))
    broken["domains"][0]["weight"] += 0.5
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    _, problems = load_taxonomy(path)
    assert any("weights sum" in p for p in problems)


def test_taxonomy_duplicate_ids_rejected(tmp_path: Path) -> None:
    taxonomy, _, _ = _real_inputs()
    broken = json.loads(json.dumps(taxonomy))
    broken["domains"][1]["id"] = broken["domains"][0]["id"]
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    _, problems = load_taxonomy(path)
    assert any("duplicate domain id" in p for p in problems)


def test_taxonomy_unknown_language_rejected(tmp_path: Path) -> None:
    taxonomy, _, _ = _real_inputs()
    broken = json.loads(json.dumps(taxonomy))
    broken["domains"][0]["subdomains"][0]["languages"] = ["xx"]
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    _, problems = load_taxonomy(path)
    assert any("not declared in taxonomy languages" in p for p in problems)


def test_sha256_text_is_stable() -> None:
    text = "نص عربي للتجربة"
    assert sha256_text(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
