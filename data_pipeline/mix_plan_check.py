"""Read-only validator for the mix taxonomy and pilot topic records (P9 gate).

The mix taxonomy is the server-side corpus expansion plan; the pilot records
are a tiny schema example used ONLY to validate the pipeline. This validator
enforces:

1. taxonomy: valid JSON, unique domain ids, unique subdomain ids, allowed
   languages per subdomain, positive weights, domain weights sum to ~1.0
2. pilot records: every line valid JSON, all required keys present, unique
   record_id and unique text_sha256 (exact dedup gate), 64-hex sha256,
   domain/subdomain must exist in the taxonomy, language must be allowed for
   that subdomain, source_id must exist in data/source_registry.jsonl and the
   record license must match the registry license for that source
3. safety: every record must carry ``pilot: true`` AND
   ``intended_use == "pipeline_validation_only"``; a record that looks like
   real data without an approved source is refused
4. integrity: if ``text`` is present, sha256(text) must equal ``text_sha256``

Exit code 0 = clean, 1 = violations. Never writes, never downloads.

Run:
    .\\.venv\\Scripts\\python.exe data_pipeline\\mix_plan_check.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "data_pipeline" / "mix_taxonomy.json"
PILOT = ROOT / "data" / "pilot" / "pilot_records.jsonl"
REGISTRY = ROOT / "data" / "source_registry.jsonl"

REQUIRED_RECORD_KEYS = (
    "record_id",
    "domain",
    "subdomain",
    "language",
    "source_id",
    "license",
    "provenance",
    "text_sha256",
    "text_chars",
    "pilot",
    "intended_use",
)

PILOT_INTENDED_USE = "pipeline_validation_only"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    problems: list[str] = []
    if not path.exists():
        return records, [f"file not found: {path}"]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name} line {number}: invalid JSON ({exc.msg})")
    return records, problems


def load_taxonomy(path: Path) -> tuple[dict, list[str]]:
    if not path.exists():
        return {}, [f"taxonomy not found: {path}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"taxonomy: invalid JSON ({exc.msg})"]

    problems: list[str] = []
    domains = data.get("domains")
    if not isinstance(domains, list) or not domains:
        return data, ["taxonomy: 'domains' must be a non-empty list"]

    seen_domains: set[str] = set()
    seen_subdomains: set[str] = set()
    allowed_languages = set(data.get("languages", {}))
    total_weight = 0.0

    for domain in domains:
        domain_id = domain.get("id")
        if not domain_id:
            problems.append("taxonomy: domain without id")
            continue
        if domain_id in seen_domains:
            problems.append(f"taxonomy: duplicate domain id '{domain_id}'")
        seen_domains.add(domain_id)

        weight = domain.get("weight")
        if not isinstance(weight, (int, float)) or weight <= 0:
            problems.append(f"taxonomy[{domain_id}]: weight must be a positive number")
        else:
            total_weight += float(weight)

        subdomains = domain.get("subdomains")
        if not isinstance(subdomains, list) or not subdomains:
            problems.append(f"taxonomy[{domain_id}]: subdomains must be a non-empty list")
            continue
        for sub in subdomains:
            sub_id = sub.get("id")
            if not sub_id:
                problems.append(f"taxonomy[{domain_id}]: subdomain without id")
                continue
            if sub_id in seen_subdomains:
                problems.append(f"taxonomy[{domain_id}]: duplicate subdomain id '{sub_id}'")
            seen_subdomains.add(sub_id)
            langs = sub.get("languages")
            if not isinstance(langs, list) or not langs:
                problems.append(f"taxonomy[{domain_id}/{sub_id}]: languages must be a non-empty list")
                continue
            for lang in langs:
                if lang not in allowed_languages:
                    problems.append(
                        f"taxonomy[{domain_id}/{sub_id}]: language '{lang}' not declared in taxonomy languages"
                    )

    if not problems and abs(total_weight - 1.0) > 0.01:
        problems.append(f"taxonomy: domain weights sum to {total_weight:.4f}, expected ~1.0")

    return data, problems


def validate_pilot_records(
    records: list[dict],
    taxonomy: dict,
    registry: dict[str, dict],
) -> list[str]:
    problems: list[str] = []
    subdomain_languages: dict[tuple[str, str], set[str]] = {}
    for domain in taxonomy.get("domains", []):
        for sub in domain.get("subdomains", []):
            subdomain_languages[(domain["id"], sub["id"])] = set(sub.get("languages", []))

    seen_record_ids: set[str] = set()
    seen_sha: set[str] = set()

    for index, record in enumerate(records, start=1):
        label = record.get("record_id") or f"record#{index}"

        for key in REQUIRED_RECORD_KEYS:
            if key not in record or record[key] in ("", None):
                problems.append(f"{label}: missing required field '{key}'")

        if record.get("pilot") is not True:
            problems.append(
                f"{label}: records without 'pilot': true are refused; "
                "real records need an approved source via the registry gate"
            )
        if record.get("intended_use") != PILOT_INTENDED_USE:
            problems.append(
                f"{label}: intended_use must be '{PILOT_INTENDED_USE}'; "
                "synthetic records must never enter training"
            )

        record_id = record.get("record_id")
        if record_id:
            if record_id in seen_record_ids:
                problems.append(f"{label}: duplicate record_id")
            seen_record_ids.add(record_id)

        sha = record.get("text_sha256")
        if isinstance(sha, str):
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                problems.append(f"{label}: text_sha256 is not a 64-char lowercase hex digest")
            elif sha in seen_sha:
                problems.append(f"{label}: duplicate text_sha256 (exact dedup violation)")
            else:
                seen_sha.add(sha)

        domain_id = record.get("domain")
        sub_id = record.get("subdomain")
        language = record.get("language")

        pair = (domain_id, sub_id)
        if domain_id and sub_id:
            if pair not in subdomain_languages:
                problems.append(f"{label}: unknown domain/subdomain '{domain_id}/{sub_id}'")
            elif language and language not in subdomain_languages[pair]:
                problems.append(
                    f"{label}: language '{language}' not allowed for '{domain_id}/{sub_id}'"
                )

        source_id = record.get("source_id")
        if source_id is not None:
            entry = registry.get(str(source_id))
            if entry is None:
                problems.append(f"{label}: source_id '{source_id}' not found in source registry")
            else:
                registry_license = str(entry.get("license", ""))
                record_license = str(record.get("license", ""))
                if record_license and record_license != registry_license:
                    problems.append(
                        f"{label}: license '{record_license}' does not match registry "
                        f"license '{registry_license}' for source '{source_id}'"
                    )
                record_provenance = str(record.get("provenance", "")).strip().upper()
                if record_provenance in ("UNVERIFIED", "UNKNOWN", ""):
                    problems.append(f"{label}: unverified provenance '{record.get('provenance')}'")

        text = record.get("text")
        if text is not None and isinstance(sha, str) and len(sha) == 64:
            if sha256_text(str(text)) != sha:
                problems.append(f"{label}: text does not hash to text_sha256")

        text_chars = record.get("text_chars")
        if text_chars is not None and (not isinstance(text_chars, int) or text_chars <= 0):
            problems.append(f"{label}: text_chars must be a positive integer")
        if text is not None and text_chars is not None and len(str(text)) != text_chars:
            problems.append(f"{label}: text_chars ({text_chars}) != len(text) ({len(str(text))})")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate mix taxonomy and pilot records.")
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry_records, registry_problems = load_jsonl(REGISTRY)
    registry = {r["source_id"]: r for r in registry_records if r.get("source_id")}

    taxonomy, problems = load_taxonomy(args.taxonomy)
    pilot_records, pilot_problems = load_jsonl(args.pilot)
    problems.extend(registry_problems)
    problems.extend(pilot_problems)
    problems.extend(validate_pilot_records(pilot_records, taxonomy, registry))

    domains = taxonomy.get("domains", [])
    subdomain_count = sum(len(d.get("subdomains", [])) for d in domains)

    report = {
        "taxonomy": str(args.taxonomy),
        "pilot": str(args.pilot),
        "domains": len(domains),
        "subdomains": subdomain_count,
        "pilot_records": len(pilot_records),
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=" * 62)
        print(" CODEXA MIX PLAN CHECK (P9)")
        print("=" * 62)
        print(f" taxonomy       : {args.taxonomy.name}")
        print(f" domains        : {len(domains)} / subdomains: {subdomain_count}")
        print(f" pilot records  : {len(pilot_records)}")
        if problems:
            print("-" * 62)
            for problem in problems:
                print(f"  [FAIL] {problem}")
        print("-" * 62)
        print(f"RESULT: {report['status']}")

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
