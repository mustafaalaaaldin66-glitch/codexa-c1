"""Read-only validator for data/source_registry.jsonl (license policy gate).

Every ingestion source must be registered BEFORE use with a dataset id,
config/split, license, provenance and status. This validator enforces the
policy in ``data/licenses/README.md`` and refuses unsafe registrations.

Rules
-----
1. every line is valid JSON with the required keys
2. ``source_id`` is unique
3. ``status`` is one of: approved, conditional_review, pending_provenance, rejected
4. ``status == approved`` requires an allowed license AND known provenance
5. a forbidden license (e.g. CC-BY-NC) can never be ``approved``
6. ``conditional_review`` licenses must not be approved while under review

Exit code 0 = clean, 1 = violations found. Never writes, never downloads.

Run:
    .\\.venv\\Scripts\\python.exe data_pipeline\\source_registry_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "source_registry.jsonl"

REQUIRED_KEYS = ("source_id", "language", "role", "license", "provenance", "status")

ALLOWED_STATUS = ("approved", "conditional_review", "pending_provenance", "rejected")

ALLOWED_LICENSES = (
    "Apache-2.0",
    "MIT",
    "CC0",
    "Public Domain",
    "ODC-By-1.0",
    "CC-BY-4.0",
)

CONDITIONAL_LICENSES = ("CC-BY-SA-4.0",)

FORBIDDEN_LICENSE_MARKERS = ("NC", "ND", "RESEARCH", "NON-COMMERCIAL", "UNKNOWN")

UNVERIFIED_PROVENANCE = ("UNVERIFIED", "", "UNKNOWN")
def load_records(path: Path) -> tuple[list[dict], list[str]]:
    """Return (records, problems) without raising on malformed input."""
    records: list[dict] = []
    problems: list[str] = []
    if not path.exists():
        return records, [f"registry not found: {path}"]
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: invalid JSON ({exc.msg})")
    return records, problems


def license_is_allowed(license_name: str) -> bool:
    return license_name.strip() in ALLOWED_LICENSES


def license_is_forbidden(license_name: str) -> bool:
    upper = license_name.upper().replace("_", "-")
    return any(marker in upper for marker in FORBIDDEN_LICENSE_MARKERS)


def validate_records(records: list[dict]) -> list[str]:
    """Return a list of policy violations (empty == clean)."""
    problems: list[str] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records, start=1):
        label = record.get("source_id") or f"record#{index}"

        for key in REQUIRED_KEYS:
            if key not in record or record[key] in ("", None):
                problems.append(f"{label}: missing required field '{key}'")

        source_id = record.get("source_id")
        if source_id:
            if source_id in seen_ids:
                problems.append(f"{label}: duplicate source_id")
            seen_ids.add(source_id)

        status = record.get("status")
        if status is not None and status not in ALLOWED_STATUS:
            problems.append(f"{label}: invalid status '{status}'")

        license_name = str(record.get("license", ""))
        provenance = str(record.get("provenance", "")).strip()

        if status == "approved":
            if not license_is_allowed(license_name):
                if license_name.strip() in CONDITIONAL_LICENSES:
                    problems.append(
                        f"{label}: approved with conditional license "
                        f"'{license_name}' (needs governance review)"
                    )
                elif license_is_forbidden(license_name):
                    problems.append(
                        f"{label}: approved with forbidden license '{license_name}'"
                    )
                else:
                    problems.append(
                        f"{label}: approved with unapproved license '{license_name}'"
                    )
            if provenance.upper() in UNVERIFIED_PROVENANCE:
                problems.append(
                    f"{label}: approved with unverified provenance '{provenance}'"
                )

        if license_is_forbidden(license_name) and status == "approved":
            problems.append(
                f"{label}: forbidden license '{license_name}' must not be approved"
            )

    return problems
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the data source registry.")
    parser.add_argument("--path", type=Path, default=REGISTRY)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    records, problems = load_records(args.path)
    problems.extend(validate_records(records))

    approved = sorted(
        r.get("source_id") for r in records if r.get("status") == "approved"
    )
    blocked = sorted(
        r.get("source_id")
        for r in records
        if r.get("status") in ("pending_provenance", "conditional_review", "rejected")
    )

    report = {
        "registry": str(args.path),
        "records": len(records),
        "approved_sources": approved,
        "not_ingestable_sources": blocked,
        "problems": problems,
        "status": "PASS" if not problems else "FAIL",
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("=" * 62)
        print(" CODEXA SOURCE REGISTRY CHECK")
        print("=" * 62)
        print(f" registry        : {args.path.name}")
        print(f" records         : {len(records)}")
        print(f" approved        : {approved}")
        print(f" not ingestable  : {blocked}")
        if problems:
            print("-" * 62)
            for problem in problems:
                print(f"  [FAIL] {problem}")
        print("-" * 62)
        print(f"RESULT: {report['status']}")

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())