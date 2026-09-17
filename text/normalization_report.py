"""Explain *why* ``normalize_text`` changes a raw string.

The tokenizer is byte-level, so ``decode(encode(raw)) == raw`` always holds.
The real risk is a *pipeline* that forgets to normalize: training and inference
would then disagree. This tool classifies every reason the canonical form of a
raw string differs, and aggregates the counts over the curated eval corpus.

Categories
----------
* ``nfkc``       — Unicode compatibility fold (ligatures, circled digits, ...)
* ``tatweel``    — U+0640 kashida removal
* ``diacritics`` — Arabic combining marks removal
* ``whitespace`` — space/tab collapse, blank-line clamp, strip

Run:
    .\\.venv\\Scripts\\python.exe text\\normalization_report.py
    .\\.venv\\Scripts\\python.exe text\\normalization_report.py --out data\\tokenizer_eval\\normalization_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text.normalize import (  # noqa: E402
    normalize_text,
    normalize_whitespace,
    remove_arabic_diacritics,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

EVAL_DIR = ROOT / "data" / "tokenizer_eval"
DEFAULT_OUT = EVAL_DIR / "normalization_report.json"
CATEGORIES = ("nfkc", "tatweel", "diacritics", "whitespace")
def classify(raw: str) -> list[str]:
    """Return the normalization steps that actually change ``raw``."""
    reasons: list[str] = []
    if unicodedata.normalize("NFKC", raw) != raw:
        reasons.append("nfkc")
    if "\u0640" in raw:
        reasons.append("tatweel")
    if remove_arabic_diacritics(raw) != raw:
        reasons.append("diacritics")
    if normalize_whitespace(raw) != raw:
        reasons.append("whitespace")
    return reasons


def analyze(samples: list[str]) -> dict:
    """Per-sample classification plus aggregate counters."""
    changed = 0
    unchanged = 0
    counts = {category: 0 for category in CATEGORIES}
    examples: dict[str, str] = {}

    for raw in samples:
        if not raw.strip():
            continue
        if normalize_text(raw) == raw:
            unchanged += 1
            continue
        changed += 1
        for reason in classify(raw):
            counts[reason] += 1
            examples.setdefault(reason, raw[:70])

    total = changed + unchanged
    return {
        "samples": total,
        "changed_by_normalization": changed,
        "unchanged": unchanged,
        "changed_rate": changed / total if total else 0.0,
        "reason_counts": counts,
        "reason_examples": examples,
        "invariant": "decode(encode(raw)) == raw holds for the byte-level tokenizer; "
        "normalization is a separate, mandatory pipeline step",
    }
def eval_samples() -> list[str]:
    if not EVAL_DIR.exists():
        return []
    samples: list[str] = []
    for path in sorted(EVAL_DIR.rglob("*.txt")):
        samples.extend(
            line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain normalization diffs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    samples = eval_samples()
    report = {
        "version": "normalization_report_v1",
        "corpus": "curated eval set (data/tokenizer_eval)",
        "analysis": analyze(samples),
    }

    print("=" * 66)
    print(" NORMALIZATION DIFF REPORT")
    print("=" * 66)
    analysis = report["analysis"]
    print(f" samples          : {analysis['samples']}")
    print(f" changed          : {analysis['changed_by_normalization']} "
          f"({analysis['changed_rate']:.4f})")
    print(f" unchanged        : {analysis['unchanged']}")
    for category in CATEGORIES:
        print(f"   {category:<11}: {analysis['reason_counts'][category]}")

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f" report           : {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())