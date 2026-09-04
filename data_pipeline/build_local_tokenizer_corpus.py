from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from text.normalize import normalize_text

SOURCE_FILE = PROJECT_ROOT / "data" / "tokenizer" / "corpus.txt"
EVAL_DIR = PROJECT_ROOT / "data" / "tokenizer_eval"

OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenizer_train_local"
MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "tokenizer_train_local_manifest.json"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simple_language_id(text: str) -> str:
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))

    if has_arabic and has_latin:
        return "mixed"

    if has_arabic:
        return "ar"

    if has_latin:
        return "en"

    return "other"


def passes_quality_filter(text: str) -> bool:
    text = text.strip()

    if len(text) < 30:
        return False

    words = text.split()

    if not words:
        return False

    unique_ratio = len(set(words)) / len(words)

    if unique_ratio < 0.30:
        return False

    return True


def load_eval_hashes() -> set[str]:
    hashes: set[str] = set()

    if not EVAL_DIR.exists():
        return hashes

    for path in EVAL_DIR.rglob("*.txt"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                normalized = normalize_text(line.strip())

                if normalized:
                    hashes.add(sha256_text(normalized))

    return hashes


def read_source() -> list[str]:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source corpus not found: {SOURCE_FILE}"
        )

    with SOURCE_FILE.open("r", encoding="utf-8") as handle:
        return [
            line.rstrip("\n")
            for line in handle
            if line.strip()
        ]


def main() -> None:
    print("=" * 58)
    print(" Codexa C1 - Local Tokenizer Corpus Builder")
    print(" Development unblock mode")
    print("=" * 58)

    lines = read_source()

    print(f"Input lines: {len(lines)}")

    eval_hashes = load_eval_hashes()
    print(f"Loaded eval hashes: {len(eval_hashes)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    categories = {
        "ar": [],
        "en": [],
        "mixed": [],
        "other": [],
    }

    seen_hashes: set[str] = set()

    stats = {
        "input": len(lines),
        "quality_rejected": 0,
        "empty_after_normalization": 0,
        "exact_duplicates": 0,
        "eval_contamination": 0,
        "accepted": 0,
    }

    for raw in lines:
        if not passes_quality_filter(raw):
            stats["quality_rejected"] += 1
            continue

        normalized = normalize_text(raw)

        if not normalized:
            stats["empty_after_normalization"] += 1
            continue

        text_hash = sha256_text(normalized)

        if text_hash in seen_hashes:
            stats["exact_duplicates"] += 1
            continue

        if text_hash in eval_hashes:
            stats["eval_contamination"] += 1
            continue

        seen_hashes.add(text_hash)

        lang = simple_language_id(normalized)

        categories[lang].append(normalized)
        stats["accepted"] += 1

    manifest_categories = []

    for category, items in categories.items():
        output_file = OUTPUT_DIR / f"{category}.txt"

        with output_file.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(item + "\n")

        text = "\n".join(items)

        manifest_categories.append(
            {
                "category": category,
                "file_path": str(
                    output_file.relative_to(PROJECT_ROOT)
                ),
                "line_count": len(items),
                "word_count": sum(
                    len(item.split()) for item in items
                ),
                "char_count": sum(
                    len(item) for item in items
                ),
                "file_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            }
        )

    manifest = {
        "version": "tokenizer-train-local-dev-v1",
        "mode": "local-development-only",
        "source_file": str(
            SOURCE_FILE.relative_to(PROJECT_ROOT)
        ),
        "normalization": "norm-v1",
        "categories": manifest_categories,
        "stats": stats,
        "notes": [
            "Development unblock only.",
            "Not the final V9 production tokenizer corpus.",
            "No near-dedup in this local mode.",
            "No external dataset download.",
        ],
    }

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("-" * 58)
    print("INPUT / FILTER STATS")

    for key, value in stats.items():
        print(f"{key:24s}: {value}")

    print("-" * 58)
    print("CATEGORY OUTPUTS")

    for item in manifest_categories:
        print(
            f"{item['category']:8s} | "
            f"{item['line_count']:5d} lines | "
            f"{item['word_count']:6d} words | "
            f"{item['char_count']:7d} chars"
        )

    print("-" * 58)
    print(f"Manifest: {MANIFEST_FILE}")
    print("STATUS: LOCAL CORPUS BUILD PASS")


if __name__ == "__main__":
    main()
