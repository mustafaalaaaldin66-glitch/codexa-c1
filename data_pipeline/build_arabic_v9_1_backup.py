from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text.normalize import normalize_text


# ============================================================
# Codexa C1 — Arabic Corpus Pipeline V9.1
# ============================================================

SOURCE_ID = "fineweb2_arb_arab"
HF_DATASET = "HuggingFaceFW/fineweb-2"
HF_CONFIG = "arb_Arab"
HF_SPLIT = "train"

TARGET_SAMPLES = 100_000
SHARD_LINES = 10_000

MIN_CHARS = 30
MAX_CHARS = 20_000
MIN_ARABIC_RATIO = 0.50
MIN_UNIQUE_WORD_RATIO = 0.30


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0

    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    letters = len(re.findall(r"[^\W\d_]", text, flags=re.UNICODE))

    if letters == 0:
        return 0.0

    return arabic / letters


def passes_quality_filter(text: str) -> bool:
    stripped = text.strip()

    if len(stripped) < MIN_CHARS:
        return False

    if len(stripped) > MAX_CHARS:
        return False

    if arabic_ratio(stripped) < MIN_ARABIC_RATIO:
        return False

    words = stripped.split()

    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < MIN_UNIQUE_WORD_RATIO:
            return False

    return True


def get_eval_hashes() -> set[str]:
    eval_dir = PROJECT_ROOT / "data" / "tokenizer_eval"

    hashes: set[str] = set()

    if not eval_dir.exists():
        return hashes

    for path in eval_dir.rglob("*.txt"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                normalized = normalize_text(line.strip())
                if normalized:
                    hashes.add(sha256_text(normalized))

    return hashes


def write_shard(shard_path: Path, lines: list[str]) -> str:
    with shard_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

    content = "\n".join(lines)
    return sha256_text(content)


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "tokenizer_train_v9_1" / "arabic"
    manifest_dir = PROJECT_ROOT / "data" / "manifests"

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        manifest_dir / "tokenizer_train_arabic_v9_1_manifest.json"
    )

    print("=" * 70)
    print("CODEXA C1 — ARABIC CORPUS PIPELINE V9.1")
    print("=" * 70)
    print(f"Dataset       : {HF_DATASET}")
    print(f"Config        : {HF_CONFIG}")
    print(f"Split         : {HF_SPLIT}")
    print(f"Target samples: {TARGET_SAMPLES:,}")
    print(f"Shard size    : {SHARD_LINES:,}")
    print("=" * 70)

    eval_hashes = get_eval_hashes()

    print(f"Eval hashes loaded: {len(eval_hashes):,}")

    ds = load_dataset(
        HF_DATASET,
        HF_CONFIG,
        split=HF_SPLIT,
        streaming=True,
    )

    seen_hashes: set[str] = set()

    shard_lines: list[str] = []
    shard_records: list[dict] = []

    accepted = 0
    rejected_quality = 0
    rejected_decontamination = 0
    rejected_duplicate = 0

    started = time.perf_counter()

    for row in ds:
        text = row.get("text", "")

        if not isinstance(text, str):
            continue

        # A record can contain paragraphs.
        for raw_line in text.splitlines():
            normalized = normalize_text(raw_line)

            if not normalized:
                continue

            if not passes_quality_filter(normalized):
                rejected_quality += 1
                continue

            digest = sha256_text(normalized)

            if digest in seen_hashes:
                rejected_duplicate += 1
                continue

            if digest in eval_hashes:
                rejected_decontamination += 1
                continue

            seen_hashes.add(digest)
            shard_lines.append(normalized)
            accepted += 1

            if len(shard_lines) >= SHARD_LINES:
                shard_number = len(shard_records)

                shard_path = (
                    output_dir
                    / f"arabic-{shard_number:05d}.txt"
                )

                file_hash = write_shard(shard_path, shard_lines)

                shard_records.append(
                    {
                        "shard": shard_number,
                        "file_path": str(
                            shard_path.relative_to(PROJECT_ROOT)
                        ),
                        "line_count": len(shard_lines),
                        "char_count": sum(map(len, shard_lines)),
                        "sha256": file_hash,
                    }
                )

                shard_lines.clear()

                elapsed = time.perf_counter() - started
                rate = accepted / elapsed if elapsed else 0.0

                print(
                    f"[SHARD {shard_number:05d}] "
                    f"accepted={accepted:,} "
                    f"rate={rate:,.1f} lines/sec"
                )

            if accepted >= TARGET_SAMPLES:
                break

        if accepted >= TARGET_SAMPLES:
            break

    if shard_lines:
        shard_number = len(shard_records)

        shard_path = (
            output_dir
            / f"arabic-{shard_number:05d}.txt"
        )

        file_hash = write_shard(shard_path, shard_lines)

        shard_records.append(
            {
                "shard": shard_number,
                "file_path": str(
                    shard_path.relative_to(PROJECT_ROOT)
                ),
                "line_count": len(shard_lines),
                "char_count": sum(map(len, shard_lines)),
                "sha256": file_hash,
            }
        )

    elapsed = time.perf_counter() - started

    manifest = {
        "version": "tokenizer-train-corpus-v9.1-arabic",
        "source_id": SOURCE_ID,
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": HF_SPLIT,
        "normalization": "norm-v1",
        "target_samples": TARGET_SAMPLES,
        "accepted_samples": accepted,
        "rejected_quality": rejected_quality,
        "rejected_duplicate": rejected_duplicate,
        "rejected_decontamination": rejected_decontamination,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_lines_per_second": round(
            accepted / elapsed if elapsed else 0.0,
            3,
        ),
        "shards": shard_records,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("V9.1 ARABIC PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Accepted       : {accepted:,}")
    print(f"Quality reject : {rejected_quality:,}")
    print(f"Duplicate reject: {rejected_duplicate:,}")
    print(f"Decontamination: {rejected_decontamination:,}")
    print(f"Elapsed        : {elapsed:.2f}s")
    print(
        f"Throughput     : "
        f"{accepted / elapsed if elapsed else 0.0:,.1f} lines/sec"
    )
    print(f"Manifest       : {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
