from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from text.normalize import normalize_text


# ============================================================
# Codexa C1 — V9.1 Fast Path
# Arabic corpus builder
# ============================================================

SOURCE_ID = "fineweb2_arb_arab"
HF_DATASET = "HuggingFaceFW/fineweb-2"
HF_CONFIG = "arb_Arab"
HF_SPLIT = "train"

PARQUET_PATH = Path(
    r"D:\hf_cache\hub\datasets--HuggingFaceFW--fineweb-2"
    r"\snapshots\af9c13333eb981300149d5ca60a8e9d659b276b9"
    r"\data\arb_Arab\train\000_00000.parquet"
)

TARGET_SAMPLES = 20_000

READ_BATCH_SIZE = 50_000
SHARD_LINES = 10_000

# Automatic interruption test.
TEST_STOP_AFTER_SHARD = True

MIN_CHARS = 30
MAX_CHARS = 20_000
MIN_ARABIC_RATIO = 0.50
MIN_UNIQUE_WORD_RATIO = 0.30

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def passes_quality_filter(text: str) -> bool:
    length = len(text)

    if length < MIN_CHARS or length > MAX_CHARS:
        return False

    letters = LETTER_RE.findall(text)

    if not letters:
        return False

    arabic = len(ARABIC_RE.findall(text))

    if arabic / len(letters) < MIN_ARABIC_RATIO:
        return False

    words = text.split()

    if words:
        if len(set(words)) / len(words) < MIN_UNIQUE_WORD_RATIO:
            return False

    return True


def get_eval_hashes() -> set[str]:
    eval_dir = PROJECT_ROOT / "data" / "tokenizer_eval"

    hashes: set[str] = set()

    if not eval_dir.exists():
        return hashes

    for path in eval_dir.rglob("*.txt"):
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                normalized = normalize_text(
                    line.rstrip("\n")
                )

                if normalized:
                    hashes.add(
                        sha256_text(normalized)
                    )

    return hashes


def atomic_write_shard(
    path: Path,
    lines: list[str],
) -> tuple[str, int]:
    temp = path.with_suffix(".tmp")

    char_count = 0
    hasher = hashlib.sha256()

    with temp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for line in lines:
            encoded = (
                line.encode("utf-8")
                + b"\n"
            )

            f.write(line)
            f.write("\n")

            hasher.update(encoded)
            char_count += len(line)

    temp.replace(path)

    return hasher.hexdigest(), char_count


def save_checkpoint(
    path: Path,
    *,
    source_row_offset: int,
    accepted: int,
    rejected_quality: int,
    rejected_duplicate: int,
    rejected_decontamination: int,
    shard_number: int,
) -> None:
    data = {
        "version": "v9.1-fastpath-row-boundary",
        "source_id": SOURCE_ID,
        "source_row_offset": source_row_offset,
        "accepted": accepted,
        "rejected_quality": rejected_quality,
        "rejected_duplicate": rejected_duplicate,
        "rejected_decontamination": rejected_decontamination,
        "shard_number": shard_number,
    }

    temp = path.with_suffix(".tmp")

    with temp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp.replace(path)


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def rebuild_seen_hashes(
    output_dir: Path,
) -> set[str]:
    hashes: set[str] = set()

    shards = sorted(
        output_dir.glob("arabic-*.txt")
    )

    if not shards:
        return hashes

    print(
        f"Rebuilding dedup state: "
        f"{len(shards)} shard(s)"
    )

    for path in shards:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.rstrip("\n")

                if line:
                    hashes.add(
                        sha256_text(line)
                    )

    print(
        f"Existing hashes: {len(hashes):,}"
    )

    return hashes


def build_manifest(
    output_dir: Path,
    manifest_path: Path,
    *,
    accepted: int,
    source_rows: int,
    rejected_quality: int,
    rejected_duplicate: int,
    rejected_decontamination: int,
    elapsed: float,
) -> None:
    shards = []

    for path in sorted(
        output_dir.glob("arabic-*.txt")
    ):
        line_count = 0
        char_count = 0
        hasher = hashlib.sha256()

        with path.open(
            "rb"
        ) as f:
            for raw in f:
                hasher.update(raw)

                if raw != b"\n":
                    line_count += 1
                    char_count += len(
                        raw.decode(
                            "utf-8"
                        ).rstrip("\n")
                    )

        shards.append(
            {
                "shard": int(
                    path.stem.split("-")[-1]
                ),
                "file_path": str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "line_count": line_count,
                "char_count": char_count,
                "sha256_file_bytes": hasher.hexdigest(),
            }
        )

    manifest = {
        "version": "tokenizer-train-corpus-v9.1-fastpath-arabic",
        "source_id": SOURCE_ID,
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": HF_SPLIT,
        "reader": "duckdb",
        "normalization": "norm-v1",
        "target_samples": TARGET_SAMPLES,
        "accepted_samples": accepted,
        "source_rows_processed": source_rows,
        "rejected_quality": rejected_quality,
        "rejected_duplicate": rejected_duplicate,
        "rejected_decontamination": rejected_decontamination,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "throughput_accepted_lines_per_second": round(
            accepted / elapsed
            if elapsed
            else 0.0,
            3,
        ),
        "shards": shards,
    }

    temp = manifest_path.with_suffix(".tmp")

    with temp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temp.replace(manifest_path)


def main() -> None:
    output_dir = (
        PROJECT_ROOT
        / "data"
        / "tokenizer_train_v9_1"
        / "arabic"
    )

    manifest_dir = (
        PROJECT_ROOT
        / "data"
        / "manifests"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        manifest_dir
        / "tokenizer_train_arabic_v9_1_checkpoint.json"
    )

    manifest_path = (
        manifest_dir
        / "tokenizer_train_arabic_v9_1_manifest.json"
    )

    print("=" * 70)
    print("CODEXA C1 — V9.1 FAST PATH")
    print("=" * 70)
    print(f"Target         : {TARGET_SAMPLES:,}")
    print(f"Read batch     : {READ_BATCH_SIZE:,}")
    print(f"Shard size     : {SHARD_LINES:,}")
    print(f"Test stop      : {TEST_STOP_AFTER_SHARD}")
    print("=" * 70)

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            PARQUET_PATH
        )

    eval_hashes = get_eval_hashes()

    print(
        f"Eval hashes: {len(eval_hashes):,}"
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    if checkpoint:
        row_offset = int(
            checkpoint["source_row_offset"]
        )
        accepted = int(
            checkpoint["accepted"]
        )
        rejected_quality = int(
            checkpoint["rejected_quality"]
        )
        rejected_duplicate = int(
            checkpoint["rejected_duplicate"]
        )
        rejected_decontamination = int(
            checkpoint["rejected_decontamination"]
        )
        shard_number = int(
            checkpoint["shard_number"]
        )

        print("=" * 70)
        print("RESUMING FROM CHECKPOINT")
        print(
            f"Source row : {row_offset:,}"
        )
        print(
            f"Accepted   : {accepted:,}"
        )
        print(
            f"Next shard : {shard_number:05d}"
        )
        print("=" * 70)

    else:
        row_offset = 0
        accepted = 0
        rejected_quality = 0
        rejected_duplicate = 0
        rejected_decontamination = 0
        shard_number = 0

    seen_hashes = rebuild_seen_hashes(
        output_dir
    )

    shard_lines: list[str] = []

    started = time.perf_counter()

    con = duckdb.connect()

    try:
        con.execute(
            "PRAGMA threads=2"
        )

        cursor = con.execute(
            "SELECT text FROM read_parquet(?)",
            [str(PARQUET_PATH)],
        )

        # Resume by discarding complete source rows.
        if row_offset:
            skipped = 0

            while skipped < row_offset:
                chunk = cursor.fetchmany(
                    min(
                        READ_BATCH_SIZE,
                        row_offset - skipped,
                    )
                )

                if not chunk:
                    break

                skipped += len(chunk)

            print(
                f"Skipped source rows: "
                f"{skipped:,}"
            )

        while accepted < TARGET_SAMPLES:
            batch_started = time.perf_counter()

            rows = cursor.fetchmany(
                READ_BATCH_SIZE
            )

            read_elapsed = (
                time.perf_counter()
                - batch_started
            )

            if not rows:
                print(
                    "SOURCE EXHAUSTED"
                )
                break

            for text, in rows:
                current_row = row_offset

                if isinstance(text, str):
                    for raw_line in text.splitlines():

                        normalized = normalize_text(
                            raw_line
                        )

                        if not normalized:
                            continue

                        if not passes_quality_filter(
                            normalized
                        ):
                            rejected_quality += 1
                            continue

                        digest = sha256_text(
                            normalized
                        )

                        if digest in seen_hashes:
                            rejected_duplicate += 1
                            continue

                        if digest in eval_hashes:
                            rejected_decontamination += 1
                            continue

                        seen_hashes.add(
                            digest
                        )

                        shard_lines.append(
                            normalized
                        )

                        accepted += 1

                        if len(shard_lines) >= SHARD_LINES:

                            shard_path = (
                                output_dir
                                / f"arabic-{shard_number:05d}.txt"
                            )

                            file_hash, char_count = (
                                atomic_write_shard(
                                    shard_path,
                                    shard_lines,
                                )
                            )

                            shard_number += 1
                            shard_lines.clear()

                            # IMPORTANT:
                            # current_row is not committed
                            # until the entire source row ends.

                            print(
                                f"[SHARD] "
                                f"{shard_number - 1:05d} | "
                                f"accepted={accepted:,} | "
                                f"read={read_elapsed:.3f}s | "
                                f"chars={char_count:,} | "
                                f"sha256={file_hash[:12]}"
                            )

                row_offset = current_row + 1

                # Commit only at source-row boundary.
                if (
                    len(shard_lines) == 0
                    or accepted >= TARGET_SAMPLES
                ):
                    save_checkpoint(
                        checkpoint_path,
                        source_row_offset=row_offset,
                        accepted=accepted,
                        rejected_quality=rejected_quality,
                        rejected_duplicate=rejected_duplicate,
                        rejected_decontamination=rejected_decontamination,
                        shard_number=shard_number,
                    )

                if (
                    accepted >= TARGET_SAMPLES
                ):
                    break

                if (
                    TEST_STOP_AFTER_SHARD
                    and shard_number > 0
                    and len(shard_lines) == 0
                ):
                    print(
                        "TEST STOP: "
                        "checkpoint committed "
                        "at source-row boundary."
                    )
                    return

    finally:
        con.close()

    if shard_lines:
        shard_path = (
            output_dir
            / f"arabic-{shard_number:05d}.txt"
        )

        file_hash, char_count = (
            atomic_write_shard(
                shard_path,
                shard_lines,
            )
        )

        shard_number += 1

        print(
            f"[FINAL SHARD] "
            f"{shard_number - 1:05d} | "
            f"lines={len(shard_lines):,} | "
            f"chars={char_count:,} | "
            f"sha256={file_hash[:12]}"
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    build_manifest(
        output_dir,
        manifest_path,
        accepted=accepted,
        source_rows=row_offset,
        rejected_quality=rejected_quality,
        rejected_duplicate=rejected_duplicate,
        rejected_decontamination=rejected_decontamination,
        elapsed=elapsed,
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(
        f"Accepted     : {accepted:,}"
    )
    print(
        f"Source rows  : {row_offset:,}"
    )
    print(
        f"Elapsed      : {elapsed:.2f}s"
    )
    print(
        f"Throughput   : "
        f"{accepted / elapsed if elapsed else 0.0:,.1f}/s"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
