"""Resume the v10 tokenizer corpus build: build arz.txt only, keep existing ar.txt, then write the combined corpus and manifest.

Does not touch data/tokenizer_train_v10/ar.txt (already built).
English target is 0 by design (weak local device).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from text.normalize import normalize_text

RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "tokenizer_train_v10"
EVAL = ROOT / "data" / "tokenizer_eval"
MANIFEST = ROOT / "data" / "manifests" / "tokenizer_train_v10.json"
COMBINED = ROOT / "data" / "tokenizer" / "corpus.txt"
LOCK_FILE = OUTPUT / ".build_arz_v10.lock"
TARGETS = {"ar": 2_000_000, "arz": 400_000, "en": 0}
SOURCES = {
    "ar": RAW / "fineweb2_arb_arab_sample",
    "arz": RAW / "fineweb2_arz_arab",
    "en": RAW / "fineweb_edu_english",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_hashes() -> set[str]:
    hashes: set[str] = set()
    for file in EVAL.rglob("*.txt") if EVAL.exists() else []:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                text = normalize_text(line)
                if text:
                    hashes.add(sha256(text))
    return hashes


def parquet_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.parquet")) if directory.exists() else []


def rows_from_parquet(connection: duckdb.DuckDBPyConnection, files: list[Path]):
    if not files:
        return
    paths = ", ".join("'" + str(path).replace("'", "''") + "'" for path in files)
    columns = [row[0] for row in connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet([{paths}], union_by_name=true)"
    ).fetchall()]
    text_column = next((name for name in ("text", "content", "document") if name in columns), None)
    if text_column is None:
        raise ValueError(f"no text column in {files[0]}: {columns}")
    query = f"SELECT {text_column} AS text FROM read_parquet([{paths}], union_by_name=true)"
    yield from connection.execute(query).fetch_record_batch(10_000)


def build_arz(connection: duckdb.DuckDBPyConnection, blocked: set[str]) -> dict:
    files = parquet_files(SOURCES["arz"])
    target = TARGETS["arz"]
    output = OUTPUT / "arz.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set(blocked)
    accepted = duplicates = invalid = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for batch in rows_from_parquet(connection, files) or []:
            for row in batch.to_pylist():
                raw_text = row["text"]
                if not isinstance(raw_text, str):
                    invalid += 1
                    continue
                text = normalize_text(raw_text)
                if not 40 <= len(text) <= 8000:
                    invalid += 1
                    continue
                digest = sha256(text)
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                handle.write(text.replace("\n", " ") + "\n")
                accepted += 1
                if accepted >= target:
                    break
            if accepted >= target:
                break
    return {
        "files": [str(path.relative_to(ROOT)) for path in files],
        "target": target,
        "accepted": accepted,
        "duplicates_or_eval": duplicates,
        "invalid_length_or_type": invalid,
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
    }


def existing_language_stats(language: str) -> dict:
    output = OUTPUT / f"{language}.txt"
    if not output.exists():
        return {
            "files": [str(SOURCES[language].relative_to(ROOT))],
            "target": TARGETS[language],
            "accepted": 0,
            "duplicates_or_eval": 0,
            "invalid_length_or_type": 0,
            "bytes": 0,
            "sha256": "",
        }
    lines = 0
    with output.open("r", encoding="utf-8") as handle:
        for _ in handle:
            lines += 1
    return {
        "files": [str(SOURCES[language].relative_to(ROOT))],
        "target": TARGETS[language],
        "accepted": lines,
        "duplicates_or_eval": 0,
        "invalid_length_or_type": 0,
        "bytes": output.stat().st_size,
        "sha256": file_sha256(output),
    }


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    free = shutil.disk_usage(ROOT).free
    if free < 2 * 1024**3:
        raise RuntimeError("insufficient free disk for tokenizer corpus")
    ar_path = OUTPUT / "ar.txt"
    if not ar_path.exists():
        raise RuntimeError("ar.txt missing; run the full builder first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another v10 corpus build is active: {LOCK_FILE}") from exc

    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.close(lock_fd)

        arz_path = OUTPUT / "arz.txt"
        if arz_path.exists() and count_lines(arz_path) == TARGETS["arz"]:
            print(f"Using completed arz corpus: {arz_path}")
            arz_stats = existing_language_stats("arz")
        else:
            connection = duckdb.connect()
            try:
                blocked = eval_hashes()
                arz_stats = build_arz(connection, blocked)
            finally:
                connection.close()

        counts = {
            "ar": existing_language_stats("ar"),
            "arz": arz_stats,
            "en": existing_language_stats("en"),
        }
        COMBINED.parent.mkdir(parents=True, exist_ok=True)
        combined_tmp = COMBINED.with_suffix(".tmp")
        with combined_tmp.open("wb") as target:
            for language in ("ar", "arz", "en"):
                source = OUTPUT / f"{language}.txt"
                if source.exists():
                    with source.open("rb") as reader:
                        shutil.copyfileobj(reader, target, length=1024 * 1024)
        combined_tmp.replace(COMBINED)
        manifest = {
            "version": "tokenizer_train_v10",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "normalizer": "text.normalize.normalize_text",
            "dedup": "exact SHA-256 of normalized text; tokenizer_eval excluded",
            "length_chars": [40, 8000],
            "counts": counts,
            "corpus": {"path": str(COMBINED.relative_to(ROOT)), "bytes": COMBINED.stat().st_size,
                       "sha256": file_sha256(COMBINED)},
        }
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()