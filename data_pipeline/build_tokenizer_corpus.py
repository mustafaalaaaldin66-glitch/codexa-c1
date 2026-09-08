"""Build the v10 tokenizer corpus from approved local parquet data only."""
from __future__ import annotations

import hashlib
import json
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
TARGETS = {"ar": 2_000_000, "arz": 400_000, "en": 0}
SOURCES = {
    "ar": RAW / "fineweb2_arb_arab_sample",
    "arz": RAW / "fineweb2_arz_arab",
    "en": RAW / "fineweb_edu_english",
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def write_language(
    connection: duckdb.DuckDBPyConnection, language: str, blocked: set[str]
) -> dict:
    files = parquet_files(SOURCES[language])
    target = TARGETS[language]
    output = OUTPUT / f"{language}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set(blocked)
    accepted = duplicates = invalid = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        if target == 0:
            return {
                "files": [str(path.relative_to(ROOT)) for path in files],
                "target": target,
                "accepted": 0,
                "duplicates_or_eval": 0,
                "invalid_length_or_type": 0,
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
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
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    free = shutil.disk_usage(ROOT).free
    if free < 2 * 1024**3:
        raise RuntimeError("insufficient free disk for tokenizer corpus")
    connection = duckdb.connect()
    blocked = eval_hashes()
    counts = {language: write_language(connection, language, blocked) for language in TARGETS}
    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    with COMBINED.open("wb") as target:
        for language in ("ar", "arz", "en"):
            source = OUTPUT / f"{language}.txt"
            if source.exists():
                target.write(source.read_bytes())
    manifest = {
        "version": "tokenizer_train_v10",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalizer": "text.normalize.normalize_text",
        "dedup": "exact SHA-256 of normalized text; tokenizer_eval excluded",
        "length_chars": [40, 8000],
        "counts": counts,
        "corpus": {"path": str(COMBINED.relative_to(ROOT)), "bytes": COMBINED.stat().st_size,
                   "sha256": hashlib.sha256(COMBINED.read_bytes()).hexdigest()},
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
