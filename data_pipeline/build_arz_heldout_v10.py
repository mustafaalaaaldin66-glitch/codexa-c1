"""Build Egyptian held-out from FineWeb2 arz test split only."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import duckdb
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from text.normalize import normalize_text
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "fineweb2_arz_arab" / "data" / "arz_Arab" / "test"
EVAL_DIR = PROJECT_ROOT / "data" / "tokenizer_eval"
TRAIN_ARZ = PROJECT_ROOT / "data" / "tokenizer_train_v10" / "arz.txt"
OUT_DIR = PROJECT_ROOT / "data" / "heldout"
OUT_TEXT = OUT_DIR / "arz_test_heldout.txt"
OUT_MANIFEST = OUT_DIR / "arz_test_heldout.manifest.json"
LOCK = OUT_DIR / ".arz_test_heldout.lock"
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def eval_hashes() -> set[str]:
    hashes: set[str] = set()
    if EVAL_DIR.exists():
        for path in EVAL_DIR.rglob("*.txt"):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    text = normalize_text(line)
                    if text:
                        hashes.add(sha256(text))
    return hashes


def train_hashes(path: Path, limit: int = 20000) -> set[str]:
    # Low-memory fingerprint: training files are already normalized, so hash
    # raw bytes of the first `limit` lines instead of re-normalizing 1.1 GB.
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            seen.add(hashlib.sha256(line).hexdigest())
            if len(seen) >= limit:
                break
    return seen


def acquire_lock() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK.open("x", encoding="ascii") as handle:
            handle.write(str(__import__("os").getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"held-out builder already running: {LOCK}") from exc


def split_long_text(text: str, limit: int = 8000) -> list[str]:
    # Test documents are whole articles (multi-paragraph). Training used
    # line-granular text, so split test documents into 40..8000-char chunks
    # on paragraph/sentence boundaries instead of rejecting whole articles.
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    parts = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        pieces = part.split(". ")
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            if i < len(pieces) - 1:
                piece += "."
            if current_len + len(piece) + 1 > limit and current:
                chunks.append(" ".join(current))
                current = [piece]
                current_len = len(piece)
            else:
                current.append(piece)
                current_len += len(" ".join(current))
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if 40 <= len(c) <= 8000]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build arz test-split held-out.")
    parser.add_argument("--max-lines", type=int, default=2000)
    parser.add_argument("--train-limit", type=int, default=20000)
    args = parser.parse_args()
    acquire_lock()
    try:
        return build(args)
    finally:
        LOCK.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> int:
    files = sorted(TEST_DIR.rglob("*.parquet"))
    if not files:
        print(f"FAIL: no test parquet files under {TEST_DIR}")
        return 1
    blocked = eval_hashes()
    train_seen = train_hashes(TRAIN_ARZ, args.train_limit)
    connection = duckdb.connect()
    try:
        paths = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
        desc = f"DESCRIBE SELECT * FROM read_parquet([{paths}], union_by_name=true)"
        columns = [row[0] for row in connection.execute(desc).fetchall()]
        text_col = next((c for c in ("text", "content", "document") if c in columns), None)
        if text_col is None:
            print(f"FAIL: no text column: {columns}")
            return 1
        accepted: list[str] = []
        seen: set[str] = set()
        stats = {"rows": 0, "accepted": 0, "dup": 0, "eval_overlap": 0,
                 "train_overlap": 0, "bad_length": 0}
        query = f"SELECT {text_col} AS text FROM read_parquet([{paths}], union_by_name=true)"
        for batch in connection.execute(query).fetch_record_batch(10000):
            for row in batch.to_pylist():
                raw = row["text"]
                stats["rows"] += 1
                if not isinstance(raw, str):
                    stats["bad_length"] += 1
                    continue
                text = normalize_text(raw)
                chunks = split_long_text(text)
                if not chunks:
                    stats["bad_length"] += 1
                    continue
                for chunk in chunks:
                    digest = sha256(chunk)
                    if digest in seen:
                        stats["dup"] += 1
                        continue
                    if digest in blocked:
                        stats["eval_overlap"] += 1
                        continue
                    if digest in train_seen:
                        stats["train_overlap"] += 1
                        continue
                    seen.add(digest)
                    accepted.append(chunk)
                    stats["accepted"] += 1
                    if len(accepted) >= args.max_lines:
                        break
                if len(accepted) >= args.max_lines:
                    break
    finally:
        connection.close()
    if not accepted:
        print("FAIL: no held-out lines accepted")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEXT.write_text("\n".join(accepted) + "\n", encoding="utf-8")
    file_digest = hashlib.sha256()
    with OUT_TEXT.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            file_digest.update(chunk)
    manifest = {
        "version": "arz_test_heldout_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalizer": "text.normalize.normalize_text",
        "dedup": "exact SHA-256 of normalized text",
        "train_hashes_checked": len(train_seen),
        "stats": stats,
        "source": {
            "dataset_id": "HuggingFaceFW/fineweb-2",
            "config": "arz_Arab",
            "split": "test",
            "files": [str(p.relative_to(PROJECT_ROOT)) for p in files],
            "license": "ODC-By-1.0",
            "provenance": "HuggingFace HuggingFaceFW/fineweb-2",
        },
        "output": {
            "path": str(OUT_TEXT.relative_to(PROJECT_ROOT)),
            "bytes": OUT_TEXT.stat().st_size,
            "lines": len(accepted),
            "sha256": file_digest.hexdigest(),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"rows={stats['rows']:,} accepted={stats['accepted']:,} dup={stats['dup']:,} "
          f"eval_overlap={stats['eval_overlap']:,} train_overlap={stats['train_overlap']:,} "
          f"bad_length={stats['bad_length']:,}")
    print(f"heldout: {OUT_TEXT} ({OUT_TEXT.stat().st_size:,} bytes)")
    print("BUILD_ARZ_HELDOUT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
