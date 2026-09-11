"""Finalize v10: combine ar.txt + arz.txt into corpus.txt atomically, then write the manifest.

Safe resume path: does NOT re-read parquet, does NOT touch ar.txt / arz.txt.
Run with: .venv\\Scripts\\python.exe data_pipeline\\finalize_v10.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "data" / "tokenizer_train_v10"
COMBINED = ROOT / "data" / "tokenizer" / "corpus.txt"
TMP = COMBINED.with_name("corpus.txt.tmp")
MANIFEST = ROOT / "data" / "manifests" / "tokenizer_train_v10.json"
SOURCES = {
    "ar": ROOT / "data" / "raw" / "fineweb2_arb_arab_sample",
    "arz": ROOT / "data" / "raw" / "fineweb2_arz_arab",
}


def copy_with_stats(source: Path, target, combined_digest: "hashlib._Hash") -> dict:
    digest = hashlib.sha256()
    line_count = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            target.write(chunk)
            digest.update(chunk)
            combined_digest.update(chunk)
            line_count += chunk.count(b"\n")
    return {
        "accepted": line_count,
        "bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    ar = V10 / "ar.txt"
    arz = V10 / "arz.txt"
    for path in (ar, arz):
        if not path.exists():
            print(f"FAIL: missing {path}")
            return 1

    print("Combining into tmp...", flush=True)
    TMP.parent.mkdir(parents=True, exist_ok=True)
    combined_digest = hashlib.sha256()
    with TMP.open("wb") as target:
        source_stats = {}
        for language, src in (("ar", ar), ("arz", arz)):
            source_stats[language] = copy_with_stats(src, target, combined_digest)
            target.flush()
            os.fsync(target.fileno())

    combined_bytes = TMP.stat().st_size
    expected = ar.stat().st_size + arz.stat().st_size
    if combined_bytes != expected:
        print(f"FAIL: size mismatch tmp={combined_bytes} expected={expected}")
        return 1

    combined_sha = combined_digest.hexdigest()
    print(f"tmp bytes={combined_bytes} sha256={combined_sha}", flush=True)

    if COMBINED.exists() and COMBINED.stat().st_size == combined_bytes:
        TMP.unlink()
        print(f"Kept existing {COMBINED.name}; size already matches.", flush=True)
    else:
        os.replace(TMP, COMBINED)
        print(f"Replaced {COMBINED.name} atomically.", flush=True)

    manifest = {
        "version": "tokenizer_train_v10",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalizer": "text.normalize.normalize_text",
        "dedup": "exact SHA-256 of normalized text; tokenizer_eval excluded",
        "length_chars": [40, 8000],
        "counts": {
            "ar": {"target": 2_000_000, "files": [str(SOURCES["ar"].relative_to(ROOT))], **source_stats["ar"]},
            "arz": {"target": 400_000, "files": [str(SOURCES["arz"].relative_to(ROOT))], **source_stats["arz"]},
            "en": {"target": 0, "accepted": 0, "bytes": 0, "sha256": ""},
        },
        "corpus": {
            "path": str(COMBINED.relative_to(ROOT)),
            "bytes": combined_bytes,
            "sha256": combined_sha,
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = MANIFEST.with_name("tokenizer_train_v10.json.tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(manifest_tmp, MANIFEST)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("FINALIZE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())