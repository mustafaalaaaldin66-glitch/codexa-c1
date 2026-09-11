"""Validate the v10 tokenizer corpus artifacts against the manifest.

READ-ONLY validator. It never rebuilds the corpus, never re-reads parquet,
and never touches existing artifacts. It only re-reads the final text files
to recompute their digest, byte size and line count.

Checks
------
1. manifest JSON parses and has the required keys
2. data/tokenizer/corpus.txt exists and its byte size matches the manifest
3. per-language files (ar.txt, arz.txt) exist
4. sha256 / bytes / line-count of each file match the manifest
5. no stale lock or *.tmp builder files remain

Exit code 0 = all checks passed, 1 = at least one check failed.

Run:
    .\\.venv\\Scripts\\python.exe data_pipeline\\validate_v10_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "tokenizer_train_v10.json"
V10 = ROOT / "data" / "tokenizer_train_v10"
CORPUS = ROOT / "data" / "tokenizer" / "corpus.txt"

LANGUAGES = ("ar", "arz", "en")

STALE_FILES = (
    V10 / ".build_arz_v10.lock",
    ROOT / "data" / "tokenizer" / "corpus.txt.tmp",
    ROOT / "data" / "manifests" / "tokenizer_train_v10.json.tmp",
)

REQUIRED_MANIFEST_KEYS = ("version", "normalizer", "counts", "corpus")

_failures: list[str] = []


def check(condition: bool, ok: str, fail: str) -> bool:
    if condition:
        print(f"  [OK]   {ok}")
        return True
    print(f"  [FAIL] {fail}")
    _failures.append(fail)
    return False


def scan(path: Path) -> dict:
    """Single streaming pass: bytes + sha256 + newline count."""
    digest = hashlib.sha256()
    size = 0
    lines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
    return {"bytes": size, "lines": lines, "sha256": digest.hexdigest()}



def main() -> int:
    print("=" * 60)
    print(" CODEXA V10 CORPUS VALIDATION")
    print("=" * 60)

    # ---- 1. manifest ---------------------------------------------------
    print("\n[1] Manifest")
    if not check(MANIFEST.exists(), f"manifest present: {MANIFEST.name}",
                 f"manifest missing: {MANIFEST}"):
        print("\nRESULT: FAIL")
        return 1

    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        check(False, "", f"manifest is not valid JSON: {exc}")
        print("\nRESULT: FAIL")
        return 1

    check(True, "manifest is valid JSON", "")
    for key in REQUIRED_MANIFEST_KEYS:
        check(key in manifest, f"manifest key '{key}' present",
              f"manifest key '{key}' missing")

    # ---- 2 & 3. corpus + language files exist --------------------------
    print("\n[2] Files")
    if not check(CORPUS.exists(), f"corpus present: {CORPUS.name}",
                 f"corpus missing: {CORPUS}"):
        print("\nRESULT: FAIL")
        return 1

    manifest_corpus = manifest.get("corpus", {})
    check(
        CORPUS.stat().st_size == manifest_corpus.get("bytes"),
        f"corpus size matches manifest ({CORPUS.stat().st_size:,} bytes)",
        f"corpus size mismatch: disk={CORPUS.stat().st_size:,} "
        f"manifest={manifest_corpus.get('bytes')}",
    )

    for language in LANGUAGES:
        path = V10 / f"{language}.txt"
        expected = manifest.get("counts", {}).get(language, {})
        if expected.get("target", 0) == 0 and not path.exists():
            check(True, f"{language}: target 0, file absent (as designed)", "")
            continue
        check(path.exists(), f"{language}: file present: {path.name}",
              f"{language}: file missing: {path}")

    # ---- 4. hashes / bytes / lines -------------------------------------
    print("\n[3] Integrity (streaming sha256 + line count)")
    for language in LANGUAGES:
        expected = manifest.get("counts", {}).get(language, {})
        path = V10 / f"{language}.txt"
        if not path.exists():
            continue
        print(f"  scanning {path.name} ({path.stat().st_size:,} bytes)...",
              flush=True)
        info = scan(path)
        check(
            info["sha256"] == expected.get("sha256"),
            f"{language}: sha256 match",
            f"{language}: sha256 mismatch "
            f"disk={info['sha256'][:12]} manifest={str(expected.get('sha256'))[:12]}",
        )
        check(
            info["bytes"] == expected.get("bytes"),
            f"{language}: byte size match ({info['bytes']:,})",
            f"{language}: byte size mismatch "
            f"disk={info['bytes']:,} manifest={expected.get('bytes')}",
        )
        check(
            info["lines"] == expected.get("accepted"),
            f"{language}: line count match ({info['lines']:,})",
            f"{language}: line count mismatch "
            f"disk={info['lines']:,} manifest accepted={expected.get('accepted')}",
        )

    print(f"  scanning {CORPUS.name} ({CORPUS.stat().st_size:,} bytes)...",
          flush=True)
    corpus_info = scan(CORPUS)
    check(
        corpus_info["sha256"] == manifest_corpus.get("sha256"),
        "corpus: sha256 match",
        f"corpus: sha256 mismatch "
        f"disk={corpus_info['sha256'][:12]} manifest={str(manifest_corpus.get('sha256'))[:12]}",
    )

    # ---- 5. no stale builder files -------------------------------------
    print("\n[4] Builder leftovers")
    for stale in STALE_FILES:
        check(not stale.exists(), f"no leftover: {stale.name}",
              f"stale builder file still present: {stale}")

    print("\n" + "=" * 60)
    if _failures:
        print(f"RESULT: FAIL ({len(_failures)} check(s) failed)")
        for failure in _failures:
            print(f"  - {failure}")
        return 1

    print("RESULT: PASS - v10 corpus matches its manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
