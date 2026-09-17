"""Evaluate a tokenizer against an independent held-out text file.

Unlike ``tokenizer/evaluate_v10.py`` (which compares two fixed artifacts against
the corpus tail), this tool takes explicit paths, so any held-out set can be
audited without touching training data or rebuilding anything.

It reports tokens/word, chars/token, unknown rate, round-trip failures, the
worst-case lines, and binds the result to the sha256 of both the held-out file
and the tokenizer artifact so a report cannot silently drift.

Run:
    .\\.venv\\Scripts\\python.exe tokenizer\\evaluate_heldout.py ^
        --heldout data\\heldout\\arz_test_heldout.txt ^
        --tokenizer tokenizer\\artifacts\\bpe32k_v10\\tokenizer.json ^
        --out data\\heldout\\arz_test_heldout.eval_v10.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

from text.normalize import normalize_text

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

DEFAULT_HELDOUT = ROOT / "data" / "heldout" / "arz_test_heldout.txt"
DEFAULT_TOKENIZER = ROOT / "tokenizer" / "artifacts" / "bpe32k_v10" / "tokenizer.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
def measure(tokenizer: Tokenizer, texts: list[str], worst_n: int = 5) -> dict:
    """Tokenize ``texts`` (after normalization) and return audited metrics."""
    unk_id = tokenizer.token_to_id("<unk>")
    words = tokens = chars = unknowns = passed = failed = 0
    scored: list[tuple[float, int, str]] = []

    for raw in texts:
        text = normalize_text(raw)
        if not text:
            continue
        ids = tokenizer.encode(text).ids
        if tokenizer.decode(ids) == text:
            passed += 1
        else:
            failed += 1
        words += len(text.split())
        tokens += len(ids)
        chars += len(text)
        if unk_id is not None:
            unknowns += sum(1 for token_id in ids if token_id == unk_id)
        scored.append((len(ids) / max(1, len(text.split())), len(ids), text[:80]))

    scored.sort(key=lambda item: item[0], reverse=True)
    return {
        "samples": passed + failed,
        "words": words,
        "chars": chars,
        "tokens": tokens,
        "tokens_per_word": tokens / words if words else 0.0,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "unk_rate": unknowns / tokens if tokens else 0.0,
        "roundtrip_passed": passed,
        "roundtrip_failed": failed,
        "worst_cases": [
            {"tokens_per_word": round(score, 4), "tokens": count, "head": head}
            for score, count, head in scored[:worst_n]
        ],
    }
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a tokenizer on a held-out file.")
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=None,
                        help="optional JSON report path")
    parser.add_argument("--max-lines", type=int, default=0, help="0 = all lines")
    args = parser.parse_args()

    heldout = args.heldout if args.heldout.is_absolute() else ROOT / args.heldout
    tokenizer_path = args.tokenizer if args.tokenizer.is_absolute() else ROOT / args.tokenizer

    if not heldout.exists():
        print(f"FAIL: held-out file not found: {heldout}")
        return 1
    if not tokenizer_path.exists():
        print(f"FAIL: tokenizer not found: {tokenizer_path}")
        return 1

    lines = read_lines(heldout)
    if not lines:
        print(f"FAIL: held-out file is empty: {heldout}")
        return 1
    if args.max_lines:
        lines = lines[: args.max_lines]

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    metrics = measure(tokenizer, lines)

    report = {
        "version": "heldout_eval_v1",
        "heldout": {
            "path": str(heldout.relative_to(ROOT)) if heldout.is_relative_to(ROOT) else str(heldout),
            "lines": len(lines),
            "sha256": sha256_file(heldout),
        },
        "tokenizer": {
            "path": str(tokenizer_path.relative_to(ROOT)) if tokenizer_path.is_relative_to(ROOT) else str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(),
        },
        "metrics": metrics,
        "status": "PASS" if metrics["roundtrip_failed"] == 0 else "FAIL",
    }

    print("=" * 66)
    print(" HELD-OUT TOKENIZER EVALUATION")
    print("=" * 66)
    print(f" held-out        : {report['heldout']['path']}")
    print(f" lines/sha       : {len(lines)} / {report['heldout']['sha256'][:12]}")
    print(f" tokenizer       : {report['tokenizer']['path']}")
    print(f" vocab/sha       : {tokenizer.get_vocab_size()} / {report['tokenizer']['sha256'][:12]}")
    print(f" tokens/word     : {metrics['tokens_per_word']:.4f}")
    print(f" chars/token     : {metrics['chars_per_token']:.4f}")
    print(f" unknown rate    : {metrics['unk_rate']:.6%}")
    print(f" round-trip      : {metrics['roundtrip_passed']} passed / {metrics['roundtrip_failed']} failed")
    if metrics["worst_cases"]:
        worst = metrics["worst_cases"][0]
        print(f" worst tpw       : {worst['tokens_per_word']} ({worst['tokens']} tokens)")

    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f" report written  : {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")

    print(f"RESULT: {report['status']}")
    return 0 if metrics["roundtrip_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())