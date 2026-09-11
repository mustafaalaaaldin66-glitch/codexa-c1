"""Evaluate the v10 tokenizer against the legacy bpe32k tokenizer.

Datasets
--------
* curated regression files under data/tokenizer_eval (arabic/english/mixed)
* a real held-out slice read from the TAIL of ar.txt and arz.txt. The sample
  tokenizer is trained on the HEAD of those files, so the tail is disjoint.

Metrics per (tokenizer, dataset): words, chars, tokens, tokens/word,
chars/token, unknown rate, round-trip pass/fail and throughput.

The report is written to data/tokenizer_eval/report_v10.json. Exit code is 1
when the v10 tokenizer fails any round-trip on the holdout.

Run:
    .\\.venv\\Scripts\\python.exe tokenizer\\evaluate_v10.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer

from text.normalize import normalize_text

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

OLD_TOKENIZER = PROJECT_ROOT / "tokenizer" / "artifacts" / "bpe32k" / "tokenizer.json"
NEW_TOKENIZER = PROJECT_ROOT / "tokenizer" / "artifacts" / "bpe32k_v10" / "tokenizer.json"
EVAL_DIR = PROJECT_ROOT / "data" / "tokenizer_eval"
V10_DIR = PROJECT_ROOT / "data" / "tokenizer_train_v10"
REPORT = EVAL_DIR / "report_v10.json"


def tail_lines(path: Path, n_bytes: int, max_lines: int) -> list[str]:
    """Read whole lines from the tail of a (possibly huge) UTF-8 file."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        start = max(0, size - n_bytes)
        handle.seek(start)
        data = handle.read()
    if start > 0:
        newline = data.find(b"\n")
        if newline != -1:
            data = data[newline + 1:]
    lines = [ln for ln in data.decode("utf-8", errors="ignore").split("\n") if ln.strip()]
    return lines[:max_lines] if max_lines else lines


def eval_file_lines(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def measure(tokenizer: Tokenizer, texts: list[str]) -> dict:
    words = tokens = chars = unknowns = passed = failed = 0
    unk_id = tokenizer.token_to_id("<unk>")
    start = perf_counter()
    for raw in texts:
        text = normalize_text(raw)
        if not text:
            continue
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        if decoded == text:
            passed += 1
        else:
            failed += 1
        words += len(text.split())
        tokens += len(enc.ids)
        chars += len(text)
        if unk_id is not None:
            unknowns += sum(1 for i in enc.ids if i == unk_id)
    elapsed = perf_counter() - start
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
        "throughput_tok_s": tokens / elapsed if elapsed else 0.0,
    }


def load(path: Path) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(path)
    return Tokenizer.from_file(str(path))



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-mb", type=int, default=2)
    parser.add_argument("--max-lines", type=int, default=4000)
    args = parser.parse_args()

    datasets: dict[str, list[str]] = {}
    for path in sorted(EVAL_DIR.rglob("*.txt")):
        name = "curated/" + path.relative_to(EVAL_DIR).with_suffix("").as_posix()
        datasets[name] = eval_file_lines(path)

    ar_tail = V10_DIR / "ar.txt"
    arz_tail = V10_DIR / "arz.txt"
    if ar_tail.exists() and arz_tail.exists():
        datasets["holdout/ar_msa"] = tail_lines(ar_tail, args.holdout_mb * 1024**2, args.max_lines)
        datasets["holdout/arz_egy"] = tail_lines(arz_tail, args.holdout_mb * 1024**2, args.max_lines)

    old = load(OLD_TOKENIZER)
    new = load(NEW_TOKENIZER)

    print("=" * 74)
    print(" TOKENIZER EVALUATION: legacy bpe32k  vs  bpe32k_v10")
    print("=" * 74)
    print(f" legacy vocab: {old.get_vocab_size():,}   v10 vocab: {new.get_vocab_size():,}")

    report = {
        "legacy_tokenizer": str(OLD_TOKENIZER.relative_to(PROJECT_ROOT)),
        "v10_tokenizer": str(NEW_TOKENIZER.relative_to(PROJECT_ROOT)),
        "legacy_vocab": old.get_vocab_size(),
        "v10_vocab": new.get_vocab_size(),
        "datasets": {},
    }

    total_failed = 0
    header = f"{'dataset':<26}{'tok/word old->new':>24}{'unk%(v10)':>12}{'rt_fail':>10}"
    print(header)
    print("-" * len(header))

    for name, texts in datasets.items():
        if not texts:
            continue
        old_m = measure(old, texts)
        new_m = measure(new, texts)
        total_failed += new_m["roundtrip_failed"]
        report["datasets"][name] = {"legacy": old_m, "v10": new_m}
        print(
            f"{name:<26}"
            f"{old_m['tokens_per_word']:>12.4f} ->{new_m['tokens_per_word']:>8.4f}"
            f"{new_m['unk_rate'] * 100:>11.4f}%"
            f"{new_m['roundtrip_failed']:>10}"
        )

    report["total_roundtrip_failed_v10"] = total_failed
    report["status"] = "PASS" if total_failed == 0 else "FAIL"

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("-" * len(header))
    print(f" total v10 round-trip failures: {total_failed}")
    print(f" report written: {REPORT.relative_to(PROJECT_ROOT)}")
    print("RESULT:", report["status"])
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
