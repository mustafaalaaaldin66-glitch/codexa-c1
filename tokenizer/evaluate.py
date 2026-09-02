from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

# ضبط مسار المشروع الرئيسي لتفادي أخطاء الاستيراد
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer

from text.normalize import normalize_text


TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer"
    / "artifacts"
    / "bpe32k"
    / "tokenizer.json"
)

EVAL_DIR = PROJECT_ROOT / "data" / "tokenizer_eval"

FILES = {
    "arabic": EVAL_DIR / "arabic.txt",
    "english": EVAL_DIR / "english.txt",
    "mixed": EVAL_DIR / "mixed.txt",
}


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file missing: {path}")

    lines = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.strip():
                lines.append(line)

    if not lines:
        raise ValueError(f"Evaluation file is empty: {path}")

    return lines


def word_count(text: str) -> int:
    return len(text.split())


def evaluate_corpus(
    tokenizer: Tokenizer,
    name: str,
    path: Path,
) -> dict[str, float]:
    raw_lines = read_lines(path)

    normalized_lines = [
        normalize_text(line)
        for line in raw_lines
    ]

    total_words = 0
    total_tokens = 0
    total_chars = 0
    total_unknown = 0

    start = perf_counter()

    for text in normalized_lines:
        if not text:
            continue

        encoding = tokenizer.encode(text)

        total_words += word_count(text)
        total_tokens += len(encoding.ids)
        total_chars += len(text)

        unk_id = tokenizer.token_to_id("<unk>")

        if unk_id is not None:
            total_unknown += sum(
                token_id == unk_id
                for token_id in encoding.ids
            )

    elapsed = perf_counter() - start

    tokens_per_second = (
        total_tokens / elapsed
        if elapsed > 0
        else 0.0
    )

    tokens_per_word = (
        total_tokens / total_words
        if total_words > 0
        else 0.0
    )

    chars_per_token = (
        total_chars / total_tokens
        if total_tokens > 0
        else 0.0
    )

    unknown_rate = (
        total_unknown / total_tokens
        if total_tokens > 0
        else 0.0
    )

    print()
    print(f"=== {name.upper()} ===")
    print(f"Lines:            {len(normalized_lines):,}")
    print(f"Words:            {total_words:,}")
    print(f"Characters:       {total_chars:,}")
    print(f"Tokens:           {total_tokens:,}")
    print(f"Tokens / word:    {tokens_per_word:.4f}")
    print(f"Chars / token:    {chars_per_token:.4f}")
    print(f"Unknown rate:     {unknown_rate:.6%}")
    print(f"Tokenizer tok/s:  {tokens_per_second:,.2f}")

    return {
        "lines": len(normalized_lines),
        "words": total_words,
        "characters": total_chars,
        "tokens": total_tokens,
        "tokens_per_word": tokens_per_word,
        "chars_per_token": chars_per_token,
        "unknown_rate": unknown_rate,
        "tokens_per_second": tokens_per_second,
    }


def round_trip_check(
    tokenizer: Tokenizer,
    name: str,
    path: Path,
) -> tuple[int, int]:
    raw_lines = read_lines(path)

    passed = 0
    failed = 0

    for raw in raw_lines:
        normalized = normalize_text(raw)

        encoded = tokenizer.encode(normalized)
        decoded = tokenizer.decode(encoded.ids)

        if decoded == normalized:
            passed += 1
        else:
            failed += 1

            print("\nROUND-TRIP FAILURE")
            print("Raw:       ", repr(raw))
            print("Normalized:", repr(normalized))
            print("Decoded:   ", repr(decoded))

    return passed, failed


def main() -> None:
    print("===================================")
    print("Codexa T0 Tokenizer Evaluation")
    print("===================================")

    tokenizer = load_tokenizer()

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    total_passed = 0
    total_failed = 0

    for name, path in FILES.items():
        evaluate_corpus(tokenizer, name, path)

        passed, failed = round_trip_check(
            tokenizer,
            name,
            path,
        )

        total_passed += passed
        total_failed += failed

    print()
    print("=== ROUND-TRIP SUMMARY ===")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")

    if total_failed != 0:
        raise RuntimeError(
            "Tokenizer round-trip failed."
        )

    print("Round-trip: 100% PASS")


if __name__ == "__main__":
    main()