from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer

from text.normalize import normalize_text
from tokenizer.metrics import chars_per_token, tokens_per_word

TOKENIZER_PATH = (
    PROJECT_ROOT / "tokenizer" / "artifacts" / "bpe32k" / "tokenizer.json"
)
EVAL_DIR = PROJECT_ROOT / "data" / "tokenizer_eval"
MANIFEST_PATH = EVAL_DIR / "manifest.json"


def get_file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()[:12]


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def read_file_lines(path: Path) -> list[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line)
    return lines


def evaluate_category(tokenizer: Tokenizer, category_dir: Path) -> dict:
    total_words = 0
    total_tokens = 0
    total_chars = 0
    total_unknown = 0
    passed_rt = 0
    failed_rt = 0

    txt_files = list(category_dir.glob("*.txt"))
    start = perf_counter()

    for file_path in txt_files:
        lines = read_file_lines(file_path)
        for raw in lines:
            normalized = normalize_text(raw)
            if not normalized:
                continue

            encoding = tokenizer.encode(normalized)
            decoded = tokenizer.decode(encoding.ids)

            if decoded == normalized:
                passed_rt += 1
            else:
                failed_rt += 1

            total_words += len(normalized.split())
            total_tokens += len(encoding.ids)
            total_chars += len(normalized)

            unk_id = tokenizer.token_to_id("<unk>")
            if unk_id is not None:
                total_unknown += sum(
                    1 for tid in encoding.ids if tid == unk_id
                )

    elapsed = perf_counter() - start

    tpw = tokens_per_word(total_tokens, total_words)
    cpt = chars_per_token(total_chars, total_tokens)
    unk_rate = total_unknown / total_tokens if total_tokens > 0 else 0.0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "words": total_words,
        "chars": total_chars,
        "tokens": total_tokens,
        "tokens_per_word": tpw,
        "chars_per_token": cpt,
        "unk_rate": unk_rate,
        "throughput_tok_s": tok_per_sec,
        "passed_rt": passed_rt,
        "failed_rt": failed_rt,
    }


def main() -> None:
    print("==================================================")
    print(" Codexa Tokenizer Evaluation Harness (v1) ")
    print("==================================================")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest missing: {MANIFEST_PATH}")

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    tokenizer = load_tokenizer()
    tok_hash = get_file_hash(TOKENIZER_PATH)

    print(f"Eval Version:    {manifest.get('version')}")
    print(f"Normalizer:      {manifest.get('normalization')}")
    print(f"Tokenizer Hash:  {tok_hash}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print("-" * 50)

    total_passed = 0
    total_failed = 0

    for category in manifest.get("categories", []):
        cat_dir = EVAL_DIR / category
        if not cat_dir.exists():
            continue

        metrics = evaluate_category(tokenizer, cat_dir)
        total_passed += metrics["passed_rt"]
        total_failed += metrics["failed_rt"]

        print(f"\n=== CATEGORY: {category.upper()} ===")
        print(f"Tokens / Word:   {metrics['tokens_per_word']:.4f}")
        print(f"Chars / Token:   {metrics['chars_per_token']:.4f}")
        print(f"Unknown Rate:    {metrics['unk_rate']:.6%}")
        print(f"Throughput:      {metrics['throughput_tok_s']:,.2f} tok/s")
        print(
            f"Round-Trip:      {metrics['passed_rt']} Passed / {metrics['failed_rt']} Failed"
        )

    print("\n" + "=" * 50)
    print("=== FINAL EVALUATION SUMMARY ===")
    print(f"Total Passed Round-Trips: {total_passed}")
    print(f"Total Failed Round-Trips: {total_failed}")

    if total_failed > 0:
        raise RuntimeError("Validation Error: Round-trip failed for some cases.")

    print("STATUS: 100% PASS - Evaluation Harness Ready!")


if __name__ == "__main__":
    main()