"""Train the production v10 BPE-32K tokenizer.

Reads an already-normalized corpus (default: data/tokenizer/corpus.txt, which
was produced by the v10 builder using text.normalize.normalize_text) and
writes a NEW artifact to tokenizer/artifacts/bpe32k_v10/. The legacy bpe32k
artifact is never touched.

A sidecar meta.json records the vocabulary size, corpus digest, normalizer
and special tokens so that results are reproducible and auditable.

This mirrors the corpus builder's normalization exactly: the corpus is
already normalized, so the trainer must NOT re-normalize (a second NFKC pass
on a 4 GB file would be wasted work). Instead a sampled idempotency check
proves the corpus lines already satisfy normalize_text(line) == line.

Run:
    .\\.venv\\Scripts\\python.exe tokenizer\\train_bpe_v10.py
    .\\.venv\\Scripts\\python.exe tokenizer\\train_bpe_v10.py --corpus runs\\v10_sample.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from text.normalize import normalize_text

CORPUS = PROJECT_ROOT / "data" / "tokenizer" / "corpus.txt"
ARTIFACTS = PROJECT_ROOT / "tokenizer" / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACTS / "bpe32k_v10"

VOCAB_SIZE = 32_768

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<mask>",
    "<ar>",
    "<en>",
    "<code>",
    "<math>",
    "<user>",
    "<assistant>",
    "<system>",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_normalized_sample(path: Path, limit: int) -> dict:
    """Prove the (already-normalized) corpus is idempotent under normalize_text."""
    checked = 0
    mismatches = 0
    example = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if not text:
                continue
            checked += 1
            if normalize_text(text) != text:
                mismatches += 1
                if example is None:
                    example = text[:80]
            if checked >= limit:
                break
    return {"checked": checked, "mismatches": mismatches, "example": example}


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the v10 BPE-32K tokenizer.")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    parser.add_argument("--norm-check", type=int, default=200)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing artifact")
    args = parser.parse_args()

    corpus = args.corpus if args.corpus.is_absolute() else PROJECT_ROOT / args.corpus
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    output_file = out_dir / "tokenizer.json"
    meta_file = out_dir / "meta.json"

    if not corpus.exists():
        print(f"FAIL: corpus not found: {corpus}")
        return 1
    if corpus.stat().st_size < 1_000_000:
        print(f"FAIL: corpus too small ({corpus.stat().st_size} bytes); "
              "the corpus build did not complete.")
        return 1
    if not args.force and (output_file.exists() or meta_file.exists()):
        print(f"FAIL: artifact already exists (use --force to overwrite): {output_file}")
        return 1

    # Guard: never write over the legacy bpe32k artifact.
    legacy = (ARTIFACTS / "bpe32k" / "tokenizer.json").resolve()
    if output_file.resolve() == legacy:
        print(f"FAIL: refusing to overwrite the legacy artifact: {legacy}")
        return 1

    print(f"Normalization idempotency check (first {args.norm_check} lines)...")
    norm = check_normalized_sample(corpus, args.norm_check)
    print(f"  checked={norm['checked']:,} mismatches={norm['mismatches']:,}")
    if norm["example"] is not None:
        print(f"  example mismatch: {norm['example']!r}")

    print(f"Corpus : {corpus}")
    print(f"Bytes  : {corpus.stat().st_size:,}")
    if norm["mismatches"]:
        print("FAIL: corpus is not idempotently normalized; refusing to train.")
        return 1
    corpus_sha = file_sha256(corpus)
    print(f"sha256 : {corpus_sha}")

    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training BPE-{args.vocab:,}...")
    tokenizer.train([str(corpus)], trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_file))

    specials = [t.content for t in tokenizer.get_added_tokens_decoder().values()]
    meta = {
        "version": "tokenizer_v10",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "normalizer": "text.normalize.normalize_text",
        "vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": 2,
        "corpus": {
            "path": str(corpus.relative_to(PROJECT_ROOT)) if corpus.is_relative_to(PROJECT_ROOT) else str(corpus),
            "bytes": corpus.stat().st_size,
            "sha256": corpus_sha,
        },
        "special_tokens": SPECIAL_TOKENS,
        "added_tokens": specials,
        "normalization_check": norm,
    }
    meta_file.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Tokenizer saved: {output_file}")
    print(f"Metadata saved : {meta_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Added tokens   : {specials}")
    print("TRAIN_BPE_V10_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())