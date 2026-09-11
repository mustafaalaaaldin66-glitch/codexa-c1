"""Train the production v10 BPE-32K tokenizer on the combined v10 corpus.

Reads data/tokenizer/corpus.txt (ar + arz + en=0), which is already
normalized and deduplicated by the corpus builder. Saves to
tokenizer/artifacts/bpe32k_v10/tokenizer.json (old artifact untouched).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

CORPUS = PROJECT_ROOT / "data" / "tokenizer" / "corpus.txt"
OUTPUT_DIR = PROJECT_ROOT / "tokenizer" / "artifacts" / "bpe32k_v10"
OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"

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


def main() -> None:
    if not CORPUS.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS}")
    if CORPUS.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Corpus too small ({CORPUS.stat().st_size} bytes); "
            "the v10 corpus build did not complete."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>",
            byte_fallback=False,
        )
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training BPE-{VOCAB_SIZE:,} on: {CORPUS}")
    tokenizer.train([str(CORPUS)], trainer)
    tokenizer.save(str(OUTPUT_FILE))

    print(f"Tokenizer saved: {OUTPUT_FILE}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()