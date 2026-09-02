import sys
from pathlib import Path

# ضبط مسار المشروع الرئيسي لتتمكن بايثون من استيراد مجلد text
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from text.normalize import normalize_text


RAW_CORPUS = PROJECT_ROOT / "data" / "tokenizer" / "corpus.txt"
NORMALIZED_CORPUS = (
    PROJECT_ROOT / "data" / "tokenizer" / "corpus.normalized.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "tokenizer" / "artifacts" / "bpe32k"
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


def normalize_corpus() -> None:
    if not RAW_CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus not found: {RAW_CORPUS}"
        )

    NORMALIZED_CORPUS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RAW_CORPUS.open(
        "r",
        encoding="utf-8",
    ) as source, NORMALIZED_CORPUS.open(
        "w",
        encoding="utf-8",
    ) as target:

        for line in source:
            normalized = normalize_text(line)

            if normalized:
                target.write(normalized + "\n")


def build_tokenizer() -> Tokenizer:
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
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(
        [str(NORMALIZED_CORPUS)],
        trainer,
    )

    return tokenizer


def main() -> None:
    print("=== Codexa T0 BPE-32K ===")

    normalize_corpus()

    tokenizer = build_tokenizer()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(OUTPUT_FILE))

    print(f"Saved tokenizer to: {OUTPUT_FILE}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()