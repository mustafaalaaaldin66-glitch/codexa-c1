from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

ROOT = Path.cwd()
CORPUS = ROOT / "data" / "tokenizer_train_v9_1" / "arabic"

FILES = sorted(CORPUS.glob("arabic-*.txt"))

if not FILES:
    raise FileNotFoundError(f"No corpus shards found in {CORPUS}")

OUTPUT_DIR = ROOT / "tokenizer" / "artifacts" / "bpe32k"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZED = ROOT / "data" / "tokenizer_train_v9_1" / "arabic_training.txt"
TOKENIZER_FILE = OUTPUT_DIR / "tokenizer.json"

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
]

with NORMALIZED.open("w", encoding="utf-8", newline="\n") as out:
    total_lines = 0

    for path in FILES:
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                line = line.rstrip("\n")
                if line:
                    out.write(line + "\n")
                    total_lines += 1

print(f"Corpus lines: {total_lines:,}")

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
    vocab_size=32_768,
    min_frequency=1,
    special_tokens=SPECIAL_TOKENS,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train([str(NORMALIZED)], trainer)
tokenizer.save(str(TOKENIZER_FILE))

print(f"Tokenizer saved: {TOKENIZER_FILE}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
