from pathlib import Path

from tokenizers import Tokenizer

from text.normalize import normalize_text


TOKENIZER_PATH = (
    Path(__file__).resolve().parent.parent
    / "tokenizer"
    / "artifacts"
    / "bpe32k"
    / "tokenizer.json"
)


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_tokenizer_loads():
    tokenizer = load_tokenizer()

    assert tokenizer.get_vocab_size() > 0


def test_arabic_encoding():
    tokenizer = load_tokenizer()

    encoded = tokenizer.encode("أنا أحب البرمجة")

    assert len(encoded.ids) > 0


def test_english_encoding():
    tokenizer = load_tokenizer()

    encoded = tokenizer.encode("I love programming")

    assert len(encoded.ids) > 0


def test_mixed_encoding():
    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(
        "أنا أحب programming و PyTorch"
    )

    assert len(encoded.ids) > 0


def test_normalized_round_trip():
    tokenizer = load_tokenizer()

    raw = "مـحَمَّد   يحب   البرمجة"

    normalized = normalize_text(raw)

    encoded = tokenizer.encode(normalized)
    decoded = tokenizer.decode(encoded.ids)

    assert decoded == normalized