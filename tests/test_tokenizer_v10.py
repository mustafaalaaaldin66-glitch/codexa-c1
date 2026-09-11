"""Strong, auditable contract tests for the v10 tokenizer.

These lock the tokenizer's externally observable behaviour so a future session
cannot silently drift the artifact, the normalizer, or the metrics. They are
SKIPPED (not failed) when the large local artifacts are absent, so `pytest`
stays green on a clean clone.

Run:
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_tokenizer_v10.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from text.normalize import normalize_text

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "tokenizer" / "artifacts" / "bpe32k_v10" / "tokenizer.json"
META_PATH = ROOT / "tokenizer" / "artifacts" / "bpe32k_v10" / "meta.json"
LEGACY_PATH = ROOT / "tokenizer" / "artifacts" / "bpe32k" / "tokenizer.json"
EVAL_DIR = ROOT / "data" / "tokenizer_eval"
V10_DIR = ROOT / "data" / "tokenizer_train_v10"

EXPECTED_VOCAB = 32_768
SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>", "<mask>", "<ar>", "<en>",
    "<code>", "<math>", "<user>", "<assistant>", "<system>",
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Raw inputs whose canonical form MUST differ (normalization is a real step).
NORMALIZATION_CHANGES = {
    "diacritics": "مُحَمَّدٌ",
    "tatweel": "مـحـمـد",
    "double_space": "أنا   أحب",
    "nfkc_circled": "①",
    "nfkc_ligature": "ﬀ",
}

# Inputs that must tokenize cleanly (zero unknown, lossless round-trip).
CLEAN_CASES = {
    "msa": "أنا أحب البرمجة كثيرا",
    "egyptian": "إزاي الحال النهاردة يا صاحبي",
    "english": "I love programming",
    "mixed": "أنا أحب programming و PyTorch",
    "url": "https://example.com/path?q=1",
    "code": "def f(x):\n    return x + 1",
    "math": "x^2 + y^2 = z^2",
    "punct": "مرحبا!!! كيف حالك؟؟",
    "emoji": "أهلا 😀🎉 مرحبا",
    "arabic_digits": "١٢٣٤٥ ٦٧٨٩٠",
}

require_tokenizer = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(),
    reason="v10 tokenizer artifact not present (local-only)",
)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@pytest.fixture(scope="module")
def unk_id(tok: Tokenizer) -> int:
    value = tok.token_to_id("<unk>")
    assert value is not None
    return value


def tail_lines(path: Path, n_bytes: int, max_lines: int) -> list[str]:
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
    return lines[:max_lines]


def curated_texts() -> list[str]:
    if not EVAL_DIR.exists():
        return []
    texts: list[str] = []
    for path in sorted(EVAL_DIR.rglob("*.txt")):
        texts.extend(
            ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()
        )
    return texts


@lru_cache(maxsize=None)
def holdout_texts(language: str, n_bytes: int = 2 * 1024**2, max_lines: int = 3000) -> tuple[str, ...]:
    path = V10_DIR / f"{language}.txt"
    if not path.exists():
        return ()
    return tuple(tail_lines(path, n_bytes, max_lines))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Artifact contract
# --------------------------------------------------------------------------

@require_tokenizer
def test_vocab_size_is_exactly_32768(tok: Tokenizer) -> None:
    assert tok.get_vocab_size() == EXPECTED_VOCAB


@require_tokenizer
def test_all_special_tokens_registered(tok: Tokenizer) -> None:
    ids = [tok.token_to_id(token) for token in SPECIAL_TOKENS]
    assert all(i is not None for i in ids), "missing special token"
    assert len(set(ids)) == len(SPECIAL_TOKENS), "duplicate special-token ids"
    assert tok.token_to_id("<pad>") == 0
    assert tok.token_to_id("<unk>") == 1


@require_tokenizer
@pytest.mark.parametrize("special", SPECIAL_TOKENS)
def test_special_token_is_single_id(tok: Tokenizer, special: str) -> None:
    assert tok.encode(special).ids == [tok.token_to_id(special)]


@pytest.mark.skipif(not META_PATH.exists(), reason="v10 meta.json not present")
def test_metadata_declares_vocab_and_hash() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    assert meta["vocab_size"] == EXPECTED_VOCAB
    assert meta["normalizer"] == "text.normalize.normalize_text"
    assert set(meta["special_tokens"]) == set(SPECIAL_TOKENS)
    assert set(meta["added_tokens"]) == set(SPECIAL_TOKENS)
    assert SHA256_RE.match(meta["corpus"]["sha256"]), "corpus sha256 malformed"
    assert meta["corpus"]["bytes"] > 0


# --------------------------------------------------------------------------
# Normalization vs raw round-trip
# --------------------------------------------------------------------------

@require_tokenizer
@pytest.mark.parametrize("name", sorted(NORMALIZATION_CHANGES))
def test_normalization_actually_changes_input(tok: Tokenizer, name: str) -> None:
    """Raw text is byte-lossless, but normalization is a real transformation."""
    raw = NORMALIZATION_CHANGES[name]
    assert normalize_text(raw) != raw, f"{name!r} should be changed by normalize_text"


@require_tokenizer
@pytest.mark.parametrize("name", sorted(NORMALIZATION_CHANGES))
def test_byte_level_roundtrip_is_lossless_on_raw(tok: Tokenizer, name: str) -> None:
    """ByteLevel BPE preserves raw bytes, so decode(encode(raw)) == raw."""
    raw = NORMALIZATION_CHANGES[name]
    assert tok.decode(tok.encode(raw).ids) == raw


@require_tokenizer
@pytest.mark.parametrize("name", sorted(CLEAN_CASES))
def test_roundtrip_after_normalization(tok: Tokenizer, name: str) -> None:
    normalized = normalize_text(CLEAN_CASES[name])
    assert tok.decode(tok.encode(normalized).ids) == normalized


@require_tokenizer
@pytest.mark.parametrize("name", sorted(NORMALIZATION_CHANGES))
def test_zero_unknown_on_normalized_hard_cases(tok: Tokenizer, unk_id: int, name: str) -> None:
    normalized = normalize_text(NORMALIZATION_CHANGES[name])
    if not normalized:
        return
    assert unk_id not in tok.encode(normalized).ids


@require_tokenizer
def test_tatweel_and_diacritics_removed() -> None:
    assert normalize_text("مُحَمَّد") == "محمد"
    assert normalize_text("مـحـمـد") == "محمد"


@require_tokenizer
def test_nfkc_applied() -> None:
    assert normalize_text("①②③") == "123"
    assert normalize_text("ﬀ") == "ff"


# --------------------------------------------------------------------------
# Corpus-level metrics (unknown rate, compression)
# --------------------------------------------------------------------------

def _metrics(tok: Tokenizer, texts: list[str]) -> dict:
    words = tokens = chars = unknowns = 0
    unk = tok.token_to_id("<unk>")
    for raw in texts:
        normalized = normalize_text(raw)
        if not normalized:
            continue
        ids = tok.encode(normalized).ids
        tokens += len(ids)
        words += len(normalized.split())
        chars += len(normalized)
        unknowns += sum(1 for i in ids if i == unk)
    return {
        "words": words,
        "tokens": tokens,
        "chars": chars,
        "unknown": unknowns,
        "tokens_per_word": tokens / words if words else 0.0,
        "chars_per_token": chars / tokens if tokens else 0.0,
        "unk_rate": unknowns / tokens if tokens else 0.0,
    }


@lru_cache(maxsize=None)
def holdout_metrics(tokenizer_path: str, language: str) -> dict:
    return _metrics(Tokenizer.from_file(tokenizer_path), list(holdout_texts(language)))


@require_tokenizer
def test_zero_unknown_on_curated(tok: Tokenizer) -> None:
    texts = curated_texts()
    if not texts:
        pytest.skip("curated eval set not present")
    assert _metrics(tok, texts)["unk_rate"] == 0.0


@pytest.mark.parametrize("language", ["ar", "arz"])
@require_tokenizer
def test_zero_unknown_on_holdout(language: str) -> None:
    if not holdout_texts(language):
        pytest.skip(f"{language} holdout not present")
    assert holdout_metrics(str(TOKENIZER_PATH), language)["unk_rate"] == 0.0


@pytest.mark.parametrize("language,words_bound", [("ar", 1.6), ("arz", 1.6)])
@require_tokenizer
def test_tokens_per_word_arabic(language: str, words_bound: float) -> None:
    if not holdout_texts(language):
        pytest.skip(f"{language} holdout not present")
    metrics = holdout_metrics(str(TOKENIZER_PATH), language)
    assert metrics["tokens_per_word"] <= words_bound, metrics
    assert metrics["chars_per_token"] >= 2.0, metrics


@require_tokenizer
def test_tokens_per_word_english(tok: Tokenizer) -> None:
    english_dir = EVAL_DIR / "english"
    if not english_dir.exists():
        pytest.skip("english eval set not present")
    texts = [
        ln
        for path in sorted(english_dir.rglob("*.txt"))
        for ln in path.read_text(encoding="utf-8").split("\n")
        if ln.strip()
    ]
    metrics = _metrics(tok, texts)
    assert metrics["tokens_per_word"] <= 3.5, metrics
    assert metrics["chars_per_token"] >= 1.2, metrics

    if LEGACY_PATH.exists():
        legacy = _metrics(Tokenizer.from_file(str(LEGACY_PATH)), texts)
        assert metrics["tokens_per_word"] <= legacy["tokens_per_word"], (
            legacy,
            metrics,
        )


@pytest.mark.parametrize("language", ["ar", "arz"])
def test_no_regression_vs_legacy(language: str) -> None:
    if not (TOKENIZER_PATH.exists() and LEGACY_PATH.exists()):
        pytest.skip("tokenizer artifacts not present")
    if not holdout_texts(language):
        pytest.skip(f"{language} holdout not present")
    legacy = holdout_metrics(str(LEGACY_PATH), language)
    new = holdout_metrics(str(TOKENIZER_PATH), language)
    assert new["tokens_per_word"] <= legacy["tokens_per_word"] + 1e-9, (legacy, new)


@require_tokenizer
def test_worst_case_examples_are_clean(tok: Tokenizer) -> None:
    """Document the hardest holdout lines; they must still tokenize cleanly."""
    texts = holdout_texts("arz", n_bytes=1024**2, max_lines=800)
    if not texts:
        pytest.skip("arz holdout not present")
    unk = tok.token_to_id("<unk>")
    scored = []
    for raw in texts:
        normalized = normalize_text(raw)
        if not normalized:
            continue
        ids = tok.encode(normalized).ids
        scored.append((len(ids) / max(1, len(normalized.split())), normalized, ids))
    scored.sort(reverse=True)
    for _, normalized, ids in scored[:5]:
        assert tok.decode(ids) == normalized
        assert unk not in ids


# --------------------------------------------------------------------------
# Robustness / determinism
# --------------------------------------------------------------------------

@require_tokenizer
def test_encoding_is_deterministic(tok: Tokenizer) -> None:
    text = "اختبار الثبات في الترميز 123 test"
    assert tok.encode(text).ids == tok.encode(text).ids


@require_tokenizer
def test_empty_and_whitespace_inputs(tok: Tokenizer) -> None:
    assert tok.encode("").ids == []
    assert normalize_text("   ") == ""
    assert tok.encode(normalize_text("   ")).ids == []


@require_tokenizer
def test_long_line_and_repetition(tok: Tokenizer) -> None:
    long_line = "ا" * 5000
    ids = tok.encode(long_line).ids
    assert len(ids) > 0
    assert tok.decode(ids) == long_line


@require_tokenizer
@pytest.mark.parametrize(
    "adversarial",
    [
        "\x00\x01\x02 control",
        "مرحبا\u200b\u200bعالم",
        "word" * 500,
        "ا" + "\u064b" * 50 + "ب",
        "🤖" * 100,
        "abc\u0301\u0302\u0303",
    ],
)
def test_adversarial_inputs_are_safe(tok: Tokenizer, unk_id: int, adversarial: str) -> None:
    ids = tok.encode(adversarial).ids
    assert unk_id not in ids
    assert tok.decode(ids) == adversarial



