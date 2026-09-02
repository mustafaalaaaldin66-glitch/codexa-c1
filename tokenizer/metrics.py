from __future__ import annotations


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def tokens_per_word(tokens: int, words: int) -> float:
    return safe_divide(tokens, words)


def chars_per_token(chars: int, tokens: int) -> float:
    return safe_divide(chars, tokens)


def tokens_per_char(tokens: int, chars: int) -> float:
    return safe_divide(tokens, chars)


def compression_ratio(chars: int, tokens: int) -> float:
    return safe_divide(chars, tokens)