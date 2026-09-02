from __future__ import annotations

import re
import unicodedata


# Arabic combining marks / harakat commonly encountered in Arabic text.
ARABIC_DIACRITIC_RANGES = (
    (0x064B, 0x0652),  # Fathatan ... Sukun
    (0x0653, 0x0655),  # Maddah, Hamza below/above
    (0x0670, 0x0670),  # Superscript Alef
)


def _is_arabic_diacritic(char: str) -> bool:
    codepoint = ord(char)

    return any(
        start <= codepoint <= end
        for start, end in ARABIC_DIACRITIC_RANGES
    )


def remove_arabic_diacritics(text: str) -> str:
    """
    Remove Arabic combining marks used as diacritics.

    This is a baseline transformation for experiments.
    It is NOT the only normalization variant we will keep forever.
    """

    return "".join(
        char
        for char in text
        if not _is_arabic_diacritic(char)
    )


def normalize_whitespace(text: str) -> str:
    """
    Normalize horizontal whitespace while preserving paragraph breaks.

    Newline characters are preserved.
    Runs of spaces/tabs inside a line become one space.
    """

    # Normalize CRLF / CR to LF first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse spaces and tabs, but do not cross newline boundaries.
    text = re.sub(r"[ \t\f\v]+", " ", text)

    # Remove spaces immediately before/after a newline.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines to at most one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text(
    text: str,
    *,
    remove_diacritics: bool = True,
) -> str:
    """
    Canonical Codexa normalization.

    Steps:
    1. Unicode NFKC
    2. Remove Tatweel
    3. Optionally remove Arabic diacritics
    4. Normalize whitespace
    """

    if not isinstance(text, str):
        raise TypeError(
            f"text must be str, got {type(text).__name__}"
        )

    # 1. Unicode compatibility normalization.
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove Tatweel / Kashida.
    text = text.replace("\u0640", "")

    # 3. Baseline: remove Arabic diacritics.
    if remove_diacritics:
        text = remove_arabic_diacritics(text)

    # 4. Whitespace normalization.
    text = normalize_whitespace(text)

    return text