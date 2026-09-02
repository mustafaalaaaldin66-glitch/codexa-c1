from text.normalize import normalize_text


def test_nfkc_compatibility():
    text = "①"

    assert normalize_text(
        text,
        remove_diacritics=False,
    ) == "1"


def test_tatweel_removed():
    text = "مـحـمـد"

    assert normalize_text(text) == "محمد"


def test_diacritics_removed_in_baseline():
    text = "مُحَمَّد"

    assert normalize_text(text) == "محمد"


def test_diacritics_can_be_preserved():
    text = "مُحَمَّد"

    normalized = normalize_text(
        text,
        remove_diacritics=False,
    )

    assert normalized == text


def test_arabic_letter_identity_is_preserved():
    text = "أ إ آ ؤ ئ ء ة ى"

    assert normalize_text(text) == text


def test_no_transliteration():
    text = "محمد"

    assert normalize_text(text) == "محمد"


def test_whitespace_normalization():
    text = "أنا    أحب\tالبرمجة"

    assert normalize_text(text) == "أنا أحب البرمجة"


def test_paragraph_breaks_preserved():
    text = "سطر أول\n\n\nسطر ثان"

    assert normalize_text(text) == "سطر أول\n\nسطر ثان"


def test_normalization_is_deterministic():
    text = "أحمد  يُحبّ البرمجةـ"

    first = normalize_text(text)
    second = normalize_text(text)

    assert first == second