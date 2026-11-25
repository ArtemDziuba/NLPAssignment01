"""Normalization must preserve recognition errors and Ukrainian distinctions."""

import pytest

from asr_rescoring.preprocessing import clean_training_sentence, normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  П’ЯТЬ,\tпʼять! ", "п'ять п'ять"),
        ("Їжак ҐАНОК Єнот Іван", "їжак ґанок єнот іван"),
        ("відео нокоуту 2024—2025", "відео нокоуту 2024 2025"),
        ("И\u0306ого", "його"),
        ("'слово' / інше_слово", "слово інше слово"),
        ("...", ""),
    ],
)
def test_normalization(raw, expected):
    assert normalize_text(raw) == expected
    assert normalize_text(normalize_text(raw)) == expected


def test_cleaning_only_applies_to_training():
    assert clean_training_sentence("https://example.com український текст") is None
    assert clean_training_sentence("English only") is None
    assert clean_training_sentence("слово") is None
    assert clean_training_sentence("дуже гарне місто", max_words=2) is None
    assert clean_training_sentence("Дуже гарне місто!") == "дуже гарне місто"
