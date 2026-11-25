"""Normalize Ukrainian consistently without correcting recognition errors."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

APOSTROPHES = re.compile("[’‘ʼ\u0060´]")
WORDS = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
UKRAINIAN = re.compile("[а-яіїєґ]", re.IGNORECASE)
VERSION = "nfc-lower-apostrophe-words-v1"


def normalize_text(text: str) -> str:
    """NFC, canonical apostrophes, lowercase, and whitespace-delimited word tokens."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    text = APOSTROPHES.sub("'", unicodedata.normalize("NFC", text)).lower()
    return " ".join(WORDS.findall(text))


def sentence_lines(lines: Iterator[str]) -> Iterator[str]:
    """Use one sentence per line from the official sentenced export.

    Do not re-segment evaluation candidates or split Ukrainian initials/abbreviations
    using an English punctuation heuristic.
    """
    for line in lines:
        sentence = line.strip()
        if sentence:
            yield sentence


def clean_training_sentence(text: str, min_words: int = 2, max_words: int = 150) -> str | None:
    """Reject empty/non-Cyrillic, overlong, and obvious markup/URL corpus lines."""
    lowered = text.lower()
    if any(marker in lowered for marker in ("http://", "https://", "<script", "<html")):
        return None
    normalized = normalize_text(text)
    length = len(normalized.split())
    if not min_words <= length <= max_words or not UKRAINIAN.search(normalized):
        return None
    return normalized
