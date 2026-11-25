"""Compute exact match, corpus edit rates, rank metrics, and token OOV coverage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def edit_distance(reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    """Levenshtein distance with linear auxiliary memory."""
    previous = list(range(len(hypothesis) + 1))
    for row, ref in enumerate(reference, 1):
        current = [row]
        for col, hyp in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (ref != hyp))
            )
        previous = current
    return previous[-1]


def aggregate_metrics(
    references: list[str], rankings: list[list[str]], cer_include_spaces: bool = True
) -> dict[str, float]:
    """Aggregate over records; edit rates use total reference words/characters."""
    if not references or len(references) != len(rankings) or any(not r for r in rankings):
        raise ValueError("Need equally sized nonempty references and rankings")
    if any(not reference.split() for reference in references):
        raise ValueError("Empty reference is invalid for corpus edit rates")
    exact = oracle = reciprocal = word_errors = char_errors = words = chars = 0
    for reference, ranking in zip(references, rankings, strict=True):
        selected = ranking[0]
        exact += selected == reference
        matches = [i for i, candidate in enumerate(ranking, 1) if candidate == reference]
        oracle += bool(matches)
        reciprocal += 1 / matches[0] if matches else 0
        word_errors += edit_distance(reference.split(), selected.split())
        words += len(reference.split())
        char_reference = reference if cer_include_spaces else reference.replace(" ", "")
        char_selected = selected if cer_include_spaces else selected.replace(" ", "")
        char_errors += edit_distance(char_reference, char_selected)
        chars += len(char_reference)
    size = len(references)
    return {
        "exact_match_accuracy": exact / size,
        "wer": word_errors / words,
        "cer": char_errors / chars,
        "mrr": reciprocal / size,
        "oracle_accuracy": oracle / size,
    }


def oov_rate(model: Any, texts: list[str]) -> float:
    """Count unknown word occurrences, excluding implicit boundary tokens."""
    unknown = total = 0
    for text in texts:
        for token in text.split():
            total += 1
            unknown += token not in model
    return unknown / total if total else 0.0
