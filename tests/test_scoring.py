"""Check the assignment denominator, sentence boundaries, ties, and overflow."""

import pytest

from asr_rescoring.scoring import rank_candidates


class FakeModel:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def score(self, text, bos, eos):
        self.calls.append((text, bos, eos))
        return self.values[text]


def test_perplexity_words_not_eos_and_length_normalization():
    model = FakeModel({"слово": -2, "два слова": -3})
    ranked = rank_candidates(model, ["слово", "Два слова!"])
    assert [candidate.index for candidate in ranked] == [1, 0]
    assert ranked[0].perplexity == pytest.approx(10**1.5)
    assert model.calls == [("слово", True, True), ("два слова", True, True)]


def test_ties_preserve_original_index():
    ranked = rank_candidates(FakeModel({"а": -2, "б": -2}), ["а", "б"])
    assert [candidate.index for candidate in ranked] == [0, 1]


def test_overflow_does_not_break_ranking():
    ranked = rank_candidates(FakeModel({"а": -400, "б": -401}), ["а", "б"])
    assert ranked[0].index == 0
    assert ranked[0].perplexity is None


@pytest.mark.parametrize("candidates", [[], ["..."]])
def test_empty_candidates_are_explicit_errors(candidates):
    with pytest.raises(ValueError):
        rank_candidates(FakeModel({}), candidates)


def test_nonfinite_scores_are_invalid():
    with pytest.raises(ValueError, match="Nonfinite"):
        rank_candidates(FakeModel({"а": float("nan")}), ["а"])
