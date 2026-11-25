"""Use hand-computable edit counts, ranking positions, and vocabulary counts."""

import pytest

from asr_rescoring.metrics import aggregate_metrics, edit_distance, oov_rate


def test_corpus_edit_rates_and_rank_metrics():
    metrics = aggregate_metrics(["а б", "в г д е"], [["а", "а б"], ["в г д е", "інше"]])
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["wer"] == pytest.approx(1 / 6)  # corpus, not mean per-record WER
    assert metrics["cer"] == pytest.approx(2 / 10)  # deleted space and letter
    assert metrics["mrr"] == 0.75
    assert metrics["oracle_accuracy"] == 1.0


def test_missing_reference_and_duplicate_matches():
    result = aggregate_metrics(["а", "б"], [["в", "а", "а"], ["в"]])
    assert result["mrr"] == 0.25
    assert result["oracle_accuracy"] == 0.5


def test_wer_can_exceed_one():
    assert aggregate_metrics(["а"], [["б в г"]])["wer"] == 3


def test_edit_distance_and_oov_occurrences():
    assert edit_distance("кіт", "кит") == 1
    assert edit_distance([], ["а"]) == 1
    assert oov_rate({"а"}, ["а б б"]) == pytest.approx(2 / 3)


def test_empty_metrics_are_rejected():
    with pytest.raises(ValueError):
        aggregate_metrics([], [])
