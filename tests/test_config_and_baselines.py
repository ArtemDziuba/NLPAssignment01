"""Test portable paths, rejected convention changes, and baseline reproducibility."""

import pytest
import yaml

from asr_rescoring.config import PROJECT_ROOT, load_config, resolve_path, snapshot
from asr_rescoring.dataset import Record
from asr_rescoring.evaluation import baseline_rankings


@pytest.mark.parametrize(
    "filename",
    ["experiments.yaml", "news_only.yaml", "wiki_only.yaml", "news_wiki.yaml", "mamaylm.yaml"],
)
def test_paths_independent_of_working_directory(tmp_path, monkeypatch, filename):
    config = load_config(PROJECT_ROOT / "configs" / filename)
    expected = resolve_path(config, "evaluation")
    monkeypatch.chdir(tmp_path)
    assert resolve_path(config, "evaluation") == expected
    assert expected == PROJECT_ROOT / "data/evaluation/ua_asr_hypotheses_500.jsonl"
    assert expected.is_file()
    assert "_root" not in snapshot(config)


def test_reject_silent_perplexity_change(tmp_path):
    config = snapshot(load_config())
    config["scoring"]["perplexity_denominator"] = "words_plus_eos"
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="assignment"):
        load_config(path)


def test_random_baseline_is_per_id_and_seeded():
    records = [Record(str(i), "еталон", ("а", "б", "в", "г")) for i in range(20)]
    ranks = baseline_rankings(records, 42, True)
    assert ranks == list(reversed(baseline_rankings(list(reversed(records)), 42, True)))
    assert ranks != baseline_rankings(records, 43, True)
    assert all(sorted(row) == [0, 1, 2, 3] for row in ranks)
    assert baseline_rankings(records, 42, False) == [[0, 1, 2, 3]] * 20
