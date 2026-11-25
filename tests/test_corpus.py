"""Test deterministic sampling, normalized leakage removal, and nested prefixes."""

import bz2

import pytest

from asr_rescoring.config import file_sha256
from asr_rescoring.corpus import create_subsets, sample_domain


def test_sampling_deduplicates_and_excludes_evaluation(tmp_path):
    archive = tmp_path / "fixture.bz2"
    with bz2.open(archive, "wt", encoding="utf-8") as stream:
        stream.write(
            "Гарне місто!\nГарне місто!\nВідомий еталон.\nінше гарне місто\nтретє мале місто\n"
        )
    settings = {"sentences_per_domain": 2, "min_sentence_words": 2, "max_sentence_words": 150}
    output = tmp_path / "sample.txt"
    sample_domain(archive, output, {"відомий еталон"}, settings, 42)
    first_hash = file_sha256(output)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(set(lines)) == 2
    assert "відомий еталон" not in lines
    # A second destination exercises reproducibility rather than cache reuse.
    again = tmp_path / "again.txt"
    sample_domain(archive, again, {"відомий еталон"}, settings, 42)
    assert file_sha256(again) == first_hash
    # Reuse must reject a corrupt cache.
    output.write_text("пошкоджений кеш\n", encoding="utf-8")
    sample_domain(archive, output, {"відомий еталон"}, settings, 42)
    assert file_sha256(output) == first_hash


def test_whole_sentence_subsets_are_strict_prefixes(tmp_path):
    sentences = [("а б", "news"), ("в г д", "court"), ("е ж", "fiction")]
    result = create_subsets(sentences, [3, 6], tmp_path)
    small = (tmp_path / "tokens_3.txt").read_bytes()
    large = (tmp_path / "tokens_6.txt").read_bytes()
    assert large.startswith(small)
    assert [row["actual_tokens"] for row in result] == [2, 5]
    assert result[1]["domain_tokens"] == {"news": 2, "court": 3}
    with pytest.raises(ValueError, match="Only"):
        create_subsets(sentences, [100], tmp_path)


def test_shared_sample_cache_reuses_across_corpus_mixtures(tmp_path, monkeypatch):
    import copy
    import json

    from asr_rescoring.config import load_config
    from asr_rescoring.corpus import prepare_corpus

    config = load_config()
    config["_root"] = tmp_path
    config["evaluation"]["expected_records"] = 1
    config["paths"]["evaluation"] = "evaluation.jsonl"
    config["paths"]["processed_corpus"] = "data/processed/mixed/corpus.txt"
    config["paths"]["subsets"] = "data/subsets/mixed"
    config["experiments"]["corpus_word_tokens"] = [5, 10]
    config["preprocessing"]["sentences_per_domain"] = 5
    config["corpus"]["sample_cache"] = "data/processed/shared"
    config["corpus"]["sources"] = [
        {"domain": domain, "filename": f"{domain}.bz2"} for domain in ("news", "wikipedia")
    ]
    (tmp_path / "evaluation.jsonl").write_text(
        json.dumps(
            {
                "utt_id": "fixture",
                "reference": "контрольний еталон",
                "hypotheses": ["контрольний еталон"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    raw = tmp_path / config["paths"]["raw_corpus"]
    raw.mkdir(parents=True)
    for source in config["corpus"]["sources"]:
        with bz2.open(raw / source["filename"], "wt", encoding="utf-8") as stream:
            for number in range(5):
                stream.write(f"{source['domain']} навчальне місто номер {number}\n")
    prepare_corpus(config, workers=1)
    sample = tmp_path / "data/processed/shared/news.sample.txt"
    before = sample.stat().st_mtime_ns

    single = copy.deepcopy(config)
    single["corpus"]["sources"] = single["corpus"]["sources"][:1]
    single["paths"]["processed_corpus"] = "data/processed/news/corpus.txt"
    single["paths"]["subsets"] = "data/subsets/news"

    def no_rescan(*args, **kwargs):
        raise AssertionError("A verified shared domain cache must not be decompressed again")

    monkeypatch.setattr(bz2, "open", no_rescan)
    prepared = prepare_corpus(single, workers=1)
    assert sample.stat().st_mtime_ns == before
    assert set(prepared["domain_manifests"]) == {"news"}
    assert prepared["subsets"][-1]["actual_tokens"] == 10
