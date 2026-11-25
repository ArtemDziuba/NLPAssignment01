"""Opt-in native KenLM checks; never install tools or download corpora in tests."""

import os

import pytest

from asr_rescoring.config import load_config
from asr_rescoring.training import verify_kenlm


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ASR_TEST_KENLM") != "1", reason="Set ASR_TEST_KENLM=1")
def test_real_kenlm_all_orders(tmp_path):
    report = verify_kenlm(load_config(), tmp_path / "kenlm.json")
    assert all(value["trained"] and value["binary_loaded"] for value in report["orders"].values())
    assert report["unigram_actual_logprob"] == pytest.approx(
        report["unigram_expected_logprob"], abs=1e-4
    )


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ASR_TEST_KENLM") != "1", reason="Set ASR_TEST_KENLM=1")
def test_tiny_end_to_end_grid_and_reuse(tmp_path):
    import bz2
    import csv
    import json

    from asr_rescoring.corpus import prepare_corpus
    from asr_rescoring.experiments import run_experiments, train_models

    config = load_config()
    config["_root"] = tmp_path
    config["paths"]["evaluation"] = "evaluation.jsonl"
    config["evaluation"]["expected_records"] = 2
    config["experiments"]["corpus_word_tokens"] = [20, 40]
    config["kenlm"]["memory"] = "128M"
    config["kenlm"]["discount_fallback"] = True
    config["preprocessing"]["sentences_per_domain"] = 10
    raw = tmp_path / config["paths"]["raw_corpus"]
    raw.mkdir(parents=True)
    for index, source in enumerate(config["corpus"]["sources"]):
        source.pop("sha256", None)  # Synthetic archives have their own computed source hashes.
        with bz2.open(raw / source["filename"], "wt", encoding="utf-8") as stream:
            for number in range(8):
                stream.write(f"навчальне місто номер {index} має парк {number}\n")
    evaluation = tmp_path / "evaluation.jsonl"
    rows = [
        {
            "utt_id": str(i),
            "reference": "контрольний еталон",
            "hypotheses": ["контрольний еталон", "контрольна помилка"],
        }
        for i in range(2)
    ]
    evaluation.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    prepare_corpus(config, workers=1)
    results = run_experiments(config, "fixture")
    assert len(results) == 6
    summary = tmp_path / config["paths"]["metrics"] / "fixture_summary.csv"
    with summary.open() as stream:
        assert len(list(csv.DictReader(stream))) == 6
    assert all(result["binary_bytes"] > 0 for result in results)
    model_path = tmp_path / "models/order_1_tokens_20/model.binary"
    previous_mtime = model_path.stat().st_mtime_ns
    train_models(config)
    assert model_path.stat().st_mtime_ns == previous_mtime
    predictions = list((tmp_path / config["paths"]["predictions"]).glob("*order_*.jsonl"))
    assert len(predictions) == 6
    assert all(len(path.read_text(encoding="utf-8").splitlines()) == 2 for path in predictions)
    config["preprocessing"]["min_sentence_words"] += 1
    with pytest.raises(ValueError, match="Preprocessing configuration changed"):
        train_models(config)
    config["preprocessing"]["min_sentence_words"] -= 1
    evaluation.write_text(evaluation.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Evaluation data changed"):
        train_models(config)
