"""Validate the actual JSONL schema without assuming eight candidates everywhere."""

import json

import pytest

from asr_rescoring.dataset import exclusion_sentences, load_dataset


def write_records(tmp_path, rows):
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def valid():
    return {"utt_id": "u1", "reference": "Гарне місто", "hypotheses": ["Гарне місто", "Гарне мито"]}


def test_preserve_order_and_raw_text(tmp_path):
    records = load_dataset(write_records(tmp_path, [valid()]), 1)
    assert records[0].hypotheses == ("Гарне місто", "Гарне мито")
    assert "гарне мито" in exclusion_sentences(records)


@pytest.mark.parametrize(
    "change",
    [
        {"utt_id": None},
        {"reference": "..."},
        {"reference": 2},
        {"hypotheses": []},
        {"hypotheses": "text"},
        {"hypotheses": [1]},
        {"hypotheses": ["..."]},
        {"reference": None},
    ],
)
def test_reject_bad_fields(tmp_path, change):
    with pytest.raises(ValueError, match=":1:"):
        load_dataset(write_records(tmp_path, [valid() | change]), 1)


def test_duplicate_id_and_count(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(write_records(tmp_path, [valid(), valid()]), 2)
    with pytest.raises(ValueError, match="Expected 500"):
        load_dataset(write_records(tmp_path, [valid()]))


def test_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_dataset(path, None)
