"""Validate evaluation JSONL and preserve raw sentences, IDs, and candidate order."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .preprocessing import normalize_text


@dataclass(frozen=True)
class Record:
    """One ASR item with immutable candidate order."""

    utt_id: str
    reference: str
    hypotheses: tuple[str, ...]


def load_dataset(path: Path, expected_records: int | None = 500) -> list[Record]:
    """Reject malformed records with useful line-number diagnostics."""
    records: list[Record] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Record must be an object")
                identifier, reference, hypotheses = (
                    item.get("utt_id"),
                    item.get("reference"),
                    item.get("hypotheses"),
                )
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("utt_id must be a nonempty string")
                if identifier in seen:
                    raise ValueError(f"Duplicate utt_id: {identifier}")
                if not isinstance(reference, str) or not normalize_text(reference):
                    raise ValueError("reference must contain word tokens")
                if not isinstance(hypotheses, list) or not hypotheses:
                    raise ValueError("hypotheses must be a nonempty list")
                if any(not isinstance(h, str) or not normalize_text(h) for h in hypotheses):
                    raise ValueError("Every hypothesis must be a string containing word tokens")
                records.append(Record(identifier, reference, tuple(hypotheses)))
                seen.add(identifier)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}:{number}: {error}") from error
    if not records:
        raise ValueError("Evaluation dataset is empty")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} records, found {len(records)}")
    return records


def exclusion_sentences(records: list[Record]) -> set[str]:
    """Use evaluation text exclusively to reject exact normalized training overlaps."""
    return {
        normalize_text(text)
        for record in records
        for text in (record.reference, *record.hypotheses)
    }
