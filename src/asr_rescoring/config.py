"""Validate settings, resolve project paths, and record reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a config; resolve its declared project root relative to the config."""
    config_path = (Path(path) if path else PROJECT_ROOT / "configs/experiments.yaml").resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    for section in (
        "paths",
        "corpus",
        "preprocessing",
        "experiments",
        "kenlm",
        "scoring",
        "evaluation",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    orders = config["experiments"]["ngram_orders"]
    sizes = config["experiments"]["corpus_word_tokens"]
    if not orders or any(type(n) is not int or n not in (1, 2, 3) for n in orders):
        raise ValueError("ngram_orders must contain orders 1, 2, or 3")
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError("corpus_word_tokens must contain positive integers")
    if len(set(orders)) != len(orders) or len(set(sizes)) != len(sizes):
        raise ValueError("Experiment orders and sizes must be unique")
    scoring = config["scoring"]
    if (
        scoring["perplexity_denominator"] != "word_tokens"
        or scoring["tie_break"] != "first_candidate"
        or scoring["log_base"] != 10
        or not scoring["bos"]
        or not scoring["eos"]
    ):
        raise ValueError("Use assignment perplexity, BOS/EOS, and first-candidate ties")
    if config["evaluation"]["match_on"] != "normalized_text":
        raise ValueError("Only documented normalized-text evaluation is supported")
    if config["experiments"]["subset_policy"] != "whole_sentence_prefix_at_most":
        raise ValueError("Unsupported subset policy")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    fixed = {
        "preprocessing": {
            "sentence_segmenter": "official_sentence_lines",
            "unicode_normalization": "NFC",
            "lowercase": True,
            "apostrophe": "'",
            "punctuation": "separate_except_internal_apostrophes",
            "collapse_whitespace": True,
            "preserve_digits": True,
            "correct_recognition_errors": False,
            "drop_empty_training_sentences": True,
            "deduplicate_training_sentences": True,
            "exclude_evaluation_overlap": True,
            "overlap_fields": ["reference", "hypotheses"],
        },
        "evaluation": {
            "id_field": "utt_id",
            "reference_field": "reference",
            "candidates_field": "hypotheses",
            "require_unique_ids": True,
            "preserve_candidate_order": True,
            "edit_rate_aggregation": "corpus",
            "mrr_missing_reference": 0.0,
            "baselines": ["first_candidate", "seeded_random"],
        },
        "scoring": {"selection": "minimum_perplexity", "empty_normalized_candidate": "error"},
    }
    for section, expected in fixed.items():
        for key, value in expected.items():
            if config[section].get(key) != value:
                raise ValueError(f"Unsupported convention {section}.{key}; expected {value!r}")
    domains = [source["domain"] for source in config["corpus"]["sources"]]
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("Corpus domains must be nonempty and unique")
    config["_root"] = (config_path.parent / config.get("project_root", "..")).resolve()
    config["_config_path"] = config_path
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a named path without relying on the working directory."""
    return (config["_root"] / config["paths"][key]).resolve()


def snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Return portable configuration without runtime-only paths."""
    return {key: value for key, value in config.items() if not key.startswith("_")}


def fingerprint(value: Any) -> str:
    """Hash JSON-serializable metadata canonically."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash files incrementally, including multi-gigabyte archives."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Replace a metadata file atomically after a complete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment() -> dict[str, str]:
    """Record interpreter and OS for experiment provenance."""
    return {"python": sys.version, "platform": platform.platform()}
