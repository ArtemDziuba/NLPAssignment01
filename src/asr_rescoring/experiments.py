"""Execute configured grids with explicit preparation, safe reuse, and result exports."""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import file_sha256, resolve_path, snapshot, write_json
from .dataset import load_dataset
from .evaluation import evaluate_baselines, evaluate_model
from .training import train_model, verify_kenlm


def configure_logging(config: dict[str, Any], name: str) -> None:
    """Log to console and a persistent file."""
    directory = resolve_path(config, "logs")
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(directory / f"{name}.log", encoding="utf-8"),
        ],
        force=True,
    )


def validate_prepared_data(config: dict[str, Any]) -> dict[str, Any]:
    """Reject stale preprocessing or changed evaluation data before any real training."""
    from .preprocessing import VERSION

    processed = resolve_path(config, "processed_corpus")
    manifest_path = processed.with_suffix(".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare corpus first: {manifest_path}")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata["evaluation_sha256"] != file_sha256(resolve_path(config, "evaluation")):
        raise ValueError("Evaluation data changed since preprocessing; prepare corpus again")
    if metadata["normalizer"] != VERSION:
        raise ValueError("Normalizer changed since preprocessing; prepare corpus again")
    if metadata["seed"] != config["seed"]:
        raise ValueError("Seed changed since preprocessing; prepare corpus again")
    if metadata["sha256"] != file_sha256(processed):
        raise ValueError("Processed corpus checksum mismatch")
    if set(metadata["domain_manifests"]) != {s["domain"] for s in config["corpus"]["sources"]}:
        raise ValueError("Corpus domains changed; prepare corpus again")
    for domain in metadata["domain_manifests"].values():
        if domain["identity"]["preprocessing"] != config["preprocessing"]:
            raise ValueError("Preprocessing configuration changed; prepare corpus again")
    for budget in config["experiments"]["corpus_word_tokens"]:
        path = resolve_path(config, "subsets") / f"tokens_{budget}.txt"
        subset_metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        expected = next((s for s in metadata["subsets"] if s["budget"] == budget), None)
        if subset_metadata != expected or file_sha256(path) != subset_metadata["sha256"]:
            raise ValueError(f"Prepared subset provenance/checksum mismatch: {path}")
    return metadata


def train_models(config: dict[str, Any], force: bool = False) -> list[Path]:
    """Train only previously prepared, checksum-verified subsets."""
    validate_prepared_data(config)
    manifests = []
    for budget in config["experiments"]["corpus_word_tokens"]:
        subset = resolve_path(config, "subsets") / f"tokens_{budget}.txt"
        metadata_path = subset.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Prepare corpus first: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if file_sha256(subset) != metadata["sha256"]:
            raise ValueError(f"Subset checksum mismatch: {subset}")
        for order in config["experiments"]["ngram_orders"]:
            manifest = train_model(config, subset, order, budget, force=force)
            manifests.append(
                resolve_path(config, "models") / manifest["model_id"] / "manifest.json"
            )
    return manifests


def save_summary(config: dict[str, Any], run_id: str, results: list[dict[str, Any]]) -> Path:
    """Export one flat table suitable for notebooks and report comparisons."""
    path = resolve_path(config, "metrics") / f"{run_id}_summary.csv"
    rows = [
        {
            key: result[key]
            for key in (
                "model_id",
                "order",
                "budget",
                "actual_tokens",
                "training_seconds",
                "binary_bytes",
            )
        }
        | result["metrics"]
        for result in results
    ]
    if not rows:
        raise ValueError("No model results to summarize")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_experiments(config: dict[str, Any], run_id: str | None = None) -> list[dict[str, Any]]:
    """Run training/evaluation on prepared data; never trigger an implicit download."""
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if not identifier or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in identifier
    ):
        raise ValueError("run_id must contain only letters, digits, underscores, or hyphens")
    configure_logging(config, identifier)
    load_dataset(resolve_path(config, "evaluation"), config["evaluation"]["expected_records"])
    write_json(resolve_path(config, "logs") / f"{identifier}_config.json", snapshot(config))
    verify_kenlm(config, resolve_path(config, "logs") / f"{identifier}_kenlm.json")
    manifests = train_models(config, force=not config["experiments"]["reuse_matching_models"])
    evaluate_baselines(config, identifier)
    results = [evaluate_model(config, manifest, identifier) for manifest in manifests]
    summary = save_summary(config, identifier, results)
    logging.info("Completed %d models; summary: %s", len(results), summary)
    return results
