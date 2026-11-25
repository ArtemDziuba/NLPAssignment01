"""Evaluate candidates and fixed baselines; save predictions, metrics, and timing."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from .config import environment, file_sha256, fingerprint, resolve_path, snapshot, write_json
from .dataset import Record, load_dataset
from .metrics import aggregate_metrics, oov_rate
from .preprocessing import normalize_text
from .scoring import KenLMScorer, rank_candidates


def baseline_rankings(records: list[Record], seed: int, randomize: bool) -> list[list[int]]:
    """Stable per-ID random permutations are independent of dataset/model iteration order."""
    result = []
    for record in records:
        indices = list(range(len(record.hypotheses)))
        if randomize:
            generator = random.Random(f"{seed}:{record.utt_id}")
            generator.shuffle(indices)
        result.append(indices)
    return result


def evaluate_model(config: dict[str, Any], manifest_path: Path, run_id: str) -> dict[str, Any]:
    """Score every record of the same validated evaluation set."""
    evaluation = resolve_path(config, "evaluation")
    records = load_dataset(evaluation, config["evaluation"]["expected_records"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["identity"]["preprocessing"] != config["preprocessing"]:
        raise ValueError("Model preprocessing differs from current configuration")
    if manifest["identity"].get("evaluation_exclusion_sha256") != file_sha256(evaluation):
        raise ValueError("Evaluation data differs from the training exclusion audit")
    binary = manifest_path.parent / manifest["binary"]
    if file_sha256(binary) != manifest["artifact_sha256"][manifest["binary"]]:
        raise ValueError(f"Model checksum mismatch: {binary}")
    loaded = time.perf_counter()
    model = KenLMScorer(binary, manifest["order"])
    load_seconds = time.perf_counter() - loaded
    normalized = [
        [normalize_text(candidate) for candidate in record.hypotheses] for record in records
    ]
    references = [normalize_text(record.reference) for record in records]
    # Explicit warmup outside measured inference; no reference text is scored.
    model.score(normalized[0][0])
    predictions, rankings = [], []
    total_seconds = 0.0
    for record, candidates in zip(records, normalized, strict=True):
        started = time.perf_counter()
        scores = rank_candidates(model, record.hypotheses)
        seconds = time.perf_counter() - started
        total_seconds += seconds
        rankings.append([score.normalized for score in scores])
        predictions.append(
            {
                "utt_id": record.utt_id,
                "reference": record.reference,
                "normalized_reference": normalize_text(record.reference),
                "selected_index": scores[0].index,
                "selected": scores[0].raw,
                "normalized_selected": scores[0].normalized,
                "inference_seconds": seconds,
                "ranking": [score.to_dict() for score in scores],
            }
        )
    metrics = aggregate_metrics(references, rankings, config["evaluation"]["cer_include_spaces"])
    metrics.update(
        {
            "raw_exact_match_accuracy": sum(
                prediction["selected"] == record.reference
                for prediction, record in zip(predictions, records, strict=True)
            )
            / len(records),
            "oov_references": oov_rate(model, references),
            "oov_hypotheses": oov_rate(model, [s for row in normalized for s in row]),
            "oov_selected": oov_rate(model, [row[0] for row in rankings]),
            "inference_seconds": total_seconds,
            "milliseconds_per_record": 1000 * total_seconds / len(records),
            "model_load_seconds": load_seconds,
        }
    )
    result = {
        "run_id": run_id,
        "model_id": manifest["model_id"],
        "order": manifest["order"],
        "budget": manifest["budget"],
        "actual_tokens": manifest["actual_tokens"],
        "records": len(records),
        "metrics": metrics,
        "training_seconds": manifest["training_seconds"],
        "binary_bytes": binary.stat().st_size,
        "evaluation_sha256": file_sha256(evaluation),
        "model_manifest_sha256": file_sha256(manifest_path),
        "config": snapshot(config),
        "environment": environment(),
        "timing_scope": "normalization, KenLM scoring and ranking; excludes metrics, OOV, I/O",
    }
    identity = fingerprint(
        {
            "evaluation": result["evaluation_sha256"],
            "manifest": result["model_manifest_sha256"],
            "config": result["config"],
        }
    )[:12]
    stem = f"{run_id}_{manifest['model_id']}_{identity}"
    prediction_path = resolve_path(config, "predictions") / f"{stem}.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(prediction_path)
    result["predictions_file"] = prediction_path.name
    write_json(resolve_path(config, "metrics") / f"{stem}.json", result)
    del model
    return result


def evaluate_baselines(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Save shared baseline selections, full rankings, and comparable metrics once."""
    evaluation = resolve_path(config, "evaluation")
    records = load_dataset(evaluation, config["evaluation"]["expected_records"])
    references = [normalize_text(record.reference) for record in records]
    output: dict[str, Any] = {
        "run_id": run_id,
        "seed": config["seed"],
        "evaluation_sha256": file_sha256(evaluation),
        "config": snapshot(config),
        "baselines": {},
    }
    selections = []
    for name, randomize in (("first_candidate", False), ("seeded_random", True)):
        started = time.perf_counter()
        indices = baseline_rankings(records, config["seed"], randomize)
        elapsed = time.perf_counter() - started
        rankings = [
            [normalize_text(record.hypotheses[i]) for i in row]
            for record, row in zip(records, indices, strict=True)
        ]
        metrics = aggregate_metrics(
            references, rankings, config["evaluation"]["cer_include_spaces"]
        )
        metrics["inference_seconds"] = elapsed
        metrics["milliseconds_per_record"] = 1000 * elapsed / len(records)
        output["baselines"][name] = metrics
        for record, row in zip(records, indices, strict=True):
            selections.append(
                {
                    "baseline": name,
                    "utt_id": record.utt_id,
                    "reference": record.reference,
                    "ranking_indices": row,
                    "selected_index": row[0],
                    "selected": record.hypotheses[row[0]],
                }
            )
    path = resolve_path(config, "predictions") / f"{run_id}_baselines.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selections), encoding="utf-8"
    )
    write_json(resolve_path(config, "metrics") / f"{run_id}_baselines.json", output)
    return output
