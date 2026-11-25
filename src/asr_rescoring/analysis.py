"""Audit saved experiment outputs and derive traceable descriptive analysis."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from typing import Any

from .config import file_sha256, resolve_path, write_json
from .dataset import load_dataset
from .metrics import aggregate_metrics
from .preprocessing import normalize_text


def is_correct(row: dict[str, Any]) -> bool:
    return row["normalized_reference"] == row["normalized_selected"]


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Unadjusted 95% Wilson interval for a fixed model's observed accuracy."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return center - margin, center + margin


def audit_run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Cross-check every saved candidate, score, metric, ID and artifact identity."""
    records = load_dataset(resolve_path(config, "evaluation"))
    results, predictions = [], {}
    expected = {
        (order, budget)
        for order in config["experiments"]["ngram_orders"]
        for budget in config["experiments"]["corpus_word_tokens"]
    }
    for path in sorted(resolve_path(config, "metrics").glob(f"{run_id}_order_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["order"], result["budget"])
        if key in predictions or key not in expected:
            raise ValueError(f"Duplicate/unexpected model result: {key}")
        prediction_path = resolve_path(config, "predictions") / result["predictions_file"]
        rows = [
            json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != len(records):
            raise ValueError(f"Incorrect prediction count: {path}")
        for record, row in zip(records, rows, strict=True):
            assert row["utt_id"] == record.utt_id and row["reference"] == record.reference
            assert row["normalized_reference"] == normalize_text(record.reference)
            ranking = row["ranking"]
            assert sorted(score["index"] for score in ranking) == list(
                range(len(record.hypotheses))
            )
            assert ranking == sorted(
                ranking, key=lambda score: (score["cross_entropy"], score["index"])
            )
            for score in ranking:
                assert score["raw"] == record.hypotheses[score["index"]]
                assert score["normalized"] == normalize_text(score["raw"])
                assert score["words"] == len(score["normalized"].split())
                assert math.isclose(
                    score["cross_entropy"],
                    -score["log10_probability"] / score["words"],
                    rel_tol=1e-12,
                )
                if score["perplexity"] is not None:
                    assert math.isclose(
                        math.log10(score["perplexity"]), score["cross_entropy"], rel_tol=1e-12
                    )
            assert row["selected_index"] == ranking[0]["index"]
            assert row["selected"] == ranking[0]["raw"]
            assert row["normalized_selected"] == ranking[0]["normalized"]
        recalculated = aggregate_metrics(
            [row["normalized_reference"] for row in rows],
            [[score["normalized"] for score in row["ranking"]] for row in rows],
            config["evaluation"]["cer_include_spaces"],
        )
        for metric, value in recalculated.items():
            assert math.isclose(value, result["metrics"][metric], abs_tol=1e-12)
        assert result["evaluation_sha256"] == file_sha256(resolve_path(config, "evaluation"))
        model_dir = resolve_path(config, "models") / result["model_id"]
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        assert file_sha256(model_dir / "manifest.json") == result["model_manifest_sha256"]
        for name, checksum in manifest["artifact_sha256"].items():
            assert file_sha256(model_dir / name) == checksum
        results.append(result)
        predictions[key] = rows
    if set(predictions) != expected:
        raise ValueError("Incomplete experiment grid")

    with (resolve_path(config, "metrics") / f"{run_id}_summary.csv").open(
        encoding="utf-8"
    ) as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == len(expected)
    for row in summary:
        result = next(r for r in results if r["model_id"] == row["model_id"])
        for metric, value in result["metrics"].items():
            assert math.isclose(float(row[metric]), value, abs_tol=1e-12)

    best_result = max(results, key=lambda r: r["metrics"]["exact_match_accuracy"])
    best_key = best_result["order"], best_result["budget"]
    best = predictions[best_key]
    bigram = predictions[2, max(config["experiments"]["corpus_word_tokens"])]
    wins = sum(is_correct(a) and not is_correct(b) for a, b in zip(best, bigram, strict=True))
    losses = sum(is_correct(b) and not is_correct(a) for a, b in zip(best, bigram, strict=True))
    discordant = wins + losses
    exact_p = (
        min(
            1.0,
            2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant,
        )
        if discordant
        else 1.0
    )
    errors = [row for row in best if not is_correct(row)]
    length_changes = Counter()
    rank_counts = Counter()
    categories = Counter()
    for row in errors:
        reference = row["normalized_reference"].split()
        selected = row["normalized_selected"].split()
        length_changes[
            "longer"
            if len(selected) > len(reference)
            else "shorter"
            if len(selected) < len(reference)
            else "equal"
        ] += 1
        rank_counts[
            next(
                i
                for i, score in enumerate(row["ranking"], 1)
                if score["normalized"] == row["normalized_reference"]
            )
        ] += 1
        collapsed = [w for i, w in enumerate(selected) if i == 0 or w != selected[i - 1]]
        if collapsed == reference and len(selected) > len(reference):
            categories["adjacent_repetition_only"] += 1
        elif row["normalized_reference"].replace("'", " ") == row["normalized_selected"]:
            categories["apostrophe_fragmentation_only"] += 1
        else:
            categories["other_or_combined"] += 1
    raw_positions, normalized_positions = Counter(), Counter()
    multi_match = 0
    matching_candidates = 0
    for record in records:
        normalized = [
            normalize_text(h) == normalize_text(record.reference) for h in record.hypotheses
        ]
        raw_positions.update(i for i, h in enumerate(record.hypotheses) if h == record.reference)
        normalized_positions.update(i for i, matches in enumerate(normalized) if matches)
        multi_match += sum(normalized) > 1
        matching_candidates += sum(normalized)

    tie_sensitivity = {}
    for key, rows in predictions.items():
        guaranteed = possible = expected_correct = 0
        distinct_text_ties = 0
        for row in rows:
            top = row["ranking"][0]["cross_entropy"]
            tied = [s for s in row["ranking"] if s["cross_entropy"] == top]
            match = [s["normalized"] == row["normalized_reference"] for s in tied]
            guaranteed += all(match)
            possible += any(match)
            expected_correct += sum(match) / len(match)
            distinct_text_ties += len({s["normalized"] for s in tied}) > 1
        tie_sensitivity[f"{key[0]}gram_{key[1]}"] = {
            "minimum_accuracy": guaranteed / len(rows),
            "maximum_accuracy": possible / len(rows),
            "random_tie_accuracy": expected_correct / len(rows),
            "top_ties_between_distinct_normalized_texts": distinct_text_ties,
        }

    baseline_path = resolve_path(config, "metrics") / f"{run_id}_baselines.json"
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = [
        json.loads(line)
        for line in (resolve_path(config, "predictions") / f"{run_id}_baselines.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for name in ("first_candidate", "seeded_random"):
        selections = [row for row in baseline_rows if row["baseline"] == name]
        assert len(selections) == len(records)
        ranked_texts = []
        for record, row in zip(records, selections, strict=True):
            assert row["utt_id"] == record.utt_id and row["reference"] == record.reference
            assert sorted(row["ranking_indices"]) == list(range(len(record.hypotheses)))
            assert row["selected_index"] == row["ranking_indices"][0]
            assert row["selected"] == record.hypotheses[row["selected_index"]]
            ranked_texts.append(
                [normalize_text(record.hypotheses[index]) for index in row["ranking_indices"]]
            )
        recalculated = aggregate_metrics(
            [normalize_text(record.reference) for record in records],
            ranked_texts,
            config["evaluation"]["cer_include_spaces"],
        )
        for metric, value in recalculated.items():
            assert math.isclose(value, baselines["baselines"][name][metric], abs_tol=1e-12)
    report = {
        "run_id": run_id,
        "audit": "passed",
        "models": len(results),
        "predictions_audited": sum(len(rows) for rows in predictions.values()),
        "candidate_scores_audited": sum(
            len(row["ranking"]) for rows in predictions.values() for row in rows
        ),
        "best_model": best_result["model_id"],
        "best_correct": sum(map(is_correct, best)),
        "best_errors": len(errors),
        "best_accuracy_wilson_95": wilson_interval(sum(map(is_correct, best)), len(best)),
        "best_vs_bigram_paired": {"wins": wins, "losses": losses, "exact_mcnemar_p": exact_p},
        "reference_positions_raw": dict(raw_positions),
        "reference_positions_normalized": dict(normalized_positions),
        "records_with_multiple_normalized_matches": multi_match,
        "matching_candidates": matching_candidates,
        "uniform_random_expected_accuracy": matching_candidates
        / sum(len(r.hypotheses) for r in records),
        "best_error_length_changes": dict(length_changes),
        "best_error_reference_ranks": dict(rank_counts),
        "best_error_categories": dict(categories),
        "tie_sensitivity": tie_sensitivity,
        "total_training_seconds": sum(r["training_seconds"] for r in results),
        "total_inference_seconds": sum(r["metrics"]["inference_seconds"] for r in results),
        "total_binary_bytes": sum(r["binary_bytes"] for r in results),
        "result_files": [
            {"model_id": r["model_id"], "predictions": r["predictions_file"]} for r in results
        ],
        "baselines": baselines["baselines"],
        "best_error_ids": [row["utt_id"] for row in errors],
    }
    write_json(resolve_path(config, "metrics") / f"{run_id}_analysis.json", report)
    return report


def vocabulary_diagnostics(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Inspect best-model vocabulary coverage without changing recorded metrics."""
    from .scoring import KenLMScorer

    metric_files = resolve_path(config, "metrics").glob(f"{run_id}_order_*.json")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in metric_files]
    best = max(results, key=lambda r: r["metrics"]["exact_match_accuracy"])
    model = KenLMScorer(
        resolve_path(config, "models") / best["model_id"] / "model.binary", best["order"]
    )
    rows = [
        json.loads(line)
        for line in (resolve_path(config, "predictions") / best["predictions_file"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    diagnostic = {}
    totals = {"correct": [0, 0], "incorrect": [0, 0]}
    for row in rows:
        tokens = row["normalized_reference"].split()
        unknown = [token for token in tokens if token not in model]
        group = "correct" if is_correct(row) else "incorrect"
        totals[group][0] += len(unknown)
        totals[group][1] += len(tokens)
        diagnostic[row["utt_id"]] = {
            "reference_oov": unknown,
            "selected_oov": [t for t in row["normalized_selected"].split() if t not in model],
        }
    output = {
        "model_id": best["model_id"],
        "items": diagnostic,
        "reference_oov_by_outcome": {
            group: {
                "unknown_tokens": counts[0],
                "total_tokens": counts[1],
                "rate": counts[0] / counts[1],
            }
            for group, counts in totals.items()
        },
    }
    write_json(resolve_path(config, "metrics") / f"{run_id}_vocabulary.json", output)
    return output
