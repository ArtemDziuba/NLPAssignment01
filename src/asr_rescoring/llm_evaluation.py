"""Run and checkpoint a local LLM evaluation with raw requests and resource evidence."""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import environment, file_sha256, fingerprint, resolve_path, snapshot, write_json
from .dataset import load_dataset
from .llm_scoring import SYSTEM_PROMPT, api_request, make_request, parse_ranking
from .metrics import aggregate_metrics
from .preprocessing import normalize_text
from .resource_monitor import ResourceMonitor, gpu_snapshot


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def run_evaluation(config: dict[str, Any], run_id: str, limit: int | None = None) -> Path:
    """The CLI explicitly opts in even while classical config keeps llm.enabled false."""
    import psutil

    settings = config["llm"]
    endpoint = settings["endpoint"]
    dataset = resolve_path(config, "evaluation")
    records = load_dataset(dataset, config["evaluation"]["expected_records"])
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("No evaluation records")
    tag = next(
        m for m in api_request(endpoint, "/api/tags")["models"] if m["name"] == settings["model"]
    )
    identity = {
        "dataset_sha256": file_sha256(dataset),
        "model_digest": tag["digest"],
        "llm_settings": settings,
        "seed": config["seed"],
        "limit": limit,
        "system_prompt": SYSTEM_PROMPT,
        "preprocessing": config["preprocessing"],
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("llm_scoring.py", "llm_evaluation.py", "resource_monitor.py")
        },
    }
    stem = f"{run_id}_{settings['model'].replace(':', '_')}"
    metric_path = resolve_path(config, "metrics") / f"{stem}.json"
    prediction_path = resolve_path(config, "predictions") / f"{stem}.jsonl"
    metadata_path = resolve_path(config, "logs") / f"{stem}_metadata.json"
    events_path = resolve_path(config, "logs") / f"{stem}_requests.jsonl"
    resources_path = resolve_path(config, "logs") / f"{stem}_resources.jsonl"
    for path in (metric_path, prediction_path, metadata_path, events_path, resources_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["identity"] != identity:
            raise ValueError("Resume identity differs: use a new run ID")
        if metadata.get("status") == "complete":
            raise ValueError("Run is already complete; use a new run ID to repeat timings")
    else:
        model_show = api_request(endpoint, "/api/show", {"model": settings["model"]})
        metadata = {
            "run_id": run_id,
            "status": "running",
            "identity": identity,
            "identity_sha256": fingerprint(identity),
            "config": snapshot(config),
            "model_tag": tag,
            "model_show": model_show,
            "ollama_version": api_request(endpoint, "/api/version"),
            "environment": {**environment(), "psutil": psutil.__version__},
            "hardware": {
                "cpu": platform.processor(),
                "logical_cpus": psutil.cpu_count(),
                "physical_cpus": psutil.cpu_count(logical=False),
                "ram_bytes": psutil.virtual_memory().total,
                "gpus": gpu_snapshot(),
            },
            "model_state_before": api_request(endpoint, "/api/ps"),
            "started_at_utc": datetime.now(UTC).isoformat(),
            "paths": {
                "predictions": str(prediction_path),
                "requests": str(events_path),
                "resources": str(resources_path),
                "metrics": str(metric_path),
            },
            "notes": [
                "Prompted candidate ranking, not teacher-forced likelihood or perplexity.",
                "Normalized candidates only; references never appear as separate gold labels.",
                "Display order is shuffled by seed and utterance ID; original indices preserved.",
                "Greedy decoding and fixed seed do not guarantee bitwise reproducibility.",
                "Single sequential pass; no cross-run latency confidence interval.",
                "OOV rate and candidate perplexity are not defined by this ranking method.",
            ],
            "resource_segments": [],
        }
        write_json(metadata_path, metadata)
    predictions = []
    if prediction_path.exists():
        predictions = [json.loads(line) for line in prediction_path.read_text("utf-8").splitlines()]
        for record, prediction in zip(records, predictions, strict=False):
            if prediction["utt_id"] != record.utt_id:
                raise ValueError("Resume predictions are not an ordered dataset prefix")
        if len(predictions) > len(records):
            raise ValueError("Too many checkpointed predictions")
    monitor = ResourceMonitor(
        resources_path, settings["memory_sample_seconds"], settings["gpu_sample_seconds"]
    )
    monitor.start()
    try:
        with events_path.open("a", encoding="utf-8") as events:

            def request(record: Any, warmup: bool = False) -> dict[str, Any]:
                payload, order = make_request(
                    settings, record.utt_id, record.hypotheses, config["seed"]
                )
                attempts = []
                for attempt in range(settings["max_attempts"]):
                    started = time.perf_counter()
                    event: dict[str, Any] = {
                        "utt_id": record.utt_id,
                        "warmup": warmup,
                        "attempt": attempt + 1,
                        "request": payload,
                        "presentation_order": order,
                        "requested_at_utc": datetime.now(UTC).isoformat(),
                    }
                    try:
                        response = api_request(
                            endpoint, "/api/chat", payload, settings["request_timeout_seconds"]
                        )
                        event["response"] = response
                        ranking = parse_ranking(response, order)
                    except Exception as error:
                        event["error"] = repr(error)
                    event["wall_seconds"] = time.perf_counter() - started
                    events.write(json.dumps(event, ensure_ascii=False) + "\n")
                    events.flush()
                    attempts.append(event)
                    if "error" not in event:
                        return {
                            "utt_id": record.utt_id,
                            "reference": record.reference,
                            "normalized_reference": normalize_text(record.reference),
                            "presentation_order": order,
                            "ranking_indices": ranking,
                            "selected_index": ranking[0],
                            "selected": record.hypotheses[ranking[0]],
                            "normalized_selected": normalize_text(record.hypotheses[ranking[0]]),
                            "normalized_ranking": [
                                normalize_text(record.hypotheses[index]) for index in ranking
                            ],
                            "inference_seconds": sum(a["wall_seconds"] for a in attempts),
                            "attempts": len(attempts),
                            "server": {k: v for k, v in response.items() if k != "message"},
                        }
                raise RuntimeError(f"No valid ranking for {record.utt_id}: {attempts[-1]['error']}")

            warmup_started = time.perf_counter()
            for record in records[: settings["warmup_records"]]:
                request(record, warmup=True)
            metadata["warmup_wall_seconds"] = time.perf_counter() - warmup_started
            metadata["model_state_loaded"] = api_request(endpoint, "/api/ps")
            write_json(metadata_path, metadata)
            with prediction_path.open("a", encoding="utf-8") as stream:
                for record in records[len(predictions) :]:
                    prediction = request(record)
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    stream.flush()
                    predictions.append(prediction)
                    if len(predictions) % 10 == 0 or len(predictions) == len(records):
                        seconds = sum(p["inference_seconds"] for p in predictions)
                        correct = sum(
                            p["normalized_selected"] == p["normalized_reference"]
                            for p in predictions
                        )
                        eta = (len(records) - len(predictions)) * seconds / len(predictions)
                        print(
                            f"{len(predictions)}/{len(records)}; correct={correct}; "
                            f"API={seconds:.1f}s; remaining~{eta / 60:.1f}min",
                            flush=True,
                        )
            metadata["model_state_after"] = api_request(endpoint, "/api/ps")
    finally:
        metadata["resource_segments"].append(monitor.stop())
        metadata["completed_records"] = len(predictions)
        write_json(metadata_path, metadata)
    references = [normalize_text(record.reference) for record in records]
    metrics = aggregate_metrics(references, [p["normalized_ranking"] for p in predictions])
    latencies = [p["inference_seconds"] for p in predictions]
    prompt_tokens = sum(p["server"].get("prompt_eval_count", 0) for p in predictions)
    output_tokens = sum(p["server"].get("eval_count", 0) for p in predictions)
    generation_seconds = sum(p["server"].get("eval_duration", 0) for p in predictions) / 1e9
    metrics.update(
        {
            "raw_exact_match_accuracy": sum(p["selected"] == p["reference"] for p in predictions)
            / len(records),
            "correct_records": sum(
                p["normalized_selected"] == r for p, r in zip(predictions, references)
            ),
            "inference_seconds": sum(latencies),
            "milliseconds_per_record": 1000 * statistics.mean(latencies),
            "latency_median_seconds": statistics.median(latencies),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "latency_min_seconds": min(latencies),
            "latency_max_seconds": max(latencies),
            "records_per_second": len(records) / sum(latencies),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": output_tokens,
            "generated_tokens_per_second": output_tokens / generation_seconds,
            "prompt_eval_cached_tokens": sum(
                p["server"].get("prompt_eval_cached_count", 0) for p in predictions
            ),
            "server_prompt_eval_seconds": sum(
                p["server"].get("prompt_eval_duration", 0) for p in predictions
            )
            / 1e9,
            "server_generation_seconds": generation_seconds,
            "server_load_seconds": sum(p["server"].get("load_duration", 0) for p in predictions)
            / 1e9,
            "retried_records": sum(p["attempts"] > 1 for p in predictions),
        }
    )
    result = {
        "run_id": run_id,
        "model_id": settings["model"],
        "records": len(records),
        "hypotheses": sum(len(r.hypotheses) for r in records),
        "method": settings["method"],
        "identity": identity,
        "metrics": metrics,
        "resources": metadata["resource_segments"],
        "model_tag": tag,
        "metadata_file": str(metadata_path),
        "predictions_file": str(prediction_path),
        "predictions_sha256": file_sha256(prediction_path),
        "gold_display_position_counts": dict(
            Counter(p["presentation_order"].index(0) + 1 for p in predictions)
        ),
        "timing_scope": (
            "Sequential local HTTP requests incl prompt and decode, validation/retries; "
            "excludes warmup, normalization, metrics and disk writes. Prefix caching on."
        ),
        "notes": metadata["notes"],
    }
    write_json(metric_path, result)
    metadata.update(status="complete", finished_at_utc=datetime.now(UTC).isoformat())
    write_json(metadata_path, metadata)
    print(f"Saved {metric_path}", flush=True)
    return metric_path
