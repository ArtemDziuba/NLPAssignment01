"""Audit saved LLM predictions and create an experiment-specific HTML report and CSV."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from asr_rescoring.config import PROJECT_ROOT, file_sha256, load_config, resolve_path, write_json
from asr_rescoring.dataset import load_dataset
from asr_rescoring.evaluation import baseline_rankings
from asr_rescoring.llm_scoring import parse_ranking, presentation_order
from asr_rescoring.metrics import aggregate_metrics
from asr_rescoring.preprocessing import normalize_text


def table(headers: list[str], rows: list[list[str]]) -> str:
    def cells(row: list[str], tag: str) -> str:
        return "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in row)

    return (
        "<table><thead><tr>"
        + cells(headers, "th")
        + "</tr></thead><tbody>"
        + "".join("<tr>" + cells(row, "td") + "</tr>" for row in rows)
        + "</tbody></table>"
    )


def build_report(metric_path: Path, others: list[Path] | None = None) -> Path:
    result = json.loads(metric_path.read_text("utf-8"))
    display_name = "MamayLM-4B" if "MamayLM" in result["model_id"] else result["model_id"]
    metadata_path = Path(result["metadata_file"])
    metadata = json.loads(metadata_path.read_text("utf-8"))
    hardware = dict(metadata["hardware"])
    cpu_path = PROJECT_ROOT / "results/logs/20261124_llm_cpu.json"
    if result["run_id"].startswith("20261124") and cpu_path.exists():
        cpu = json.loads(cpu_path.read_text("utf-8-sig"))
        hardware["cpu_model"] = cpu["ProcessorNameString"]
        hardware["supplemental_cpu_file_sha256"] = file_sha256(cpu_path)
    config = load_config()
    evaluation_path = resolve_path(config, "evaluation")
    records = load_dataset(evaluation_path, config["evaluation"]["expected_records"])
    predictions_path = Path(result["predictions_file"])
    predictions = [json.loads(line) for line in predictions_path.read_text("utf-8").splitlines()]
    if metadata["status"] != "complete" or len(records) != len(predictions):
        raise ValueError("Need a completed full-dataset evaluation")
    if file_sha256(predictions_path) != result["predictions_sha256"]:
        raise ValueError("Prediction checksum mismatch")
    if file_sha256(evaluation_path) != result["identity"]["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    events = [
        json.loads(line)
        for line in Path(metadata["paths"]["requests"]).read_text("utf-8").splitlines()
    ]
    valid_events = {e["utt_id"]: e for e in events if not e["warmup"] and "error" not in e}
    for record, prediction in zip(records, predictions, strict=True):
        assert record.utt_id == prediction["utt_id"]
        assert record.reference == prediction["reference"]
        assert prediction["presentation_order"] == presentation_order(
            record.utt_id, len(record.hypotheses), result["identity"]["seed"]
        )
        event = valid_events[record.utt_id]
        ranking = parse_ranking(event["response"], prediction["presentation_order"])
        assert ranking == prediction["ranking_indices"]
        assert prediction["selected_index"] == ranking[0]
        assert prediction["selected"] == record.hypotheses[ranking[0]]
        assert prediction["normalized_ranking"] == [
            normalize_text(record.hypotheses[i]) for i in ranking
        ]
        assert prediction["normalized_selected"] == normalize_text(prediction["selected"])
        request_candidates = json.loads(event["request"]["messages"][1]["content"])
        assert request_candidates == [
            {"number": position + 1, "text": normalize_text(record.hypotheses[index])}
            for position, index in enumerate(prediction["presentation_order"])
        ]
    references = [normalize_text(record.reference) for record in records]
    audited = aggregate_metrics(references, [p["normalized_ranking"] for p in predictions])
    assert all(abs(result["metrics"][key] - value) < 1e-12 for key, value in audited.items())
    assert (
        abs(
            sum(p["inference_seconds"] for p in predictions)
            - result["metrics"]["inference_seconds"]
        )
        < 1e-8
    )
    baselines = {}
    for label, randomize in (
        ("Original first candidate (position leakage)", False),
        ("Seeded shuffled first candidate", True),
    ):
        order = baseline_rankings(records, result["identity"]["seed"], randomize)
        rankings = [
            [normalize_text(r.hypotheses[i]) for i in indices]
            for r, indices in zip(records, order, strict=True)
        ]
        baselines[label] = aggregate_metrics(references, rankings)
    classical_path = PROJECT_ROOT / "results/metrics/20261124_domain_comparison.csv"
    classical = list(csv.DictReader(classical_path.open(encoding="utf-8")))
    comparison = []
    for corpus, order in (("all_five", "3"), ("news_only", "2"), ("news_only", "3")):
        row = next(
            r
            for r in classical
            if r["corpus"] == corpus and r["order"] == order and r["budget"] == "10000000"
        )
        comparison.append(
            {
                "model": f"KenLM {corpus} {order}-gram / 10M words",
                "method": "perplexity",
                "accuracy": float(row["exact_match_accuracy"]),
                "wer": float(row["wer"]),
                "cer": float(row["cer"]),
                "mrr": float(row["mrr"]),
                "ms_per_record": float(row["milliseconds_per_record"]),
                "disk_bytes": int(row["binary_bytes"]),
                "timing_source": "historical Docker CPU run",
            }
        )
    metrics = result["metrics"]
    llm_results = [result]
    for path in others or []:
        other = json.loads(path.read_text("utf-8"))
        if other["identity"]["dataset_sha256"] != result["identity"]["dataset_sha256"]:
            raise ValueError("Compared LLMs used different evaluation data")
        if other["model_id"] != result["model_id"]:
            llm_results.append(other)
    memory_rows = []
    for item in llm_results:
        m = item["metrics"]
        name = "MamayLM-4B" if "MamayLM" in item["model_id"] else item["model_id"]
        comparison.append(
            {
                "model": name,
                "method": item["method"],
                "accuracy": m["exact_match_accuracy"],
                "wer": m["wer"],
                "cer": m["cer"],
                "mrr": m["mrr"],
                "ms_per_record": m["milliseconds_per_record"],
                "disk_bytes": item["model_tag"]["size"],
                "timing_source": "Ollama GPU local HTTP run",
            }
        )
        other_metadata = json.loads(Path(item["metadata_file"]).read_text("utf-8"))
        model_gpu = next(
            row["size_vram"]
            for row in other_metadata["model_state_loaded"]["models"]
            if row["name"] == item["model_id"]
        )
        memory_rows.append(
            [
                name,
                f"{max(s['peak_service_rss_bytes'] for s in item['resources']) / 2**30:.3f}",
                f"{model_gpu / 2**30:.3f}",
                f"{max(s['peak_device_gpu_used_mib'] for s in item['resources']) / 1024:.3f}",
            ]
        )
    csv_path = metric_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    all_models = [
        {
            "model": f"KenLM {row['corpus']} {row['order']}-gram / {int(row['budget']):,} words",
            "method": "perplexity",
            "accuracy": float(row["exact_match_accuracy"]),
            "wer": float(row["wer"]),
            "cer": float(row["cer"]),
            "mrr": float(row["mrr"]),
            "ms_per_record": float(row["milliseconds_per_record"]),
            "disk_bytes": int(row["binary_bytes"]),
            "timing_source": "historical Docker CPU run",
        }
        for row in classical
    ] + comparison[3:]
    all_models.sort(key=lambda row: row["accuracy"], reverse=True)
    all_models_path = PROJECT_ROOT / "results/metrics" / f"{result['run_id']}_all_models.csv"
    with all_models_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(all_models)
    failures = [p for p in predictions if p["normalized_selected"] != p["normalized_reference"]]
    successes = [p for p in predictions if p["normalized_selected"] == p["normalized_reference"]]
    resources = result["resources"]
    peak_ram = max(s["peak_service_rss_bytes"] for s in resources)
    peak_gpu = max(s["peak_device_gpu_used_mib"] for s in resources)
    loaded = metadata["model_state_loaded"]["models"]
    allocation = next(m["size_vram"] for m in loaded if m["name"] == result["model_id"])
    audit = {
        "run_id": result["run_id"],
        "records_audited": len(predictions),
        "candidate_rankings_audited": sum(len(p["ranking_indices"]) for p in predictions),
        "aggregate_metrics": audited,
        "baselines": baselines,
        "comparison": comparison,
        "all_models_csv": str(all_models_path),
        "failures": len(failures),
        "prediction_sha256": file_sha256(predictions_path),
        "requests_sha256": file_sha256(Path(metadata["paths"]["requests"])),
        "resources_sha256": file_sha256(Path(metadata["paths"]["resources"])),
        "gold_display_position_counts": result["gold_display_position_counts"],
        "selected_display_position_counts": dict(
            Counter(p["presentation_order"].index(p["selected_index"]) + 1 for p in predictions)
        ),
        "selected_first_displayed_count": sum(
            p["selected_index"] == p["presentation_order"][0] for p in predictions
        ),
        "unchanged_display_order_rankings": sum(
            p["ranking_indices"] == p["presentation_order"] for p in predictions
        ),
    }
    resource_rows = [
        json.loads(line)
        for line in Path(metadata["paths"]["resources"]).read_text("utf-8").splitlines()
    ]
    intervals = [
        b["monotonic_seconds"] - a["monotonic_seconds"]
        for a, b in zip(resource_rows, resource_rows[1:])
    ]
    audit["actual_resource_sampling"] = {
        "samples": len(resource_rows),
        "median_interval_seconds": statistics.median(intervals),
        "max_interval_seconds": max(intervals),
    }
    write_json(metric_path.with_name(metric_path.stem + "_audit.json"), audit)
    stats_rows = [
        [key, f"{metrics[key]:.6f}"]
        for key in (
            "exact_match_accuracy",
            "raw_exact_match_accuracy",
            "correct_records",
            "wer",
            "cer",
            "mrr",
            "oracle_accuracy",
            "inference_seconds",
            "milliseconds_per_record",
            "latency_median_seconds",
            "latency_p95_seconds",
            "latency_min_seconds",
            "latency_max_seconds",
            "records_per_second",
            "prompt_tokens",
            "generated_tokens",
            "generated_tokens_per_second",
            "prompt_eval_cached_tokens",
            "server_prompt_eval_seconds",
            "server_generation_seconds",
            "server_load_seconds",
            "retried_records",
        )
    ]
    stats_rows += [
        ["Model download / stored layers", f"{result['model_tag']['size'] / 1e9:.3f} GB"],
        ["Peak combined inference-service RSS", f"{peak_ram / 2**30:.3f} GiB"],
        ["Loaded model GPU allocation reported by Ollama", f"{allocation / 2**30:.3f} GiB"],
        ["Peak device-wide GPU usage", f"{peak_gpu:.0f} MiB"],
        ["Warmup wall time", f"{metadata['warmup_wall_seconds']:.3f} seconds"],
    ]
    comparison_table = table(
        ["Model", "Accuracy", "WER", "CER", "MRR", "ms/record", "Disk MB"],
        [
            [
                r["model"],
                f"{r['accuracy']:.1%}",
                f"{r['wer']:.3%}",
                f"{r['cer']:.3%}",
                f"{r['mrr']:.4f}",
                f"{r['ms_per_record']:.3f}",
                f"{r['disk_bytes'] / 1e6:.2f}",
            ]
            for r in comparison
        ],
    )
    all_models_table = table(
        ["Model", "Accuracy", "WER", "CER", "ms/record", "Disk MB"],
        [
            [
                r["model"],
                f"{r['accuracy']:.1%}",
                f"{r['wer']:.3%}",
                f"{r['cer']:.3%}",
                f"{r['ms_per_record']:.3f}",
                f"{r['disk_bytes'] / 1e6:.2f}",
            ]
            for r in all_models
        ],
    )
    example_table = table(
        ["Result", "Utterance", "Reference", "Selected"],
        [
            [label, p["utt_id"], p["reference"], p["selected"]]
            for label, rows in (("Failure", failures[:8]), ("Correct", successes[:3]))
            for p in rows
        ],
    )
    baseline_table = table(
        ["Baseline", "Accuracy", "WER", "CER"],
        [
            [label, f"{m['exact_match_accuracy']:.1%}", f"{m['wer']:.3%}", f"{m['cer']:.3%}"]
            for label, m in baselines.items()
        ],
    )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{html.escape(display_name)} ASR evaluation — experiment results</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#18212b}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border:1px solid #d8dee7;padding:8px;text-align:left}}
th{{background:#edf3f9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:16px}}
small{{color:#526171}}h1,h2{{line-height:1.2}}.note{{border-left:4px solid #3978ac;padding:10px 16px;background:#f0f6fc}}</style>
<h1>{html.escape(display_name)} ASR candidate evaluation</h1>
<p>Experiment-specific report, 24 November 2026. Run: <code>{html.escape(result["run_id"])}</code>.
This is not the final assignment report.</p>
<p class="note"><b>{metrics["correct_records"]} / {len(records)} correct ({metrics["exact_match_accuracy"]:.1%})</b>;
WER {metrics["wer"]:.3%}; CER {metrics["cer"]:.3%}. All {result["hypotheses"]} candidate positions were ranked.</p>
<h2>Method and controls</h2>
<p>The installed <b>{html.escape(result["model_id"])}</b> model uses Q4_K_M quantization and
{metadata["model_show"]["model_info"]["general.parameter_count"]:,} language-model parameters.
It ranks eight normalized candidate sentences through Ollama's local chat API. The output is a validated
complete permutation; the first item is selected. This is prompted ranking, not candidate log-probability
or perplexity scoring. No training or fine-tuning was performed.</p>
<p>Requests contain the system instruction and normalized candidates only. References are used exclusively
after inference for scoring. A deterministic shuffle based on seed 42 and utterance ID removes the
dataset's fixed gold-first presentation; original candidate indices and every shuffle are retained.
Normalization and corpus-aggregated WER/CER match the classical experiments; CER includes spaces.</p>
<h2>Accuracy comparison</h2>{comparison_table}
<details><summary>All {len(all_models)} evaluated models</summary>{all_models_table}</details>
<p>Classical results come from the preserved 36-model comparison. Accuracy is measured on the same data,
but the methods and presentation differ: KenLM ties favor the original first index, while LLMs see shuffled
candidates. Timing is descriptive: historical Docker CPU KenLM calls versus current GPU Ollama requests,
including prompt processing and generation. These are not controlled measurements of architecture alone.</p>
<h2>Baselines and candidate position</h2>{baseline_table}
<p>The reference is the original first hypothesis in every record. The original first-candidate baseline
therefore reaches 100% without language understanding. The shuffled-first baseline uses the same permutations
as the LLMs. Gold displayed-position counts: {html.escape(str(result["gold_display_position_counts"]))}.</p>
<p>This model selected the first displayed candidate in {audit["selected_first_displayed_count"]} / 500 records.
It returned the entire unchanged display order in {audit["unchanged_display_order_rankings"]} / 500 records.
Strong preference for presentation order indicates a limitation of this prompted-ranking configuration;
these results do not establish performance under a different prompt or likelihood-based rescoring.</p>
<h2>Runtime memory</h2>
{table(["Model", "Peak service RAM GiB", "Model GPU allocation GiB", "Peak device GPU GiB"], memory_rows)}
<h2>Measured parameters</h2>{table(["Parameter / metric", "Value"], stats_rows)}
<p>Inference time sums sequential warm local HTTP requests, including server processing, generation,
validation and any retries. It excludes warmup, local normalization, metric computation and file writes.
Ollama prefix caching remains enabled and cached token counts are recorded. This is one evaluation pass,
not repeated-run latency benchmarking. One warmup request is saved separately.</p>
<p>RAM sampling requests a 0.25-second interval and sums Ollama GUI, service and llama-runner working sets; shared
pages may be counted more than once. GPU sampling targets one second across the whole device, including
other applications. Ollama's loaded-model GPU allocation is a separate measurement. Neither disk size nor
GPU allocation should be read as the peak RAM footprint. Short-lived peaks between samples may be missed.
No matched KenLM runtime-RAM measurement was collected in this run.</p>
<p>Sampling includes warmup. Process inspection and GPU queries add overhead: the observed median
sampling interval was {statistics.median(intervals):.3f} seconds, maximum {max(intervals):.3f} seconds,
with {len(resource_rows)} samples. These are sampled peaks, not instrumented allocator maxima.</p>
<h2>Examples from saved predictions</h2>{example_table}
<p>Failures can include plausible alternate word forms or spellings: without audio or context, linguistic
plausibility cannot always identify the reference. Examples are actual outputs, with no correction.</p>
<p>Primary accuracy uses normalized text. Raw exact match also counts punctuation and casing differences
and is reported separately. Pretraining overlap for these third-party pretrained LLMs is unknown;
the classical models' exact training-exclusion audit cannot be applied to their unavailable pretraining data.</p>
<h2>Reproducibility</h2><p>Ollama version: {html.escape(metadata["ollama_version"]["version"])}.
Model digest: <code>{html.escape(result["model_tag"]["digest"])}</code>.
Dataset SHA-256: <code>{result["identity"]["dataset_sha256"]}</code>.</p>
<h3>Hardware</h3><pre>{html.escape(json.dumps(hardware, indent=2))}</pre>
<h3>Explicit inference settings</h3><pre>{html.escape(json.dumps(result["identity"]["llm_settings"], indent=2))}</pre>
<p>The generic configuration remains disabled for automatic classical runs; this run was explicitly invoked
through <code>scripts/evaluate_llm.py</code>. Greedy decoding, fixed seed, model digest, prompt,
source hashes and package versions are saved; identical results across hardware or backend versions are not guaranteed.</p>
<h3>System prompt</h3><pre>{html.escape(result["identity"]["system_prompt"])}</pre>
<h3>Artifact locations</h3><pre>{html.escape(json.dumps(metadata["paths"], indent=2))}</pre>
<p>Every selected index and full ranking was re-audited against the dataset and raw response. Aggregate metrics,
candidate mappings, dataset hash and prediction checksum were independently checked when generating this report.</p>
</html>"""
    destination = PROJECT_ROOT / "report" / f"{result['run_id']}_results.html"
    destination.write_text(body, encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--compare", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(build_report(args.metrics.resolve(), [path.resolve() for path in args.compare]))
