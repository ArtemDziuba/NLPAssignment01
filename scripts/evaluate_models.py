"""Evaluate configured saved models and shared baselines without training."""

import argparse
from datetime import UTC, datetime

import _bootstrap  # noqa: F401

from asr_rescoring.config import load_config, resolve_path
from asr_rescoring.evaluation import evaluate_baselines, evaluate_model
from asr_rescoring.experiments import configure_logging, save_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    configure_logging(config, run_id)
    manifests = [
        resolve_path(config, "models") / f"order_{order}_tokens_{budget}" / "manifest.json"
        for budget in config["experiments"]["corpus_word_tokens"]
        for order in config["experiments"]["ngram_orders"]
    ]
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"Train the configured model first: {manifest}")
    evaluate_baselines(config, run_id)
    results = [evaluate_model(config, path, run_id) for path in manifests]
    save_summary(config, run_id, results)
