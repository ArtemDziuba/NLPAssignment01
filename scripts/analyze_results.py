"""Audit saved experiment results; optionally inspect the best model's vocabulary."""

import argparse
import json

import _bootstrap  # noqa: F401

from asr_rescoring.analysis import audit_run, vocabulary_diagnostics
from asr_rescoring.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kenlm", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    report = audit_run(config, args.run_id)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "audit",
                    "models",
                    "best_model",
                    "best_correct",
                    "best_errors",
                    "best_error_categories",
                    "best_error_length_changes",
                    "best_vs_bigram_paired",
                    "total_training_seconds",
                )
            },
            indent=2,
        )
    )
    if args.kenlm:
        vocabulary_diagnostics(config, args.run_id)
