"""Explicitly evaluate an already installed Ollama model; never download or train one."""

import argparse
from datetime import UTC, datetime

import _bootstrap  # noqa: F401

from asr_rescoring.config import load_config
from asr_rescoring.llm_evaluation import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_llm"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run_evaluation(load_config(args.config), args.run_id, args.limit)
