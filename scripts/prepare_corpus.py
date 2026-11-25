"""Stream all corpus domains once and create deterministic nested subsets."""

import argparse
import logging

import _bootstrap  # noqa: F401

from asr_rescoring.config import load_config
from asr_rescoring.corpus import prepare_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Cache samples for completed downloads without making subsets",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare_corpus(load_config(args.config), workers=args.workers, sample_only=args.sample_only)
