"""Verify KenLM and train the configured models from prepared subsets."""

import argparse

import _bootstrap  # noqa: F401

from asr_rescoring.config import load_config, resolve_path
from asr_rescoring.experiments import configure_logging, train_models
from asr_rescoring.training import verify_kenlm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config, "training")
    verify_kenlm(config, resolve_path(config, "logs") / "kenlm_check.json")
    if not args.check_only:
        train_models(config, force=args.force)
