"""Run the full configured grid on prepared data; never download implicitly."""

import argparse

import _bootstrap  # noqa: F401

from asr_rescoring.config import load_config
from asr_rescoring.experiments import run_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_experiments(load_config(args.config), args.run_id)
