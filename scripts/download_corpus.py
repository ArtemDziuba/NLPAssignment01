"""Download all configured corpus domains with resume and checksum manifests."""

import argparse
import logging

import _bootstrap  # noqa: F401

from asr_rescoring.config import load_config
from asr_rescoring.download import download_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_corpus(load_config(args.config))
