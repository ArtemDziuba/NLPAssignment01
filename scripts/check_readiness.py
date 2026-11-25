"""Check data integrity and readiness without running the experiment grid."""

import argparse
import json
import shutil

import _bootstrap  # noqa: F401

from asr_rescoring.config import file_sha256, load_config, resolve_path, write_json
from asr_rescoring.dataset import exclusion_sentences, load_dataset
from asr_rescoring.experiments import validate_prepared_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--verify-archives", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    records = load_dataset(
        resolve_path(config, "evaluation"), config["evaluation"]["expected_records"]
    )
    metadata = validate_prepared_data(config)
    excluded = exclusion_sentences(records)
    subsets = []
    previous = b""
    for budget in sorted(config["experiments"]["corpus_word_tokens"]):
        path = resolve_path(config, "subsets") / f"tokens_{budget}.txt"
        content = path.read_bytes()
        if not content.startswith(previous):
            raise ValueError(f"Subsets are not nested: {path}")
        sentences = content.decode("utf-8").splitlines()
        if len(sentences) != len(set(sentences)) or excluded.intersection(sentences):
            raise ValueError(f"Duplicate or evaluation-overlapping training sentence: {path}")
        previous = content
        subsets.append(json.loads(path.with_suffix(".json").read_text(encoding="utf-8")))
    archives = []
    for source in config["corpus"]["sources"]:
        path = resolve_path(config, "raw_corpus") / source["filename"]
        provenance = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if path.stat().st_size != provenance["bytes"] or provenance["url"] != source["url"]:
            raise ValueError(f"Archive size/source mismatch: {path}")
        if args.verify_archives and file_sha256(path) != provenance["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {path}")
        if source.get("sha256") and source["sha256"] != provenance["sha256"]:
            raise ValueError(f"Archive differs from pinned source hash: {path}")
        archives.append(provenance)
    kenlm_report = resolve_path(config, "logs") / "kenlm_check.json"
    report = {
        "status": "data_ready",
        "evaluation_records": len(records),
        "evaluation_sha256": file_sha256(resolve_path(config, "evaluation")),
        "archives": archives,
        "archive_checksums_verified": args.verify_archives,
        "subsets": subsets,
        "processed_sentences": metadata["sentences"],
        "free_disk_bytes": shutil.disk_usage(config["_root"]).free,
        "kenlm_check": (
            json.loads(kenlm_report.read_text(encoding="utf-8")) if kenlm_report.exists() else None
        ),
    }
    write_json(resolve_path(config, "logs") / "readiness.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("status", "evaluation_records", "free_disk_bytes")}, indent=2
        )
    )
    for row in subsets:
        print(
            f"{row['budget']:>10,} budget: {row['actual_tokens']:>10,} words; "
            f"{row['sentences']:,} sentences; domains={row['domain_tokens']}"
        )
