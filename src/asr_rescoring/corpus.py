"""Stream compressed UberText, sample unique sentences, and create nested subsets."""

from __future__ import annotations

import bz2
import hashlib
import heapq
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .config import file_sha256, fingerprint, resolve_path, write_json
from .dataset import exclusion_sentences, load_dataset
from .preprocessing import VERSION, clean_training_sentence, sentence_lines


def priority(sentence: str, seed: int) -> int:
    """A stable random priority independent of Python hash randomization and input order."""
    return int.from_bytes(
        hashlib.blake2b(f"{seed}\0{sentence}".encode(), digest_size=16).digest(), "big"
    )


def _initialize_worker_logging() -> None:
    """Expose preparation progress in fresh Windows worker processes too."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def sample_domain(
    archive: Path,
    output: Path,
    excluded: set[str],
    settings: dict[str, Any],
    seed: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Scan a complete archive once and retain the lowest-priority unique sentences.

    Memory is bounded by sentences_per_domain. Duplicate retained sentences share a
    priority, so repeated source text never increases its selection probability.
    """
    source_hash = file_sha256(archive)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError(f"Archive differs from configured hash: {archive}")
    identity = {
        "archive_sha256": source_hash,
        "preprocessing": settings,
        "normalizer": VERSION,
        "seed": seed,
        "exclusion_hash": fingerprint(sorted(excluded)),
    }
    manifest_path = output.with_suffix(".json")
    if output.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") == identity and previous["sha256"] == file_sha256(output):
            return previous

    capacity = settings["sentences_per_domain"]
    if type(capacity) is not int or capacity < 1:
        raise ValueError("sentences_per_domain must be a positive integer")
    heap: list[tuple[int, str]] = []
    retained: set[str] = set()
    counts: Counter[str] = Counter()
    with bz2.open(archive, "rt", encoding="utf-8", errors="strict") as stream:
        for text in sentence_lines(stream):
            counts["input_sentences"] += 1
            sentence = clean_training_sentence(
                text, settings["min_sentence_words"], settings["max_sentence_words"]
            )
            if sentence is None:
                counts["rejected_cleaning"] += 1
                continue
            if sentence in excluded:
                counts["rejected_evaluation_overlap"] += 1
                continue
            counts["eligible_occurrences"] += 1
            if sentence in retained:
                counts["duplicate_retained_occurrences"] += 1
                continue
            rank = priority(sentence, seed)
            if len(heap) < capacity:
                heapq.heappush(heap, (-rank, sentence))
                retained.add(sentence)
            elif (rank, sentence) < (-heap[0][0], heap[0][1]):
                _, removed = heapq.heapreplace(heap, (-rank, sentence))
                retained.remove(removed)
                retained.add(sentence)
            if counts["input_sentences"] % 1000000 == 0:
                logging.info("%s: scanned %d sentences", archive.name, counts["input_sentences"])

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    ordered = sorted(retained, key=lambda sentence: (priority(sentence, seed), sentence))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for sentence in ordered:
            stream.write(sentence + "\n")
    temporary.replace(output)
    manifest = {
        "identity": identity,
        "counts": dict(counts),
        "sentences": len(ordered),
        "tokens": sum(len(s.split()) for s in ordered),
        "sha256": file_sha256(output),
    }
    write_json(manifest_path, manifest)
    return manifest


def create_subsets(
    sentences: list[tuple[str, str]], budgets: list[int], directory: Path
) -> list[dict[str, Any]]:
    """Write strict nested whole-sentence prefixes; do not skip a long boundary sentence."""
    directory.mkdir(parents=True, exist_ok=True)
    total = sum(len(sentence.split()) for sentence, _ in sentences)
    if total < max(budgets):
        raise ValueError(
            f"Only {total:,} words available; need {max(budgets):,}. "
            "Increase preprocessing.sentences_per_domain."
        )
    results = []
    for budget in sorted(budgets):
        path = directory / f"tokens_{budget}.txt"
        temporary = path.with_suffix(".tmp")
        tokens = 0
        domains: Counter[str] = Counter()
        domain_tokens: Counter[str] = Counter()
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for sentence, domain in sentences:
                words = len(sentence.split())
                if tokens + words > budget:
                    break
                stream.write(sentence + "\n")
                tokens += words
                domains[domain] += 1
                domain_tokens[domain] += words
        if not tokens:
            temporary.unlink()
            raise ValueError(f"Token budget {budget} cannot contain the first sentence")
        temporary.replace(path)
        metadata = {
            "budget": budget,
            "actual_tokens": tokens,
            "sentences": sum(domains.values()),
            "domain_sentences": dict(domains),
            "domain_tokens": dict(domain_tokens),
            "filename": path.name,
            "sha256": file_sha256(path),
        }
        write_json(path.with_suffix(".json"), metadata)
        results.append(metadata)
    return results


def prepare_corpus(
    config: dict[str, Any], workers: int = 2, sample_only: bool = False
) -> dict[str, Any]:
    """Prepare every domain once, then interleave samples and export reusable subsets."""
    if config["preprocessing"]["version"] != VERSION:
        raise ValueError(f"Preprocessing version must match implementation: {VERSION}")
    records = load_dataset(
        resolve_path(config, "evaluation"), config["evaluation"]["expected_records"]
    )
    excluded = exclusion_sentences(records)
    processed = resolve_path(config, "processed_corpus")
    processed.parent.mkdir(parents=True, exist_ok=True)
    # Different mixtures can reuse the same verified domain sample without rescanning.
    sample_cache = config["corpus"].get("sample_cache")
    sample_directory = (
        (config["_root"] / sample_cache).resolve() if sample_cache else processed.parent
    )
    jobs = []
    for source in config["corpus"]["sources"]:
        archive = resolve_path(config, "raw_corpus") / source["filename"]
        if not archive.is_file():
            if sample_only:
                logging.info("Not downloaded yet; skipping %s", source["domain"])
                continue
            raise FileNotFoundError(f"Download missing archive first: {archive}")
        jobs.append(
            (
                archive,
                sample_directory / f"{source['domain']}.sample.txt",
                excluded,
                config["preprocessing"],
                config["seed"],
                source.get("sha256"),
            )
        )
    if workers == 1:
        manifests = [sample_domain(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_initialize_worker_logging
        ) as pool:
            futures = [pool.submit(sample_domain, *job) for job in jobs]
            manifests = [future.result() for future in futures]

    if sample_only:
        return {"domain_samples": manifests}

    # Equal sentence contribution per domain, not equal token counts. Preserve all
    # configured samples; a hash-sorted order within domains makes prefixes reproducible.
    streams = [job[1].read_text(encoding="utf-8").splitlines() for job in jobs]
    unique: set[str] = set()
    sentences: list[tuple[str, str]] = []
    for row in zip_longest(*streams):
        for source, sentence in zip(config["corpus"]["sources"], row, strict=True):
            if sentence and sentence not in unique:
                unique.add(sentence)
                sentences.append((sentence, source["domain"]))
    subsets = create_subsets(
        sentences, config["experiments"]["corpus_word_tokens"], resolve_path(config, "subsets")
    )
    temporary = processed.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for sentence, _ in sentences:
            stream.write(sentence + "\n")
    temporary.replace(processed)
    metadata = {
        "normalizer": VERSION,
        "seed": config["seed"],
        "evaluation_sha256": file_sha256(resolve_path(config, "evaluation")),
        "domain_manifests": dict(
            zip([s["domain"] for s in config["corpus"]["sources"]], manifests, strict=True)
        ),
        "mixture": "round_robin_equal_sentences; global normalized deduplication",
        "sentences": len(sentences),
        "sha256": file_sha256(processed),
        "subsets": subsets,
    }
    write_json(processed.with_suffix(".json"), metadata)
    logging.info("Prepared %d unique sentences and %d nested subsets", len(sentences), len(subsets))
    return metadata
