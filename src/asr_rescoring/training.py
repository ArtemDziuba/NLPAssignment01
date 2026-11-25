"""Train KenLM with provenance, verified reuse, and explicit unigram compatibility."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import environment, file_sha256, fingerprint, resolve_path, write_json


def toolchain(config: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint both native executables and the Python binding for safe reuse."""
    import kenlm

    result: dict[str, Any] = {"binding_sha256": file_sha256(Path(kenlm.__file__))}
    for name in ("lmplz", "build_binary"):
        executable = shutil.which(config["kenlm"][name])
        if not executable:
            raise RuntimeError(f"KenLM executable not on PATH: {config['kenlm'][name]}")
        result[name] = {"path": executable, "sha256": file_sha256(Path(executable))}
    provenance = Path(config["kenlm"].get("toolchain_manifest", "/opt/kenlm/toolchain.json"))
    if provenance.exists():
        result["build"] = json.loads(provenance.read_text(encoding="utf-8"))
    return result


def unigram_container(source: Path, destination: Path) -> dict[str, float]:
    """Wrap true unigram probabilities in ARPA with zero observed bigrams.

    KenLM 0.3.0 rejects an order-one ARPA in its ngram loader. This adds only
    an empty second-order section and zero backoffs. It does not train a bigram.
    Scoring explicitly resets context for every token.
    """
    probabilities: dict[str, float] = {}
    entries: list[tuple[str, str]] = []
    active = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip() == "\\1-grams:":
            active = True
            continue
        if line.startswith("\\") and active:
            break
        if active and line.strip():
            fields = line.split()
            if len(fields) < 2:
                raise ValueError("Malformed unigram ARPA")
            logprob, word = fields[:2]
            probabilities[word] = float(logprob)
            entries.append((logprob, word))
    if not {"<s>", "</s>", "<unk>"} <= probabilities.keys():
        raise ValueError("Unigram ARPA lacks required boundary/unknown entries")
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"\\data\\\nngram 1={len(entries)}\nngram 2=0\n\n\\1-grams:\n")
        for logprob, word in entries:
            stream.write(f"{logprob}\t{word}\t0\n")
        stream.write("\n\\2-grams:\n\n\\end\\\n")
    return probabilities


def train_model(
    config: dict[str, Any], subset: Path, order: int, budget: int, force: bool = False
) -> dict[str, Any]:
    """Train one model atomically; reuse only checksum- and parameter-matching artifacts."""
    if order not in (1, 2, 3):
        raise ValueError("Unsupported n-gram order")
    if not subset.is_file():
        raise FileNotFoundError(f"Prepare the subset first: {subset}")
    tools = toolchain(config)
    identity = {
        "subset_sha256": file_sha256(subset),
        "effective_order": order,
        "kenlm": config["kenlm"],
        "preprocessing": config["preprocessing"],
        "toolchain": tools,
        "trainer_version": 1,
        "trainer_sha256": file_sha256(Path(__file__)),
        "evaluation_exclusion_sha256": (
            file_sha256(resolve_path(config, "evaluation"))
            if resolve_path(config, "evaluation").is_file()
            else None
        ),
    }
    model_dir = resolve_path(config, "models") / f"order_{order}_tokens_{budget}"
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") == identity and all(
            (model_dir / name).is_file() and file_sha256(model_dir / name) == checksum
            for name, checksum in previous["artifact_sha256"].items()
        ):
            logging.info("Reusing model %s", model_dir.name)
            return previous

    log_path = model_dir / "training.log"
    started = time.perf_counter()
    # KenLM uses mmap and on-disk sorting: keep working files on the Linux filesystem.
    with tempfile.TemporaryDirectory(prefix="asr_training_") as temporary:
        work = Path(temporary)
        arpa = work / "model.arpa"
        command = [
            tools["lmplz"]["path"],
            "-o",
            str(order),
            "-S",
            config["kenlm"]["memory"],
            "-T",
            str(work),
            "--text",
            str(subset),
            "--arpa",
            str(arpa),
        ]
        if config["kenlm"].get("discount_fallback", False):
            command.append("--discount_fallback")
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=log)
            load_arpa = arpa
            if order == 1:
                if config["kenlm"]["unigram_compatibility"] != "zero_bigram_container":
                    raise RuntimeError("Select documented zero_bigram_container for KenLM 0.3.0")
                load_arpa = work / "model.container.arpa"
                unigram_container(arpa, load_arpa)
            binary = work / "model.binary"
            build_command = [
                tools["build_binary"]["path"],
                config["kenlm"]["binary_type"],
                str(load_arpa),
                str(binary),
            ]
            log.write("COMMAND: " + json.dumps(build_command) + "\n")
            log.flush()
            subprocess.run(build_command, check=True, stdout=log, stderr=log)
        # Validate load and finite scoring before committing a model.
        from .scoring import KenLMScorer

        scorer = KenLMScorer(binary, order)
        probe = next(line.strip() for line in subset.open(encoding="utf-8") if line.strip())
        import math

        if not math.isfinite(scorer.score(probe)):
            raise RuntimeError("Trained model failed scoring validation")
        del scorer
        artifacts = [arpa, binary] + ([load_arpa] if order == 1 else [])
        for artifact in artifacts:
            destination = model_dir / artifact.name
            pending = destination.with_suffix(destination.suffix + ".pending")
            shutil.copyfile(artifact, pending)
            pending.replace(destination)
    elapsed = time.perf_counter() - started
    manifest = {
        "identity": identity,
        "model_id": model_dir.name,
        "order": order,
        "budget": budget,
        "actual_tokens": sum(len(line.split()) for line in subset.open(encoding="utf-8")),
        "binary": "model.binary",
        "training_seconds": elapsed,
        "environment": environment(),
        "container_order": 2 if order == 1 else order,
        "artifact_sha256": {p.name: file_sha256(p) for p in model_dir.glob("model.*")},
        "artifact_bytes": {p.name: p.stat().st_size for p in model_dir.glob("model.*")},
        "identity_hash": fingerprint(identity),
    }
    write_json(manifest_path, manifest)
    logging.info("Trained %s in %.3f seconds", model_dir.name, elapsed)
    return manifest


def verify_kenlm(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Actually test native unigram training/loading and all configured effective orders."""
    import kenlm

    from .scoring import KenLMScorer

    report: dict[str, Any] = {"toolchain": toolchain(config), "orders": {}}
    with tempfile.TemporaryDirectory(prefix="kenlm_probe_") as directory:
        work = Path(directory)
        # Deliberately artificial fixture; fallback discounts are explicit and smoke-only.
        fixture = work / "fixture.txt"
        fixture.write_text(
            "київ місто україни\nльвів місто україни\n"
            "київ гарне місто\nукраїна має міста\n"
            "ми любимо україну\nмісто має парк\n" * 4,
            encoding="utf-8",
        )
        import copy

        smoke = copy.deepcopy(config)
        smoke["_root"] = work
        smoke["paths"]["models"] = "models"
        smoke["kenlm"]["memory"] = "128M"
        smoke["kenlm"]["discount_fallback"] = True
        for order in (1, 2, 3):
            manifest = train_model(smoke, fixture, order, 100)
            model_dir = work / "models" / manifest["model_id"]
            if order == 1:
                try:
                    kenlm.Model(str(model_dir / "model.arpa"))
                    native_load = True
                except OSError:
                    native_load = False
                values = unigram_container(
                    model_dir / "model.arpa", model_dir / "probe.container.arpa"
                )
                scorer = KenLMScorer(model_dir / "model.binary", 1)
                sentence = "київ невідомеслово"
                expected = sum(values.get(t, values["<unk>"]) for t in sentence.split() + ["</s>"])
                actual = scorer.score(sentence)
                if abs(expected - actual) > 1e-4:
                    raise RuntimeError("Unigram compatibility changed ARPA probabilities")
                report["native_unigram_arpa_load"] = native_load
                report["unigram_expected_logprob"] = expected
                report["unigram_actual_logprob"] = actual
            else:
                scorer = KenLMScorer(model_dir / "model.binary", order)
            report["orders"][str(order)] = {
                "trained": True,
                "binary_loaded": True,
                "log10_probability": scorer.score("київ місто україни"),
                "container_order": manifest["container_order"],
            }
            del scorer
    write_json(output, report)
    return report
