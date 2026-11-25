"""Build the full assignment-structured Markdown report from saved measurements."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from asr_rescoring.config import PROJECT_ROOT, file_sha256, load_config, resolve_path
from asr_rescoring.dataset import load_dataset
from asr_rescoring.llm_scoring import make_request, parse_ranking
from asr_rescoring.metrics import aggregate_metrics
from asr_rescoring.preprocessing import normalize_text

STUDIES = [
    ("all_five", "experiments", "20261124_main", "All five domains"),
    ("news_only", "news_only", "20261124_news_only", "News only"),
    ("wiki_only", "wiki_only", "20261124_wiki_only", "Wikipedia only"),
    ("news_wiki", "news_wiki", "20261124_news_wiki", "News + Wikipedia"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    return "\n".join(
        [
            "| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    )


def percent(value: Any, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def link(path: Path | str, label: str) -> str:
    relative = Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    return f"[{label}](../{relative})"


def build_report() -> Path:
    root = PROJECT_ROOT
    configs = {key: load_config(root / "configs" / f"{name}.yaml") for key, name, _, _ in STUDIES}
    labels = {key: label for key, _, _, label in STUDIES}
    analyses = {
        key: read_json(root / "results/metrics" / f"{run}_analysis.json")
        for key, _, run, _ in STUDIES
    }
    prepared = {
        key: read_json(resolve_path(configs[key], "processed_corpus").with_suffix(".json"))
        for key, _, _, _ in STUDIES
    }
    comparison_path = root / "results/metrics/20261124_domain_comparison.csv"
    with comparison_path.open(encoding="utf-8") as stream:
        classical = list(csv.DictReader(stream))
    provenance = read_json(root / "report/20261124_domain_comparison.json")
    assert len(classical) == 36 and file_sha256(comparison_path) == provenance["csv_sha256"]
    by = {(r["corpus"], int(r["order"]), int(r["budget"])): r for r in classical}
    evaluation = resolve_path(configs["all_five"], "evaluation")
    records = load_dataset(evaluation)
    evaluation_hash = file_sha256(evaluation)
    results, predictions, manifests = {}, {}, {}
    for key, _, run, _ in STUDIES:
        for path in sorted((root / "results/metrics").glob(f"{run}_order_*.json")):
            result = read_json(path)
            identity = key, result["order"], result["budget"]
            assert result["evaluation_sha256"] == evaluation_hash
            results[identity] = result
            prediction_path = root / "results/predictions" / result["predictions_file"]
            predictions[identity] = [
                json.loads(line) for line in prediction_path.read_text("utf-8").splitlines()
            ]
            assert len(predictions[identity]) == 500
            assert (
                result["metrics"]["raw_exact_match_accuracy"]
                == result["metrics"]["exact_match_accuracy"]
            )
            assert result["metrics"]["oracle_accuracy"] == 1
            model_path = resolve_path(configs[key], "models") / result["model_id"] / "manifest.json"
            assert file_sha256(model_path) == result["model_manifest_sha256"]
            manifests[identity] = read_json(model_path)
    assert len(results) == 36

    llm_paths = [
        root / "results/metrics/20261124_qwen35_4b_qwen3.5_4b.json",
        next((root / "results/metrics/20261124_mamaylm_4b_hf.co").rglob("*Q4_K_M.json")),
    ]
    llms = [read_json(path) for path in llm_paths]
    metadata = [read_json(Path(r["metadata_file"])) for r in llms]
    for result, meta in zip(llms, metadata, strict=True):
        assert (
            meta["status"] == "complete" and result["identity"]["dataset_sha256"] == evaluation_hash
        )
        path = Path(result["predictions_file"])
        assert file_sha256(path) == result["predictions_sha256"]
        saved = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        events = [
            json.loads(line)
            for line in Path(meta["paths"]["requests"]).read_text("utf-8").splitlines()
        ]
        valid = {e["utt_id"]: e for e in events if not e["warmup"] and "error" not in e}
        assert len(saved) == len(records)
        for record, prediction in zip(records, saved, strict=True):
            payload, order = make_request(
                result["identity"]["llm_settings"], record.utt_id, record.hypotheses, 42
            )
            event = valid[record.utt_id]
            assert event["request"] == payload and prediction["presentation_order"] == order
            assert parse_ranking(event["response"], order) == prediction["ranking_indices"]
            assert (
                prediction["utt_id"] == record.utt_id
                and prediction["reference"] == record.reference
            )
            assert prediction["selected"] == record.hypotheses[prediction["ranking_indices"][0]]
            assert prediction["normalized_ranking"] == [
                normalize_text(record.hypotheses[i]) for i in prediction["ranking_indices"]
            ]
        calculated = aggregate_metrics(
            [normalize_text(r.reference) for r in records], [p["normalized_ranking"] for p in saved]
        )
        assert all(
            abs(value - result["metrics"][name]) < 1e-12 for name, value in calculated.items()
        )
        assert (
            abs(sum(p["inference_seconds"] for p in saved) - result["metrics"]["inference_seconds"])
            < 1e-8
        )

    llm_report = (root / "report/20261124_llm_experiment_results.md").read_text("utf-8")

    def llm_section(number: int) -> str:
        match = re.search(rf"^## {number}\. .*?\n(.*?)(?=^## |\Z)", llm_report, re.M | re.S)
        if not match:
            raise ValueError(f"Missing authored LLM report section {number}")
        return re.sub(r"^### ", "#### ", match.group(1).strip(), flags=re.M)

    sections: list[str] = []

    def add(text: str) -> None:
        sections.append(text.strip())

    add("""# Improving Speech Recognition Output Using a Classical Language Model

## Full experimental report

**Experiments:** 24 November 2026. **Report prepared:** 25 November 2026.

This Markdown report follows the activity structure and evaluation criteria in the [official assignment](../readme.md). It covers all completed experiments: 36 KenLM models across four corpus selections, plus local Qwen3.5-4B and MamayLM-4B ranking experiments. It consolidates the earlier experiment reports. The required 5-7 page PDF and source-code delivery are separate deliverables; this comprehensive Markdown source includes appendices that will need editing for that page limit.

### Overview

The task is to select the best sentence from an ASR-like candidate list. A unigram uses individual-word statistics, a bigram adds the preceding word, and a trigram adds the preceding two words. Context can help identify grammatical endings and implausible substitutions. A locally plausible sentence can still differ from the reference, and length-normalized probability can favor repeated words.

The best recorded normalized accuracy was **92.8% (464/500)**, reached by news-only bigram and trigram models at a nominal 10M-word budget. The original five-domain trigram reached **90.2%**; Wikipedia-only and news-plus-Wikipedia trigrams reached **86.0%** and **91.6%**. Qwen3.5-4B reached **88.4%** and MamayLM-4B **18.6%** under a separately documented prompted-ranking protocol. Candidate-order bias, exact-score ties and measurement scopes limit interpretation.

## Tools and Resources

- KenLM 0.3.0, with recorded native build, executable and Python-binding hashes; [training code](../src/asr_rescoring/training.py) and [Docker environment](../Dockerfile).
- UberText 2.0 sentence-split Ukrainian exports: court, fiction, news, social and Wikipedia. Source URLs and pinned archive checksums are in [experiments.yaml](../configs/experiments.yaml). The Wikipedia experiment uses this export, not an independently downloaded Wikipedia dump.
- Python with `uv`, YAML configurations, [pyproject.toml](../pyproject.toml) and [uv.lock](../uv.lock).
- Ollama 0.40.17 for the optional pretrained local LLMs.
- Reusable [source modules](../src/asr_rescoring), [scripts](../scripts) and [notebooks](../notebooks).

## 1. Language Model Training

### 1.1 Evaluation data and controls
""")
    add(f"""The fixed evaluation input [{evaluation.name}](../data/evaluation/{evaluation.name}) contains **500 unique utterances**, each with a reference and eight hypotheses: **4,000 candidate positions per model**. The input is synthetic ASR-like text; no audio is decoded. All models use the same evaluation records.

The file moved into `data/evaluation/` on 25 November without a content change. Current configs use the new path; historical snapshots retain the original path. Its SHA-256 is:

```text
{evaluation_hash}
```

### 1.2 Training corpus and sampling
""")
    archive_rows = []
    archive_total = 0
    for source in configs["all_five"]["corpus"]["sources"]:
        domain = source["domain"]
        size = (resolve_path(configs["all_five"], "raw_corpus") / source["filename"]).stat().st_size
        archive_total += size
        domain_meta = prepared["all_five"]["domain_manifests"][domain]
        archive_rows.append(
            [
                domain,
                f"{size / 1e6:.2f}",
                f"{domain_meta['counts']['input_sentences']:,}",
                f"{domain_meta['sentences']:,}",
                domain_meta["counts"].get("rejected_evaluation_overlap", 0),
            ]
        )
    add(
        table(
            [
                "Domain",
                "Compressed archive MB",
                "Sentences scanned",
                "Retained original sample",
                "Exact evaluation-overlap occurrences excluded",
            ],
            archive_rows,
        )
    )
    add(f"""The five archives occupy **{archive_total / 1e9:.3f} decimal GB**. Full archives were scanned; a deterministic bounded sample retained 200,000 sentences per domain in the original grid. After global deduplication, its prepared pool contains **{prepared["all_five"]["sentences"]:,} sentences**.

The additional news/Wikipedia studies use a separate **750,000-sentence sample per included domain**. The original 200,000-sentence news and Wikipedia samples each supply only about 3.2M words, insufficient for a 10M-word single-domain experiment. This sampling-capacity change affects available training text; normalization and evaluation rules remain fixed.

### 1.3 Preprocessing and leakage prevention

1. Preserve the official corpus sentence boundaries and keep evaluation candidates whole.
2. Apply NFC, canonicalize apostrophes, lowercase, extract word/digit tokens, and collapse whitespace. Preserve internal apostrophes; remove other punctuation and symbols. Hyphens become token boundaries.
3. For training only, reject empty/non-Cyrillic lines, obvious URL/HTML content and sentences outside 2-150 normalized words.
4. Exclude exact normalized evaluation references and hypotheses before sampling; deduplicate retained sentences, including across mixed domains.
5. Assign deterministic hash priorities using seed 42 and normalized sentence text; retain the lowest-priority unique sentences, sorted by priority.
6. Interleave domains round-robin by sentence. Take nested whole-sentence prefixes for nominal 100K, 1M and 10M word budgets, stopping before a sentence would exceed the budget.

Training and evaluation share normalization. There is no spelling correction, transliteration, stemming or recognition-error repair. Exact exclusion does not rule out paraphrases, partial overlap or related source material. The 39 original news exclusions count archive occurrences, not distinct evaluation records.

Mixed-domain interleaving balances sentence opportunities, not words. Actual word shares at the largest budget are:
""")
    shares = []
    for key in ("all_five", "news_wiki"):
        subset = prepared[key]["subsets"][-1]
        for domain, words in subset["domain_tokens"].items():
            shares.append(
                [labels[key], domain, f"{words:,}", percent(words / subset["actual_tokens"], 2)]
            )
    add(table(["Selection", "Domain", "Words at nominal 10M", "Word share"], shares))
    add("""Court supplies about 40% of the original mixture's words despite roughly equal sentence counts. Under a fixed total budget, adding domains reduces news exposure. These shares matter when interpreting domain effects.

### 1.4 Actual training dataset sizes
""")
    add(
        table(
            ["Selection", "Nominal words", "Actual words", "Sentences"],
            [
                [label, f"{s['budget']:,}", f"{s['actual_tokens']:,}", f"{s['sentences']:,}"]
                for key, _, _, label in STUDIES
                for s in prepared[key]["subsets"]
            ],
        )
    )
    add("""Counts can be slightly below the nominal budget because whole sentences are preserved. Within each corpus selection, smaller subsets are strict prefixes of larger ones. Configurations share orders, budgets and seed; they do not share identical constituent sentences across different domains.

### 1.5 Training configuration and unigram compatibility

Models were trained sequentially using `lmplz -o ORDER -S 2G`, with Linux-local scratch files, then converted using `build_binary probing`. ARPA and binary artifacts were retained. Real experiments use no pruning or fallback discounts. Bigram/trigram estimation uses KenLM's modified Kneser-Ney estimator. Logs/manifests retain native commands, training time, actual tokens, subset/trainer/toolchain identities and artifact hashes.

An initial native attempt using Windows-mounted scratch failed; Linux-local temporary files resolved it. Successful training times exclude that failed attempt and exclude corpus download/preparation.

KenLM 0.3.0 estimates unigrams but its generic loader rejects pure order-one ARPA. The explicit compatibility mode preserves original unigram probabilities in a container with zero observed bigrams and zero backoffs. Scoring resets context for each word and EOS. Effective order remains **one**; no bigram observations are trained. Native checks verified the scored sum against source ARPA, including an OOV token. These per-token Python calls also affect unigram latency.

The full grid is **4 corpus selections x 3 orders x 3 budgets = 36 classical models**. The two optional LLMs use pretrained weights and have no controlled pretraining-word budget or training time in this experiment.

## 2. Sentence Scoring

### 2.1 Log probability and assignment perplexity

Each candidate uses the assignment's BOS/EOS-enabled log10 probability and word-only denominator:

```python
normalized = normalize_text(candidate)
log10_probability = scorer.score(normalized, bos=True, eos=True)
N = len(normalized.split())
cross_entropy = -log10_probability / N
perplexity = 10 ** cross_entropy
```

EOS contributes to probability, while `N` excludes boundary tokens. KenLM's built-in perplexity has a different denominator and is deliberately not used. Ranking directly by cross-entropy gives the same ordering while avoiding perplexity overflow. Select the minimum, break exact ties by original index, and reject empty normalized candidates. Save raw/normalized text, index, word count, log probability, cross-entropy and perplexity for every candidate.

### 2.2 Normalization, length and unknown words

Normalization can make distinct raw candidates equivalent: `$2` and `$ 2` both yield the token `2`; a spelling change within a remaining word still matters. No candidate is repaired.

Dividing by word count permits different lengths but can favor an extra predictable repeated word: adding a low-cost token can reduce average cost while making the sentence incorrect. This is demonstrated below.

KenLM maps OOV words to an unknown-word probability. Different unseen spellings and higher-order backoff can therefore produce exact ties. OOV rates count unknown word occurrences, excluding implicit boundaries, in references, hypotheses and selected text. Improved coverage does not guarantee correct names or rankings.

## 3. Evaluation

### 3.1 Metrics

Primary normalized exact-match accuracy is the fraction of 500 records for which `normalize(selected) == normalize(reference)`. It is whole-sentence matching: any remaining word-content difference makes the sentence incorrect. Raw accuracy compares original strings including punctuation, case and spacing. Raw and normalized accuracy coincide for all 36 classical runs under their original-order tie rule. Both are retained for the LLMs because their shuffled rankings can choose different raw but equivalent normalized candidates.

Additional metrics are corpus WER, CER including spaces, MRR, oracle accuracy, OOV rates, timing and disk size. WER/CER divide total Levenshtein edits by total reference words/characters. MRR averages the reciprocal rank of the first normalized reference match. All 38 models have 100% oracle accuracy: a matching candidate exists in every record.

### 3.2 Baselines and source ordering
""")
    baseline = analyses["all_five"]
    add(
        table(
            ["Baseline", "Accuracy", "WER", "CER", "MRR"],
            [
                [
                    name,
                    percent(m["exact_match_accuracy"]),
                    percent(m["wer"], 4),
                    percent(m["cer"], 4),
                    f"{m['mrr']:.5f}",
                ]
                for name, m in baseline["baselines"].items()
            ],
        )
    )
    add(f"""The reference is originally candidate zero in every record, so original-first selection reaches 100% without language understanding. The random baseline uses deterministic per-ID permutations; its observed 15.6% differs from the analytical uniform-random expectation, **{percent(baseline["uniform_random_expected_accuracy"], 3)}**. There are **{baseline["records_with_multiple_normalized_matches"]} records with multiple normalized matches**, totaling **{baseline["matching_candidates"]} matching candidate positions**.

Classical models preserve original order and first-index ties. LLMs see shuffled presentation with hidden original indices. The accuracy comparison therefore includes different tie/presentation treatments and does not show that either method beats the perfect positional baseline.

### 3.3 N-gram comparison: all four experimental grids

The four tables cover every classical configuration. Percentages are normalized accuracy on the same 500 records.
""")
    explanations = {
        "all_five": "All orders improve with data in this grid. At 10M, the trigram leads the unigram by 12.8 percentage points but the bigram by only 0.8: most of the observed context gain is already present at order two.",
        "news_only": "Bigrams slightly outperform trigrams at 100K and 1M, then tie at 92.8% at 10M. Higher order is not automatically better. At the largest budget, the bigram also has lower WER and a smaller binary.",
        "wiki_only": "Bigrams lead trigrams at 100K and 1M; the trigram leads at 10M. The unigram dips from 74.2% to 73.6% between 100K and 1M before reaching 76.6%. More training text is not strictly monotonic at every intermediate point, although every order improves from the smallest to largest budget.",
        "news_wiki": "All orders improve with data. The trigram maintains a small lead over the bigram and reaches 91.6% at 10M. Adding Wikipedia does not exceed the best news-only result at the same total word budget.",
    }
    for key, _, run, label in STUDIES:
        add(f"#### {label}: `{run}`")
        add(
            table(
                ["Nominal words", "Actual words", "Unigram", "Bigram", "Trigram"],
                [
                    [
                        f"{b:,}",
                        f"{int(by[key, 1, b]['actual_tokens']):,}",
                        *[percent(by[key, n, b]["exact_match_accuracy"]) for n in (1, 2, 3)],
                    ]
                    for b in (100000, 1000000, 10000000)
                ],
            )
        )
        add(explanations[key])
    add("### 3.4 Training dataset size analysis")
    add(
        table(
            ["Corpus", "Unigram gain, 100K to 10M", "Bigram gain", "Trigram gain"],
            [
                [
                    label,
                    *[
                        f"{100 * (float(by[key, n, 10000000]['exact_match_accuracy']) - float(by[key, n, 100000]['exact_match_accuracy'])):.1f} pp"
                        for n in (1, 2, 3)
                    ],
                ]
                for key, _, _, label in STUDIES
            ],
        )
    )
    add("""Contextual models gain more than unigrams from the 100-fold nominal increase in every corpus. More observations can improve context estimates and vocabulary coverage. However, one deterministic sample ordering is used, and the Wikipedia unigram dip shows that larger subsets do not guarantee improvement at each point. Frequency shifts, sparse contexts, backoff and ties can affect selection; these runs do not isolate a cause for every fluctuation.

The original five-domain trigram shows the following coverage relationship:
""")
    add(
        table(
            ["Budget", "Accuracy", "Reference OOV", "Hypotheses OOV", "Selected OOV", "Binary MiB"],
            [
                [
                    f"{b:,}",
                    percent(by["all_five", 3, b]["exact_match_accuracy"]),
                    *[
                        percent(by["all_five", 3, b][k], 3)
                        for k in ("oov_references", "oov_hypotheses", "oov_selected")
                    ],
                    f"{int(by['all_five', 3, b]['binary_bytes']) / 2**20:.2f}",
                ]
                for b in (100000, 1000000, 10000000)
            ],
        )
    )
    add("""Reference OOV falls from about 28.08% to 2.32% while accuracy rises. This supports coverage as part of the size explanation. In-vocabulary substitutions and repetition remain, so coverage alone cannot explain correctness.

### 3.5 Domain comparison at a fixed budget
""")
    add(
        table(
            ["Order", "All five", "News", "Wikipedia", "News + Wikipedia"],
            [
                [
                    n,
                    *[
                        percent(by[key, n, 10000000]["exact_match_accuracy"])
                        for key, _, _, _ in STUDIES
                    ],
                ]
                for n in (1, 2, 3)
            ],
        )
    )
    add("""News-only leads for each order at 10M. Closer genre and vocabulary alignment with news-like evaluation titles is a plausible explanation, not proof that news is universally the best Ukrainian ASR corpus. At fixed budget, the original mixture allocates many words to court, and news-plus-Wikipedia replaces about half the news exposure with Wikipedia. Neither more domains nor more general text automatically improves the recorded accuracy.

The following representatives maximize recorded accuracy within each grid. For the tied news models, the smaller bigram is shown as a practical representative, not a statistically established winner.
""")
    best = []
    for key, _, _, label in STUDIES:
        n, b = map(
            int, re.fullmatch(r"order_(\d+)_tokens_(\d+)", analyses[key]["best_model"]).groups()
        )
        best.append((key, n, b, by[key, n, b]))
    add(
        table(
            ["Corpus", "Order", "Accuracy / correct", "WER", "CER", "MRR", "Binary MiB"],
            [
                [
                    labels[key],
                    n,
                    f"{percent(r['exact_match_accuracy'])} / {round(float(r['exact_match_accuracy']) * 500)}",
                    percent(r["wer"], 4),
                    percent(r["cer"], 4),
                    f"{float(r['mrr']):.5f}",
                    f"{int(r['binary_bytes']) / 2**20:.2f}",
                ]
                for key, n, b, r in best
            ],
        )
    )
    bi, tri = predictions["news_only", 2, 10000000], predictions["news_only", 3, 10000000]
    bi_only = sum(
        a["normalized_selected"] == a["normalized_reference"]
        and b["normalized_selected"] != b["normalized_reference"]
        for a, b in zip(bi, tri, strict=True)
    )
    tri_only = sum(
        b["normalized_selected"] == b["normalized_reference"]
        and a["normalized_selected"] != a["normalized_reference"]
        for a, b in zip(bi, tri, strict=True)
    )
    add(f"""Both news models select 464 correct sentences. The bigram alone is correct in **{bi_only} records** and the trigram alone in **{tri_only} records**. WER is {percent(by["news_only", 2, 10000000]["wer"], 4)} versus {percent(by["news_only", 3, 10000000]["wer"], 4)}; binaries are {int(by["news_only", 2, 10000000]["binary_bytes"]) / 2**20:.2f} versus {int(by["news_only", 3, 10000000]["binary_bytes"]) / 2**20:.2f} MiB. Equal sentence accuracy need not mean identical predictions or error severity.

### 3.6 Qualitative successes and failures

These examples use the original all-five-domain 10M models. Numerical costs are the trigram's `-log10(P)/N` for each displayed alternative; lower is better.
""")
    tri_by_id = {r["utt_id"]: r for r in predictions["all_five", 3, 10000000]}
    uni_by_id = {r["utt_id"]: r for r in predictions["all_five", 1, 10000000]}
    examples = [
        (
            "Context resolves a case ending",
            "ua_title_0005",
            True,
            "The unigram selects an incorrect case ending while the trigram favors the reference. This is consistent with contextual statistics resolving a word-form ambiguity.",
        ),
        (
            "Context distinguishes a real-word substitution",
            "ua_title_0246",
            True,
            "The alternatives contain different real words. The unigram's frequent word is incorrect in this verb-object context, while the trigram selects the reference.",
        ),
        (
            "An extra repeated word reduces average cost",
            "ua_title_0024",
            False,
            "The selected sentence repeats a word but obtains a lower average cost. This is a failure of the length-normalized objective despite vocabulary coverage.",
        ),
        (
            "Apostrophe fragmentation changes tokenization",
            "ua_title_0317",
            False,
            "An internal apostrophe stays within a word; a fragmented apostrophe creates token boundaries. Changed token count and context alter perplexity. The preprocessing deliberately does not repair this error.",
        ),
        (
            "Vocabulary coverage does not guarantee a correct name",
            "ua_title_0049",
            False,
            "The wrong name spelling is preferred. Stored vocabulary diagnostics identify both name spellings as known words. Local probability does not guarantee the factual spelling of a named entity.",
        ),
    ]
    for title, identifier, success, explanation in examples:
        row = tri_by_id[identifier]
        alternative = uni_by_id[identifier]["selected"] if success else row["selected"]
        ref = next(s for s in row["ranking"] if s["index"] == 0)
        alt = next(s for s in row["ranking"] if s["normalized"] == normalize_text(alternative))
        assert (row["normalized_selected"] == row["normalized_reference"]) == success
        add(f"#### {title}: `{identifier}`")
        add(
            table(
                ["Item", "Saved sentence", "Trigram cost"],
                [
                    ["Reference", row["reference"], f"{ref['cross_entropy']:.6f}"],
                    [
                        "Unigram selection" if success else "Trigram selection",
                        alternative,
                        f"{alt['cross_entropy']:.6f}",
                    ],
                ],
            )
        )
        add(explanation)
    add(
        "The following deterministic pattern diagnostics summarize errors in the best representative of each corpus. Other/combined errors have not all been manually annotated."
    )
    add(
        table(
            [
                "Corpus / order",
                "Total errors",
                "Adjacent repetition only",
                "Apostrophe fragmentation only",
                "Other / combined",
                "Longer than reference",
            ],
            [
                [
                    f"{labels[key]} / {n}",
                    analyses[key]["best_errors"],
                    analyses[key]["best_error_categories"].get("adjacent_repetition_only", 0),
                    analyses[key]["best_error_categories"].get("apostrophe_fragmentation_only", 0),
                    analyses[key]["best_error_categories"].get("other_or_combined", 0),
                    analyses[key]["best_error_length_changes"].get("longer", 0),
                ]
                for key, n, b, r in best
            ],
        )
    )
    add("""Repetition is the largest identified category in each selected model. A repetition penalty would be a new scoring method and has not been silently added.

### 3.7 Exact-score ties and uncertainty

Gold-first order benefits classical first-index decisions when distinct normalized sentences tie for best score. The following analytical diagnostics use saved scores; they do not change official predictions or represent new scored runs.
""")
    tie_rows = []
    for key, n, b, r in best:
        tie = analyses[key]["tie_sensitivity"][f"{n}gram_{b}"]
        tie_rows.append(
            [
                f"{labels[key]}, {n}-gram",
                percent(r["exact_match_accuracy"]),
                percent(tie["random_tie_accuracy"], 2),
                percent(tie["minimum_accuracy"]),
                tie["top_ties_between_distinct_normalized_texts"],
            ]
        )
    add(
        table(
            [
                "Representative",
                "Recorded accuracy",
                "Uniform top-tie expectation",
                "Minimum with different tie choices",
                "Distinct-text top ties",
            ],
            tie_rows,
        )
    )
    ci, paired = baseline["best_accuracy_wilson_95"], baseline["best_vs_bigram_paired"]
    add(f"""The original best trigram has an unadjusted 95% Wilson interval of **{percent(ci[0], 2)} to {percent(ci[1], 2)}**. This is descriptive for a fixed model and is not adjusted for selecting the best of nine. Against the original 10M bigram it has **{paired["wins"]} unique wins and {paired["losses"]} unique losses**, with exact paired McNemar p = **{paired["exact_mcnemar_p"]}**. The observed 0.8-point gain is not presented as established superiority.

### 3.8 Classical training and inference resources
""")
    add(
        table(
            [
                "Grid",
                "Sum of training seconds",
                "Inference seconds, nine models",
                "Binary storage MiB",
            ],
            [
                [
                    label,
                    f"{analyses[key]['total_training_seconds']:.3f}",
                    f"{analyses[key]['total_inference_seconds']:.6f}",
                    f"{analyses[key]['total_binary_bytes'] / 2**20:.2f}",
                ]
                for key, _, _, label in STUDIES
            ],
        )
    )
    add("""Classical inference includes normalization, scoring and ranking after one warmup; it excludes model loading, metrics, OOV scans and writes. These are single-run timings. Unigram context resets add Python overhead. Per-model training, loading and storage appear in Appendix B. No classical peak runtime-RAM measurement is inferred from these file sizes.

## 4. Bonus: Comparison with Modern Large Language Models

### 4.1 Method and evaluation controls

The already-installed Qwen3.5-4B and MamayLM-4B Q4_K_M models ran sequentially through local Ollama on the same machine. Each received an English ranking instruction and the same shuffled normalized Ukrainian candidates. Original indices and gold labels were hidden. The response had to be a complete permutation of all eight displayed candidate numbers; its first item was selected. No training/fine-tuning occurred, and neither run needed a retry.

This is prompted ranking, not teacher-forced candidate likelihood/perplexity. Differences from KenLM include the scoring objective, presentation, tokenizer and backend. No LLM candidate perplexity or word-level OOV rate is produced.

### 4.2 Accuracy and ranking quality
""")
    add(llm_section(4).split("#### Historical KenLM comparison")[0])
    add(
        "Qwen's 88.4% accuracy is below the best news KenLM's 92.8% and the original trigram's 90.2%. Its lower CER than the news models does not reverse the sentence-accuracy comparison: these metrics measure different aspects of error. MamayLM's result must be interpreted alongside its positional preference."
    )
    for title, number in [
        ("4.3 Raw versus normalized accuracy", 3),
        ("4.4 Positional preference", 5),
        ("4.5 Speed and token accounting", 6),
        ("4.6 Memory and storage", 7),
        ("4.7 Hardware, software and complete settings", 8),
        ("4.8 LLM qualitative examples", 9),
    ]:
        add(f"### {title}")
        add(llm_section(number))
    add("""## 5. Discussion and Limitations

Context and additional text improve recorded selection accuracy overall. The original grid improves with order and size, while the other grids contain higher-order and intermediate-size exceptions. News-only is strongest at 10M under this evaluation, plausibly reflecting genre alignment. These findings describe one fixed synthetic ASR-like set and one sampling seed.

The 10M news bigram is an attractive resource/accuracy representative: it ties the trigram on sentence accuracy, has slightly lower WER and a smaller binary. This does not show that trigrams are inherently inferior. Repetition, exact ties and factual name errors remain important failure modes.

Qwen performs useful ranking under the shared prompt but does not exceed the best classical accuracy. MamayLM mostly follows presentation order; a separately documented Ukrainian-prompt or likelihood experiment would be needed before broader conclusions about its capability.

Material limitations are:

1. Gold-first input yields a perfect positional baseline and favors classical first-index ties. LLM presentation is shuffled, so accuracy comparisons include different tie/presentation handling.
2. Normalization removes punctuation/symbol/spacing differences. Raw matching is retained to expose that distinction.
3. Exact training exclusion applies to the classical corpora, but cannot rule out paraphrases/related material. LLM pretraining overlap is unknown.
4. Equal total word budgets do not imply equal news exposure or vocabulary. The original and domain studies use different sampling capacities.
5. There are no real-audio decoding experiments, independent evaluation sets or repeated training seeds. The same 500 records support all comparisons.
6. Historical Docker/CPU scoring and current GPU/HTTP ranking have different timing scopes. No isolated architecture speedup or repeated-run latency confidence interval is claimed.
7. RAM/GPU measurements are sampled and have different attribution scopes. Matching KenLM runtime-RAM measurements remain pending; disk size cannot replace them.

## 6. Reproducibility, Assignment Coverage and Deliverables

### 6.1 Artifact provenance and validation

Preparation, training, scoring, evaluation and visualization are separate modules. Scripts/notebooks reuse them. Configuration paths resolve from the project root. Seeds, source/subset/model hashes, toolchain identities, raw predictions, scores/rankings, LLM prompts/digests and resource samples are saved. Current configs resolve the moved evaluation file; historical snapshots preserve the experiment-time path.

All 36 classical models were re-audited for report preparation: **18,000 predictions and 144,000 candidate scores**. Both LLM runs were checked against exact requests, raw responses, complete permutations, selected indices, checksums, metrics and timing: **1,000 predictions and 8,000 ranked positions**. Together these are **19,000 record predictions and 152,000 candidate positions**. No training or new model inference was performed to write this report.

The current native path/regression suite has 51 passing tests; two opt-in KenLM integration tests were skipped in that invocation. Earlier native Linux checks and model checksums are retained in run logs. This report's tables are generated from saved measurements, not invented or estimated outcomes.
""")
    add(
        f"- {link(comparison_path, 'Classical 36-model CSV')}; SHA-256 `{file_sha256(comparison_path)}`.\n- {link(root / 'results/metrics/20261124_qwen35_4b_all_models.csv', 'Combined 38-model CSV')}.\n- {link(root / 'report/20261124_llm_experiment_results.md', 'Detailed LLM-only Markdown report')}."
    )
    add(
        table(
            ["Run / selection", "Config", "Models", "Summary", "Audit"],
            [
                [
                    f"{run}: {label}",
                    link(root / "configs" / f"{name}.yaml", name),
                    link(resolve_path(configs[key], "models"), "Directory"),
                    link(root / "results/metrics" / f"{run}_summary.csv", "CSV"),
                    link(root / "results/metrics" / f"{run}_analysis.json", "JSON"),
                ]
                for key, name, run, label in STUDIES
            ],
        )
    )
    add(
        table(
            [
                "LLM run",
                "Metrics",
                "Predictions",
                "Requests/responses",
                "Resource samples",
                "Metadata",
            ],
            [
                [
                    r["run_id"],
                    link(path, "JSON"),
                    link(r["predictions_file"], "JSONL"),
                    link(md["paths"]["requests"], "JSONL"),
                    link(md["paths"]["resources"], "JSONL"),
                    link(r["metadata_file"], "JSON"),
                ]
                for r, md, path in zip(llms, metadata, llm_paths, strict=True)
            ],
        )
    )
    add("""### 6.2 Commands

The report-generation command reads saved evidence and writes this Markdown file. The new-run commands below were not executed while authoring it. Use fresh IDs for new inference runs.

```powershell
uv sync --locked --extra llm --group dev
uv run --locked --extra llm python scripts/build_full_report.py
uv run --locked --extra llm python -m pytest -q

# Inspect existing data/results without training
uv run --locked python scripts/check_readiness.py
uv run --locked python scripts/analyze_results.py --run-id 20261124_main
uv run --locked python scripts/run_domain_experiments.py --compare-only

# NEW LLM runs, only when deliberately requested
uv run --locked --extra llm python scripts/evaluate_llm.py --run-id NEW_QWEN_RUN
uv run --locked --extra llm python scripts/evaluate_llm.py --config configs/mamaylm.yaml --run-id NEW_MAMAY_RUN
```

Keep LLMs sequential and avoid unrelated inference during measurement. Classical training requires the separate Docker/Linux KenLM environment documented in [development.md](../docs/development.md).

### 6.3 Assignment coverage

| Assignment requirement | Evidence in this report |
| --- | --- |
| Training corpus, dataset size, preprocessing | Section 1; actual counts, word shares, sampling and leakage rules |
| Sentence log probability and required perplexity | Section 2; BOS/EOS and word-only denominator |
| Reference-based evaluation | Section 3.1; exact match, WER/CER, MRR and baselines |
| N-gram comparison, 10-point criterion | Section 3.3; four complete grids and Appendix A |
| Dataset-size analysis, 7-point criterion | Sections 1.4 and 3.4; gains and non-monotonic exception |
| Report quality/error analysis, 3-point criterion | Sections 3.6-3.7 and 5; actual cases, mechanisms and limitations |
| Optional LLM comparison, 5-point bonus | Section 4; accuracy, speed and LLM memory; matched KenLM runtime RAM remains pending |
| Results available through code | Modules, scripts, notebooks and saved artifacts |
| Final PDF, 5-7 pages | Pending conversion/editing; this comprehensive Markdown has longer appendices |
| Source-code repository/archive | Local implementation exists; final delivery remains pending |

## Appendix A. All 38 Models: Accuracy and Efficiency

Accuracy/WER/CER use normalized text. Classical timing is historical Docker/CPU scoring; LLM timing is GPU/HTTP prompted ranking. Stored MB is decimal disk size, not RAM. LLM pretraining counts are not controlled or recorded here.
""")
    all_models = [
        {
            "label": f"{labels[r['corpus']]}, {r['order']}-gram, {int(r['budget']):,} words",
            "accuracy": float(r["exact_match_accuracy"]),
            "wer": float(r["wer"]),
            "cer": float(r["cer"]),
            "mrr": float(r["mrr"]),
            "ms": float(r["milliseconds_per_record"]),
            "bytes": int(r["binary_bytes"]),
        }
        for r in classical
    ]
    for name, result in zip(("Qwen3.5-4B Q4_K_M", "MamayLM-4B Q4_K_M"), llms, strict=True):
        m = result["metrics"]
        all_models.append(
            {
                "label": name,
                "accuracy": m["exact_match_accuracy"],
                "wer": m["wer"],
                "cer": m["cer"],
                "mrr": m["mrr"],
                "ms": m["milliseconds_per_record"],
                "bytes": result["model_tag"]["size"],
            }
        )
    all_models.sort(key=lambda row: row["accuracy"], reverse=True)
    assert len(all_models) == 38
    add(
        table(
            ["Model / nominal words", "Accuracy", "WER", "CER", "MRR", "ms/record", "Stored MB"],
            [
                [
                    r["label"],
                    percent(r["accuracy"]),
                    percent(r["wer"], 4),
                    percent(r["cer"], 4),
                    f"{r['mrr']:.5f}",
                    f"{r['ms']:.5f}",
                    f"{r['bytes'] / 1e6:.2f}",
                ]
                for r in all_models
            ],
        )
    )
    add("""## Appendix B. Every Classical Configuration: Coverage, Training and Storage

OOV columns are occurrence rates. Inference is total time for 500 records; Appendix A gives mean latency. Loading and training are separate. ARPA/binary sizes are distinct disk artifacts; no column measures runtime peak RAM.
""")
    for key, _, _, label in STUDIES:
        add(f"### {label}")
        data = []
        for budget in (100000, 1000000, 10000000):
            for order in (1, 2, 3):
                r, model = by[key, order, budget], manifests[key, order, budget]
                data.append(
                    [
                        f"{budget:,}",
                        order,
                        percent(r["oov_references"], 3),
                        percent(r["oov_hypotheses"], 3),
                        percent(r["oov_selected"], 3),
                        f"{float(r['training_seconds']):.3f}",
                        f"{float(r['model_load_seconds']):.4f}",
                        f"{float(r['inference_seconds']):.6f}",
                        f"{model['artifact_bytes']['model.arpa'] / 2**20:.2f}",
                        f"{int(r['binary_bytes']) / 2**20:.2f}",
                    ]
                )
        add(
            table(
                [
                    "Budget",
                    "Order",
                    "Ref OOV",
                    "Hyp OOV",
                    "Selected OOV",
                    "Training s",
                    "Load s",
                    "Inference s",
                    "ARPA MiB",
                    "Binary MiB",
                ],
                data,
            )
        )
    add("""## Appendix C. All Classical Exact-Tie Diagnostics

These are analytical alternatives derived from saved exact top ties, not new scored runs. Uniform tie accuracy averages the correct fraction within each top tie. Minimum accuracy is a worst-case choice from those same ties. Official scores and first-index predictions remain unchanged.
""")
    diagnostics = []
    for key, _, _, label in STUDIES:
        for budget in (100000, 1000000, 10000000):
            for order in (1, 2, 3):
                tie = analyses[key]["tie_sensitivity"][f"{order}gram_{budget}"]
                diagnostics.append(
                    [
                        label,
                        f"{budget:,}",
                        order,
                        percent(by[key, order, budget]["exact_match_accuracy"]),
                        percent(tie["random_tie_accuracy"], 2),
                        percent(tie["minimum_accuracy"]),
                        tie["top_ties_between_distinct_normalized_texts"],
                    ]
                )
    add(
        table(
            [
                "Corpus",
                "Budget",
                "Order",
                "Recorded accuracy",
                "Uniform top-tie expectation",
                "Minimum top-tie accuracy",
                "Distinct-text top ties",
            ],
            diagnostics,
        )
    )
    output = root / "report/full_assignment_report.md"
    output.write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build_report())
