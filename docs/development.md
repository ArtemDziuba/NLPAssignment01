# Development and experiment guide

The official assignment remains unchanged in [readme.md](../readme.md).
The pipeline collects all five UberText 2.0 domains and prepares the nine configured
models: orders 1/2/3, each with 100K/1M/10M word-token budgets. The full experimental
grid is a separate explicit command; setup and tests do not run it.

## Windows setup

Python 3.13.16 is pinned for Windows editing, tests, and notebooks. Use the project
root for the commands below. All data paths are resolved from the configured project
root, independent of the current working directory.

~~~powershell
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv sync --locked --group analysis
uv run --locked --group analysis jupyter lab notebooks
~~~

Runtime dependencies are PyYAML and an explicitly enabled KenLM extra.
Development uses pytest/Ruff; notebooks use the separate analysis group.
Commit uv.lock. Deliberate dependency changes require uv lock and review.

## Linux KenLM environment

This machine has Docker Desktop with its WSL2 backend, rather than a general-purpose
Ubuntu WSL distribution. The Dockerfile provides a Linux environment without changing
the Windows Python installation. It pins the base-image digest, uv, KenLM source archive
and SHA256. Native utilities and bindings use KenLM 0.3.0; compiler, build settings,
package versions, and executable/binding hashes are recorded as provenance.

~~~powershell
docker build --tag asr-rescoring .
docker run --rm --mount "type=bind,source=$PWD,target=/workspace" --workdir /workspace --env PYTHONPATH=/workspace/src asr-rescoring scripts/train_models.py --check-only
~~~

A separate /opt/venv inside the image avoids sharing the Windows .venv with Linux.
Mounted source code is used through PYTHONPATH and script bootstrapping, so edits
do not require rebuilding the image unless dependencies or the toolchain change.
The image also works without a mount for help/tests on its packaged source;
real corpus data and generated models should be persisted through the mount.

For integration tests, including a complete tiny synthetic experiment grid:

~~~powershell
docker run --rm --mount "type=bind,source=$PWD,target=/workspace" --workdir /workspace --env PYTHONPATH=/workspace/src --env ASR_TEST_KENLM=1 asr-rescoring -m pytest -q
~~~

No real corpus download or large model training occurs in these tests. Synthetic
test predictions live only in pytest temporary directories and are not assignment
results. Default Windows tests skip native KenLM integration.

A regular WSL2 Linux checkout can also run uv sync --locked --extra kenlm after
the native utilities are built from the same release and placed on PATH. Keep its
environment separate. External compiler/system packages are outside the Python lock;
the container records their installed versions, but a fresh apt repository may change.

## Data collection and preparation

The authoritative [UberText page](https://lang.org.ua/en/ubertext/) provides sentence-split
exports for court, fiction, news, social, and wikipedia. The five compressed archives
are approximately 5 GB combined. The [corpus paper](https://aclanthology.org/2023.unlp-1.1/)
describes the source collection and processing. The project page does not supply an
explicit corpus-wide redistribution license or cryptographic checksums; retain source
attribution and local checksum manifests, and do not confuse the paper's license with
a license for the underlying corpus.

~~~powershell
uv run --locked python scripts/download_corpus.py
uv run --locked python scripts/prepare_corpus.py --workers 2
uv run --locked python scripts/check_readiness.py --verify-archives
~~~

Downloads resume using HTTP Range and ETag/If-Range. Completed archives are reused
only after SHA256 verification. JSON sidecars record URL, size, headers, download time,
and local hash. A hash detects later local changes; it is not publisher authentication.
The observed archive hashes are also pinned in experiments.yaml; downloading a changed
upstream archive fails explicitly instead of silently changing the training inputs.
Interrupted downloads retain .part files.

Preparation streams bz2 directly, with no full decompressed copy. It scans every
configured archive once, normalizes and cleans sentences, rejects evaluation overlaps,
and retains the lowest seeded hash priorities for up to 200K unique normalized sentences
per domain. This is bounded-memory uniform sampling over eligible unique normalized
sentences within each domain. Domain sample caches include source and preprocessing
identities and output hashes. Increase sentences_per_domain if larger future budgets
need more text. --sample-only caches samples for already completed downloads without
creating final subsets; the normal command requires all five domains.

The samples are interleaved by domain in the configured order with global normalized
deduplication. This deliberately gives approximately equal sentence representation,
rather than weighting domains by archive size; word contributions differ and are saved
in each subset manifest. Model comparisons use this same fixed mixture.

Subsets are nested whole-sentence prefixes, with budgets as upper bounds. A boundary
sentence is never skipped or cut just to hit a number. Report actual token counts and
the shortfall, not just nominal 100K/1M/10M labels. Insufficient available tokens fail
clearly. The processed corpus is the reusable sampled mixture, not the entire expanded
multi-billion-token UberText collection.

Normalization v1:
- Unicode NFC and lowercase.
- Canonicalize curly/modifier/backtick/acute apostrophes to ASCII apostrophe.
- Retain Unicode letters/numbers, with apostrophes only inside words.
- Turn other punctuation into separators and collapse whitespace.
- Preserve digits, recognition errors, and Ukrainian letter distinctions.
- No spelling correction, transliteration, stemming, or lemmatization.

The sentenced source already supplies segmentation. Evaluation candidates remain whole
sentences. Corpus-only cleaning excludes empty, non-Cyrillic, URL/obvious HTML lines,
and sentences outside configurable 2–150 word limits.

## Evaluation data and leakage

The supplied dataset is stored at `data/evaluation/ua_asr_hypotheses_500.jsonl`,
resolved through `paths.evaluation` in every configuration. It was moved from the
project root on 25 November 2026 and remains byte-identical to the experiment input
(SHA-256 `426c324d25fe15ebb12ec27a48b74e6d26c133d3721c804185b6c101794a76b0`).
Historical configuration snapshots retain the path used at experiment time; audits
resolve the current configured location and verify the original content checksum.
The file remains included in Git and the Docker build context; other corpus data
is excluded from the image context.
The relocation was verified across all five configs, prepared corpora/subsets,
all 36 saved KenLM model audits, and both 500-record LLM report audits. The LLM
reports also rebuilt from a working directory outside the project, using temporary
outputs to preserve the saved experiment artifacts. A network-disabled Docker COPY
test confirmed the dataset passes the build-context exclusions. Regression checks
passed 51 tests, with two opt-in KenLM integration tests skipped, plus Ruff checks.
It contains 500 unique utt_id strings, references, and lists of string hypotheses
(eight per record in this file). Validation requires nonempty normalized sentences,
unique IDs and the configured record count; it does not hardcode eight candidates.

All normalized evaluation references and hypotheses are used only as an exclusion set,
never as training text or a basis for choosing preprocessing by accuracy. Exact normalized
training overlaps are removed before sampling. This audit does not detect every partial
overlap or paraphrase; the corpus itself may contain related source material.

Training rejects stale preparation manifests when the evaluation file, seed, domains,
normalizer, settings, processed text, or subset checksums change.

## Training and unigram compatibility

~~~powershell
docker run --rm --mount "type=bind,source=$PWD,target=/workspace" --workdir /workspace --env PYTHONPATH=/workspace/src asr-rescoring scripts/train_models.py
~~~

Each model records actual tokens, ARPA/binary sizes, native commands in training logs,
training time, input/config/toolchain identity, and artifact hashes. Reuse requires a
matching identity and matching artifact hashes; --force retrains explicitly.
Models are trained sequentially with the configured 2G KenLM memory budget.
Native working files live in the Linux temporary directory, then completed artifacts
are copied atomically to models/. A real-data attempt using Windows-mounted temporary
files crashed in KenLM; using Linux-local scratch resolved it without changing data,
estimation settings, or scoring conventions. Keep scratch on the Linux filesystem.
Fallback discounts are disabled for real experiments; tiny synthetic checks enable
them explicitly because artificial fixtures do not have realistic count-of-counts.

KenLM 0.3.0 can estimate unigrams but its generic ngram ARPA loader rejects order-one
files. The explicit zero_bigram_container compatibility mode preserves the original
unigram ARPA probabilities, adds zero backoffs and an empty second-order section solely
for loading/build_binary, and resets KenLM context for each word and EOS during scoring.
No bigram observations are trained for the effective unigram model. Metadata distinguishes
effective order 1 from container order 2. The native smoke test verifies the scored sum
against the original ARPA values, including an OOV word, and records native-load behavior.
Never label a genuinely trained bigram model as a unigram.
Unigram timing includes per-token Python calls for context resets; account for that
implementation overhead when interpreting speed comparisons.

Sources: [KenLM build instructions](https://github.com/kpu/kenlm),
[loader restriction](https://github.com/kpu/kenlm/blob/master/lm/model.cc),
[Python scoring implementation](https://github.com/kpu/kenlm/blob/master/python/kenlm.pyx).

## Scoring, metrics, and running experiments

Score with BOS/EOS, log10 probabilities, and the assignment formula:
perplexity = 10 ** (-log10_probability / word_count), excluding boundary tokens from
the denominator. KenLM's built-in perplexity includes EOS in the denominator and is
therefore not used. Ranking compares cross entropy directly to avoid overflow.
An unrepresentable displayed perplexity is null; the log probability and cross entropy
remain available. Exact ties choose the original first candidate. Empty normalized
candidates/references fail validation; none of the 500 records is silently dropped.

Primary accuracy compares normalized selected and reference text; raw exact-match
accuracy is also saved. WER/CER use corpus edit totals over total reference lengths,
and CER includes spaces by default. MRR uses the first matching rank, with zero when
absent. Oracle accuracy measures reference availability among candidates.
OOV rates count token occurrences separately for references, all hypotheses, and
selected sentences, excluding implicit boundaries. Uniform-random and first-candidate
baselines reuse deterministic per-ID selections across models.

Inference timing includes normalization, scoring and ranking, after a warmup; loading,
metric calculation, vocabulary checks, and output I/O are outside that interval.
Baseline selection/permutation timing is reported separately; it does not perform LM work.

Run the real nine-model grid only when ready:
~~~powershell
docker run --rm --mount "type=bind,source=$PWD,target=/workspace" --workdir /workspace --env PYTHONPATH=/workspace/src asr-rescoring scripts/run_experiments.py
~~~

Alternatively evaluate existing configured models with scripts/evaluate_models.py.
The runner saves real predictions with full rankings and selected indices, aggregate
JSON metrics, a comparison CSV, config snapshots, logs, source/model hashes, and timing.
It never downloads implicitly. The three notebooks cover dataset EDA, result plots,
and successful/failed predictions; they display actual saved data only.

The optional LLM component remains independent and disabled in the classical pipeline.
Explicit local evaluation is available through `scripts/evaluate_llm.py` (see below).
No API key or external inference call is part of the classical pipeline. Write the
required final PDF report only after collecting real experimental results.

## Completed experiment and report

Run 20261124_main completed all nine configurations. Its metrics and predictions are
in results/metrics and results/predictions; the summary is
results/metrics/20261124_main_summary.csv. The measured best LM is the 10M trigram at
90.2% exact match. The supplied data always puts the reference first, so the positional
baseline is perfect and exact-tie sensitivity is material; read the report's limitations.

The seven-page PDF and standalone HTML are report/experiment_results.pdf and
report/experiment_results.html. To rebuild the authored report from the saved artifacts:

~~~powershell
uv sync --locked --group analysis --group report
uv run --locked --group analysis --group report python scripts/build_report.py --run-id 20261124_main
~~~

The report command verifies all saved predictions, metrics and model hashes first.
Its narrative is authored for this run and refuses a different accuracy profile until
reviewed. scripts/analyze_results.py performs the same saved-artifact audit independently;
add --kenlm inside Linux for per-record vocabulary diagnostics. Position/tie sensitivity
is computed analytically from saved scores and does not alter official predictions.

## News / Wikipedia domain experiments

Three additional configurations preserve the original five-domain run:
`configs/news_only.yaml`, `configs/wiki_only.yaml`, and `configs/news_wiki.yaml`.
Each uses the same orders, token budgets, seed, normalization, scoring, and evaluation.
The two-domain mixture uses the established round-robin sentence policy; manifests
record the actual word shares. These are experiment comparisons, not the final report.

The original 200K-sentence samples are too small for a 10M-word single-domain model.
These configurations share a separate 750K-sentence-per-domain sample cache under
`data/processed/domain_samples_750k`. The existing archives are scanned once to enlarge
the samples; all three preparations reuse that cache. Their processed corpora, subsets,
and model directories are separate, so old results remain intact.

Inside the existing Linux container environment:

~~~bash
python scripts/run_domain_experiments.py --prepare --prefix 20261124
~~~

This prepares the three corpora, runs 27 models, audits their saved predictions/model
hashes, and exports three result tables plus a 36-model comparison including the
original grid. Outputs are `report/20261124_domain_comparison.html` and
`results/metrics/20261124_domain_comparison.csv`. Run IDs end in `news_only`,
`wiki_only`, and `news_wiki`; each configuration has its own `models/<name>/` directory.
Regenerate the tables without training using `--compare-only` (also works on Windows).
The original mixed-run summary is checked for preservation.

## Local LLM evaluation

The user authorized evaluation of the installed `qwen3.5:4b`, followed by
`hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF:Q4_K_M` in Ollama.
These are experiment results, not the final assignment report. No model training,
fine-tuning, or implicit model download is performed. The evaluator permits only
an explicitly configured loopback HTTP endpoint.

Both models use the same English instruction, normalized Ukrainian candidate text,
seeded per-record candidate shuffles, greedy decoding, 4096-token context, 128-token
output cap, batch size 512 and eight CPU threads. Thinking is disabled. Full request
settings, Ollama version, model digest, template, parameter count, quantization,
configuration snapshot, dataset/source hashes and hardware are saved. Shuffles are
based only on seed 42 and utterance ID. Original indices remain unchanged in the data.
The prompt builder accepts no reference argument, and original candidate indices
are not displayed to the model. This prevents using the dataset's gold-first ordering.

Ollama exposes output-token probabilities rather than teacher-forced probabilities
for all supplied candidates through the endpoint used here. This experiment therefore
uses **prompted candidate ranking**, not LLM perplexity. An output must be a complete
permutation of the displayed candidate numbers. Invalid responses are logged and
retried at most once; the evaluator never substitutes the first/gold candidate.
The first ranked candidate is evaluated with the existing normalized accuracy,
corpus WER/CER, MRR and oracle metrics; raw accuracy is reported separately.

~~~powershell
uv sync --locked --extra llm --group dev
uv run --locked --extra llm python scripts/evaluate_llm.py --run-id YOUR_QWEN_RUN
uv run --locked --extra llm python scripts/evaluate_llm.py --config configs/mamaylm.yaml --run-id YOUR_MAMAY_RUN
~~~

Keep models sequential and avoid unrelated inference during measurements. Use a new
run ID for an independent repeat; the same ID resumes a matching incomplete run.
Predictions are checkpointed after every validated response. The model, source,
dataset and configuration identity must match before resuming. Small `--limit` runs
are for smoke checks and are not accepted as full reports.

Raw requests/responses, model metadata, warmup requests and resource samples are
stored under `results/logs`; predictions under `results/predictions`; metrics under
`results/metrics`. HF model identifiers produce nested artifact directories. The
metadata gives each exact path. The classical default remains `llm.enabled: false`;
invoking the dedicated LLM script is the explicit opt-in.

Timing is one sequential warm HTTP pass with prefix caching enabled. It includes
prompt evaluation, generation and retries; excludes normalization, warmup, metric
calculation and file writes. Per-record latency, percentile summaries, token counts,
cache counts and server prompt/decode/load durations are retained. Greedy decoding
and a fixed seed do not guarantee identical outputs across backend/hardware versions.

The optional `psutil` dependency records the combined RSS of Ollama GUI/service and
llama runner processes. Shared pages can be counted repeatedly. GPU sampling uses
`nvidia-smi` and includes other applications on the device; Ollama's reported model
GPU allocation is saved separately. Sampling targets are 0.25 seconds for RAM and
one second for GPU, but inspection overhead increases actual intervals. The report
calculates observed intervals from the sample timestamps; peaks may be missed.
Preflight loads, warmups and measured inference are identified separately in logs.

`scripts/report_llm.py METRICS_JSON --compare OTHER_METRICS_JSON` independently checks
dataset/prediction hashes, every raw response, shuffle mapping, selection, full ranking
and aggregate metrics before producing an HTML experiment report, comparison CSV,
all-model CSV and audit JSON. It includes original-first and shuffled-first baselines
and selection-position diagnostics. Historical CPU KenLM timings and current GPU
Ollama timings have different scopes, and no matched KenLM runtime-memory benchmark
has been collected. Pretraining overlap cannot be audited for the pretrained LLMs.

Completed runs are `20261124_qwen35_4b` and `20261124_mamaylm_4b`, executed sequentially
on the same RTX 4060 Laptop GPU. Each produced 500 valid rankings (4,000 candidate
positions), with zero retries. Reports and all-model CSVs contain 38 models: the
preserved 36 classical models plus these two LLMs.

| Model | Normalized exact match | Raw exact match | WER | Mean ms/record | Total inference s |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 88.4% | 77.6% | 1.586% | 570.84 | 285.42 |
| MamayLM-4B Q4_K_M | 18.6% | 15.0% | 13.727% | 770.53 | 385.27 |

Peak combined service RSS was 7.696 GiB for Qwen and 7.160 GiB for MamayLM. Loaded-model
GPU allocations reported by Ollama were 2.925 and 2.566 GiB respectively; sampled
device-wide GPU peaks were 3,767 and 2,829 MiB. Median observed resource-sample intervals
were 1.288 and 1.488 seconds; the configured shorter target intervals were not achieved.
Both runs logged zero resource-monitor errors. Raw timestamped samples are retained.

MamayLM selected the first displayed candidate in 482/500 records and returned the
unchanged display order in 233/500. Qwen's corresponding counts were 34 and nine.
The shared shuffled-first baseline reaches 15.6% normalized accuracy; original-first
reaches 100%. MamayLM's result therefore primarily exposes positional preference
under this English structured-ranking prompt. It does not establish the model's
accuracy under Ukrainian prompts or teacher-forced likelihood rescoring. The report
generator independently audited all 1,000 predictions and 8,000 ranked positions.

The native regression suite passed 47 tests, with two opt-in KenLM integration tests
skipped. Ruff checks passed. The official assignment text remains unchanged.

## Full assignment-structured Markdown report

The user requested a comprehensive Markdown report following the official README.
`report/full_assignment_report.md` covers all 36 classical models and both LLM runs,
with training sources, actual dataset sizes, preprocessing, assignment perplexity,
four complete accuracy grids, size/domain explanations, real error examples, optional
LLM measurements, and full metric/resource/tie-diagnostic appendices. Earlier reports
remain intact. The final 5-7 page PDF and source-code delivery remain pending.

Regenerate the Markdown from the saved artifacts without training or inference:

~~~powershell
uv run --locked --extra llm python scripts/build_full_report.py
~~~

The command also reuses the authored LLM report's detailed narrative and tables,
checks the classical comparison checksum, and verifies LLM requests, responses,
rankings, metrics and prediction checksums. For a fresh complete classical artifact
audit, use `scripts/analyze_results.py` with each run ID and its matching config.
