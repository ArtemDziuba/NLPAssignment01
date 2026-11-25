# Improving Speech Recognition Output Using a Classical Language Model

## Project overview

A Ukrainian ASR candidate rescoring experiment using classical n-gram language models trained with KenLM on UberText 2.0. Each of 500 synthetic ASR-like records has a reference and eight candidates. The model ranks the candidates by assignment-defined perplexity and selects the lowest-scoring one.

The assignment requires 1-, 2- and 3-gram comparisons, multiple corpus sizes, exact-match accuracy, qualitative error analysis and a 5–7-page PDF report. An optional comparison with modern LLMs covers accuracy, speed and memory. This project evaluates 36 classical configurations (four corpus selections × three budgets × three orders) and two pretrained LLMs.

## Results at a glance

| Model | Normalized accuracy | WER | Mean ms/record |
| --- | ---: | ---: | ---: |
| News, 10M, bigram | **92.8%** | 0.869% | 0.040 |
| News, 10M, trigram | **92.8%** | 0.891% | 0.041 |
| All five, 10M, trigram | 90.2% | 1.347% | 0.066 |
| Qwen3.5-4B | 88.4% | 1.586% | 570.838 |
| MamayLM-4B | 18.6% | 13.727% | 770.535 |

The original first-candidate baseline is 100% because the dataset always puts the reference first. Classical ties favor that order, while LLM candidates are shuffled. The comparison is not a demonstration of improved real-ASR decoding. Timing scopes differ and matched KenLM runtime RAM is unavailable.

## Reports and supporting evidence

- [Short assignment report](report/debloated_report.md) — Markdown source for the 5–7-page submission.
- [Detailed classical results](report/supplementary/classical_results.md) — full grids, actual corpus sizes, full error examples, ties, resources and all 38 model metrics.
- [Detailed LLM experiment report](report/20261124_llm_experiment_results.md) — prompt, settings, model digests, accuracy, latency, memory and positional preference.
- [Original five-domain experiment](report/supplementary/experiment_results.html) — saved figures and tables.
- [Domain comparison](report/supplementary/20261124_domain_comparison.html) ([JSON](report/supplementary/20261124_domain_comparison.json)).
- [Qwen run](report/supplementary/20261124_qwen35_4b_results.html) and [MamayLM run](report/supplementary/20261124_mamaylm_4b_results.html) — original experiment-specific exports.

The supplementary documents are not intended to be appended to the submission PDF. Repository paths below are taken from the existing report; the source repository itself was not included in the documentation edit.

## Experimental configuration

- Training: [UberText 2.0](https://lang.org.ua/en/ubertext/) (court, fiction, news, social, Wikipedia); nested 100K, 1M and 10M word budgets; seed 42.
- Evaluation: `data/evaluation/ua_asr_hypotheses_500.jsonl`, 500 records × 8 candidates. SHA-256: `426c324d25fe15ebb12ec27a48b74e6d26c133d3721c804185b6c101794a76b0`.
- Normalize consistently, exclude exact evaluation matches before training, retain original candidate errors. Minimize `-log10_probability / word_count`, with BOS/EOS enabled and boundary tokens excluded from the word count.
- Classical environment: KenLM 0.3.0, Python 3.11.18, Docker/Linux on Windows/WSL2. LLM environment: Windows 11, Ollama 0.40.17, Python 3.13.16, i7-13650HX, RTX 4060 Laptop GPU, approximately 16 GB RAM. See the supplementary reports for details.

## Artifact provenance and validation

Preparation, training, scoring, evaluation and visualization are separate modules. Scripts/notebooks reuse them. Configuration paths resolve from the project root. Seeds, source/subset/model hashes, toolchain identities, raw predictions, scores/rankings, LLM prompts/digests and resource samples are saved. Current configs resolve the moved evaluation file; historical snapshots preserve the experiment-time path.

The original full-report audit recorded all 36 classical models: **18,000 predictions and 144,000 candidate scores**. Both LLM runs were checked against exact requests, raw responses, complete permutations, selected indices, checksums, metrics and timing: **1,000 predictions and 8,000 ranked positions**. Together these are **19,000 record predictions and 152,000 candidate positions**. No training or new model inference was performed to write this report.

The current native path/regression suite has 51 passing tests; two opt-in KenLM integration tests were skipped in that invocation. Earlier native Linux checks and model checksums are retained in run logs. The revised documentation uses the saved results; it does not rerun the original audit or tests.

- [Classical 36-model CSV](results/metrics/20261124_domain_comparison.csv); SHA-256 `6c9f52a7d1b3099b82b898e030c3646c5ffaa24ab21e8caa91d4567326c73a7b`.
- [Combined 38-model CSV](results/metrics/20261124_qwen35_4b_all_models.csv).
- [Detailed LLM-only Markdown report](report/20261124_llm_experiment_results.md).

| Run / selection | Config | Models | Summary | Audit |
| --- | --- | --- | --- | --- |
| 20261124_main: All five domains | [experiments](configs/experiments.yaml) | [Directory](models) | [CSV](results/metrics/20261124_main_summary.csv) | [JSON](results/metrics/20261124_main_analysis.json) |
| 20261124_news_only: News only | [news_only](configs/news_only.yaml) | [Directory](models/news_only) | [CSV](results/metrics/20261124_news_only_summary.csv) | [JSON](results/metrics/20261124_news_only_analysis.json) |
| 20261124_wiki_only: Wikipedia only | [wiki_only](configs/wiki_only.yaml) | [Directory](models/wiki_only) | [CSV](results/metrics/20261124_wiki_only_summary.csv) | [JSON](results/metrics/20261124_wiki_only_analysis.json) |
| 20261124_news_wiki: News + Wikipedia | [news_wiki](configs/news_wiki.yaml) | [Directory](models/news_wiki) | [CSV](results/metrics/20261124_news_wiki_summary.csv) | [JSON](results/metrics/20261124_news_wiki_analysis.json) |

| LLM run | Metrics | Predictions | Requests/responses | Resource samples | Metadata |
| --- | --- | --- | --- | --- | --- |
| 20261124_qwen35_4b | [JSON](results/metrics/20261124_qwen35_4b_qwen3.5_4b.json) | [JSONL](results/predictions/20261124_qwen35_4b_qwen3.5_4b.jsonl) | [JSONL](results/logs/20261124_qwen35_4b_qwen3.5_4b_requests.jsonl) | [JSONL](results/logs/20261124_qwen35_4b_qwen3.5_4b_resources.jsonl) | [JSON](results/logs/20261124_qwen35_4b_qwen3.5_4b_metadata.json) |
| 20261124_mamaylm_4b | [JSON](results/metrics/20261124_mamaylm_4b_hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF_Q4_K_M.json) | [JSONL](results/predictions/20261124_mamaylm_4b_hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF_Q4_K_M.jsonl) | [JSONL](results/logs/20261124_mamaylm_4b_hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF_Q4_K_M_requests.jsonl) | [JSONL](results/logs/20261124_mamaylm_4b_hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF_Q4_K_M_resources.jsonl) | [JSON](results/logs/20261124_mamaylm_4b_hf.co/INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF_Q4_K_M_metadata.json) |

## Setup and reproducibility

The existing report-generation command reads saved evidence and regenerates the original comprehensive report, not the shortened report. The new-run commands below were not executed while authoring it. Use fresh IDs for new inference runs.

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

Keep LLMs sequential and avoid unrelated inference during measurement. Classical training requires the separate Docker/Linux KenLM environment documented in [development.md](docs/development.md).


## Report export

Convert only `report/debloated_report.md`, not the full report or the supplements. For example, with Pandoc installed, from the repository root:

```powershell
pandoc report/debloated_report.md -o report/debloated_report.docx
```

Open the DOCX and export to PDF. Check page count with the actual submission template (font, margins and spacing affect pagination). The short Markdown was checked with a default Pandoc DOCX conversion and LibreOffice PDF export: six pages. This is a formatting check, not a claim that every Word template produces six pages.

## Assignment coverage

| Criterion | Main report section |
| --- | --- |
| Corpus, dataset size, preprocessing | 1 |
| Log-probability, perplexity, metrics | 2 |
| N-gram comparison (10 points) | 3, Table 1 |
| Dataset-size analysis (7 points) | 3, Tables 1–2 |
| Error analysis and interpretation (3 points) | 4 and 6 |
| Optional LLM comparison (5 points) | 5, Table 4 (KenLM runtime RAM missing) |

The PDF and source-code repository/archive are the final deliverables. The full experiment exports remain supporting evidence.
