# Assignment completion checklist

Status updated on 25 November 2026 against the official [readme.md](../readme.md).
Evidence includes all four classical grids, both LLM runs, saved artifacts, executed
notebooks, and recorded tests. The official assignment text is unchanged.

**Current position:** the classical-LM implementation, required experiments, and
experiment analysis are complete. A comprehensive [assignment-structured Markdown
report](../report/full_assignment_report.md) now covers all 38 models and their
explanations. Final PDF editing/conversion and final source-code delivery remain
pending. The existing experiment PDF/HTML retain their experiment-analysis scope.

Legend: `[x]` completed and evidenced; `[ ]` pending. Optional bonus work is identified
separately. This checklist records completion, not an estimated grade.

## Main task in the README

- [x] Train a classical n-gram language model with KenLM.
  Nine trained models and manifests exist under `models/`: orders 1, 2, and 3 at
  each of the three corpus budgets.
- [x] Score each candidate sentence using perplexity.
  [scoring.py](../src/asr_rescoring/scoring.py) computes log10 probabilities and the
  assignment's word-count-normalized perplexity.
- [x] Select the best candidate.
  Minimum perplexity is selected, with deterministic first-index tie breaking.
  Nine prediction files preserve the full rankings and selected indices.
- [x] Analyze performance and failure cases.
  [analysis.py](../src/asr_rescoring/analysis.py), the executed error-analysis notebook,
  and the [experiment analysis](../report/experiment_results.html) contain measured
  comparisons, concrete Ukrainian examples, and limitations.

## Activity 1: language-model training

- [x] Train models with orders from 1 to 3 grams.
  All three orders were trained at 100K, 1M, and 10M nominal word-token budgets.
  The unigram uses a documented zero-bigram storage container and context-reset
  scoring that preserves the true unigram probabilities; it is not a trained bigram.
- [x] Document the training corpus.
  UberText 2.0 sentence-split exports from court, fiction, news, social, and wikipedia;
  source URLs and pinned hashes are in [experiments.yaml](../configs/experiments.yaml).
- [x] Document the training dataset sizes.
  Actual sizes are **99,987**, **999,999**, and **9,999,984** word tokens. The small
  shortfalls preserve whole sentences. Subset manifests record counts and hashes.
- [x] Document preprocessing choices.
  [development.md](development.md) and configuration describe normalization, upstream
  sentence segmentation, cleaning, deduplication, evaluation-overlap exclusions,
  seeded sampling, the five-domain mixture, and nested subsets.

## Activity 2: sentence scoring

- [x] Compute each candidate's sentence log-probability with the trained LM.
  Saved scores include BOS/EOS consistently and use log base 10.
- [x] Compare different sentence lengths using the assignment formula.
  `perplexity = 10 ** (-log10_probability / N)`, with `N` equal to word count,
  excluding boundary tokens. KenLM's different built-in denominator is not used.
- [x] Select the sentence with the lowest perplexity.
  Ranking uses the equivalent cross-entropy value to avoid numerical overflow.
  Saved predictions and scores were audited for all nine models.

## Activity 3: evaluation

- [x] Compare selected sentences against the references.
  All models use the same 500 records and all eight candidates per record.
- [x] Compute exact-match accuracy.
  Both normalized and raw accuracy are saved; they coincide for every LM in this run.
- [x] Perform qualitative error analysis.
  The experiment analysis includes successful case-ending and real-word selections,
  plus failures involving repetition, apostrophe fragmentation, and name spelling.
- [x] Experiment with different n-gram orders.
  Unigram, bigram, and trigram comparisons are complete.
- [x] Experiment with different training dataset sizes.
  The three nested size configurations are complete for each order.

### Completed accuracy comparison

| Nominal budget | Actual word tokens | Unigram | Bigram | Trigram |
|---|---:|---:|---:|---:|
| 100K | 99,987 | 73.6% | 78.0% | 78.6% |
| 1M | 999,999 | 76.0% | 82.4% | 83.4% |
| 10M | 9,999,984 | 77.4% | 89.4% | 90.2% |

Evidence: [comparison CSV](../results/metrics/20261124_main_summary.csv),
[executed results notebook](../notebooks/02_experiment_results.ipynb), and
[saved-output audit](../results/metrics/20261124_main_analysis.json).

## Evaluation criteria in the README

### N-gram comparison — 10-point criterion

- [x] Comparison table for 1-gram, 2-gram, and 3-gram models.
- [x] Accuracy results and a discussion of the observed differences.
- [x] Results available through code and the executed results notebook.
- [x] Comparison table and discussion available in the experiment-analysis PDF.
- [ ] Incorporate the chosen comparison table and reviewed discussion into the
  **final assignment PDF**. Existing experiment material is ready to use.

### Training dataset size analysis — 7-point criterion

- [x] Train on different corpus sizes.
- [x] Provide a comparison table with dataset size and performance.
- [x] Analyze increasing corpus size: accuracy rises for all three orders in this run;
  the experiment analysis also examines vocabulary coverage and model size.
- [ ] Incorporate the size comparison and reviewed interpretation into the
  **final assignment PDF**.

### Report quality — 3-point criterion

- [x] Produce explanations of why the LM succeeds in selected cases.
- [x] Produce explanations of where it fails.
- [x] Include concrete examples from actual predictions.
- [x] Identify limitations affecting interpretation: every reference is first,
  exact ties favor that ordering, and the evaluation is synthetic ASR-like data.
- [ ] Write and review these materials as the final assignment narrative.
  Availability of experiment-analysis material does not complete the final report.

### Modern LLM comparison — optional 5-point bonus

- [x] Implement and run local modern LLM candidate ranking: Qwen3.5-4B and
  MamayLM-4B Q4_K_M, 500 records each. This uses prompted ranking rather than
  candidate perplexity; references are hidden and display order is shuffled.
- [x] Compare classical and modern model accuracy. Qwen normalized accuracy is
  **88.4%**, MamayLM **18.6%**; the best existing news KenLM is **92.8%**.
  See the [Qwen experiment](../report/20261124_qwen35_4b_results.html) and
  [MamayLM experiment](../report/20261124_mamaylm_4b_results.html).
- [x] Measure and discuss inference speed. Qwen: **570.84 ms/record**; MamayLM:
  **770.53 ms/record**. Reports distinguish current GPU/HTTP inference from historical
  Docker/CPU KenLM timings; these are not controlled architecture-only benchmarks.
- [x] Measure LLM runtime resources: sampled peak service RSS, device-wide GPU use,
  and Ollama model GPU allocation. These are recorded separately from disk sizes.
- [ ] Complete the classical/LLM runtime-memory comparison: matching KenLM runtime
  RAM measurements remain pending. Disk sizes alone do not complete this comparison.
- [ ] Incorporate the reviewed optional analysis into the final assignment PDF,
  if submitting the bonus. MamayLM selected the first displayed candidate in
  **482/500** cases; its result is specific to this prompted-ranking configuration.

Skipping this section does not leave a mandatory README requirement incomplete.

## Required deliverables

- [ ] **Final PDF report, 5–7 pages.** Pending. The existing
  [seven-page experiment PDF](../report/experiment_results.pdf) is a report on
  `20261124_main`, as requested, rather than the finalized assignment report.
- [x] **Source implementation exists locally.** Modules, scripts, tests,
  configuration, dependency lockfile, Docker recipe, and executed notebooks exist.
- [ ] **Finalize source-code delivery as a GitHub repository or archive.**
  A local Git repository and an `origin` remote are configured. At this check,
  implementation changes and new files were still uncommitted/untracked, and no
  submission archive was present at the project root. Remote contents were not
  checked. Commit/package the intended complete version and deliver it through
  one of the README's accepted routes.

## Remaining required work

1. Review and condense [the full Markdown report](../report/full_assignment_report.md)
   into the final 5–7 page PDF. Retain candidate-position and tie-bias limitations;
   the comprehensive Markdown appendices exceed that page budget.
2. Finalize and deliver the complete source code through GitHub or an archive.

The README does not require another experimental grid to complete the mandatory
classical-LM work. Additional experiments may be useful research choices; the
modern LLM comparison remains optional.

## Additional completed work beyond the README minimum

- WER, CER, MRR, oracle accuracy, OOV rates, and deterministic selection baselines.
- Training/inference timing, binary sizes, per-record predictions, and model reuse.
- Source/model checksums and an audit of 4,500 predictions / 36,000 candidate scores.
- Three executed notebooks with no cell errors.
- A recorded native regression run with **39 tests passed**, zero failures/errors.

These strengthen the project but do not replace the pending final deliverables.

## Additional completed domain experiments

- [x] News only: all nine size/order configurations (`20261124_news_only`).
- [x] Wikipedia only: all nine configurations (`20261124_wiki_only`).
- [x] News + Wikipedia: all nine configurations (`20261124_news_wiki`).
- [x] Three experiment tables and a comparison of all 36 models, including the
  original nine: [domain comparison](../report/20261124_domain_comparison.html) and
  [full CSV](../results/metrics/20261124_domain_comparison.csv).
- [x] All three new corpora audited for nested subsets, normalized duplicates, and
  exact evaluation overlap. Original mixed-corpus results and assignment text preserved.

These are additional experiment results; the final assignment report remains pending.
