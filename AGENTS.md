# Project development instructions

- Treat the existing `readme.md` (README.md on Windows) as the source of truth for
  assignment requirements. Never modify the official assignment without explicit permission.
- Current stage: project scaffold only. Do not download corpora, train models, run
  experiments, fabricate results, or write the final report during setup.
- Keep code simple, modular, readable, and reproducible. Avoid unnecessary classes,
  abstractions, frameworks, and dependencies.
- Use Python with `uv`; maintain `pyproject.toml` and commit the generated `uv.lock`.
  Keep KenLM optional for Windows development; use a separate WSL2 environment for KenLM.
- Use type hints, meaningful names, and `pathlib` for filesystem operations.
- Resolve configured relative paths from the project root, independent of working directory.
- Separate preparation, training, scoring, evaluation, and visualization. Reuse
  `src/asr_rescoring` from scripts and notebooks; never duplicate pipeline logic there.
- Put experiment parameters in `configs/experiments.yaml`. Fix seeds, preserve candidate
  order, record source checksums, preprocessing settings, and tool versions.
- Preserve `ua_asr_hypotheses_500.jsonl` in place. Its observed schema is `utt_id`
  (string), `reference` (string), and `hypotheses` (list of strings), with 500 records.
  Validate actual inputs rather than inventing another schema.
- Never train on evaluation records. Exclude normalized evaluation references and
  hypotheses from training and audit overlap before building nested corpus subsets.
- Apply identical normalization to training and evaluation without spelling correction,
  transliteration, stemming, or other recognition-error repair.
- Follow the assignment's perplexity definition: log10 score with BOS/EOS enabled,
  denominator equal to word count excluding boundary tokens. Do not silently use
  KenLM's built-in perplexity, which includes EOS in the denominator.
- Preserve evaluation conventions across experiments, including normalization, tie
  breaking, empty-input policy, baselines, and metric aggregation.
- Verify installed KenLM unigram training, loading, and scoring support before running
  the grid. Fail clearly if unsupported; never silently substitute a bigram model.
- Add focused tests as preprocessing, dataset validation, scoring, and metrics are
  implemented. Use tiny fixtures and fake scorers for unit tests; make real KenLM
  integration tests opt-in. No large downloads or training in the default test suite.
- Never fabricate experimental results. Save real per-record predictions and aggregate
  metrics together with reproducibility metadata; reuse models only when inputs match.
- Document important decisions and limitations in `docs/development.md`. Keep optional
  LLM work independent of required classical models and free of implicit API calls.
