# Development setup and implementation plan

The official assignment is preserved in `../readme.md`. This document supplements it.
This stage contains packaging, configuration, module/script docstrings, and unexecuted
notebook outlines only. There is no working pipeline, trained model, result, or final report.

## Environment

Use `uv` from the project root. `.python-version` pins Python 3.13.16, which is already
available on this Windows machine; package metadata permits Python 3.11 through 3.13.
The Python/KenLM combination still needs verification inside WSL2 before training.

```powershell
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
# Optional notebook dependencies:
uv sync --locked --group analysis
uv run --locked --group analysis jupyter lab notebooks
```

`uv sync` creates a local `.venv`; activation is unnecessary with `uv run`.
The default `dev` group contains pytest and Ruff. `analysis` adds JupyterLab, the Python
kernel, pandas, and matplotlib. The runtime dependency is PyYAML; avoid extra libraries
until implementation needs them. The `kenlm` extra pins the Python binding separately.
Dependency groups and optional extras follow [uv's project dependency conventions](https://docs.astral.sh/uv/concepts/projects/dependencies/).
Commit `uv.lock` and use `--locked` for reproducible syncs. Deliberate dependency changes
require regenerating the lock with `uv lock` and reviewing the diff.

Once tests exist, run `uv run --locked pytest`. No tests are supplied for docstring-only
modules, and pytest currently reports no tests collected. Add behavioral tests alongside
the first implementations; do not treat an empty test suite as pipeline validation.
Scripts are placeholders with no CLI or operational behavior yet.

## Windows and KenLM

Use Windows for editing, data inspection, and notebooks. Prefer running the entire
training **and scoring** process within WSL2, with its own Linux `uv` and `.venv`.
Never share a Windows virtual environment with WSL2. For larger runs, a separate
checkout on the WSL filesystem avoids heavy I/O through a mounted Windows drive;
retain the same config and evaluation checksum in either checkout.

The [KenLM repository](https://github.com/kpu/kenlm) documents CMake-based native builds,
`lmplz` for estimation, and `build_binary` for model conversion. The Python binding
alone is not the complete training toolchain. [PyPI's KenLM 0.3.0 release](https://pypi.org/project/kenlm/0.3.0/#files)
provides a source distribution, so do not assume a native Windows wheel is available.
Compiler, CMake, and relevant build libraries must be prepared separately, with approval
before any system-level installation. No WSL distribution, compiler, or KenLM build
has been installed or verified by this scaffold.

After those prerequisites are ready, inside WSL2:

```bash
uv sync --locked --extra kenlm
command -v lmplz
command -v build_binary
uv run --locked --extra kenlm python -c 'import kenlm; print(kenlm.__file__)'
```

Keep `--extra kenlm` on later `uv run` commands that need the binding, since sync is exact.
Record the C++ source revision, compiler, build options, and binding version separately
from `uv.lock`; the lock cannot reproduce external system utilities.

Before the real grid, verify training, ARPA loading, binary conversion, and scoring
for orders 1, 2, and 3 on a tiny local fixture. Unigram support in the installed tools
is unresolved. If unsupported, stop and document a reviewed approach; never silently
skip order 1 or label a higher-order model as a unigram.

## Data and configuration decisions

- The provided `ua_asr_hypotheses_500.jsonl` remains at the root, unchanged and trackable.
  Inspection found 500 records, 500 unique `utt_id` strings, nonempty `reference`
  strings, and eight nonempty string `hypotheses` per record. No scores or candidate
  objects are assumed. The future validator should require nonempty candidate lists,
  without unnecessarily hardcoding eight candidates.
- `data/evaluation/` is reserved for future explicitly imported evaluation resources.
  The current config points directly to the supplied root file; do not duplicate it.
- `configs/experiments.yaml` is an initial specification, not a validated executable
  interface yet. Paths resolve from the project root. Its Cartesian product defines
  nine models: orders 1/2/3 with budgets 100K/1M/10M normalized word tokens.
- Choose one [UberText release/export](https://lang.org.ua/en/ubertext/), verify its
  license and format, and record source URLs and checksums before downloading once.
  Release, download URL, and sentence segmenter are deliberately unset pending inspection.
- Proposed normalization is Unicode NFC, lowercase, canonical apostrophes, punctuation
  separation except within apostrophized words, and whitespace collapse, with digits
  preserved. Specify exact apostrophe mappings and segmentation rules when implementing
  and version them. Apply the same normalization to evaluation references and candidates,
  without spelling repair or linguistic rewriting. Training alone uses sentence segmentation.
- Remove duplicate training sentences and normalized exact overlaps with evaluation
  references **and** hypotheses before sampling. Evaluation text may be used for exclusion,
  never for training or selecting preprocessing based on accuracy. Exact overlap checks
  do not detect all paraphrases or partial overlaps; document the audit's limits.
- Seed one reproducible training sentence order and take whole-sentence prefixes for
  nested subsets. Budgets are upper bounds: preserve sentence boundaries and record
  actual tokens rather than truncating a sentence to force an exact count. Fail if
  the source cannot supply the largest budget; record any boundary shortfall.
- Model reuse must compare corpus/subset checksums, preprocessing version, order,
  configuration, and KenLM build provenance. File existence alone is insufficient.

## Evaluation conventions to implement

The assignment scores with `model.score(text, bos=True, eos=True)` and defines
`perplexity = 10 ** (-logprob / N)`, where `N` is the normalized word count excluding
boundary tokens. Select minimum perplexity; compare `-logprob / N` to avoid overflow.
Do not put literal `<s>` or `</s>` in text. The [KenLM Python implementation](https://github.com/kpu/kenlm/blob/master/python/kenlm.pyx)
uses `N + 1` in its built-in `perplexity()` to include EOS. Implement the assignment's
formula explicitly and keep that convention fixed across all models.

Preserve input candidate order and select the first candidate on exact score ties.
Reject empty normalized references/candidates explicitly rather than dividing by zero
or silently dropping evaluation records. Use the same 500 IDs for every evaluation.
Preserve raw strings alongside normalized strings and selected candidate indices.

Primary accuracy compares normalized selected text with normalized reference text.
WER and CER aggregate edit counts over total reference words/characters; CER includes
spaces. MRR uses the first normalized reference match in the full candidate ranking,
with zero when absent. Oracle accuracy is the fraction with any matching candidate.
Measure OOV tokens against each model's vocabulary separately for references, all
hypotheses, and selected text, excluding BOS/EOS. Define first-candidate and seeded
uniform-random baselines once and reuse the same selections for all models.

Save config snapshots, input hashes, actual corpus sizes, model sizes, training time,
software/build versions, and aggregate metrics with real predictions. Time scoring
and ranking with a monotonic timer; report model loading and preprocessing separately.
The optional LLM module stays disabled and independent; no provider dependency is added.

## Next implementation stages

1. Implement config loading/path resolution and dataset validation. Test bad JSONL,
   missing/incorrect fields, empty text/lists, duplicate IDs, and expected record count.
2. Inspect the selected corpus export. Implement and test normalization/segmentation,
   leakage filtering, provenance, and deterministic nested subsets; then prepare data once.
3. Verify KenLM compatibility on tiny fixtures, particularly unigrams. Implement training,
   manifests, timing, sizes, and reuse checks before attempting larger models.
4. Implement scoring with a fake model in unit tests. Cover base-10 perplexity, word
   denominator, BOS/EOS, empty input, different sentence lengths, and stable ties.
5. Implement/test exact match, corpus WER/CER, MRR, oracle and OOV, then baselines and
   predictions/timing exports. Use hand-computable metric examples, not invented results.
6. Implement experiment orchestration, run the nine configurations, then populate the
   notebooks with real tables, plots, and successful/failed examples. Write the required
   5–7 page PDF report only after results exist; consider LLM comparison last.
