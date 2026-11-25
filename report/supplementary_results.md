# Supplementary classical language model results

Companion to the [short report](../debloated_report.md). The tables and detailed examples below are retained from the original assignment report (saved experiments of 24 November 2026). No experiments were rerun for this edit. This file is not part of the 5–7-page report.

## 1. Language Model Training

### 1.1 Evaluation data and controls

The fixed evaluation input [ua_asr_hypotheses_500.jsonl](../../data/evaluation/ua_asr_hypotheses_500.jsonl) contains 500 unique utterances, each with a reference and eight hypotheses: 4,000 candidate positions per model. The input is synthetic ASR-like text. All models use the same evaluation records.

The file was moved into `data/evaluation/` from the root. Current configs use the new path; historical snapshots retain the original path. Its SHA-256 is:

```text
426c324d25fe15ebb12ec27a48b74e6d26c133d3721c804185b6c101794a76b0
```

### 1.2 Training corpus and sampling

| Domain | Compressed archive MB | Sentences scanned | Retained original sample | Exact evaluation-overlap occurrences excluded |
| --- | --- | --- | --- | --- |
| court | 314.21 | 6,448,755 | 200,000 | 0 |
| fiction | 416.65 | 16,937,108 | 200,000 | 0 |
| news | 3565.95 | 105,237,866 | 200,000 | 39 |
| social | 90.79 | 3,064,084 | 200,000 | 0 |
| wikipedia | 841.64 | 24,365,667 | 200,000 | 0 |

The five archives occupy 5.23 GB. Full archives were scanned; a deterministic bounded sample retained 200,000 sentences per domain in the original grid. After global deduplication, its prepared pool contains 998,334 sentences.

The additional news/Wikipedia studies use a separate 750,000-sentence sample per included domain. The original 200,000-sentence news and Wikipedia samples each supply only about 3.2M words, insufficient for a 10M-word single-domain experiment. This sampling-capacity change affects available training text; normalization and evaluation rules remain fixed.

### 1.3 Preprocessing and leakage prevention

1. Preserve the official corpus sentence boundaries and keep evaluation candidates whole.
2. Apply NFC, canonicalize apostrophes, lowercase, extract word/digit tokens, and collapse whitespace. Preserve internal apostrophes; remove other punctuation and symbols. Hyphens become token boundaries.
3. For training only, reject empty/non-Cyrillic lines, obvious URL/HTML content and sentences outside 2-150 normalized words.
4. Exclude exact normalized evaluation references and hypotheses before sampling; deduplicate retained sentences, including across mixed domains.
5. Assign deterministic hash priorities using seed 42 and normalized sentence text; retain the lowest-priority unique sentences, sorted by priority.
6. Interleave domains round-robin by sentence. Take nested whole-sentence prefixes for nominal 100K, 1M and 10M word budgets, stopping before a sentence would exceed the budget.

Training and evaluation share normalization. There is no spelling correction, transliteration, stemming or recognition-error repair. Exact exclusion does not rule out paraphrases, partial overlap or related source material. The 39 original news exclusions count archive occurrences, not distinct evaluation records.

Mixed-domain interleaving balances sentence opportunities, not words. Actual word shares at the largest budget are:

| Selection | Domain | Words at nominal 10M | Word share |
| --- | --- | --- | --- |
| All five domains | court | 3,998,102 | 39.98% |
| All five domains | fiction | 1,249,427 | 12.49% |
| All five domains | news | 1,628,320 | 16.28% |
| All five domains | social | 1,506,498 | 15.07% |
| All five domains | wikipedia | 1,617,637 | 16.18% |
| News + Wikipedia | news | 5,015,119 | 50.15% |
| News + Wikipedia | wikipedia | 4,984,881 | 49.85% |

Court supplies about 40% of the original mixture's words despite roughly equal sentence counts. Under a fixed total budget, adding domains reduces news exposure. These shares matter when interpreting domain effects.

### 1.4 Actual training dataset sizes

| Selection | Nominal words | Actual words | Sentences |
| --- | --- | --- | --- |
| All five domains | 100,000 | 99,987 | 5,030 |
| All five domains | 1,000,000 | 999,999 | 49,975 |
| All five domains | 10,000,000 | 9,999,984 | 500,646 |
| News only | 100,000 | 99,994 | 6,056 |
| News only | 1,000,000 | 999,979 | 61,444 |
| News only | 10,000,000 | 9,999,995 | 613,743 |
| Wikipedia only | 100,000 | 99,995 | 6,168 |
| Wikipedia only | 1,000,000 | 999,985 | 61,984 |
| Wikipedia only | 10,000,000 | 9,999,983 | 618,737 |
| News + Wikipedia | 100,000 | 99,980 | 6,060 |
| News + Wikipedia | 1,000,000 | 999,981 | 61,610 |
| News + Wikipedia | 10,000,000 | 10,000,000 | 616,145 |

### 1.5 Training configuration and unigram compatibility

Models were trained sequentially using `lmplz -o ORDER -S 2G`, with Linux-local scratch files, then converted using `build_binary probing`. ARPA and binary artifacts were retained. Bigram/trigram estimation uses KenLM's modified Kneser-Ney estimator. Logs/manifests retain native commands, training time, actual tokens, subset/trainer/toolchain identities and artifact hashes.

KenLM 0.3.0 estimates unigrams but its generic loader rejects pure order-one ARPA. The explicit compatibility mode preserves original unigram probabilities in a container with zero observed bigrams and zero backoffs. Scoring resets context for each word and EOS. Effective order remains one; no bigram observations are trained. Native checks verified the scored sum against source ARPA, including an OOV token. These per-token Python calls also affect unigram latency.

The full grid is 4 corpus selections x 3 orders x 3 budgets = 36 classical models. The two optional LLMs use pretrained weights and have no controlled pretraining-word budget or training time in this experiment.

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

| Baseline | Accuracy | WER | CER | MRR |
| --- | --- | --- | --- | --- |
| first_candidate | 100.0% | 0.0000% | 0.0000% | 1.00000 |
| seeded_random | 15.6% | 14.3788% | 3.4729% | 0.36882 |

The reference is originally candidate zero in every record, so original-first selection reaches 100% without language understanding. The random baseline uses deterministic per-ID permutations; its observed 15.6% is close to the analytical uniform-random expectation, 15.425%. There are 117 records with multiple normalized matches, totaling 617 matching candidate positions.

Classical models preserve original order and first-index ties. LLMs see shuffled presentation with hidden original indices. The accuracy comparison therefore includes different tie/presentation treatments and does not show that either method beats the perfect positional baseline.

### 3.3 N-gram comparison: all four experimental grids

The four tables cover every classical configuration. Percentages are normalized accuracy on the same 500 records.

#### All five domains: `20261124_main`

| Nominal words | Actual words | Unigram | Bigram | Trigram |
| --- | --- | --- | --- | --- |
| 100,000 | 99,987 | 73.6% | 78.0% | 78.6% |
| 1,000,000 | 999,999 | 76.0% | 82.4% | 83.4% |
| 10,000,000 | 9,999,984 | 77.4% | 89.4% | 90.2% |

All orders improve with data in this grid. At 10M, the trigram leads the unigram by 12.8 percentage points but the bigram by only 0.8: most of the observed context gain is already present at order two.

#### News only: `20261124_news_only`

| Nominal words | Actual words | Unigram | Bigram | Trigram |
| --- | --- | --- | --- | --- |
| 100,000 | 99,994 | 76.2% | 80.8% | 80.4% |
| 1,000,000 | 999,979 | 77.0% | 88.4% | 87.8% |
| 10,000,000 | 9,999,995 | 78.6% | 92.8% | 92.8% |

Bigrams slightly outperform trigrams at 100K and 1M, then tie at 92.8% at 10M. Higher order is not automatically better. At the largest budget, the bigram also has lower WER and a smaller binary.

#### Wikipedia only: `20261124_wiki_only`

| Nominal words | Actual words | Unigram | Bigram | Trigram |
| --- | --- | --- | --- | --- |
| 100,000 | 99,995 | 74.2% | 77.2% | 77.0% |
| 1,000,000 | 999,985 | 73.6% | 81.4% | 80.8% |
| 10,000,000 | 9,999,983 | 76.6% | 85.6% | 86.0% |

Bigrams lead trigrams at 100K and 1M; the trigram leads at 10M. The unigram dips from 74.2% to 73.6% between 100K and 1M before reaching 76.6%. More training text is not strictly monotonic at every intermediate point, although every order improves from the smallest to largest budget.

#### News + Wikipedia: `20261124_news_wiki`

| Nominal words | Actual words | Unigram | Bigram | Trigram |
| --- | --- | --- | --- | --- |
| 100,000 | 99,980 | 75.4% | 79.2% | 79.4% |
| 1,000,000 | 999,981 | 76.4% | 86.2% | 86.8% |
| 10,000,000 | 10,000,000 | 77.6% | 91.4% | 91.6% |

All orders improve with data. The trigram maintains a small lead over the bigram and reaches 91.6% at 10M. Adding Wikipedia does not exceed the best news-only result at the same total word budget.

### 3.4 Training dataset size analysis

| Corpus | Unigram gain, 100K to 10M | Bigram gain | Trigram gain |
| --- | --- | --- | --- |
| All five domains | 3.8 pp | 11.4 pp | 11.6 pp |
| News only | 2.4 pp | 12.0 pp | 12.4 pp |
| Wikipedia only | 2.4 pp | 8.4 pp | 9.0 pp |
| News + Wikipedia | 2.2 pp | 12.2 pp | 12.2 pp |

Contextual models gain more than unigrams from the 100-fold nominal increase in every corpus. More observations can improve context estimates and vocabulary coverage. However, one deterministic sample ordering is used, and the Wikipedia unigram dip shows that larger subsets do not guarantee improvement at each point. Frequency shifts, sparse contexts, backoff and ties can affect selection.

The original five-domain trigram shows the following coverage relationship:

| Budget | Accuracy | Reference OOV | Hypotheses OOV | Selected OOV | Binary MiB |
| --- | --- | --- | --- | --- | --- |
| 100,000 | 78.6% | 28.084% | 35.332% | 27.898% | 4.50 |
| 1,000,000 | 83.4% | 8.970% | 17.663% | 9.151% | 32.27 |
| 10,000,000 | 90.2% | 2.324% | 10.868% | 2.366% | 224.18 |

Reference OOV falls from about 28.08% to 2.32% while accuracy rises. This supports coverage as part of the size explanation. In-vocabulary substitutions and repetition remain, so coverage alone cannot explain correctness.

### 3.5 Domain comparison at a fixed budget

| Order | All five | News | Wikipedia | News + Wikipedia |
| --- | --- | --- | --- | --- |
| 1 | 77.4% | 78.6% | 76.6% | 77.6% |
| 2 | 89.4% | 92.8% | 85.6% | 91.4% |
| 3 | 90.2% | 92.8% | 86.0% | 91.6% |

News-only leads for each order at 10M. At fixed budget, the original mixture allocates many words to court, and news-plus-Wikipedia replaces about half the news exposure with Wikipedia. Neither more domains nor more general text automatically improves the recorded accuracy.

The following representatives maximize recorded accuracy within each grid. For the tied news models, the smaller bigram is shown as a practical representative, not a statistically established winner.

| Corpus | Order | Accuracy / correct | WER | CER | MRR | Binary MiB |
| --- | --- | --- | --- | --- | --- | --- |
| All five domains | 3 | 90.2% / 451 | 1.3467% | 0.6927% | 0.94983 | 224.18 |
| News only | 2 | 92.8% / 464 | 0.8688% | 0.4448% | 0.96367 | 85.33 |
| Wikipedia only | 3 | 86.0% / 430 | 1.9331% | 0.9786% | 0.92590 | 278.60 |
| News + Wikipedia | 3 | 91.6% / 458 | 1.1295% | 0.6291% | 0.95750 | 269.72 |

Both news models select 464 correct sentences. The bigram alone is correct in 4 records and the trigram alone in 4 records. WER is 0.8688% versus 0.8905%; binaries are 85.33 versus 239.38 MiB. Equal sentence accuracy need not mean identical predictions or error severity.

### 3.6 Qualitative successes and failures

These examples use the original all-five-domain 10M models. Numerical costs are the trigram's `-log10(P)/N` for each displayed alternative; lower is better.

#### Context resolves a case ending: `ua_title_0005`

| Item | Saved sentence | Trigram cost |
| --- | --- | --- |
| Reference | Волонтери через брак коштів припиняють вивезення тіл загиблих із зони АТО | 2.914421 |
| Unigram selection | Волонтери через брак коштів припиняють вивезення тіл загиблих із зоні АТО | 3.310689 |

The unigram selects an incorrect case ending while the trigram favors the reference. This is consistent with contextual statistics resolving a word-form ambiguity.

#### Context distinguishes a real-word substitution: `ua_title_0246`

| Item | Saved sentence | Trigram cost |
| --- | --- | --- |
| Reference | Польща депортує росіянина, який запустив дрон над офісом президента | 4.083764 |
| Unigram selection | Польща депортує росіянина, який запустив трон над офісом президента | 4.209800 |

The alternatives contain different real words. The unigram's frequent word is incorrect in this verb-object context, while the trigram selects the reference.

#### An extra repeated word reduces average cost: `ua_title_0024`

| Item | Saved sentence | Trigram cost |
| --- | --- | --- |
| Reference | Вчені відтворили звучання відомої середньовічної мелодії | 5.412263 |
| Trigram selection | Вчені відтворили звучання відомої відомої середньовічної мелодії | 5.379568 |

The selected sentence repeats a word but obtains a lower average cost. This is a failure of the length-normalized objective despite vocabulary coverage.

#### Apostrophe fragmentation changes tokenization: `ua_title_0317`

| Item | Saved sentence | Trigram cost |
| --- | --- | --- |
| Reference | Джиро д'Італія. Подвійний успіх Берналя | 5.351907 |
| Trigram selection | Джиро д ' Італія. Подвійний успіх Берналя | 4.641739 |

An internal apostrophe stays within a word; a fragmented apostrophe creates token boundaries. Changed token count and context alter perplexity. The preprocessing deliberately does not repair this error.

#### Vocabulary coverage does not guarantee a correct name: `ua_title_0049`

| Item | Saved sentence | Trigram cost |
| --- | --- | --- |
| Reference | Атлетіко обіграв Бетіс на виїзді | 4.799914 |
| Trigram selection | Атлетіко обіграв Бетис на виїзді | 4.550764 |

The wrong name spelling is preferred. Stored vocabulary diagnostics identify both name spellings as known words. Local probability does not guarantee the factual spelling of a named entity.

The following deterministic pattern diagnostics summarize errors in the best representative of each corpus. Other/combined errors have not all been manually annotated.

| Corpus / order | Total errors | Adjacent repetition only | Apostrophe fragmentation only | Other / combined | Longer than reference |
| --- | --- | --- | --- | --- | --- |
| All five domains / 3 | 49 | 30 | 2 | 17 | 44 |
| News only / 2 | 36 | 26 | 2 | 8 | 32 |
| Wikipedia only / 3 | 70 | 44 | 1 | 25 | 60 |
| News + Wikipedia / 3 | 42 | 29 | 2 | 11 | 38 |

Repetition is the largest identified category in each selected model. A repetition penalty would be a new scoring method and has not been silently added.

### 3.7 Exact-score ties and uncertainty

Gold-first order benefits classical first-index decisions when distinct normalized sentences tie for best score. The following analytical diagnostics use saved scores; they do not change official predictions or represent new scored runs.

| Representative | Recorded accuracy | Uniform top-tie expectation | Minimum with different tie choices | Distinct-text top ties |
| --- | --- | --- | --- | --- |
| All five domains, 3-gram | 90.2% | 87.58% | 84.8% | 28 |
| News only, 2-gram | 92.8% | 91.08% | 89.0% | 19 |
| Wikipedia only, 3-gram | 86.0% | 81.52% | 77.0% | 47 |
| News + Wikipedia, 3-gram | 91.6% | 90.33% | 88.8% | 14 |

The original best trigram has an unadjusted 95% Wilson interval of **87.28% to 92.51%**. This is descriptive for a fixed model and is not adjusted for selecting the best of nine. Against the original 10M bigram it has **5 unique wins and 1 unique losses**, with exact paired McNemar p = **0.21875**. The observed 0.8-point gain is not presented as established superiority.

### 3.8 Classical training and inference resources

| Grid | Sum of training seconds | Inference seconds, nine models | Binary storage MiB |
| --- | --- | --- | --- |
| All five domains | 33.563 | 0.228197 | 390.70 |
| News only | 34.351 | 0.211842 | 402.28 |
| Wikipedia only | 38.960 | 0.218829 | 494.91 |
| News + Wikipedia | 36.749 | 0.219001 | 469.30 |

Classical inference includes normalization, scoring and ranking after one warmup; it excludes model loading, metrics, OOV scans and writes. These are single-run timings. Unigram context resets add Python overhead. Per-model training, loading and storage appear in Appendix B. No classical peak runtime-RAM measurement is inferred from these file sizes.

## Appendix A. All 38 Models: Accuracy and Efficiency

Accuracy/WER/CER use normalized text. Classical timing is historical Docker/CPU scoring; LLM timing is GPU/HTTP prompted ranking. Stored MB is decimal disk size, not RAM. LLM pretraining counts are not controlled or recorded here.

| Model / nominal words | Accuracy | WER | CER | MRR | ms/record | Stored MB |
| --- | --- | --- | --- | --- | --- | --- |
| News only, 2-gram, 10,000,000 words | 92.8% | 0.8688% | 0.4448% | 0.96367 | 0.03955 | 89.48 |
| News only, 3-gram, 10,000,000 words | 92.8% | 0.8905% | 0.4734% | 0.96333 | 0.04125 | 251.01 |
| News + Wikipedia, 3-gram, 10,000,000 words | 91.6% | 1.1295% | 0.6291% | 0.95750 | 0.04165 | 282.82 |
| News + Wikipedia, 2-gram, 10,000,000 words | 91.4% | 1.1295% | 0.6482% | 0.95583 | 0.04057 | 110.77 |
| All five domains, 3-gram, 10,000,000 words | 90.2% | 1.3467% | 0.6927% | 0.94983 | 0.06573 | 235.07 |
| All five domains, 2-gram, 10,000,000 words | 89.4% | 1.4770% | 0.7149% | 0.94567 | 0.04164 | 90.49 |
| News only, 2-gram, 1,000,000 words | 88.4% | 1.7376% | 0.7531% | 0.93815 | 0.04107 | 16.10 |
| Qwen3.5-4B Q4_K_M | 88.4% | 1.5856% | 0.3527% | 0.93317 | 570.83843 | 3324.17 |
| News only, 3-gram, 1,000,000 words | 87.8% | 1.8245% | 0.8388% | 0.93515 | 0.03912 | 35.86 |
| News + Wikipedia, 3-gram, 1,000,000 words | 86.8% | 1.8679% | 0.9183% | 0.92868 | 0.04077 | 39.09 |
| News + Wikipedia, 2-gram, 1,000,000 words | 86.2% | 1.9983% | 0.9405% | 0.92568 | 0.04105 | 18.69 |
| Wikipedia only, 3-gram, 10,000,000 words | 86.0% | 1.9331% | 0.9786% | 0.92590 | 0.04127 | 292.13 |
| Wikipedia only, 2-gram, 10,000,000 words | 85.6% | 1.9983% | 1.0200% | 0.92390 | 0.04152 | 120.08 |
| All five domains, 3-gram, 1,000,000 words | 83.4% | 2.6064% | 1.1693% | 0.90968 | 0.03945 | 33.83 |
| All five domains, 2-gram, 1,000,000 words | 82.4% | 2.8019% | 1.2170% | 0.90352 | 0.03998 | 15.68 |
| Wikipedia only, 2-gram, 1,000,000 words | 81.4% | 2.7802% | 1.2487% | 0.90113 | 0.03969 | 19.74 |
| News only, 2-gram, 100,000 words | 80.8% | 3.2146% | 1.4171% | 0.89425 | 0.03789 | 2.69 |
| Wikipedia only, 3-gram, 1,000,000 words | 80.8% | 2.9105% | 1.2964% | 0.89880 | 0.03965 | 39.94 |
| News only, 3-gram, 100,000 words | 80.4% | 3.3449% | 1.4330% | 0.89227 | 0.03950 | 4.92 |
| News + Wikipedia, 3-gram, 100,000 words | 79.4% | 3.4970% | 1.4426% | 0.88477 | 0.04490 | 5.19 |
| News + Wikipedia, 2-gram, 100,000 words | 79.2% | 3.4970% | 1.4235% | 0.88366 | 0.03963 | 2.94 |
| All five domains, 3-gram, 100,000 words | 78.6% | 3.9096% | 1.4711% | 0.87814 | 0.03783 | 4.71 |
| News only, 1-gram, 10,000,000 words | 78.6% | 2.9105% | 1.5283% | 0.88848 | 0.06127 | 15.82 |
| All five domains, 2-gram, 100,000 words | 78.0% | 3.9748% | 1.4997% | 0.87490 | 0.04319 | 2.59 |
| News + Wikipedia, 1-gram, 10,000,000 words | 77.6% | 3.1060% | 1.6523% | 0.88308 | 0.06439 | 24.88 |
| All five domains, 1-gram, 10,000,000 words | 77.4% | 3.3449% | 1.5855% | 0.87832 | 0.06490 | 20.90 |
| Wikipedia only, 2-gram, 100,000 words | 77.2% | 3.9314% | 1.6396% | 0.87087 | 0.03786 | 3.04 |
| News only, 1-gram, 1,000,000 words | 77.0% | 3.4318% | 1.6078% | 0.87573 | 0.06161 | 4.77 |
| Wikipedia only, 3-gram, 100,000 words | 77.0% | 3.9965% | 1.6554% | 0.86977 | 0.04271 | 5.28 |
| Wikipedia only, 1-gram, 10,000,000 words | 76.6% | 3.3449% | 1.7158% | 0.87525 | 0.06334 | 30.07 |
| News + Wikipedia, 1-gram, 1,000,000 words | 76.4% | 3.6273% | 1.6872% | 0.87227 | 0.06270 | 6.33 |
| News only, 1-gram, 100,000 words | 76.2% | 3.8879% | 1.6396% | 0.86679 | 0.06243 | 1.18 |
| All five domains, 1-gram, 1,000,000 words | 76.0% | 3.8445% | 1.6205% | 0.86722 | 0.06337 | 5.24 |
| News + Wikipedia, 1-gram, 100,000 words | 75.4% | 4.1268% | 1.6904% | 0.86159 | 0.06233 | 1.37 |
| Wikipedia only, 1-gram, 100,000 words | 74.2% | 4.5178% | 1.8651% | 0.85335 | 0.06951 | 1.47 |
| All five domains, 1-gram, 100,000 words | 73.6% | 4.6916% | 1.7666% | 0.85122 | 0.06031 | 1.17 |
| Wikipedia only, 1-gram, 1,000,000 words | 73.6% | 4.2137% | 1.8461% | 0.85565 | 0.06211 | 7.19 |
| MamayLM-4B Q4_K_M | 18.6% | 13.7272% | 3.3522% | 0.46999 | 770.53493 | 2489.89 |

## Appendix B. Every Classical Configuration: Coverage, Training and Storage

OOV columns are occurrence rates. Inference is total time for 500 records; Appendix A gives mean latency. Loading and training are separate. ARPA/binary sizes are distinct disk artifacts; no column measures runtime peak RAM.

### All five domains

| Budget | Order | Ref OOV | Hyp OOV | Selected OOV | Training s | Load s | Inference s | ARPA MiB | Binary MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100,000 | 1 | 28.084% | 35.332% | 27.925% | 0.552 | 0.0122 | 0.030154 | 0.70 | 1.12 |
| 100,000 | 2 | 28.084% | 35.332% | 27.935% | 0.382 | 0.0179 | 0.021593 | 3.77 | 2.47 |
| 100,000 | 3 | 28.084% | 35.332% | 27.898% | 0.454 | 0.0360 | 0.018915 | 8.87 | 4.50 |
| 1,000,000 | 1 | 8.970% | 17.663% | 9.316% | 0.816 | 0.0302 | 0.031684 | 3.24 | 4.99 |
| 1,000,000 | 2 | 8.970% | 17.663% | 9.203% | 1.222 | 0.1318 | 0.019991 | 25.25 | 14.96 |
| 1,000,000 | 3 | 8.970% | 17.663% | 9.151% | 2.119 | 0.1113 | 0.019723 | 69.59 | 32.27 |
| 10,000,000 | 1 | 2.324% | 10.868% | 2.544% | 3.075 | 0.1103 | 0.032452 | 13.01 | 19.93 |
| 10,000,000 | 2 | 2.324% | 10.868% | 2.407% | 8.303 | 0.2691 | 0.020820 | 160.08 | 86.30 |
| 10,000,000 | 3 | 2.324% | 10.868% | 2.366% | 16.639 | 1.2691 | 0.032866 | 517.24 | 224.18 |

### News only

| Budget | Order | Ref OOV | Hyp OOV | Selected OOV | Training s | Load s | Inference s | ARPA MiB | Binary MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100,000 | 1 | 21.633% | 29.688% | 21.483% | 0.647 | 0.0117 | 0.031214 | 0.73 | 1.13 |
| 100,000 | 2 | 21.633% | 29.688% | 21.591% | 0.386 | 0.0227 | 0.018944 | 3.99 | 2.56 |
| 100,000 | 3 | 21.633% | 29.688% | 21.662% | 0.478 | 0.0313 | 0.019748 | 9.42 | 4.69 |
| 1,000,000 | 1 | 5.908% | 15.180% | 6.233% | 0.669 | 0.0350 | 0.030806 | 2.95 | 4.55 |
| 1,000,000 | 2 | 5.908% | 15.180% | 6.029% | 1.368 | 0.0667 | 0.020534 | 27.34 | 15.36 |
| 1,000,000 | 3 | 5.908% | 15.180% | 6.068% | 2.393 | 0.1507 | 0.019561 | 76.34 | 34.20 |
| 10,000,000 | 1 | 1.672% | 10.484% | 1.847% | 2.801 | 0.0929 | 0.030635 | 9.84 | 15.08 |
| 10,000,000 | 2 | 1.672% | 10.484% | 1.683% | 8.114 | 0.3323 | 0.019774 | 168.74 | 85.33 |
| 10,000,000 | 3 | 1.672% | 10.484% | 1.682% | 17.494 | 1.1445 | 0.020626 | 579.96 | 239.38 |

### Wikipedia only

| Budget | Order | Ref OOV | Hyp OOV | Selected OOV | Training s | Load s | Inference s | ARPA MiB | Binary MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100,000 | 1 | 31.277% | 38.204% | 30.904% | 0.616 | 0.0159 | 0.034753 | 0.85 | 1.40 |
| 100,000 | 2 | 31.277% | 38.204% | 31.001% | 0.514 | 0.0229 | 0.018928 | 4.32 | 2.90 |
| 100,000 | 3 | 31.277% | 38.204% | 30.975% | 0.493 | 0.0382 | 0.021355 | 9.63 | 5.04 |
| 1,000,000 | 1 | 11.425% | 19.675% | 11.538% | 0.816 | 0.0407 | 0.031056 | 4.42 | 6.86 |
| 1,000,000 | 2 | 11.425% | 19.675% | 11.476% | 1.411 | 0.1028 | 0.019847 | 30.96 | 18.82 |
| 1,000,000 | 3 | 11.425% | 19.675% | 11.485% | 2.287 | 0.1666 | 0.019824 | 79.57 | 38.09 |
| 10,000,000 | 1 | 3.562% | 11.822% | 3.708% | 3.492 | 0.1031 | 0.031672 | 17.99 | 28.68 |
| 10,000,000 | 2 | 3.562% | 11.822% | 3.618% | 9.257 | 0.3460 | 0.020760 | 207.37 | 114.51 |
| 10,000,000 | 3 | 3.562% | 11.822% | 3.599% | 20.074 | 1.1225 | 0.020634 | 625.87 | 278.60 |

### News + Wikipedia

| Budget | Order | Ref OOV | Hyp OOV | Selected OOV | Training s | Load s | Inference s | ARPA MiB | Binary MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100,000 | 1 | 22.676% | 30.502% | 22.662% | 0.693 | 0.0191 | 0.031165 | 0.85 | 1.31 |
| 100,000 | 2 | 22.676% | 30.502% | 22.706% | 0.420 | 0.0333 | 0.019816 | 4.23 | 2.80 |
| 100,000 | 3 | 22.676% | 30.502% | 22.680% | 0.474 | 0.0281 | 0.022451 | 9.64 | 4.95 |
| 1,000,000 | 1 | 6.560% | 15.446% | 6.816% | 0.857 | 0.0334 | 0.031352 | 3.88 | 6.04 |
| 1,000,000 | 2 | 6.560% | 15.446% | 6.617% | 1.468 | 0.0770 | 0.020526 | 30.10 | 17.83 |
| 1,000,000 | 3 | 6.560% | 15.446% | 6.578% | 2.389 | 0.1297 | 0.020384 | 80.05 | 37.28 |
| 10,000,000 | 1 | 1.629% | 10.194% | 1.803% | 3.390 | 0.0881 | 0.032195 | 15.46 | 23.73 |
| 10,000,000 | 2 | 1.629% | 10.194% | 1.614% | 9.034 | 0.4036 | 0.020287 | 197.77 | 105.64 |
| 10,000,000 | 3 | 1.629% | 10.194% | 1.636% | 18.023 | 1.0400 | 0.020824 | 622.60 | 269.72 |

## Appendix C. All Classical Exact-Tie Diagnostics

These are analytical alternatives derived from saved exact top ties, not new scored runs. Uniform tie accuracy averages the correct fraction within each top tie. Minimum accuracy is a worst-case choice from those same ties. Official scores and first-index predictions remain unchanged.

| Corpus | Budget | Order | Recorded accuracy | Uniform top-tie expectation | Minimum top-tie accuracy | Distinct-text top ties |
| --- | --- | --- | --- | --- | --- | --- |
| All five domains | 100,000 | 1 | 73.6% | 46.18% | 25.8% | 253 |
| All five domains | 100,000 | 2 | 78.0% | 49.21% | 27.6% | 265 |
| All five domains | 100,000 | 3 | 78.6% | 49.81% | 28.2% | 265 |
| All five domains | 1,000,000 | 1 | 76.0% | 66.01% | 56.2% | 105 |
| All five domains | 1,000,000 | 2 | 82.4% | 72.44% | 62.8% | 104 |
| All five domains | 1,000,000 | 3 | 83.4% | 73.34% | 63.6% | 104 |
| All five domains | 10,000,000 | 1 | 77.4% | 74.93% | 72.6% | 27 |
| All five domains | 10,000,000 | 2 | 89.4% | 86.68% | 83.8% | 30 |
| All five domains | 10,000,000 | 3 | 90.2% | 87.58% | 84.8% | 28 |
| News only | 100,000 | 1 | 76.2% | 55.13% | 37.8% | 204 |
| News only | 100,000 | 2 | 80.8% | 59.17% | 41.4% | 207 |
| News only | 100,000 | 3 | 80.4% | 58.84% | 41.2% | 206 |
| News only | 1,000,000 | 1 | 77.0% | 70.80% | 65.0% | 67 |
| News only | 1,000,000 | 2 | 88.4% | 81.51% | 74.8% | 71 |
| News only | 1,000,000 | 3 | 87.8% | 81.21% | 74.8% | 68 |
| News only | 10,000,000 | 1 | 78.6% | 77.18% | 75.6% | 15 |
| News only | 10,000,000 | 2 | 92.8% | 91.08% | 89.0% | 19 |
| News only | 10,000,000 | 3 | 92.8% | 91.18% | 89.2% | 18 |
| Wikipedia only | 100,000 | 1 | 74.2% | 44.72% | 23.2% | 272 |
| Wikipedia only | 100,000 | 2 | 77.2% | 47.54% | 25.8% | 274 |
| Wikipedia only | 100,000 | 3 | 77.0% | 47.10% | 25.2% | 276 |
| Wikipedia only | 1,000,000 | 1 | 73.6% | 60.61% | 48.4% | 133 |
| Wikipedia only | 1,000,000 | 2 | 81.4% | 68.01% | 55.4% | 136 |
| Wikipedia only | 1,000,000 | 3 | 80.8% | 67.34% | 54.6% | 136 |
| Wikipedia only | 10,000,000 | 1 | 76.6% | 72.38% | 68.2% | 44 |
| Wikipedia only | 10,000,000 | 2 | 85.6% | 81.22% | 76.8% | 46 |
| Wikipedia only | 10,000,000 | 3 | 86.0% | 81.52% | 77.0% | 47 |
| News + Wikipedia | 100,000 | 1 | 75.4% | 51.57% | 31.8% | 230 |
| News + Wikipedia | 100,000 | 2 | 79.2% | 54.44% | 33.8% | 240 |
| News + Wikipedia | 100,000 | 3 | 79.4% | 54.74% | 34.2% | 238 |
| News + Wikipedia | 1,000,000 | 1 | 76.4% | 69.15% | 62.4% | 76 |
| News + Wikipedia | 1,000,000 | 2 | 86.2% | 78.66% | 71.4% | 77 |
| News + Wikipedia | 1,000,000 | 3 | 86.8% | 79.26% | 72.0% | 77 |
| News + Wikipedia | 10,000,000 | 1 | 77.6% | 76.40% | 75.0% | 13 |
| News + Wikipedia | 10,000,000 | 2 | 91.4% | 90.03% | 88.4% | 16 |
| News + Wikipedia | 10,000,000 | 3 | 91.6% | 90.33% | 88.8% | 14 |
