# Improving Speech Recognition Output Using a Classical Language Model

## Overview

This study evaluates Ukrainian ASR candidate rescoring using KenLM language models trained on UberText 2.0. Across 36 classical configurations, the highest recorded normalized exact-match accuracy was **92.8% (464/500)**, achieved by news-only bigram and trigram models trained on approximately 10 million words. The original five-domain trigram reached 90.2%. Two optional prompted LLM baselines, Qwen3.5-4B and MamayLM-4B, achieved 88.4% and 18.6%. 

## 1. Language model training

### Data and preprocessing

The evaluation dataset contains 500 unique Ukrainian utterances, each with a reference and eight hypotheses (4,000 candidate positions per model). It remains fixed across experiments and is excluded from training.

Training uses five sentence-split domains from [UberText 2.0](https://lang.org.ua/en/ubertext/): court, fiction, news, social media and Wikipedia. The original experiment scanned approximately 156 million sentence occurrences (5.23 GB compressed) and retained up to 200,000 unique sentences per domain. After cross-domain deduplication, the pool contained 998,334 sentences. Additional experiments used up to 750,000 sentences per included domain to support larger single-domain budgets.

Both corpus and hypotheses undergo Unicode NFC normalization, lowercasing, apostrophe standardization, word/digit tokenization and whitespace collapse. Internal apostrophes are preserved; other punctuation is removed or treated as token boundaries. Training additionally excludes URL/markup content, non-Cyrillic or empty lines, and sentences outside 2–150 words. Exact normalized matches to evaluation references and candidates are excluded before sampling; spelling correction, stemming and transliteration are not applied.

Deterministic hash-priority sampling (seed 42), deduplication and round-robin sentence mixing produce nested whole-sentence subsets. The nominal budgets are 100K, 1M and 10M word tokens. Actual counts can be slightly smaller to avoid cutting sentences. Mixed domains are balanced by sentence count rather than word count.

### Model configurations

Four corpus selections (all five domains, news, Wikipedia, news + Wikipedia) are each evaluated at three budgets and three n-gram orders, giving **36 models**. Models of different orders within the same corpus/budget use identical training text. A unigram estimates independent word probabilities; a bigram conditions on one previous word and a trigram on two. Higher orders capture local syntax but suffer from sparse contexts.

KenLM 0.3.0 was run in a Linux container on Windows/WSL2 using `lmplz -o ORDER -S 2G`, followed by `build_binary probing`. Bigram/trigram estimation uses modified Kneser–Ney smoothing and backoff. For order one, the generic loader requires a compatibility container with zero observed bigrams and zero backoffs. Scoring resets context for each word and EOS, preserving the original unigram probabilities; native checks verified the resulting scores against source ARPA, including an unknown word.

## 2. Sentence scoring and evaluation

Each candidate is normalized and scored independently. With $L=\log_{10}P(S,\mathrm{EOS}\mid\mathrm{BOS})$ and $N$ normalized words, the assignment-defined perplexity is:

$$\operatorname{PPL}(S)=10^{-L/N}.$$

```python
L = scorer.score(normalize_text(candidate), bos=True, eos=True)
N = len(normalize_text(candidate).split())
perplexity = 10 ** (-L / N)
```

The lowest-perplexity candidate is selected by minimizing the equivalent average cost $-L/N$. EOS contributes to the probability but not the denominator, as specified in the assignment; KenLM's built-in perplexity includes EOS in the denominator and is not used. Empty normalized candidates are rejected. Exact score ties favor the original candidate index.

The primary metric is normalized exact-match accuracy: the fraction of 500 selected sentences matching their normalized references. Raw exact match additionally preserves punctuation, case and spacing. Supplementary metrics are corpus word error rate (WER) and character error rate (CER), calculated from total edit counts divided by reference lengths, mean reciprocal rank (MRR) of the first correct candidate, and reference out-of-vocabulary (OOV) rate.

**Baselines and evaluation bias.** Oracle accuracy is 100% because every record contains a correct candidate. The first original candidate also achieves 100% accuracy because it is always the reference. Seeded random selection achieves 15.6%; its analytical expectation is 15.425% because 117 records contain a second normalized match (617 correct positions in total). Therefore the experiment evaluates linguistic ranking, not improvement over the original order. Gold-first tie-breaking can inflate classical accuracy and is examined below.

## 3. Classical model results

### N-gram order and training dataset size

Table 1. Normalized exact-match accuracy (%) on the same 500 records. Budgets denote nominal word counts.

| Corpus | Budget | 1-gram | 2-gram | 3-gram |
| --- | ---: | ---: | ---: | ---: |
| All five | 100K | 73.6 | 78.0 | 78.6 |
| All five | 1M | 76.0 | 82.4 | 83.4 |
| All five | 10M | 77.4 | 89.4 | 90.2 |
| News | 100K | 76.2 | 80.8 | 80.4 |
| News | 1M | 77.0 | 88.4 | 87.8 |
| News | 10M | 78.6 | **92.8** | **92.8** |
| Wikipedia | 100K | 74.2 | 77.2 | 77.0 |
| Wikipedia | 1M | 73.6 | 81.4 | 80.8 |
| Wikipedia | 10M | 76.6 | 85.6 | 86.0 |
| News + Wikipedia | 100K | 75.4 | 79.2 | 79.4 |
| News + Wikipedia | 1M | 76.4 | 86.2 | 86.8 |
| News + Wikipedia | 10M | 77.6 | 91.4 | 91.6 |

All contextual models outperform the corresponding unigrams. In the original 10M grid, moving from unigram to bigram improves accuracy by 12.0 percentage points (pp), while adding another context word contributes only 0.8 pp. The trigram advantage is not universal: news and Wikipedia bigrams lead their trigrams at smaller budgets, consistent with sparse higher-order contexts. The original 10M trigram's small advantage over its bigram is not statistically established (exact paired McNemar p = 0.21875).

Increasing the corpus from 100K to 10M words improves every order within every selection. Across the four grids, unigram gains are 2.2–3.8 pp; bigram gains are 8.4–12.2 pp; trigram gains are 9.0–12.4 pp. The Wikipedia unigram nevertheless declines from 74.2% at 100K to 73.6% at 1M. More data is beneficial overall, particularly for context estimates, but is not guaranteed to improve each intermediate result.

For the original five-domain trigram, reference OOV falls from 28.08% to 8.97% to 2.32% as the budget grows, while accuracy rises from 78.6% to 83.4% to 90.2%. Greater vocabulary coverage reduces reliance on unknown-word probabilities, although valid-word substitutions and repetition remain unresolved.

### Corpus domain comparison

Table 2. Best recorded model per corpus; all use the 10M budget. The news bigram is the smaller of two accuracy-tied models.

| Corpus / order | Actual words | Accuracy | WER | Ref. OOV | Binary MiB |
| --- | --- | --- | --- | --- | --- |
| All five / 3 | 9,999,984 | 90.2% | 1.347% | 2.32% | 224.18 |
| News / 2 | 9,999,995 | 92.8% | 0.869% | 1.67% | 85.33 |
| Wikipedia / 3 | 9,999,983 | 86.0% | 1.933% | 3.56% | 278.60 |
| News + Wikipedia / 3 | 10,000,000 | 91.6% | 1.129% | 1.63% | 269.72 |

News-only leads at each order at 10M words, plausibly because the evaluation sentences resemble news headlines. The five-domain mixture allocates about 40% of its words to court and only 16.3% to news; news + Wikipedia allocates about half to news. Thus adding domains at a fixed total budget also replaces in-domain text. These results do not isolate domain effects from the sampling-capacity change or prove general superiority of news for all ASR tasks.

The news bigram and trigram each select 464 correct sentences, but each has four unique correct predictions missed by the other. The bigram is a practical choice because it has marginally lower WER (0.869% vs. 0.891%) and a smaller binary (85.33 vs. 239.38 MiB).

### Effect of exact-score ties

When distinct normalized sentences share the lowest score, original-index tie-breaking favors the gold-first ordering. From saved scores, uniform random selection among exact top ties yields expected accuracy of 87.58% for the original best trigram (recorded 90.2%) and 91.08% for the news bigram (recorded 92.8%). For the news trigram the corresponding expectation is 91.18%. These are analytical expectations, not rerun predictions. The full tie analysis is supplied separately.

## 4. Qualitative error analysis

The following examples are from the original 10M five-domain models. Costs are the trigram's $-L/N$ (lower is better). Excerpts highlight the changed text; scores are for complete sentences.

**Case ending — success (ua_title_0005).** The unigram selects «із **зоні** АТО» (cost 3.310689), whereas the trigram selects the reference «із **зони** АТО» (2.914421). Context helps resolve Ukrainian case agreement.

**Real-word substitution — success (ua_title_0246).** The unigram selects «запустив **трон** над офісом президента» (4.209800); the trigram selects «запустив **дрон** над офісом президента» (4.083764). Both words exist, but the verb–object context favors the reference.

**Repetition — failure (ua_title_0024).** The reference is «Вчені відтворили звучання відомої середньовічної мелодії» (5.412263). The trigram selects «Вчені відтворили звучання **відомої відомої** середньовічної мелодії» (5.379568). An additional predictable word can lower the length-normalized average cost despite introducing an error.

**Tokenization — failure (ua_title_0317).** The incorrect «Джиро **д ' Італія**» (4.641739) is preferred to «Джиро **д'Італія**» (5.351907). Fragmentation changes token boundaries and the denominator. Normalization intentionally does not repair this recognition error.

**Named entity — failure (ua_title_0049).** The trigram prefers «Атлетіко обіграв **Бетис** на виїзді» (4.550764) to the reference spelling «Атлетіко обіграв **Бетіс** на виїзді» (4.799914). Both name variants occur in the vocabulary; linguistic probability does not ensure factual spelling.

Table 3. Automatically identified error patterns among the best model per corpus (not exhaustive manual annotations).

| Corpus | Errors | Repetition only | Apostrophe only | Other / combined |
| --- | ---: | ---: | ---: | ---: |
| All five | 49 | 30 | 2 | 17 |
| News | 36 | 26 | 2 | 8 |
| Wikipedia | 70 | 44 | 1 | 25 |
| News + Wikipedia | 42 | 29 | 2 | 11 |

Repetition is the dominant identified failure pattern: 26 of 36 news-bigram errors are adjacent repetition only, and 32 of its errors select longer sentences. A repetition penalty or alternative length objective is a possible future experiment, not part of the reported scoring method. Missing broader context also limit resolution of plausible alternatives.

## 5. Bonus: comparison with modern LLMs

Qwen3.5-4B and MamayLM-4B (Q4_K_M) were evaluated through local Ollama without fine-tuning. Both received the same normalized hypotheses in a deterministic per-record shuffled order (seed 42), without original indices or gold labels. An English structured prompt requested a complete ranking of eight candidates; all responses were valid without retries. The best ranked candidate was selected. This is prompted ranking rather than LLM perplexity scoring.

Table 4. Accuracy and measured inference resources. KenLM models run on CPU; LLMs use local GPU/HTTP inference.

| Model | Norm. accuracy | Raw accuracy | WER | ms/record | Disk MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| All five 3-gram | 90.2% | 90.2% | 1.347% | 0.066 | 235.07 |
| News 2-gram | 92.8% | 92.8% | 0.869% | 0.040 | 89.48 |
| News 3-gram | 92.8% | 92.8% | 0.891% | 0.041 | 251.01 |
| Qwen3.5-4B | 88.4% | 77.6% | 1.586% | 570.838 | 3,324.17 |
| MamayLM-4B | 18.6% | 15.0% | 13.727% | 770.535 | 2,489.89 |

Qwen ranks substantially better than the 15.6% shuffled-first baseline, but trails the best recorded KenLM models by 4.4 pp in normalized accuracy. Its CER (0.353%) is lower than the news bigram's (0.445%), showing that fewer character edits need not imply more wholly correct sentences. MamayLM selected the first displayed candidate in 482/500 records (96.4%), versus 34/500 for Qwen. Its low accuracy is strongly associated with positional preference under this prompt and decoding setup, not proof of poor Ukrainian language modelling in general.

The hardware was an Intel i7-13650HX, approximately 16 GB RAM and an RTX 4060 Laptop GPU (8 GB VRAM). Qwen and MamayLM took 285.419 and 385.267 seconds for the 500 requests. KenLM timings exclude loading and include normalization/scoring/ranking; LLM timings include local HTTP, generation and validation but exclude normalization, warmup and reporting. They are single-pass historical measurements, not an architecture-only benchmark.

Sampled combined Ollama service RAM peaked at 7.696 GiB for Qwen and 7.160 GiB for MamayLM. Ollama reported model GPU allocations of 2.925 and 2.566 GiB, respectively. Service working sets may double-count shared pages and sampling may miss peaks. Matched KenLM runtime RAM was not measured. Detailed settings, token counts and resource diagnostics are available in the supplementary LLM report.

## 6. Conclusions and limitations

The main findings are that local context improves Ukrainian candidate selection, larger corpora disproportionately benefit contextual models, and news-domain training performs best on this particular headline-like dataset. A 10M-word news bigram offers the strongest recorded accuracy with lower storage cost than its tied trigram. Most identified remaining errors arise from repetition, while names and tokenization also cause failures.

The gold-first data order and exact ties favor the classical evaluation; LLM presentation is shuffled, so comparisons are not fully controlled. Normalization removes some raw differences, exact training exclusion cannot rule out related material, and LLM pretraining overlap is unknown. Only one fixed synthetic evaluation set and sampling seed were used, without independent test data or repeated timings. 

Full per-model results, exact training sizes, error examples and tie diagnostics are provided in the [supplementary classical results](supplementary/classical_results.md); the [LLM experiment report](20261124_llm_experiment_results.md) contains complete bonus measurements. Reproduction commands and artifact links are documented in the repository README.

**Resources:** [KenLM](https://github.com/kpu/kenlm); [UberText](https://lang.org.ua/en/ubertext/).
