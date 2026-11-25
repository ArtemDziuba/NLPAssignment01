# Improving Speech Recognition Output Using a Classical Language Model

Modern Automatic Speech Recognition (ASR) systems rarely produce a single final transcription.
Instead, they generate multiple candidate sentence outputs (also called N-best hypotheses), which differ by small spelling, grammatical, or morphological errors.

In this assignment, you will use a classical n-gram Language Model (LM) to rescore these candidate outputs and select the most fluent and linguistically plausible sentence.

Although large neural language models exist today, classical language models are still widely used in speech recognition systems because they are:

- Fast — inference takes milliseconds even on CPUs
- Memory-efficient — models are small and easy to deploy
- Reliable and predictable — no hallucinations, stable behavior
- Easy to integrate — commonly used in ASR decoding pipelines (e.g., Kaldi, Vosk)

For example, a classical LM can reliably correct:

- spelling errors (березня → перезня)
- missing function words
- grammatical inconsistencies

You are given a dataset of 500 sentence outputs, where each record contains:

- a reference sentence
- several ASR-like candidate outputs with realistic recognition errors

Your task is to:

1. Train a classical n-gram language model (using KenLM)
2. Score each candidate sentence (using perplexity score)
3. Select the best candidate
4. Analyze performance and failure cases

## Tools and Resources
- KenLM (official repository): https://github.com/kpu/kenlm
- Suggested Ukrainian training corpus: 
    - UberText — https://lang.org.ua/en/ubertext/

(You may also use Ukrainian Wikipedia or other public Ukrainian text corpora.)

## Activity Instructions
### 1. Language Model Training
- Train n-gram language models with different orders from 1 to 3-grams
- Clearly document:
    - training corpus
    - dataset size

preprocessing choices

### 2. Sentence Scoring
Use the trained language model to compute the log-probability of each candidate sentence.

Example:
```
def sentence_logprob(sent: str) -> float:
    # log10 probability
    return model.score(sent, bos=True, eos=True)
```
To compare sentences of different lengths, use perplexity:

Perplexity = $10^{-\frac{log_{10}P}{N}}$
where N is the number of words in the sentence and P is the sentence logprob.

The sentence with the lowest perplexity should be selected as the best output.

### 3. Evaluation
- Compare the selected sentence against the reference.
- Evaluate performance using:
    - exact-match accuracy
    - qualitative error analysis
- Perform experiments with:
    - different n-gram orders
    - different training dataset sizes

## Scoring (25 points total)

### 10 points — N-gram Comparison

- Comparison table for 1-gram, 2-gram, and 3-gram models
- Must include:
    - accuracy results
    - brief discussion
- Results must be shown both in code and in the PDF report

### 7 points — Training Dataset Size Analysis

- Train the LM on different corpus sizes
- Provide a comparison table showing:
    - dataset size
    - performance
- Analyze how increasing data size affects results

### 3 points — Report Quality

- Clear explanations of:
    - why the language model works in some cases
    - where it fails
- Meaningful error analysis with concrete examples

### 5 points — Bonus (Optional)

- Compare classical LM rescoring with a modern large language model
- Discuss:
    - accuracy
    - inference speed
    - memory usage
- This section is optional and will not penalize students who skip it

### Deliverables
1. PDF report (5–7 pages)
2. Source code (GitHub repository or archive)