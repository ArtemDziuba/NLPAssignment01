"""Score candidates with assignment perplexity and deterministic ranking."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .preprocessing import normalize_text


class KenLMScorer:
    """Load KenLM; reset context for a unigram stored in a zero-bigram container."""

    def __init__(self, path: Path, effective_order: int) -> None:
        try:
            import kenlm
        except ImportError as error:
            raise RuntimeError(
                "KenLM is required: run inside Docker or WSL with the kenlm extra"
            ) from error
        self.model = kenlm.Model(str(path))
        self.effective_order = effective_order

    def score(self, text: str, bos: bool = True, eos: bool = True) -> float:
        """Return log10 probability, including EOS exactly once."""
        if self.effective_order == 1:
            # Each word is queried through KenLM with empty context. The container
            # adds no observed bigrams and cannot introduce context dependence.
            tokens = text.split() + (["</s>"] if eos else [])
            return sum(self.model.score(token, bos=False, eos=False) for token in tokens)
        return self.model.score(text, bos=bos, eos=eos)

    def __contains__(self, token: str) -> bool:
        return token in self.model


@dataclass(frozen=True)
class CandidateScore:
    """Keep original indices and sentences alongside every normalized model score."""

    index: int
    raw: str
    normalized: str
    log10_probability: float
    cross_entropy: float
    perplexity: float | None
    words: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rank_candidates(model: Any, candidates: tuple[str, ...] | list[str]) -> list[CandidateScore]:
    """Rank by -log10(P)/N, equivalent to minimum perplexity without overflow."""
    if not candidates:
        raise ValueError("Cannot rank an empty candidate list")
    scored = []
    for index, raw in enumerate(candidates):
        normalized = normalize_text(raw)
        words = len(normalized.split())
        if not words:
            raise ValueError(f"Candidate {index} has no word tokens")
        logprob = float(model.score(normalized, bos=True, eos=True))
        if not math.isfinite(logprob):
            raise ValueError(f"Nonfinite log probability for candidate {index}")
        cross_entropy = -logprob / words
        perplexity = 10.0**cross_entropy if cross_entropy < 308 else None
        scored.append(
            CandidateScore(index, raw, normalized, logprob, cross_entropy, perplexity, words)
        )
    return sorted(scored, key=lambda candidate: (candidate.cross_entropy, candidate.index))
