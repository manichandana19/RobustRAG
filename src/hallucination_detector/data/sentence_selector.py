from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Sequence

from hallucination_detector.data.schema import RAGTruthRecord
from hallucination_detector.modeling.tfidf_logistic import normalize_text

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
_NEGATION_TERMS = {
    "no", "not", "never", "none", "neither", "nor", "cannot", "can't",
    "doesn't", "don't", "didn't", "won't", "wouldn't", "isn't", "aren't",
    "wasn't", "weren't", "hardly", "scarcely", "barely", "false", "untrue",
}


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences with fallback to newline or segment chunks.
    Ensures empty or non-splittable strings return at least one non-empty string if text exists.
    """
    clean_text = normalize_text(text)
    if not clean_text:
        return []

    # If newlines exist, break by newlines first
    paragraphs = [p.strip() for p in clean_text.split("\n") if p.strip()]
    sentences: list[str] = []

    for para in paragraphs:
        splits = _SENTENCE_SPLIT_RE.split(para)
        for s in splits:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)

    if not sentences and clean_text:
        sentences.append(clean_text)

    return sentences


def extract_keywords(text: str) -> set[str]:
    """Extract lowercase alphanumeric terms."""
    return set(_WORD_RE.findall(text.lower()))


def extract_numbers_and_dates(text: str) -> set[str]:
    """Extract numeric, percentage, and date digits."""
    return set(_NUMBER_RE.findall(text))


def extract_negation_markers(text: str) -> set[str]:
    """Extract negation cue words."""
    words = set(_WORD_RE.findall(text.lower()))
    return words.intersection(_NEGATION_TERMS)


@dataclass
class SentenceSelector:
    """
    Context sentence ranker and selector with contradiction awareness.

    Design principles:
    1. Lexical Relevance (TF-IDF / BM25 style IDF):
       Computes sentence relevance to Question and Answer without acting as an echo chamber.
    2. Contradiction Preservation:
       Boosts sentences that share named entities / numbers with the answer but contain
       distinct terms or explicit negation cues (high probability of conflict).
    3. Provenance & Original Ordering:
       Selected sentences are always reconstructed in their original sequential order in the context.
    """

    idf: dict[str, float] = field(default_factory=dict)
    default_idf: float = 1.0

    @classmethod
    def fit_from_records(cls, records: Sequence[RAGTruthRecord]) -> "SentenceSelector":
        """Fit document frequency statistics strictly on the training records."""
        from collections import Counter

        doc_freq: Counter[str] = Counter()
        n_docs = len(records)

        for record in records:
            tokens = extract_keywords(
                f"{normalize_text(record.question)} {normalize_text(record.context)}"
            )
            for token in tokens:
                doc_freq[token] += 1

        idf: dict[str, float] = {}
        for token, count in doc_freq.items():
            # Standard smooth IDF formula
            idf[token] = math.log((1 + n_docs) / (1 + count)) + 1.0

        return cls(idf=idf, default_idf=1.0)

    def _score_sentence(
        self,
        sentence: str,
        q_tokens: set[str],
        a_tokens: set[str],
        a_numbers: set[str],
        a_negations: set[str],
        alpha: float = 0.4,
    ) -> float:
        """
        Compute sentence relevance score.
        Score = alpha * Score(Q) + (1 - alpha) * Score(A) + Contradiction_Bonus
        """
        s_tokens = extract_keywords(sentence)
        if not s_tokens:
            return 0.0

        # Term overlap with Question
        q_overlap = s_tokens.intersection(q_tokens)
        q_score = sum(self.idf.get(t, self.default_idf) for t in q_overlap)

        # Term overlap with Answer
        a_overlap = s_tokens.intersection(a_tokens)
        a_score = sum(self.idf.get(t, self.default_idf) for t in a_overlap)

        # Contradiction bonus:
        # If sentence shares numbers or entities with the answer, but contains negation or
        # differs in polarity, it holds higher evidentiary value for hallucination detection.
        s_numbers = extract_numbers_and_dates(sentence)
        s_negations = extract_negation_markers(sentence)

        contra_bonus = 0.0
        # Shared numerical topic but discrepancy check
        if a_numbers and s_numbers:
            if not a_numbers.issubset(s_numbers):
                contra_bonus += 2.0  # Divergent number claim

        # Polarity discordance (one is negated, the other is not)
        if bool(a_negations) != bool(s_negations) and len(a_overlap) >= 2:
            contra_bonus += 2.5

        return (alpha * q_score) + ((1.0 - alpha) * a_score) + contra_bonus

    def select_sentences(
        self,
        question: str,
        context: str,
        answer: str,
        max_sentences: int | None = None,
        max_words: int | None = None,
    ) -> list[tuple[int, str]]:
        """
        Rank sentences and return top selections preserved in their original context order.
        Returns list of (original_index, sentence_text).
        """
        sentences = split_into_sentences(context)
        if not sentences:
            return []

        # Fallback: if very short context, return all
        if len(sentences) <= 2:
            return list(enumerate(sentences))

        q_tokens = extract_keywords(question)
        a_tokens = extract_keywords(answer)
        a_numbers = extract_numbers_and_dates(answer)
        a_negations = extract_negation_markers(answer)

        scored_sentences: list[tuple[int, float, str]] = []
        for idx, s in enumerate(sentences):
            score = self._score_sentence(
                sentence=s,
                q_tokens=q_tokens,
                a_tokens=a_tokens,
                a_numbers=a_numbers,
                a_negations=a_negations,
            )
            # Prioritize opening sentences slightly (lead bias in document context)
            lead_bias = 1.0 / (idx + 1)
            final_score = score + lead_bias
            scored_sentences.append((idx, final_score, s))

        # Rank by score descending
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        selected: list[tuple[int, str]] = []
        total_words = 0

        for idx, score, s in scored_sentences:
            words_in_s = len(s.split())
            if max_words is not None and (total_words + words_in_s > max_words) and selected:
                continue
            selected.append((idx, s))
            total_words += words_in_s

            if max_sentences is not None and len(selected) >= max_sentences:
                break

        # Re-sort to preserve original document chronological flow
        selected.sort(key=lambda x: x[0])
        return selected
