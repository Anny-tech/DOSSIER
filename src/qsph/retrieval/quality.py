"""Extraction-quality assessment: the gate that decides when to escalate.

The tiered extractor uses a cheap rule-based parser first and an expensive LLM
reader only as a fallback. This module is the gate between them. Its honest
scope, stated up front:

    It detects extraction FAILURE (garbled/empty/mangled text), NOT extraction
    CORRECTNESS (whether the extracted N(E_F) value is right). Correctness is the
    physics-critic's job, downstream. A cheap deterministic gate cannot verify
    that a number is the right number; it CAN tell that the text is too broken to
    trust, which is exactly the routing decision we need.

The signals are chosen to catch the documented failure modes of rule-based
parsers on scientific PDFs:
  * near-empty output          -> scanned/image PDF or parser crash
  * low alphabetic density     -> ligature/math mangling, symbol noise
  * high broken-word rate      -> column-splicing (words split mid-token)
  * few well-formed sentences  -> layout destroyed reading order
  * missing expected terms     -> the paper's topic terms absent despite the
                                  paper being about them (optional, query-aware)

Thresholds are explicit and tunable, deliberately mirroring the physics-critic's
auditable-thresholds philosophy. They should be validated on a labelled set of
good/bad extractions (see docs) before being trusted in production; the defaults
are conservative starting points, not final science.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A "word" for our purposes: a run of letters, optionally with internal digits
# (e.g. Bi2Te3). Used to measure broken-word rate and alphabetic density.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_SENTENCE_END_RE = re.compile(r"[.!?]\s")


@dataclass(frozen=True)
class QualityThresholds:
    """Tunable gate thresholds. Validate on labelled extractions before trusting.

    Every threshold is a knob a reviewer can argue with; none are hidden in the
    logic below.
    """

    min_chars: int = 200
    # Fraction of characters that must be alphabetic/space (vs symbol noise).
    min_alpha_ratio: float = 0.55
    # Max fraction of "words" that look broken (single-char fragments, common
    # with column splicing and hyphenation errors).
    max_broken_word_rate: float = 0.20
    # Minimum number of well-formed sentences expected in a real article page.
    min_sentences: int = 3
    # Minimum average word length; column-spliced garbage skews very low.
    min_avg_word_len: float = 3.0


@dataclass
class QualityReport:
    """Result of assessing one extraction."""

    score: float  # 0..1, higher is better
    acceptable: bool  # score passed the gate -> no fallback needed
    reasons: list[str] = field(default_factory=list)  # why it failed, if it did
    metrics: dict[str, float] = field(default_factory=dict)  # raw measurements

    def summary(self) -> str:
        verdict = "ACCEPTABLE" if self.acceptable else "POOR -> escalate"
        detail = "; ".join(self.reasons) if self.reasons else "all checks passed"
        return f"quality={self.score:.2f} [{verdict}]: {detail}"


def assess_extraction(
    text: str,
    *,
    thresholds: QualityThresholds | None = None,
    expected_terms: list[str] | None = None,
) -> QualityReport:
    """Assess whether extracted text is good enough to use without LLM fallback.

    Args:
        text: the extracted text to assess.
        thresholds: tunable gate; defaults to conservative QualityThresholds.
        expected_terms: optional topic terms the paper should contain (e.g. the
            material or property being searched). If provided and NONE appear,
            that is a strong signal extraction dropped the relevant content.

    Returns:
        QualityReport. `acceptable=False` means escalate to the LLM reader.

    The score is a weighted combination of the individual signals, each in 0..1,
    so partial degradation lowers the score smoothly rather than tripping a
    single hard flag. The gate then thresholds the score AND applies hard
    vetoes (e.g. near-empty text always escalates regardless of other signals).
    """
    t = thresholds or QualityThresholds()
    reasons: list[str] = []
    metrics: dict[str, float] = {}

    stripped = text.strip()
    n_chars = len(stripped)
    metrics["n_chars"] = float(n_chars)

    # Hard veto 1: near-empty extraction. Nothing else matters.
    if n_chars < t.min_chars:
        return QualityReport(
            score=0.0,
            acceptable=False,
            reasons=[f"near-empty extraction ({n_chars} < {t.min_chars} chars)"],
            metrics=metrics,
        )

    # Alphabetic density.
    alpha_or_space = sum(1 for c in stripped if c.isalpha() or c.isspace())
    alpha_ratio = alpha_or_space / n_chars
    metrics["alpha_ratio"] = alpha_ratio

    # Word-level signals.
    words = _WORD_RE.findall(stripped)
    n_words = len(words)
    metrics["n_words"] = float(n_words)
    if n_words == 0:
        return QualityReport(
            score=0.0,
            acceptable=False,
            reasons=["no recognisable words in extraction"],
            metrics=metrics,
        )

    broken = sum(1 for w in words if len(w) == 1)
    broken_rate = broken / n_words
    metrics["broken_word_rate"] = broken_rate

    avg_word_len = sum(len(w) for w in words) / n_words
    metrics["avg_word_len"] = avg_word_len

    # Sentence count.
    n_sentences = len(_SENTENCE_END_RE.findall(stripped)) + 1
    metrics["n_sentences"] = float(n_sentences)

    # Optional topic-term presence.
    terms_present = True
    if expected_terms:
        low = stripped.lower()
        terms_present = any(term.lower() in low for term in expected_terms)
        metrics["expected_terms_present"] = float(terms_present)

    # --- score each signal in 0..1, collecting failure reasons ---------------
    def _ratio_score(value: float, threshold: float, *, higher_better: bool) -> float:
        if higher_better:
            return min(1.0, value / threshold) if threshold > 0 else 1.0
        # lower better: full score at 0, zero score at/above threshold
        return max(0.0, 1.0 - value / threshold) if threshold > 0 else 1.0

    alpha_score = _ratio_score(alpha_ratio, t.min_alpha_ratio, higher_better=True)
    if alpha_ratio < t.min_alpha_ratio:
        reasons.append(
            f"low alphabetic density ({alpha_ratio:.2f} < {t.min_alpha_ratio})"
        )

    broken_score = _ratio_score(
        broken_rate, t.max_broken_word_rate, higher_better=False
    )
    if broken_rate > t.max_broken_word_rate:
        reasons.append(
            f"high broken-word rate ({broken_rate:.2f} > {t.max_broken_word_rate})"
        )

    wordlen_score = _ratio_score(avg_word_len, t.min_avg_word_len, higher_better=True)
    if avg_word_len < t.min_avg_word_len:
        reasons.append(
            f"low average word length ({avg_word_len:.2f} < {t.min_avg_word_len})"
        )

    sentence_score = _ratio_score(
        float(n_sentences), float(t.min_sentences), higher_better=True
    )
    if n_sentences < t.min_sentences:
        reasons.append(
            f"too few sentences ({n_sentences} < {t.min_sentences})"
        )

    if not terms_present:
        reasons.append("none of the expected topic terms found")

    # Weighted overall score. Broken-word and alpha density are the strongest
    # indicators of the column-splice/ligature failures we most want to catch.
    score = (
        0.30 * broken_score
        + 0.30 * alpha_score
        + 0.20 * sentence_score
        + 0.20 * wordlen_score
    )
    # Topic-term absence applies a multiplicative penalty (it is a strong signal
    # that the *relevant* content was lost, even if the page has clean text).
    if not terms_present:
        score *= 0.5

    acceptable = not reasons  # any failing signal -> escalate, to be safe
    return QualityReport(
        score=round(score, 3),
        acceptable=acceptable,
        reasons=reasons,
        metrics=metrics,
    )
