"""Tiered extractor tests: routing logic and provenance, with fake backends.

Proves the core behaviour: good primary extraction is used as-is (no LLM cost);
poor primary extraction escalates to the fallback; and the outcome correctly
records which backend produced the text so downstream can tag low-confidence
(LLM-read) content.
"""

from __future__ import annotations

from pathlib import Path

from qsph.retrieval.tiered_extractor import TieredExtractor

GOOD_TEXT = (
    "We report density functional theory calculations of Bi2Te3. The density of "
    "states shows a pronounced peak below the Fermi level, consistent with the "
    "large Seebeck coefficient. Antimony doping shifts the Fermi level. These "
    "findings agree with prior photoemission measurements of the surface states."
)

# Column-spliced garbage, long enough to clear the near-empty veto.
BAD_TEXT = " ".join("W e r e p o r t d e n s i t y f u n c t i o n a l".split()) * 6


class FakeBackend:
    def __init__(self, text: str, tag: str):
        self.text = text
        self.tag = tag
        self.called = False

    def extract(self, path: Path) -> str:
        self.called = True
        return self.text


def test_good_primary_used_no_fallback():
    primary = FakeBackend(GOOD_TEXT, "primary")
    fallback = FakeBackend("LLM READ TEXT", "fallback")
    tx = TieredExtractor(primary, fallback=fallback)

    outcome = tx.extract_with_outcome(Path("dummy.pdf"))
    assert outcome.backend_used == "primary"
    assert not outcome.escalated
    assert not outcome.low_confidence_source
    assert primary.called
    assert not fallback.called  # fallback NOT invoked -> no LLM cost


def test_poor_primary_escalates_to_fallback():
    primary = FakeBackend(BAD_TEXT, "primary")
    fallback = FakeBackend(GOOD_TEXT, "fallback")
    tx = TieredExtractor(primary, fallback=fallback)

    outcome = tx.extract_with_outcome(Path("dummy.pdf"))
    assert outcome.escalated
    assert outcome.backend_used == "llm_fallback"
    assert outcome.low_confidence_source  # LLM-read -> weaker provenance
    assert fallback.called
    assert outcome.text == GOOD_TEXT
    # The primary's quality report explains WHY it escalated.
    assert not outcome.primary_quality.acceptable


def test_no_fallback_returns_primary_with_quality_flag():
    """Without a fallback, poor primary text is returned but flagged, not hidden."""
    primary = FakeBackend(BAD_TEXT, "primary")
    tx = TieredExtractor(primary, fallback=None)

    outcome = tx.extract_with_outcome(Path("dummy.pdf"))
    assert outcome.backend_used == "primary"
    assert not outcome.escalated
    assert not outcome.primary_quality.acceptable  # flagged as poor


def test_extract_returns_plain_text():
    primary = FakeBackend(GOOD_TEXT, "primary")
    tx = TieredExtractor(primary)
    assert tx.extract(Path("dummy.pdf")) == GOOD_TEXT


def test_expected_terms_drive_escalation():
    """Clean text that lacks the searched-for content escalates so the LLM can
    look for it (the parser may have dropped the relevant section)."""
    off_topic_clean = (
        "The compressive strength of reinforced concrete increased with curing "
        "time under cyclic loading across all laboratory samples tested here."
    )
    primary = FakeBackend(off_topic_clean, "primary")
    fallback = FakeBackend(GOOD_TEXT, "fallback")
    tx = TieredExtractor(
        primary, fallback=fallback, expected_terms=["Bi2Te3", "density of states"]
    )
    outcome = tx.extract_with_outcome(Path("dummy.pdf"))
    assert outcome.escalated  # topic terms absent -> escalate to double-check
