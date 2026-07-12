"""Quality-assessor tests.

The gate must ACCEPT clean scientific text and ESCALATE the real failure modes:
near-empty output, column-splicing garbage, ligature/symbol noise, and dropped
topic content. We build representative good/bad samples so the gate is validated
against the actual failure signatures, not toy strings.
"""

from __future__ import annotations

from qsph.retrieval.quality import (
    QualityThresholds,
    assess_extraction,
)

# A clean, well-extracted scientific paragraph.
GOOD = (
    "We report density functional theory calculations of the electronic "
    "structure of Bi2Te3. The density of states shows a pronounced peak just "
    "below the Fermi level, consistent with the large Seebeck coefficient "
    "observed experimentally. The computed value of N(E_F) is moderate. Antimony "
    "substitution shifts the Fermi level toward the valence band edge. These "
    "results are in good agreement with prior angle-resolved photoemission "
    "measurements of the surface states."
)

# Column-splicing failure: words broken apart, lines interleaved. Long enough
# to clear the near-empty veto so the broken-word signal is what triggers.
COLUMN_SPLICE = (
    "W e r e p o r t d e n s i t y f u n c t i o n a l t h e o r y a n d t h e "
    "s u r f a c e s t a t e s w e r e m e a s u r e d b y A R P E S o n t h e "
    "s a m p l e a n d t h e F e r m i l e v e l w a s d e t e r m i n e d "
    "f r o m t h e c o m p u t e d b a n d s t r u c t u r e o f t h e "
    "c r y s t a l w i t h g o o d a g r e e m e n t o v e r a l l r a n g e"
)

# Ligature/math mangling: heavy symbol noise, low alphabetic density. Long
# enough to clear the near-empty veto so alphabetic-density is what triggers.
SYMBOL_NOISE = (
    "!@#$%^&*()_+ =XX III ttcc 0.3pm0.1 SSPP ~~~~ |||| <<<>>> %%%% &&&& "
    "]]][[[ ///bbb ..... ooo SSS aaa xxxddd nnn =YY III ttcc SSPP ~~~~ "
    "|||| <<<>>> %%%% &&&& ]]][[[ ///bbb ..... ooo SSS aaa xxxddd nnn "
    "@@@ ### $$$ %%% ^^^ &&& *** ((( ))) +++ === {{{ }}} [[[ ]]] ||| \\\\\\ "
)


def test_good_extraction_accepted():
    report = assess_extraction(GOOD)
    assert report.acceptable
    assert report.score > 0.8
    assert not report.reasons


def test_empty_extraction_escalates():
    report = assess_extraction("")
    assert not report.acceptable
    assert report.score == 0.0
    assert "near-empty" in report.reasons[0]


def test_near_empty_extraction_escalates():
    report = assess_extraction("Bi2Te3.")
    assert not report.acceptable
    assert "near-empty" in report.reasons[0]


def test_column_splice_escalates():
    report = assess_extraction(COLUMN_SPLICE)
    assert not report.acceptable
    # The broken-word rate and/or average word length should flag it.
    assert any(
        "broken-word" in r or "average word length" in r for r in report.reasons
    )


def test_symbol_noise_escalates():
    report = assess_extraction(SYMBOL_NOISE)
    assert not report.acceptable
    assert any("alphabetic density" in r for r in report.reasons)


def test_expected_terms_present_passes():
    report = assess_extraction(GOOD, expected_terms=["Bi2Te3", "Seebeck"])
    assert report.acceptable


def test_expected_terms_absent_penalises():
    # Clean text, but about a totally different topic than the search terms.
    off_topic = (
        "The mechanical properties of reinforced concrete were studied under "
        "cyclic loading. Compressive strength increased with curing time across "
        "all tested samples in the laboratory over several weeks of observation."
    )
    report = assess_extraction(
        off_topic, expected_terms=["Bi2Te3", "density of states"]
    )
    # Text is clean but the relevant content is absent -> escalate to check.
    assert not report.acceptable
    assert any("expected topic terms" in r for r in report.reasons)
    assert report.score < 0.6  # multiplicative penalty applied


def test_thresholds_are_tunable():
    # With a very lax gate, borderline text passes; with a strict one it fails.
    borderline = "Short text with only two sentences here. Second one now."
    lax = assess_extraction(
        borderline, thresholds=QualityThresholds(min_sentences=1, min_chars=20)
    )
    strict = assess_extraction(
        borderline, thresholds=QualityThresholds(min_sentences=10, min_chars=20)
    )
    assert lax.acceptable
    assert not strict.acceptable


def test_metrics_populated():
    report = assess_extraction(GOOD)
    assert "alpha_ratio" in report.metrics
    assert "broken_word_rate" in report.metrics
    assert "n_words" in report.metrics
    assert report.metrics["n_words"] > 0
