"""§11.3 — the aligned word-level comparison against 27 CFR 16.21.

The brief's rule was a byte-exact match. Only 73.5 % of warning-bearing corpus images
produce one (§2.6, §14.1), so a byte-exact green rule would falsely deny green to roughly
one compliant label in four. These ten tests pin the replacement: word-for-word at the word
(that is, the legal) unit, tolerant of character-level OCR noise.

Each case is driven through the shipped verdict rule, so what is asserted is what an agent
would actually be shown.
"""

import pytest
from conftest import check_by_id, make_read, make_result

from app import verdict
from app.statute import CLAUSE_1, STATUTE_TEXT, STATUTE_TOKENS

REAL_DEFECTS = {"substituted", "missing", "extra"}


def text_check(region_text: str, **read_over):
    """The `warning_text` check for a label whose statement reads as `region_text`."""
    from app.config import get_settings

    settings = get_settings()
    read = make_read(region_text=region_text, page_text=region_text, **read_over)
    return check_by_id(verdict.evaluate(make_result(read=read), None, settings), "warning_text")


def swap(old: str, new: str) -> str:
    assert old in STATUTE_TEXT, f"{old!r} is not in the statute"
    return STATUTE_TEXT.replace(old, new, 1)


def test_an_exact_statement_is_green_with_no_defects():
    """§11.3.13 — the reference case."""
    check = text_check(STATUTE_TEXT)
    assert check.state == "green"
    diff = check.evidence["diff"]
    assert [d["s"] for d in diff] == ["equal"] * 43


def test_a_substituted_function_word_is_amber_not_green():
    """§6.2 refinement (adversarial-review catch): the one-function-word tolerance exists
    for OCR *dropping* a small word, not for a *different* word being printed. "the
    ability" for "your ability" is a wording change — an agent decides, never a green."""
    check = text_check(swap("your ability", "the ability"))
    assert check.state == "amber", check.detail
    assert "the" in check.detail or "your" in check.detail


def test_a_one_glyph_near_miss_outside_the_fold_is_suspect_and_amber():
    """§6.2 suspect bucket — pinned by the live-deploy catch: the box read "probloms" for
    "problems." on blurred tiny type (e/o is not a fold class), and the old rule made that
    a critical substitution — a compliant label machine-rejected. A suspect must force
    amber: never red, never green."""
    check = text_check(swap("problems.", "probloms."))
    assert check.state == "amber", check.detail
    states = {d["s"] for d in check.evidence["diff"]}
    assert "suspect" in states and "substituted" not in states
    assert check.evidence["suspect_words"] == 1
    assert "probloms" in check.detail and "problems" in check.detail
    assert check.citation == "27 CFR 16.21"


def test_a_case_slip_inside_a_word_is_noise_not_a_deviation():
    """§11.3.14 — `AcCording`: the recogniser's case slip, not the label's wording."""
    check = text_check(swap("According", "AcCording"))
    assert check.state == "green"
    entry = next(d for d in check.evidence["diff"] if d["w"] == "AcCording")
    assert entry["s"] == "noise"
    assert entry["expected"] == "According"


def test_a_confusable_glyph_is_noise_not_a_deviation():
    """§11.3.15 — `WARNlNG`, lowercase L for capital I. The classic OCR confusion."""
    check = text_check(swap("WARNING:", "WARNlNG:"))
    assert check.state == "green"
    entry = next(d for d in check.evidence["diff"] if d["w"].startswith("WARN"))
    assert entry["s"] == "noise"


def test_a_reworded_content_word_is_a_rejection():
    """§11.3.16 — "health issues" for "health problems" is the deviation Jenny rejects.

    `fuzz.ratio("issues", "problems")` is 28.6, nowhere near the 80 noise threshold, so no
    plausible OCR fold can excuse it.
    """
    check = text_check(swap("health problems.", "health issues."))
    assert check.state == "red"
    entry = next(d for d in check.evidence["diff"] if d.get("expected", "").startswith("problems"))
    assert entry["s"] == "substituted"
    assert "issues" in check.detail and "problems" in check.detail, (
        f"the detail must name both words so an agent can see the deviation: {check.detail!r}"
    )


def test_a_missing_clause_is_caught_by_coverage_not_by_defect_counting():
    """§11.3.17 — clause (2) absent.

    `partial_ratio` scores clause (1) alone at 100 against the whole statute, so a global
    similarity score cannot see this. Per-clause coverage can, which is why §6.2 computes it
    independently of the opcode counts.
    """
    check = text_check(CLAUSE_1)
    assert check.state == "red"
    coverage = check.evidence["clause_coverage"]
    assert coverage["1"] >= 0.9
    assert coverage["2"] < 0.5, f"clause (2) coverage should collapse, got {coverage['2']}"


def test_one_dropped_article_stays_green():
    """§11.3.18 — a dropped `the` is an OCR artefact essentially every time.

    Jenny's requirement is word-for-word, so any content word changing is a deviation and
    one is enough; function words get the carve-out. That asymmetry is the whole rule.
    """
    check = text_check(swap("According to the Surgeon", "According to Surgeon"))
    assert check.state == "green"
    assert any(d["s"] == "missing" for d in check.evidence["diff"])


def test_three_dropped_function_words_are_no_longer_dismissible():
    """§11.3.19 — one is noise; three is a read (or a label) that needs a human."""
    damaged = STATUTE_TEXT
    for old, new in [
        ("According to the Surgeon", "According to Surgeon"),
        ("because of the risk", "because of risk"),
        ("a car or operate", "a car operate"),
    ]:
        damaged = damaged.replace(old, new, 1)
    check = text_check(damaged)
    assert check.state == "amber"


def test_label_copy_bleeding_in_at_the_edges_is_ignored():
    """§11.3.20 — the band crop catches neighbouring copy; that is not an insertion."""
    check = text_check(f"Bottled by Old Tom Distillery {STATUTE_TEXT} 750 mL 45% Alc./Vol.")
    assert check.state == "green"
    assert not [d for d in check.evidence["diff"] if d["s"] in REAL_DEFECTS]


def test_an_extra_word_inside_the_statement_is_not_green():
    """§11.3.21 — an interior insertion changes what the label says."""
    check = text_check(swap("Consumption of alcoholic", "Consumption of premium alcoholic"))
    assert check.state != "green"
    assert any(d["s"] == "extra" and d["w"] == "premium" for d in check.evidence["diff"])


def test_the_diff_accounts_for_every_statute_token_exactly_once():
    """§11.3.22 — the diff is the evidence the UI renders; it may not lose a word.

    A reviewer reading the panel must see all 43 statutory words, each with a state.
    """
    check = text_check(swap("birth defects.", "birth defect."))
    accounted = [d for d in check.evidence["diff"] if d["s"] != "extra"]
    expected = [d.get("expected", d["w"]) for d in accounted]
    assert expected == list(STATUTE_TOKENS), (
        "every statute token must appear exactly once, in order, whatever its state"
    )
    assert len(accounted) == 43


def test_an_unreliable_read_can_never_produce_a_rejection():
    """§11.3 (the TRUST gate, §6.2) — the asymmetric-cost rule at the verdict layer.

    label_027 (glare) and label_017 (small type on a rotated frame) both read badly and are
    both compliant. Rejecting them would be the expensive error, so a read that fails TRUST
    is amber whatever the diff says.
    """
    from app.config import get_settings

    settings = get_settings()
    garbled = swap("health problems.", "heaith prcblems,")
    check = text_check(garbled, prefix_score=88.0)
    assert check.state == "amber", "prefix_score below text_trust_min must never reject"
    assert settings.text_trust_min == pytest.approx(95.0)
    shown = f"{check.title} {check.detail}".lower()
    assert "read" in shown and "review" in shown, (
        "an agent must be told this is a reading problem held for review, not a violation: "
        f"{check.title!r} / {check.detail!r}"
    )
