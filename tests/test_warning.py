"""§11.2 — reading order, the case rule, the escalation policy, and absence.

Everything here replays cached pass-1 output (`fixtures/ocr_pass1.json`) or hand-built
quads. No OCR runs.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest
from conftest import make_ocr_result, make_read, make_result

from app import statute, warning

BOLD_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "bold"

# The six lines a 750 mL label wraps the statement onto, in reading order.
WARNING_LINES = [
    "GOVERNMENT WARNING: (1) According to the",
    "Surgeon General, women should not drink alcoholic",
    "beverages during pregnancy because of the risk of",
    "birth defects. (2) Consumption of alcoholic beverages",
    "impairs your ability to drive a car or operate",
    "machinery, and may cause health problems.",
]

# The v_missing class of spike/corpus/ground_truth.json: seven labels that carry no health
# warning at all. Ground truth, copied here so the test runs without the git-ignored corpus.
LABELS_WITHOUT_A_WARNING = {
    "label_005.png",
    "label_013.png",
    "label_023.png",
    "label_032.png",
    "label_033.png",
    "label_042.png",
    "label_052.png",
}


def line_quad(row: int, width: float, height: float = 40.0, x0: float = 90.0):
    """A level, left-aligned text box on row `row`."""
    y0 = 100.0 + row * (height + 12.0)
    return np.asarray(
        [[x0, y0], [x0 + width, y0], [x0 + width, y0 + height], [x0, y0 + height]],
        dtype=np.float64,
    )


def test_reading_order_reassembles_shuffled_boxes_into_statute_order():
    """§11.2.6 — the detector emits boxes in arbitrary order; assembly must fix it."""
    lines = [
        {"text": text, "score": 0.99, "quad": line_quad(row, 20.0 * len(text))}
        for row, text in enumerate(WARNING_LINES)
    ]
    shuffled = [lines[i] for i in (3, 0, 5, 2, 4, 1)]
    text = warning.assemble_text(make_ocr_result(shuffled), deskew=False)
    assert text == " ".join(WARNING_LINES), (
        "lines were re-assembled out of order; the statute cannot match a scrambled page"
    )
    # The deskew path (§4.3) must not disturb a page that is already level.
    assert warning.assemble_text(make_ocr_result(shuffled), deskew=True) == text


def test_reading_order_falls_back_to_a_plain_join_when_geometry_is_useless():
    """§11.2.7 — degenerate quads must not raise, and must not lose text."""
    zero = np.zeros((4, 2), dtype=np.float64)
    lines = [{"text": text, "score": 0.99, "quad": zero.copy()} for text in WARNING_LINES]
    text = warning.assemble_text(make_ocr_result(lines), deskew=True)
    assert text == " ".join(WARNING_LINES)


def test_warning_presence_over_the_whole_corpus(cached_ocr, settings):
    """§11.2.8 — 49 present / 7 absent, 0 false positives, 0 false negatives.

    The anchor is a fuzzy match, so this is the measurement that says the 80 cut has room
    on both sides (§2.6). Driven through the shipped verdict rule, not a re-implementation.
    """
    from app import verdict

    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    wrong = []
    for filename, entry in sorted(cached_ocr.items()):
        result = make_result(page_text=entry["page_text"], read=make_read(entry["page_text"]))
        state = verdict.evaluate(result, None, settings)
        check = next(c for c in state.checks if c.id == "warning_present")
        found = check.state == "green"
        expected = filename not in LABELS_WITHOUT_A_WARNING
        if expected and found:
            confusion["tp"] += 1
        elif expected:
            confusion["fn"] += 1
            wrong.append((filename, check.state, check.detail))
        elif found:
            confusion["fp"] += 1
            wrong.append((filename, check.state, check.detail))
        else:
            confusion["tn"] += 1
    assert confusion == {"tp": 49, "fp": 0, "tn": 7, "fn": 0}, f"{confusion}; wrong: {wrong}"


def test_case_is_read_by_uppercase_ratio_not_string_equality(settings):
    """§11.2.9 — one OCR glyph slip must not be reported as a Title Case violation.

    Strict equality scores 0.9388 on this corpus because the recogniser returns GoVERNMENT
    on 3 of 49 images; the uppercase ratio scores 1.000 (§2.7).
    """
    cases = {
        "GOVERNMENT WARNING": ("caps", 1.0),
        "GoVERNMENT WARNING": ("caps", 16 / 17),  # one glyph slipped: still capitals
        "Government Warning": ("title", 2 / 17),  # the classic rejection
        "GOVERNMENT Warning": ("other", 11 / 17),  # genuinely mixed: an agent decides
    }
    for prefix, (expected_case, expected_ratio) in cases.items():
        text = prefix + statute.STATUTE_TEXT[len(statute.PREFIX) :]
        lines = [{"text": text, "score": 0.99, "quad": line_quad(0, 900.0)}]
        read = warning.read_warning(make_ocr_result(lines), deskew=False)
        assert read.upper_ratio == pytest.approx(expected_ratio, abs=0.01), prefix
        assert read.prefix_case == expected_case, f"{prefix!r} classified {read.prefix_case!r}"
    # And the cuts sit in wide empty space between those measurements (§6.3).
    assert settings.case_title_max < 0.5 < settings.case_caps_min


def test_escalation_policy_has_no_confidence_term(settings):
    """§11.2.10 — fires on prefix < 75, region < 95, case != caps, or an empty page.

    Engine confidence was measured useless as a quality signal (§2.3): garbled reads carry
    the same 0.98–0.99 confidence as perfect ones. The parameter is deleted from the
    signature, not defaulted off, so nobody can re-enable it.
    """
    parameters = inspect.signature(warning.should_escalate).parameters
    assert not [p for p in parameters if "conf" in p.lower()], (
        f"should_escalate grew a confidence parameter: {list(parameters)}"
    )

    clean = make_read(region_score=100.0, prefix_score=100.0, prefix_case="caps", n_lines=13)
    assert warning.should_escalate(clean, settings)[0] is False

    for damaged, why in [
        (make_read(prefix_score=70.0, region_score=100.0, prefix_case="caps"), "prefix"),
        (make_read(prefix_score=100.0, region_score=90.0, prefix_case="caps"), "region"),
        (make_read(prefix_score=100.0, region_score=100.0, prefix_case="title"), "case"),
        (make_read(prefix_score=100.0, region_score=100.0, prefix_case="caps", n_lines=0), "empty"),
    ]:
        fire, reason = warning.should_escalate(damaged, settings)
        assert fire is True, f"{why}: expected escalation"
        assert reason, f"{why}: escalation must say why"


def test_a_worse_second_pass_is_discarded(monkeypatch, settings, bold_fixtures):
    """§11.2.11 — the region-score-only rule of §4.4 step 6.

    The enhancement ladder must be non-harmful by construction: if pass 2 comes back worse
    than pass 1, pass 1 is what the verdict sees. Both passes are canned here, so this runs
    the real ladder with no engine.
    """
    from app import ocr, pipeline

    entry = bold_fixtures["compliant"]
    damaged = [dict(line) for line in entry["lines"]]
    # Title Case on the prefix: a pass-1 read that implies a rejection is always re-verified
    # before the app rejects (§4.2's require_caps clause).
    damaged[0]["text"] = damaged[0]["text"].replace("GOVERNMENT WARNING:", "Government Warning:")
    first = make_ocr_result(damaged)
    second = make_ocr_result([dict(entry["lines"][0], text="GOVERNMENT")])  # a much worse read
    canned = iter([first, second])

    def fake_run(img, long_side):
        return next(canned)

    monkeypatch.setattr(ocr, "run", fake_run)
    if hasattr(pipeline, "run"):
        monkeypatch.setattr(pipeline, "run", fake_run, raising=False)

    result = pipeline.analyse(BOLD_FIXTURE_DIR / entry["file"], settings)
    assert result.passes == 2, "the truncated statement should have escalated"
    assert result.read.region_score == pytest.approx(
        warning.read_warning(first, settings.reading_order_deskew).region_score
    ), "pass 2 scored worse and must have been discarded"


def test_absence_is_a_rejection_only_when_the_page_is_readable(cached_ocr, settings):
    """§11.2.12 — "this label has no warning" vs "this photograph is unreadable".

    A label with brand, class, ABV, net contents and a producer line carries 20+ tokens, so
    a genuinely warning-less label is a red; a black frame is an amber, which is what an
    agent does today — reject the image and ask for a better one.
    """
    from app import verdict

    entry = cached_ocr["label_023.png"]  # no warning, but a fully legible label
    assert len(entry["page_text"].split()) >= settings.min_tokens_for_absence
    rich = verdict.evaluate(
        make_result(page_text=entry["page_text"], read=make_read(entry["page_text"])),
        None,
        settings,
    )
    check = next(c for c in rich.checks if c.id == "warning_present")
    assert check.state == "red"

    dark = "BLACK ROCK"  # two tokens: an unreadable photograph
    empty = verdict.evaluate(
        make_result(page_text=dark, read=make_read(dark)),
        None,
        settings,
    )
    check = next(c for c in empty.checks if c.id == "warning_present")
    assert check.state == "amber", "an unreadable image must never be reported as a violation"
