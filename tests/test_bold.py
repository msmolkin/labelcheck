"""§11.4 — 27 CFR 16.22(a)(2), the stroke-weight measurement.

Every case here is a synthetic patch drawn by `fixtures/gen_fixtures.py` (Georgia, whose
bold and regular weights separate cleanly at these sizes) together with the exact word
quads used to draw it. That makes the stage testable with no OCR engine and no corpus, and
it keeps the failure mode of each test unambiguous: one property of the drawing changes at
a time.

The manifest also records what the spike's measurement produced on each patch. That is
evidence about the fixture, not an expectation — the assertions below are the §5.2 rule.
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import make_ocr_result

from app import bold

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "bold"
APP_DIR = Path(__file__).resolve().parents[1] / "app"


def measure(bold_fixtures, case: str, settings):
    """Run the shipped bold stage over one synthetic patch."""
    entry = bold_fixtures[case]
    img = cv2.imread(str(FIXTURE_DIR / entry["file"]))
    assert img is not None, f"missing fixture {entry['file']}; run tests/fixtures/gen_fixtures.py"
    lines = make_ocr_result(entry["lines"]).lines
    return bold.measure(img, lines, settings)


def test_a_bold_prefix_over_a_regular_body_passes(bold_fixtures, settings):
    """§11.4.23 — the compliant arrangement: prefix heavier than the body."""
    result = measure(bold_fixtures, "compliant", settings)
    assert result.state == "pass"
    assert result.ratio >= settings.bold_t_pass
    assert result.likely_violation is None


def test_a_uniformly_weighted_statement_fails(bold_fixtures, settings):
    """§11.4.24 — nothing is bold, so 16.22(a)(2)'s first requirement is unmet."""
    result = measure(bold_fixtures, "prefix_not_bold", settings)
    assert result.state == "fail"
    assert result.ratio <= settings.bold_t_fail
    assert result.likely_violation == "prefix_not_bold"


def test_an_all_bold_statement_fails_and_names_the_body(bold_fixtures, settings):
    """§11.4.25 — the second requirement: the remainder must NOT be bold.

    The subtype is decided against the label's own regular copy, and it is 92.9 % accurate
    on the corpus — so the verdict presents it as the likely violation, never as fact
    (§5.3). What must be right is that the statement is rejected either way.
    """
    result = measure(bold_fixtures, "body_bold", settings)
    assert result.state == "fail"
    assert result.likely_violation == "body_bold"
    assert result.subtype_value >= settings.bold_t_ref


def test_the_defocus_gate_stops_a_blurred_violation_from_passing(bold_fixtures, settings):
    """§11.4.26 — the highest-value test in the suite.

    `bold_blur_sweep.json` measured the failure directly: at sigma >= 3.0 the ratio of a
    same-weight (violating) statement inflates to 1.374 and 1.702 — past `bold_t_pass`, so
    a confident false PASS on a label that is actually in breach. This patch is the same
    violation, defocused: the gate must abstain before the ratio bands are ever consulted.
    """
    result = measure(bold_fixtures, "blurred_violation", settings)
    assert result.state == "amber"
    assert result.gate == "defocus"
    assert result.sharpness < settings.bold_t_sharp
    if result.ratio is not None:
        assert result.ratio > settings.bold_t_fail, (
            f"the fixture must reproduce the inflated-ratio failure mode (ratio "
            f"{result.ratio:.3f} should sit above the fail band); redraw it with "
            "tests/fixtures/gen_fixtures.py --bold if the drawing changed"
        )


def test_type_too_small_to_measure_abstains(bold_fixtures, settings):
    """§11.4.27 — below ~11 px cap height the ratio carries no signal (§5.2).

    This patch would otherwise be a confident PASS, which is exactly why the abstain has to
    be evaluated before the bands rather than after them.
    """
    result = measure(bold_fixtures, "tiny", settings)
    assert result.state == "amber"
    assert result.gate == "tiny"
    assert result.cap_px < settings.bold_min_cap_px


def test_the_skeleton_of_a_bar_is_a_one_pixel_line_of_its_length(settings):
    """§11.4.28 — Zhang-Suen thinning is what makes ink area / length a stroke width.

    A 3-px-thick horizontal bar must thin to a single mid-row line. Thinning erodes the two
    ends by about a pixel and a half each, which is why the length is asserted as a
    tolerance rather than an equality.
    """
    length = 40
    image = np.zeros((11, length + 10), dtype=np.uint8)
    image[4:7, 5 : 5 + length] = 1
    skeleton = bold.zhang_suen(image)

    rows = sorted({int(r) for r, _ in np.argwhere(skeleton)})
    assert rows == [5], f"a 3-px bar must thin to its middle row, got rows {rows}"
    assert length - 4 <= int(skeleton.sum()) <= length


def test_only_the_measured_estimator_ships(settings):
    """§11.4.29 — `area_skel` is the shipped estimator and the only one reachable.

    `dt2mean`, `dt2skel` and `area_perim` tie it on balanced accuracy, but in the
    three-state rule `area_perim` produces two hard errors where `area_skel` produces none
    (§5.1). Shipping the zoo would invite a future reader to switch estimators on a
    tie-breaking number that does not survive the three-state rule.
    """
    source = (APP_DIR / "bold.py").read_text(encoding="utf-8")
    assert "zhang_suen" in source
    for rejected in ("area_perim", "dt2mean", "dt2skel", "ink_density", "erosion"):
        assert not re.search(rf"\bdef\s+\w*{rejected}", source), (
            f"{rejected} was measured and cut; it must not be reachable in shipped code"
        )
    assert not [f for f in settings.__dataclass_fields__ if "estimator" in f], (
        "the estimator is not a configuration choice"
    )


def test_a_prefix_fragmented_by_the_recogniser_is_still_recovered(bold_fixtures, settings):
    """§11.4.30 — the recogniser returns GOVERNM + ENT + WA + RNING:.

    A token-equality test misses that completely, which would leave every fragmented read
    with no prefix to measure and an amber it does not deserve.
    """
    result = measure(bold_fixtures, "fragmented_prefix", settings)
    assert result.n_prefix_tokens == 4, "all four fragments belong to the prefix"
    assert result.state == "pass"
    assert result.ratio == pytest.approx(
        measure(bold_fixtures, "compliant", settings).ratio, rel=0.10
    ), "the same ink measured through four fragments should give the same weight"


def test_a_page_with_no_warning_block_is_not_found_never_a_violation(settings):
    """§11.4 (§4.6) — pass 1 found no statement: the bold check abstains.

    Reporting a bold violation because the block was never located would be a rejection
    manufactured out of a detection failure.
    """
    image = np.full((300, 900, 3), 255, dtype=np.uint8)
    lines = make_ocr_result(
        [
            {
                "text": "OLD TOM DISTILLERY",
                "score": 0.99,
                "quad": [[50, 50], [500, 50], [500, 90], [50, 90]],
                "words": [
                    {"text": "OLD", "quad": [[50, 50], [120, 50], [120, 90], [50, 90]]},
                    {"text": "TOM", "quad": [[130, 50], [210, 50], [210, 90], [130, 90]]},
                ],
            }
        ]
    ).lines
    result = bold.measure(image, lines, settings)
    assert result.state == "not_found"
    assert result.ratio is None
