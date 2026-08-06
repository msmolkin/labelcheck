"""§11.8 — the golden set: the whole tool, end to end, on labelled images.

Marked `slow` because every test here runs the OCR engine. The expectations live in
`tests/golden.json` and are derived from `spike/corpus/ground_truth.json` by the documented
mapping — never from observed behaviour, which would make the suite a tautology.

Images come from `app/samples/` (committed, so a fresh clone can run this) and, when the
git-ignored corpus is present, from `spike/corpus/` as well.
"""

import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import CORPUS, SAMPLES

pytestmark = pytest.mark.slow


def analyse_dir(directory: Path, settings) -> dict[str, dict]:
    """Run the shipped pipeline and verdict over every image in a directory.

    Keyed by filename stem, because `app/samples/` re-encodes `label_001.png` to
    `label_001.jpg` (§8.4) and the golden expectations are written against the corpus names.
    """
    from app import csvio, pipeline, verdict

    rows, _ = csvio.read_records(directory / "records.csv")
    records = {row["record_id"]: row for row in rows}
    out = {}
    for path in sorted(directory.glob("label_*.*")):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        started = time.perf_counter()
        result = pipeline.analyse(path, settings)
        decision = verdict.evaluate(result, records.get(path.stem), settings)
        out[path.stem] = {
            "file": path.name,
            "overall": decision.overall,
            "states": {check.id: check.state for check in decision.checks},
            "checks": {check.id: check for check in decision.checks},
            "result": result,
            "ms": (time.perf_counter() - started) * 1000.0,
        }
    return out


@pytest.fixture(scope="session")
def sample_outcomes(settings) -> dict[str, dict]:
    if not (SAMPLES / "records.csv").is_file():
        pytest.skip("app/samples/ has not been built yet")
    return analyse_dir(SAMPLES, settings)


@pytest.fixture(scope="session")
def ground_truth_optional(settings):
    """(ground truth, outcomes) over the full 56-image corpus, or (None, None) without it.

    The corpus is git-ignored, so the full confusion matrices of tests 60, 61 and 62 can
    only be asserted where it exists (§11.9). Their 22-image equivalents always run.
    """
    path = CORPUS / "ground_truth.json"
    if not path.is_file():
        return None, None
    truth = {row["file"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
    return truth, analyse_dir(CORPUS, settings)


def report(request, line: str) -> None:
    """Print through pytest's terminal writer so the table survives `-q` capture."""
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    (reporter.write_line if reporter else print)(line)


def test_the_golden_set_covers_every_class_it_claims_to(golden):
    """§11.8 (acceptance 4) — a future edit cannot silently narrow the set.

    The 22 were chosen so that all 9 degradation classes, all 8 warning statuses, all 5
    font families, both dark-background labels and all 5 record mismatches appear at least
    once. If the file stops covering one of them, the golden run stops meaning anything.
    """
    labels = golden["labels"]
    required = golden["_coverage_required"]
    assert len(labels) == 22

    for field in ("degradation", "warning_status", "font_family"):
        present = {entry["ground_truth"][field] for entry in labels.values()}
        assert present == set(required[field]), f"{field}: missing {set(required[field]) - present}"

    dark = [name for name, e in labels.items() if e["ground_truth"]["dark_background"]]
    assert len(dark) >= required["dark_background_min"], dark
    mismatches = {n for n, e in labels.items() if not e["ground_truth"]["expected_record_match"]}
    assert mismatches == set(required["record_mismatches"])
    assert sum(1 for e in labels.values() if e["basis"] == "tightened") == 8
    assert sum(1 for e in labels.values() if e["basis"] == "known_hard") == 3


def test_every_golden_label_lands_inside_its_allowed_verdicts(golden, sample_outcomes, request):
    """§11.8.59 — zero wrong-coloured greens, zero wrong-coloured reds.

    A green on a violating label is a missed rejection; a red on a compliant one is a
    wrongly rejected application. Both are the errors this tool exists to avoid.
    """
    wrong = []
    for filename, entry in golden["labels"].items():
        stem = filename.rsplit(".", 1)[0]
        outcome = sample_outcomes.get(stem)
        assert outcome, f"{stem} is missing from app/samples/"
        if outcome["overall"] not in entry["allowed"]:
            wrong.append(
                f"{stem}: got {outcome['overall']}, allowed {entry['allowed']} "
                f"({entry['basis']}: {entry['why']}) states={outcome['states']}"
            )
    for line in wrong:
        report(request, f"  golden miss: {line}")
    assert not wrong, f"{len(wrong)} of 22 golden labels landed outside their allowed verdicts"


def test_warning_presence_on_the_golden_images(golden, sample_outcomes, ground_truth_optional):
    """§11.8.60 — presence over the 22, and the full 49/7/0/0 when the corpus is present."""
    for filename, entry in golden["labels"].items():
        stem = filename.rsplit(".", 1)[0]
        expected = "red" if entry["ground_truth"]["warning_status"] == "v_missing" else "green"
        got = sample_outcomes[stem]["states"]["warning_present"]
        assert got == expected, f"{stem}: warning_present {got}, expected {expected}"

    truth, outcomes = ground_truth_optional
    if truth is None:
        pytest.skip("full-corpus confusion matrix needs spike/corpus/")
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for name, row in truth.items():
        state = outcomes[Path(name).stem]["states"]["warning_present"]
        present = state == "green"
        has_warning = row["warning_status"] != "v_missing"
        confusion["tp" if present else "fn"] += 1 if has_warning else 0
        confusion["fp" if present else "tn"] += 0 if has_warning else 1
    assert confusion == {"tp": 49, "fp": 0, "tn": 7, "fn": 0}


def test_case_classification_on_the_golden_images(golden, sample_outcomes, ground_truth_optional):
    """§11.8.61 — Title Case is the canonical rejection; capitals must never be one."""
    for filename, entry in golden["labels"].items():
        stem = filename.rsplit(".", 1)[0]
        gw_case = entry["ground_truth"]["gw_case"]
        state = sample_outcomes[stem]["states"]["warning_case"]
        if gw_case == "caps":
            assert state == "green", f"{stem}: capitals reported as {state}"
        elif gw_case == "title":
            assert state == "red", f"{stem}: Title Case reported as {state}"

    truth, outcomes = ground_truth_optional
    if truth is None:
        pytest.skip("full-corpus case counts need spike/corpus/")
    counts = {"caps": 0, "title": 0}
    for name, row in truth.items():
        if row["gw_case"] not in counts:
            continue
        counts[row["gw_case"]] += 1
        state = outcomes[Path(name).stem]["states"]["warning_case"]
        assert state == ("green" if row["gw_case"] == "caps" else "red"), name
    assert counts == {"caps": 42, "title": 7}


def test_bold_makes_no_hard_errors(golden, sample_outcomes, ground_truth_optional, request):
    """§11.8.62 — no confident call in the wrong direction; amber is allowed and counted.

    A hard error is a bold-compliant label reported red, or a bold violation reported
    green. Amber is not an error: it is the designed output when the ink cannot be
    measured (§5.2), and the count is reported so it can be watched.
    """
    amber = 0
    for filename, entry in golden["labels"].items():
        stem = filename.rsplit(".", 1)[0]
        truth = entry["ground_truth"]
        if truth["warning_status"] == "v_missing":
            continue
        compliant = truth["gw_bold"] and not truth["body_bold"]
        state = sample_outcomes[stem]["states"]["warning_bold"]
        amber += state == "amber"
        assert state != ("red" if compliant else "green"), (
            f"{stem}: bold reported {state} for a {'compliant' if compliant else 'violating'} label"
        )
    report(request, f"  bold amber on the golden set: {amber} of 20 warning-bearing images")

    truth, outcomes = ground_truth_optional
    if truth is None:
        pytest.skip("full-corpus amber rate needs spike/corpus/")
    ambers = [
        name
        for name, row in truth.items()
        if row["warning_status"] != "v_missing"
        and outcomes[Path(name).stem]["states"]["warning_bold"] == "amber"
    ]
    report(request, f"  bold amber over the corpus: {len(ambers)} of 49 — {sorted(ambers)}")
    assert len(ambers) <= 12, f"{len(ambers)} ambers over 49 warning-bearing images"


def test_record_mismatches_are_caught_and_matches_are_not(golden, sample_outcomes):
    """§11.8.63 — all 5 known mismatches red; every genuinely matching record green."""
    field_ids = ("brand", "class_type", "alcohol_content", "net_contents")
    for filename, entry in golden["labels"].items():
        stem = filename.rsplit(".", 1)[0]
        states = [sample_outcomes[stem]["states"][check_id] for check_id in field_ids]
        if entry["ground_truth"]["expected_record_match"]:
            assert "red" not in states, f"{stem}: a matching record was rejected — {states}"
        else:
            assert "red" in states, f"{stem}: a known mismatch was not caught — {states}"


def test_the_golden_set_fits_the_latency_budget(sample_outcomes, settings, request):
    """§11.8.64 — the p50/p95/max table, and the reading-order deskew A/B (§4.3, §14.9).

    The deskew of box centroids before line clustering is new and unmeasured, so it ships
    behind a knob and the golden run prints both settings. The default flips on evidence.
    """
    times = sorted(outcome["ms"] for outcome in sample_outcomes.values())
    p50 = statistics.median(times)
    p95 = times[max(0, int(round(0.95 * len(times))) - 1)]
    report(request, "")
    report(request, f"  golden set: {len(times)} images at {settings.ocr_threads} OCR threads")
    report(
        request,
        f"  per-image ms — p50 {p50:.0f} · p95 {p95:.0f} · max {times[-1]:.0f} "
        f"· budget {settings.budget_ms}",
    )

    without = analyse_dir(SAMPLES, replace(settings, reading_order_deskew=False))
    changed = [
        stem
        for stem, outcome in sample_outcomes.items()
        if outcome["overall"] != without[stem]["overall"]
    ]
    report(request, "  READING_ORDER_DESKEW A/B (§14.9):")
    for label, outcomes in (("on ", sample_outcomes), ("off", without)):
        greens = sum(o["overall"] == "green" for o in outcomes.values())
        ambers = sum(o["overall"] == "amber" for o in outcomes.values())
        reds = sum(o["overall"] == "red" for o in outcomes.values())
        report(request, f"    deskew {label}: green {greens} · amber {ambers} · red {reds}")
    report(request, f"    verdicts that differ: {changed or 'none'}")

    assert p50 < settings.budget_ms, (
        f"median {p50:.0f} ms exceeds the {settings.budget_ms} ms budget"
    )
