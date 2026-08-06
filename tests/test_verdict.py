"""§11.5 (40–46) — the roll-up, and the rules that keep uncertainty out of the extremes.

The design invariant every one of these serves: OCR uncertainty must never produce a false
green, and must never auto-reject a compliant label. Amber is where uncertainty goes.
"""

from conftest import check_by_id, make_bold, make_read, make_record, make_result

from app import db, verdict
from app.config import get_settings
from app.statute import STATUTE_TEXT

SETTINGS = get_settings()
COMPLIANT_PAGE = f"OLD TOM DISTILLERY Straight Bourbon Whiskey 45% Alc./Vol. 750 mL {STATUTE_TEXT}"


def evaluate(page_text: str = COMPLIANT_PAGE, record: dict | None = None, **over):
    read = over.pop("read", None) or make_read(region_text=STATUTE_TEXT, page_text=page_text)
    return verdict.evaluate(make_result(page_text=page_text, read=read, **over), record, SETTINGS)


def test_the_roll_up_is_worst_state_wins():
    """§11.5.40 — any red is a red; any amber without a red is an amber."""
    assert evaluate().overall == "green"
    assert evaluate(bold=make_bold(state="amber", gate="ratio", ratio=1.12)).overall == "amber"

    title_case = STATUTE_TEXT.replace("GOVERNMENT WARNING", "Government Warning")
    page = COMPLIANT_PAGE.replace(STATUTE_TEXT, title_case)
    red = evaluate(page, read=make_read(region_text=title_case, page_text=page))
    assert check_by_id(red, "warning_case").state == "red"
    assert red.overall == "red"
    # ...and one red outranks any number of ambers.
    assert (
        evaluate(
            page,
            read=make_read(region_text=title_case, page_text=page),
            bold=make_bold(state="amber", gate="ratio"),
        ).overall
        == "red"
    )


def test_the_tool_is_useful_with_no_application_record():
    """§11.5.41 — the four statutory checks do not need a record; the field checks say so."""
    result = evaluate(record=None)
    for check_id in ("brand", "class_type", "alcohol_content", "net_contents"):
        check = check_by_id(result, check_id)
        assert check.state == "na"
        assert "record" in check.detail.lower(), check.detail
    for check_id in ("warning_present", "warning_text", "warning_case", "warning_bold"):
        assert check_by_id(result, check_id).state != "na"
    assert result.overall == "green", "`na` checks never affect the roll-up"


def test_an_unmeasurable_bold_check_is_amber_never_red():
    """§11.5.42 — `not_found` and both abstain gates mean "we could not measure it".

    A rejection manufactured out of a measurement the tool could not take is the failure
    mode this whole three-state design exists to prevent.
    """
    for state, gate in (("not_found", None), ("amber", "defocus"), ("amber", "tiny")):
        bold = make_bold(state=state, gate=gate, ratio=None, sharpness=0.04, cap_px=9.0)
        check = check_by_id(evaluate(bold=bold), "warning_bold")
        assert check.state == "amber", f"bold {state}/{gate} produced {check.state}"
        assert check.detail, "an amber must tell the agent why it could not be measured"


def test_an_agent_decision_never_overwrites_the_machine_verdict(tmp_path):
    """§11.5.43 — §6.8: the machine verdict is immutable.

    A compliance tool that rewrites its own finding when a human disagrees has destroyed
    the audit trail. Both are stored; the export carries both columns.
    """
    conn = db.connect(tmp_path / "labelcheck.db")
    db.init_schema(conn)
    batch_id = db.create_batch(conn, label="test")
    image_id = db.add_image(conn, batch_id, filename="label_002.png", stored_path="x.jpg")
    db.claim_next_queued(conn)
    db.finish_image(conn, image_id, overall="red", result_json="{}")

    db.record_adjudication(conn, image_id, "warning_case", "confirm", note="agent agrees")
    stored = db.get_image(conn, image_id)
    assert stored["overall"] == "red", "the machine verdict must survive adjudication untouched"
    adjudications = db.adjudications_for_image(conn, image_id)
    assert [(a["check_id"], a["decision"]) for a in adjudications] == [("warning_case", "confirm")]
    conn.close()


def test_every_check_carries_a_citation_key():
    """§11.5.44 — present for every check; explicitly None where no citation exists.

    A missing key would render as a blank in the UI and read as an oversight. An invented
    Part 4/5/7 citation would be worse.
    """
    result = evaluate(record=make_record())
    assert {c.id for c in result.checks} == set(verdict.CHECK_IDS)
    for check in result.checks:
        assert hasattr(check, "citation")
        if check.id.startswith("warning_"):
            assert check.citation and check.citation.startswith("27 CFR 16."), check.id
        else:
            assert check.citation is None, f"{check.id} must not invent a citation"


def test_a_verdict_reached_in_reduced_mode_cannot_reject():
    """§11.5.45 — §6.7: a rejection the system did not have time to verify is not one.

    When the pass-2 deadline fires, the image was checked by the fast pass only. Any red
    demotes to amber and the reason is visible.
    """
    title_case = STATUTE_TEXT.replace("GOVERNMENT WARNING", "Government Warning")
    page = COMPLIANT_PAGE.replace(STATUTE_TEXT, title_case)
    read = make_read(region_text=title_case, page_text=page)

    confident = evaluate(page, read=read)
    assert confident.overall == "red"

    reduced = evaluate(
        page,
        read=make_read(region_text=title_case, page_text=page),
        low_confidence=True,
        passes=1,
        pass2_skipped_reason="budget: 1500 ms remaining, pass 2 needs 1700 ms",
    )
    assert reduced.overall == "amber"
    assert not [c for c in reduced.checks if c.state == "red"]
    notes = " ".join(str(i) for i in reduced.informational) + " ".join(
        c.detail for c in reduced.checks
    )
    assert "reduced" in notes.lower(), "the reduced mode must be visible to the agent"


def test_the_type_size_panel_is_informational_only():
    """§11.5.46 — §16.22(b) needs physical scale a photograph does not carry.

    The minimums are surfaced from the declared net contents, with the limitation stated,
    and they never move the traffic light.
    """
    result = evaluate(record=make_record(net_contents="750 mL"))
    panel = next(item for item in result.informational if item["id"] == "type_size")
    text = panel["text"]
    assert "750" in text and "2 mm" in text.replace("2mm", "2 mm")
    assert "25" in text, "16.22(a)(4)'s characters-per-inch limit travels with the minimum"
    assert "photograph" in text.lower(), "the limitation must be stated, not implied"
    assert result.overall == "green"
    assert panel["id"] not in {c.id for c in result.checks}
