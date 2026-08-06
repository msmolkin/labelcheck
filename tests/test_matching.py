"""§11.5 (31–39) — the label against the application record.

These four checks carry no CFR citation (§6.6). What they must get right is the
normalisation: Dave's `STONE'S THROW` on the label and `Stone's Throw` in the record are
the same brand, and no threshold should have to be loosened to say so.

Numbers are parsed numerically, never fuzzily: `40%` against `38%` is a rejection, and a
fuzzy score would call those strings 96 % similar.
"""

import pytest
from conftest import check_by_id, make_read, make_record, make_result

from app import verdict
from app.config import get_settings
from app.statute import STATUTE_TEXT

SETTINGS = get_settings()


def label_page(brand: str, class_type: str, abv: str, net_contents: str) -> str:
    """A realistic page of label copy: the front panel, then the statutory statement."""
    return f"{brand} {class_type} {abv} {net_contents} {STATUTE_TEXT}"


def field_check(check_id: str, page_text: str, record: dict):
    read = make_read(region_text=STATUTE_TEXT, page_text=page_text)
    result = make_result(page_text=page_text, read=read)
    return check_by_id(verdict.evaluate(result, record, SETTINGS), check_id)


@pytest.mark.parametrize(
    ("on_label", "in_record", "expected"),
    [
        ("STONE'S THROW", "Stone's Throw", "green"),  # §11.5.31 case only
        ("SMITH & SONS", "Smith & Sons", "green"),  # §11.5.32 ampersand
        ("CUVÉE ROYALE", "Cuvee Royale", "green"),  # §11.5.33 accent folding
        ("STONE’S THROW", "Stone's Throw", "green"),  # §11.5.34 curly apostrophe
        ("OLD TOM DISTILLERY", "Black-Rock Reserve", "red"),  # §11.5.35 a real mismatch
    ],
)
def test_brand_normalisation_decides_the_match(on_label, in_record, expected):
    """§11.5.31–35 — the normaliser does the work, not a loose threshold."""
    page = label_page(on_label, "London Dry Gin", "47% Alc./Vol. (94 Proof)", "750 mL")
    check = field_check("brand", page, make_record(brand_name=in_record))
    assert check.state == expected, f"{on_label!r} vs {in_record!r}: {check.detail}"
    assert check.citation is None, "record-match checks carry no invented CFR citation (§6.6)"


def test_a_different_alcohol_content_is_a_rejection_showing_both_values():
    """§11.5.36 — `40%` and `38%` are 96 % similar as strings and a rejection in fact."""
    page = label_page("OLD TOM DISTILLERY", "Straight Bourbon Whiskey", "40% Alc./Vol.", "750 mL")
    check = field_check("alcohol_content", page, make_record(alcohol_content="38% Alc./Vol."))
    assert check.state == "red"
    shown = f"{check.detail} {check.evidence}"
    assert "40" in shown and "38" in shown, f"an agent must see both values: {shown}"


def test_wine_without_a_proof_clause_still_matches():
    """§11.5.37 — wine has no proof clause; its absence is not a discrepancy."""
    page = label_page("CUVÉE ROYALE", "Napa Valley Cabernet Sauvignon", "13.5% Alc./Vol.", "750 mL")
    check = field_check(
        "alcohol_content",
        page,
        make_record(brand_name="Cuvée Royale", alcohol_content="13.5% Alc./Vol."),
    )
    assert check.state == "green"


def test_net_contents_compare_by_volume_not_by_string():
    """§11.5.38 — `1 L`, `1L` and `1000 mL` are one volume; 750 mL and 700 mL are not.

    The strict-token read rate for net contents was 0.89–0.91 purely because of how the
    recogniser splits `1 L`, which is why the comparison de-spaces and converts to mL.
    """
    record = make_record(net_contents="1 L")
    for on_label in ("1 L", "1L", "1000 mL"):
        page = label_page(
            "OLD TOM DISTILLERY", "Straight Bourbon Whiskey", "45% Alc./Vol.", on_label
        )
        check = field_check("net_contents", page, record)
        assert check.state == "green", f"{on_label!r} should equal 1 L: {check.detail}"

    page = label_page("OLD TOM DISTILLERY", "Straight Bourbon Whiskey", "45% Alc./Vol.", "700 mL")
    check = field_check("net_contents", page, make_record(net_contents="750 mL"))
    assert check.state == "red"


def test_a_proof_that_is_not_twice_the_percentage_is_flagged_for_review():
    """§11.5.39 — an internally inconsistent label: 40 % cannot be 90 proof.

    Neither figure can be preferred over the other, so this is an amber with a note rather
    than a rejection of a value that might be the correct one.
    """
    page = label_page(
        "OLD TOM DISTILLERY", "Straight Bourbon Whiskey", "40% Alc./Vol. (90 Proof)", "750 mL"
    )
    check = field_check("alcohol_content", page, make_record(alcohol_content="40% Alc./Vol."))
    assert check.state == "amber"
    assert "proof" in f"{check.detail} {check.evidence}".lower()


def test_class_type_uses_a_comparison_that_cannot_hide_a_missing_word():
    """§11.5 (§6.5) — `token_set_ratio` scores a dropped word at 100; it is not used.

    "Kentucky Straight Bourbon Whiskey" against "Straight Bourbon Whiskey" is a real
    discrepancy an agent has to see. The corpus contains no class/type mismatch, so this
    rule is unvalidated by measurement — the identity case is pinned, and the limitation is
    stated rather than papered over.
    """
    page = label_page("OLD TOM DISTILLERY", "Straight Bourbon Whiskey", "45% Alc./Vol.", "750 mL")
    assert field_check("class_type", page, make_record()).state == "green"

    mismatch = field_check(
        "class_type", page, make_record(class_type="Kentucky Straight Bourbon Whiskey")
    )
    assert mismatch.state != "green", (
        "a dropped qualifier must not score 100; token_set_ratio would hide it"
    )
