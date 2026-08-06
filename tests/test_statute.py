"""§11.1 — the statute itself. Five tests, none of them optional.

The text of 27 CFR 16.21 is the one string in this system that must never drift: a
"helpful" edit to it silently redefines every comparison the tool makes.
"""

import hashlib

from app import statute


def test_statute_text_matches_its_pinned_digest():
    """§11.1.1 — nobody "fixes a typo" in the law without this failing."""
    digest = hashlib.sha256(statute.STATUTE_TEXT.encode("utf-8")).hexdigest()
    assert digest == statute.STATUTE_SHA256, (
        "STATUTE_TEXT no longer matches its pinned SHA-256. If 27 CFR 16.21 was "
        "genuinely amended, re-derive the text from docs/regulatory-basis.md and update "
        f"STATUTE_SHA256 deliberately. Computed {digest}."
    )


def test_statute_tokens_are_43_and_preserve_punctuation():
    """§11.1.2 — the token tuple is what the word-level diff aligns against."""
    tokens = statute.STATUTE_TOKENS
    assert len(tokens) == 43, f"expected 43 statute tokens, got {len(tokens)}"
    assert tokens[:3] == ("GOVERNMENT", "WARNING:", "(1)")
    assert tokens[-1] == "problems."
    assert " ".join(tokens) == statute.STATUTE_TEXT, (
        "punctuation must survive tokenisation: the clause markers (1) and (2) are how "
        "clause coverage is computed (§6.2)"
    )


def test_the_two_clauses_reassemble_the_statement():
    """§11.1.3 — CLAUSE_1 and CLAUSE_2 partition the statement, losing nothing.

    The clauses are stored stripped, so the identity carries the separating space. Giving
    CLAUSE_1 a trailing space would make bare concatenation work but would corrupt it as a
    fuzzy-match target for `prefix_score`.
    """
    assert statute.CLAUSE_1 + " " + statute.CLAUSE_2 == statute.STATUTE_TEXT
    assert statute.CLAUSE_1.startswith(statute.PREFIX)
    assert statute.CLAUSE_2.startswith("(2)")


def test_minimum_type_size_by_container_volume():
    """§11.1.4 — the 16.22(b) table, including both boundaries."""
    assert statute.min_type_size_mm(237) == 1
    assert statute.min_type_size_mm(238) == 2
    assert statute.min_type_size_mm(750) == 2
    assert statute.min_type_size_mm(3000) == 2
    assert statute.min_type_size_mm(3001) == 3
    # 16.22(a)(4) travels with it: the informational panel quotes both.
    assert statute.max_chars_per_inch(2.0) == 25
    assert statute.max_chars_per_inch(3.0) == 12


def test_every_check_id_has_a_citation_entry():
    """§11.1.5 — a check with no CITATIONS entry would be published uncited.

    Record-match checks map to None on purpose (§6.6): the mandatory label-content rules
    live in Parts 4/5/7, and inventing a citation in a Treasury deliverable is not an
    option. The entry must exist; its value may be None.
    """
    from app import verdict

    for check_id in verdict.CHECK_IDS:
        assert check_id in statute.CITATIONS, (
            f"check {check_id!r} has no entry in statute.CITATIONS — add one, using None "
            "if the requirement is outside the documented regulatory basis"
        )
