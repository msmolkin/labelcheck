"""27 CFR Part 16 — the single source of truth for every statutory string.

The statement below is byte-exact from `docs/regulatory-basis.md` §16.21, which was taken
from the official govinfo annual-edition XML (`CFR-2024-title27-vol1-part16.xml`). Nothing
else in the codebase may retype it: import from here. `STATUTE_SHA256` is pinned so a test
fails loudly if anyone "fixes a typo" in the law.
"""

import hashlib

STATUTE_TEXT = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)
STATUTE_SHA256 = "35e1f5d39ee341ac7c114f8159956cb0cc1981b94e4ffeee194ff5060bf99fbc"

# The two numbered clauses. `CLAUSE_1 + " " + CLAUSE_2 == STATUTE_TEXT`.
CLAUSE_1, CLAUSE_2 = STATUTE_TEXT.split(" (2) ")[0], "(2) " + STATUTE_TEXT.split(" (2) ")[1]

PREFIX = "GOVERNMENT WARNING"

# Whitespace-split tokens with punctuation preserved, so `" ".join(STATUTE_TOKENS)`
# reproduces STATUTE_TEXT exactly. `(1)` and `(2)` survive as clause markers. The word
# diff (app/textdiff.py) strips trailing punctuation for matching only.
STATUTE_TOKENS = tuple(STATUTE_TEXT.split())

# A changed content word is a deviation; a dropped article almost never is (§6.2).
FUNCTION_WORDS = frozenset({"the", "to", "a", "of", "and", "or", "your"})

# Warning checks cite Part 16. Record-match checks compare the label against the
# application record, a requirement that lives in Parts 4/5/7 and is outside the
# regulatory basis this tool documents — so they carry None rather than an invented
# citation (§6.6).
CITATIONS: dict[str, str | None] = {
    "warning_present": "27 CFR 16.21",
    "warning_text": "27 CFR 16.21",
    "warning_case": "27 CFR 16.22(a)(2)",
    "warning_bold": "27 CFR 16.22(a)(2)",
    "brand": None,
    "class_type": None,
    "alcohol_content": None,
    "net_contents": None,
    "type_size": "27 CFR 16.22(b)",
}

# §16.22 format requirements, with an honest note on what a photograph can settle.
FORMAT_RULES = (
    ("27 CFR 16.22(a)(1)", "Readily legible; on a contrasting background", "partial"),
    ("27 CFR 16.22(a)(2)", '"GOVERNMENT WARNING" in capitals and bold; remainder not bold', "yes"),
    ("27 CFR 16.22(a)(3)", "Not compressed or distorted so as to be illegible", "partial"),
    ("27 CFR 16.22(a)(4)", "Maximum characters per inch: 40 at 1 mm, 25 at 2 mm, 12 at 3 mm", "no"),
    ("27 CFR 16.22(b)", "Minimum type size by container volume (see min_type_size_mm)", "no"),
    ("27 CFR 16.22(c)", "The label bearing the statement is firmly affixed", "no"),
)

# §16.22(b): (upper bound in mL, minimum type size in mm). Anything larger needs 3 mm.
MIN_TYPE_SIZE_MM = ((237.0, 1.0), (3000.0, 2.0))

# §16.22(a)(4): (type size in mm, maximum characters per inch), largest size first.
MAX_CHARS_PER_INCH = ((3.0, 12), (2.0, 25))


def min_type_size_mm(volume_ml: float) -> float:
    """Minimum type size §16.22(b) requires for a container of this volume."""
    for upper_ml, size_mm in MIN_TYPE_SIZE_MM:
        if volume_ml <= upper_ml:
            return size_mm
    return 3.0


def max_chars_per_inch(type_mm: float) -> int:
    """Maximum characters per inch §16.22(a)(4) allows at this type size.

    The regulation tabulates three sizes; a size between two entries takes the limit of
    the largest tabulated size it reaches. Informational only — neither this nor
    `min_type_size_mm` is verifiable from an uncalibrated photograph.
    """
    for size_mm, chars in MAX_CHARS_PER_INCH:
        if type_mm >= size_mm:
            return chars
    return 40


def statute_digest() -> str:
    """SHA-256 of `STATUTE_TEXT`, for the test that pins it against `STATUTE_SHA256`."""
    return hashlib.sha256(STATUTE_TEXT.encode("utf-8")).hexdigest()
