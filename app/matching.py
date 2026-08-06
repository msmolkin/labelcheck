"""Normalisation and the three-state comparison of a label against an application record.

Two label writers spell one brand four ways: `STONE'S THROW`, `Stone's Throw`, `Stone’s
Throw`, `STONES THROW`. None of those is a discrepancy. `normalize()` folds exactly the
differences that carry no meaning and nothing else, so the fuzzy score that follows works on
already-clean strings. The numeric fields never come through that path at all: "40%" and
"48%" are two characters apart and one of them is a rejection (§6.5).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from app.fields import Abv, Volume, format_number

# Curly quotes, primes and every dash Unicode offers, folded to their ASCII spelling before
# the character filter runs. NFKD has already handled accents by this point.
_PUNCTUATION_FOLD = str.maketrans(
    {
        **dict.fromkeys("‘’‛′´`", "'"),
        **dict.fromkeys("“”″", '"'),
        **dict.fromkeys("‐‑‒–—―−", "-"),
    }
)
_KEEP = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

# Proof is printed as a whole number and alcohol content to one decimal, so these are the
# precision of the printed declaration rather than tuning knobs (§6.5).
PROOF_TOLERANCE = 0.5
_PROOF_ONE_SIDE = (
    "One side states a proof figure and the other does not, which is not itself a discrepancy: "
    "wine labels carry no proof clause."
)


def normalize(text: str) -> str:
    """Fold away the differences between two spellings of the same words: NFKD, drop combining
    marks, ASCII punctuation, `&` becomes "and", lowercase, keep only `[a-z0-9 ]`, collapse
    whitespace. Dropped punctuation becomes a space, so "Alc./Vol." stays two words."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    folded = stripped.translate(_PUNCTUATION_FOLD).replace("&", " and ").lower()
    return _SPACES.sub(" ", _KEEP.sub(" ", folded)).strip()


def despace(text: str) -> str:
    """`normalize()` with the spaces removed, so "1 L" and "1L" are the same read."""
    return normalize(text).replace(" ", "")


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens: "Alc./Vol." → ["alc", "vol"], "(1)" → ["1"]."""
    return _TOKEN.findall((text or "").lower())


def similarity(a: str, b: str, *, ignore_spacing: bool = True) -> float:
    """0-100 similarity of two normalised strings. `token_set_ratio` is deliberately absent:
    it scores "Kentucky Straight Bourbon Whiskey" against "Straight Bourbon Whiskey" at 100,
    hiding a real discrepancy (§6.5)."""
    na, nb = normalize(a), normalize(b)
    best = max(fuzz.ratio(na, nb), fuzz.token_sort_ratio(na, nb))
    if ignore_spacing:
        best = max(best, fuzz.ratio(na.replace(" ", ""), nb.replace(" ", "")))
    return float(best)


def best_window(page_text: str, target: str, *, ignore_spacing: bool = True) -> tuple[str, float]:
    """Find the run of words on the label that best matches `target`.

    Windows of n-1, n and n+1 words are scored, which is stricter than a `partial_ratio` over
    the whole page (that can score high on an accidental substring) while tolerating one
    dropped or extra word. The text returned is sliced from the original page, so the evidence
    shows "STONE'S THROW" as the label prints it. From `spike/eval_ocr.py` (§13).
    """
    spans = [m.span() for m in _TOKEN.finditer((page_text or "").lower())]
    n = len(tokens(target))
    if not spans or not n:
        return "", 0.0
    best_text, best_score = "", 0.0
    for width in sorted({max(1, n - 1), n, n + 1}):
        for start in range(0, max(1, len(spans) - width + 1)):
            end = min(start + width, len(spans)) - 1
            candidate = page_text[spans[start][0] : spans[end][1]]
            score = similarity(candidate, target, ignore_spacing=ignore_spacing)
            if score > best_score:
                best_text, best_score = candidate, score
    return best_text, best_score


@dataclass(frozen=True)
class FieldMatch:
    """One record-field comparison: the state, what each side said, and why."""

    state: str  # "green" | "amber" | "red"
    label: str  # what the label appears to declare
    record: str  # what the application record declares
    score: float | None = None  # fuzzy score, for the two text fields only
    note: str | None = None  # a sentence that qualifies a green or an amber
    code: str | None = None  # bare reason code; the verdict prefixes it (§7.6)


def compare_similar(
    label: str, record: str, score: float, *, green: float, amber: float, readable: bool
) -> FieldMatch:
    """Three-state comparison of two text fields from an already-computed score. When the
    label text never came back (`readable=False`) the value was not located, which is an amber
    ("we could not check"), never a red ("they disagree")."""
    if not readable:
        return FieldMatch("amber", label, record, score, "Not found on the label.", "notfound")
    if score >= green:
        return FieldMatch("green", label, record, score)
    if score >= amber:
        return FieldMatch("amber", label, record, score, code="unclear")
    return FieldMatch("red", label, record, score, code="mismatch")


def compare_alcohol(label: Abv, record: Abv) -> FieldMatch:
    """Compare alcohol content numerically (§6.5). Proof is a cross-check, not a verdict.

    `label` is parsed from the whole page, so the evidence shows the numbers it found rather
    than the page they came from; the record's own wording is short and is shown verbatim.
    """
    shown, declared = label.display, record.text
    if record.percent is None:
        return FieldMatch("amber", shown, declared, note="Record value is not a percentage.")
    if label.percent is None:
        return FieldMatch(
            "amber", shown, declared, note="Alcohol content not readable.", code="unreadable"
        )
    if abs(label.percent - record.percent) > Abv.TOLERANCE:
        code = f"{format_number(label.percent)}!={format_number(record.percent)}"
        return FieldMatch("red", shown, declared, code=code)
    if label.proof is not None and abs(label.proof - 2 * label.percent) > PROOF_TOLERANCE:
        return FieldMatch(
            "amber",
            shown,
            declared,
            note=(
                f"The label's proof figure ({format_number(label.proof)}) is not twice its "
                f"percentage ({format_number(label.percent)}); the label contradicts itself."
            ),
            code="proof_inconsistent",
        )
    if (label.proof is None) != (record.proof is None):
        return FieldMatch("green", shown, declared, note=_PROOF_ONE_SIDE)
    return FieldMatch("green", shown, declared)


def compare_volume(label: list[Volume], record: Volume | None, record_text: str) -> FieldMatch:
    """Compare net contents by converted millilitres, never by string (§6.5).

    `record_text` is the raw declaration, kept for the evidence even when it did not parse.
    """
    shown = " / ".join(v.text for v in label)
    declared = record.text if record else record_text.strip()
    if record is None:
        return FieldMatch("amber", shown, declared, note="Record value is not a volume.")
    if not label:
        return FieldMatch(
            "amber", shown, declared, note="Net contents not readable.", code="unreadable"
        )
    hit = min(label, key=lambda v: abs(v.ml - record.ml))
    if abs(hit.ml - record.ml) > Volume.TOLERANCE_ML:
        return FieldMatch("red", hit.text, declared, code=f"{hit.text}!={record.text}")
    note = None
    if hit.unit != record.unit:
        note = (
            f"The label declares {hit.text} and the record {record.text}; "
            f"the same volume in different units."
        )
    return FieldMatch("green", hit.text, declared, note=note)
