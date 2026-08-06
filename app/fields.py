"""Numeric reading of the two record fields that must never be matched fuzzily.

"40%" and "48%" differ by one character and by a rejection, so alcohol content and net
contents are parsed into numbers and compared as numbers (§6.5). Unit conversion is part of
the read: "1 L" on the label and "1000 mL" in the record agree. Brand and class/type are the
opposite problem, free text whose spelling varies without meaning, so they are located and
scored in `app.matching` (§13 puts `best_window` there).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

# 27 CFR 4/5/7 declarations: "45% Alc./Vol. (90 Proof)", "13.5% Alc./Vol.". The lookbehind
# is load-bearing rather than tidiness: without it "100% DE AGAVE" matches its last two
# digits and the label reads as 0.0 % alcohol, which rejects a compliant tequila.
_PERCENT = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*%")
_PROOF = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*proof", re.IGNORECASE)
# Label copy prints percentages that are not alcohol content — "100% DE AGAVE", "100% CORN" —
# and prints them above the declaration, so position alone picks the marketing number. These
# are the words a §4/5/7 declaration is actually made of.
_ABV_CONTEXT = re.compile(r"\b(?:alc|vol|proof)", re.IGNORECASE)

# Net contents. Longer unit spellings come first so "1 litre" cannot match as "1 l" + "itre";
# the trailing lookahead forces that backtracking. Only the units §6.5 lists are accepted —
# an unrecognised unit reads as "unparseable" (amber), never as a guess.
_VOLUME = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(ml|milli\s?li(?:ter|tre)s?|cl|centi\s?li(?:ter|tre)s?|fl\.?\s?oz\.?|oz\.?"
    r"|pts?\.?|qts?\.?|gal(?:lons?)?|l|li(?:ter|tre)s?)"
    r"(?![a-z])",
    re.IGNORECASE,
)
_ML_PER_UNIT = {
    "ml": 1.0,
    "milliliter": 1.0,
    "cl": 10.0,
    "centiliter": 10.0,
    "l": 1000.0,
    "liter": 1000.0,
    "fl oz": 29.5735295625,  # US fluid ounce
    "oz": 29.5735295625,  # on a beverage label "oz" is always the fluid ounce
    "pt": 473.176473,
    "qt": 946.352946,
    "gal": 3785.411784,
}


@dataclass(frozen=True)
class Abv:
    """An alcohol-content declaration: the raw text plus whatever numbers it yielded."""

    text: str
    percent: float | None = None
    proof: float | None = None

    # Printed to one decimal place, so half of the smallest printed step. This is the
    # precision of the declaration itself, not a tuning knob.
    TOLERANCE: ClassVar[float] = 0.05

    @property
    def display(self) -> str:
        """Just the numbers, for evidence read beside the record's own wording."""
        if self.percent is None:
            return ""
        proof = f" ({format_number(self.proof)} proof)" if self.proof is not None else ""
        return f"{format_number(self.percent)}%{proof}"


@dataclass(frozen=True)
class Volume:
    """A net-contents declaration, converted to millilitres."""

    text: str
    value: float
    unit: str
    ml: float

    # §6.5: equal within half a millilitre.
    TOLERANCE_ML: ClassVar[float] = 0.5


def format_number(value: float) -> str:
    """Render a parsed number the way a label prints it: 40, not 40.0; 13.5 stays 13.5."""
    return f"{value:g}"


def _declared_percent(raw: str) -> re.Match[str] | None:
    """The percentage the alcohol declaration is made of, not the first one on the page.

    Five of the corpus labels print "100% DE AGAVE" on the front panel and "40% Alc./Vol."
    in the mandatory declaration below it, so "first match wins" reads the marketing copy.
    Candidates are ranked by character distance to the nearest "alc"/"vol"/"proof"; ties keep
    the earliest. With none of those words anywhere on the page there is nothing to rank by,
    so the first percentage is the only available reading — which is also the single-field
    case, where the record supplies "45% Alc./Vol." and the context word is present anyway.
    """
    candidates = list(_PERCENT.finditer(raw))
    contexts = [c.span() for c in _ABV_CONTEXT.finditer(raw)]
    if not candidates or not contexts:
        return candidates[0] if candidates else None

    def distance(match: re.Match[str]) -> int:
        start, end = match.span()
        return min(max(0, c_start - end, start - c_end) for c_start, c_end in contexts)

    return min(candidates, key=distance)


def parse_abv(text: str) -> Abv:
    """Read the percentage and the optional proof figure out of a declaration or a page."""
    raw = (text or "").strip()
    percent = _declared_percent(raw)
    proof = _PROOF.search(raw)
    return Abv(
        text=raw,
        percent=float(percent.group(1)) if percent else None,
        proof=float(proof.group(1)) if proof else None,
    )


def _canonical_unit(unit: str) -> str:
    """Fold a matched unit spelling ("Litres", "fl. oz.", "GAL") onto a `_ML_PER_UNIT` key."""
    folded = re.sub(r"[.\s]", "", unit.lower()).rstrip("s")
    for prefix, canonical in (
        ("floz", "fl oz"),
        ("milli", "milliliter"),
        ("centi", "centiliter"),
        ("li", "liter"),
        ("gal", "gal"),
    ):
        if folded.startswith(prefix):
            return canonical
    return folded


def parse_volumes(text: str) -> list[Volume]:
    """Every volume declaration in the text, in order, converted to millilitres. A page can
    carry a dual declaration ("355 mL / 12 fl oz"); both state the same net contents, so either
    one matching the record is a match."""
    found: list[Volume] = []
    for match in _VOLUME.finditer(text or ""):
        unit = _canonical_unit(match.group(2))
        per_ml = _ML_PER_UNIT.get(unit)
        if per_ml is None:
            continue
        value = float(match.group(1))
        found.append(
            Volume(
                text=" ".join(match.group(0).split()),
                value=value,
                unit=unit,
                ml=value * per_ml,
            )
        )
    return found


def parse_volume(text: str) -> Volume | None:
    """The single volume a record field declares, or None if it does not parse."""
    found = parse_volumes(text)
    return found[0] if found else None
