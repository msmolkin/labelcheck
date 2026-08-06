"""Turn one analysed image plus its application record into a three-state verdict (§6).

Amber is the designed home for uncertainty, and it is what keeps a bad photograph from becoming
a false green or a false rejection. Two rules carry that. The **TRUST gate** in `_warning_text`:
if clause (1) did not come back legible, the tool may not reject the label at all, however bad
the rest of the read looks. The **`low_confidence` demotion** in `evaluate`: a rejection the
system had no time to verify is not a rejection. Every `title` and `detail` is written for the
compliance agent reading the review panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

from app import fields, matching, textdiff
from app.config import Settings
from app.fields import format_number
from app.statute import CITATIONS, PREFIX, STATUTE_TEXT, max_chars_per_inch, min_type_size_mm

if TYPE_CHECKING:  # typing only: app.pipeline is the vision half, not imported at runtime
    from app.pipeline import PipelineResult

_FIELD_NAMES = {
    "brand": "Brand name",
    "class_type": "Class/type designation",
    "alcohol_content": "Alcohol content",
    "net_contents": "Net contents",
}
CHECK_IDS = ("warning_present", "warning_text", "warning_case", "warning_bold")
CHECK_IDS += tuple(_FIELD_NAMES)
# The record's CSV columns share the check ids apart from the brand.
_RECORD_COLUMN = {"brand": "brand_name"}
# Record-match checks carry citation=None: the duty to match the application lives in 27 CFR
# Parts 4/5/7, outside the regulatory basis this tool documents, and inventing a citation in a
# Treasury deliverable is not an option (§6.6).
RECORD_BASIS = "Compared against the application record."
_SEVERITY = {"na": 0, "green": 1, "amber": 2, "red": 3}
_REDUCED = " Checked in reduced mode (the fast pass only), so this was not confirmed."
_CODE_KEYS = {"class_type": "class", "alcohol_content": "abv", "net_contents": "net"}
_HOLD = " Left for a person to judge."
_SPAN_MIN = 70.0  # alignment score under which we will not claim to have found the words
_REDUCED_NOTE = (
    "The time budget did not allow the second, closer read of this image, so any deviation is "
    "reported for review rather than as a rejection."
)
_BOLD_AMBER = {
    "defocus": "This photograph is too blurred at the statement to measure stroke weight.",
    "tiny": "The statement is printed too small here to measure stroke weight.",
    "ratio": "The measured stroke weight falls between the pass and the fail band.",
}


@dataclass
class Check:
    """One check's finding. `evidence` is JSON-serialisable and drives the review panel.

    `evidence["note_code"]` carries the compact reason code the CSV export joins into its
    `notes` column (§7.6). It is present on every check that is not green.
    """

    id: str
    state: str  # "green" | "amber" | "red" | "na"
    title: str
    detail: str
    citation: str | None
    evidence: dict[str, Any] = field(default_factory=dict)
    crop: str | None = None  # "band" | "bold" | None


@dataclass
class Verdict:
    """The roll-up an agent sees first, plus every check behind it."""

    overall: str  # "green" | "amber" | "red" | "error"
    checks: list[Check] = field(default_factory=list)
    informational: list[dict] = field(default_factory=list)


def _check(cid, state, title, detail, ev, crop=None, code=None) -> Check:
    """Assemble one check, taking its citation from the single statute table."""
    if code:
        ev["note_code"] = code
    return Check(cid, state, title, detail, CITATIONS.get(cid), ev, crop)


def _na(cid: str, detail: str) -> Check:
    return Check(cid, "na", "Not assessed", detail, CITATIONS.get(cid), {})


def evaluate(p: PipelineResult, record: dict | None, s: Settings) -> Verdict:
    """Apply §6's rules to one `PipelineResult`. Pure: no I/O, no engine, no clock."""
    present = _warning_present(p, s)
    checks = [present]
    if present.state == "green":
        checks += [_warning_text(p, s), _warning_case(p, s), _warning_bold(p, s)]
    else:
        why = "No statement was found" if present.state == "red" else "This image was unreadable"
        checks += [_na(cid, f"{why}, so this was not assessed.") for cid in CHECK_IDS[1:4]]
    checks += _record_checks(p, record, s)
    informational = [_type_size(record, p.page_text)]
    if p.low_confidence:
        checks = [_demote(c) for c in checks]
        informational.append({"id": "low_confidence", "text": _REDUCED_NOTE, "citation": None})
    return Verdict(overall=roll_up(checks), checks=checks, informational=informational)


def roll_up(checks: list[Check]) -> str:
    """Worst state wins. `na` checks never affect the outcome (§6)."""
    states = [c.state for c in checks if c.state != "na"]
    return max(states, key=lambda state: _SEVERITY[state], default="green")


def _demote(c: Check) -> Check:
    """§6.7: a red the pipeline had no time to verify becomes an amber."""
    if c.state != "red":
        return c
    ev = dict(c.evidence, demoted_from="red")
    return Check(c.id, "amber", c.title, c.detail + _REDUCED, c.citation, ev, c.crop)


# --------------------------------------------------------- warning checks (27 CFR Part 16)


def _warning_present(p: PipelineResult, s: Settings) -> Check:
    """§6.1. The token floor separates "no warning on this label" from "unreadable photo"."""
    anchor = float(fuzz.partial_ratio(PREFIX.lower(), (p.page_text or "").lower()))
    ev: dict[str, Any] = {"anchor_score": round(anchor, 1), "page_tokens": len(p.page_tokens)}
    if anchor >= s.anchor_min:
        detail = "The health warning statement was located on the label."
        return _check("warning_present", "green", "Health warning present", detail, ev)
    if len(p.page_tokens) >= s.min_tokens_for_absence:
        detail = (
            "The label text read clearly and carries no health warning statement, which "
            "27 CFR 16.21 requires on every container of alcoholic beverage."
        )
        title = "No health warning statement found"
        return _check("warning_present", "red", title, detail, ev, code="present:absent")
    detail = "Too little text came back from this photograph to say whether a statement is there."
    title = "This image could not be read"
    return _check("warning_present", "amber", title, detail, ev, code="present:unreadable")


def _warning_text(p: PipelineResult, s: Settings) -> Check:
    """§6.2. The TRUST gate runs first: an unreliable read may never produce a rejection."""
    diff = textdiff.compare(p.read.region_text, s)
    ev: dict[str, Any] = {
        "prefix_score": round(p.read.prefix_score, 1),
        "region_score": round(p.read.region_score, 1),
        "diff": diff.diff,
        "clause_coverage": {k: round(v, 3) for k, v in diff.clause_coverage.items()},
        "noise_words": diff.noise_words,
        "suspect_words": diff.suspect_words,
        "critical_defects": diff.critical_defects,
        "expected": STATUTE_TEXT,
    }
    if p.read.prefix_score < s.text_trust_min:
        detail = (
            "The statement did not come back clearly enough to compare word for word, so this "
            "label is held for review rather than rejected."
        )
        title = "Warning text could not be read reliably"
        return _check("warning_text", "amber", title, detail, ev, "band", "text:unreadable")
    if diff.state == "green":
        detail = "The statement matches 27 CFR 16.21 word for word."
        if diff.noise_words:
            detail += f" {diff.noise_words} word(s) show OCR noise and were read through."
        title = "Warning text matches 27 CFR 16.21"
        return _check("warning_text", "green", title, detail, ev, "band")
    if diff.state == "amber" and diff.suspects and not diff.defects:
        expected, got = diff.suspects[0]
        more = f" (and {len(diff.suspects) - 1} more like it)" if len(diff.suspects) > 1 else ""
        detail = (
            f'The label reads "{got}" where the statement requires "{expected}"{more} — one '
            "letter apart, which is more often the camera than the label. Confirm against "
            "the image."
        )
        title = "Warning text is one letter off — confirm against the image"
        return _check("warning_text", "amber", title, detail, ev, "band", "text:suspect")
    detail, code = _text_finding(diff)
    if diff.state == "red":
        title = "Warning text does not match 27 CFR 16.21"
        return _check("warning_text", "red", title, detail, ev, "band", code)
    title = "Warning text could not be confirmed word for word"
    return _check("warning_text", "amber", title, detail, ev, "band", "text:unconfirmed")


def _text_finding(diff: textdiff.TextDiff) -> tuple[str, str]:
    """Name the deviations in an agent's words, and the CSV reason code that stands for them."""
    parts = [f"Clause ({n}) of the required statement is missing." for n in diff.missing_clauses]
    for d in diff.defects[:3]:
        if d.kind == "substituted":
            parts.append(f'The label reads "{d.got}" where the statement requires "{d.expected}".')
        elif d.kind == "missing":
            parts.append(f'The word "{d.expected}" is missing.')
        else:
            parts.append(f'The label adds "{d.got}" inside the statement.')
    if len(diff.defects) > 3:
        parts.append(f"{len(diff.defects) - 3} further difference(s) are marked in the diff.")
    critical = next((d for d in diff.defects if d.critical), None)
    if diff.missing_clauses:
        code = f"text:clause{diff.missing_clauses[0]}_missing"
    elif critical is not None:
        code = f"text:{critical.expected or '-'}!={critical.got or '-'}"
    else:
        code = "text:mismatch"
    return " ".join(parts) or "Some words could not be matched to the statement.", code


def _warning_case(p: PipelineResult, s: Settings) -> Check:
    """§6.3. An uppercase ratio, never string equality: OCR returns `GoVERNMENT` (§2.7)."""
    ratio = float(p.read.upper_ratio)
    span = _prefix_span(p.read.region_text) or _prefix_span(p.read.full_text)
    ev: dict[str, Any] = {"upper_ratio": round(ratio, 4), "span": span}
    if p.read.prefix_case is None or span is None:
        detail = 'The words "GOVERNMENT WARNING" were not located precisely enough to judge case.'
        title = "Capitalisation could not be checked"
        return _check("warning_case", "amber", title, detail, ev, "band", "case:notfound")
    read_as = f'Read as "{span}" ({round(ratio * 100)}% of letters capitalised).'
    if ratio >= s.case_caps_min:
        title = '"GOVERNMENT WARNING" is in capital letters'
        return _check("warning_case", "green", title, read_as, ev, "band")
    if ratio <= s.case_title_max:
        detail = f"{read_as} 27 CFR 16.22(a)(2) requires those two words in capital letters."
        title = '"GOVERNMENT WARNING" is not in capital letters'
        return _check("warning_case", "red", title, detail, ev, "band", "case:title")
    detail = f"{read_as} That is neither clearly capitals nor clearly title case."
    title = "Capitalisation is unclear"
    return _check("warning_case", "amber", title, detail, ev, "band", "case:mixed")


def _prefix_span(text: str) -> str | None:
    """The label's own spelling of "GOVERNMENT WARNING", for the evidence panel. Located by
    alignment, not position: label copy regularly bleeds into the front of the cropped region
    ("750 mL GOVERNMENT WARNING: ..."), so the words are not reliably the first two."""
    align = fuzz.partial_ratio_alignment(PREFIX.lower(), (text or "").lower())
    if align is None or align.score < _SPAN_MIN:
        return None
    return text[align.dest_start : align.dest_end].strip(":;,. ") or None


def _warning_bold(p: PipelineResult, s: Settings) -> Check:
    """§6.4. The red names the requirement; the subtype is only ever the *likely* one (§5.3)."""
    b = p.bold
    keys = ("ratio", "sharpness", "cap_px", "gate", "likely_violation")
    ev: dict[str, Any] = {k: getattr(b, k) for k in keys}
    if b.state == "pass":
        detail = (
            f"The prefix measures {b.ratio:.2f} times the stroke weight of the body text, where "
            f"{s.bold_t_pass:.2f} or more is required."
        )
        title = '"GOVERNMENT WARNING" is bold and the remainder is not'
        return _check("warning_bold", "green", title, detail, ev, "bold")
    if b.state == "fail":
        likely = (
            "the whole statement appears to be bold"
            if b.likely_violation == "body_bold"
            else 'the words "GOVERNMENT WARNING" appear not to be bold'
        )
        detail = (
            f"The prefix and the body carry the same stroke weight ({b.ratio:.2f}, where "
            f"{s.bold_t_pass:.2f} or more is required), so one of the two bold requirements is "
            f"broken. Most likely {likely}."
        )
        title = "The bold requirement of 27 CFR 16.22(a)(2) is not met"
        code = f"bold:{b.likely_violation or 'fail'}"
        return _check("warning_bold", "red", title, detail, ev, "bold", code)
    if b.state == "not_found":
        detail = "The warning block was not located precisely enough to measure stroke weight."
        title = "Stroke weight could not be measured"
        return _check("warning_bold", "amber", title, detail + _HOLD, ev, "bold", "bold:not_found")
    detail = _BOLD_AMBER.get(b.gate or "", _BOLD_AMBER["ratio"]) + _HOLD
    title = "Stroke weight could not be measured reliably"
    return _check("warning_bold", "amber", title, detail, ev, "bold", f"bold:{b.gate or 'amber'}")


# ----------------------------------------------------- application-record checks (§6.5, §6.6)


def _record_checks(p: PipelineResult, record: dict | None, s: Settings) -> list[Check]:
    """The four record comparisons. With no record they are `na` and the tool still works."""
    if record is None:
        no_record = "No application record was supplied for this image."
        return [_na(cid, no_record) for cid in _FIELD_NAMES]
    if record.get("pairing") == "ambiguous":
        detail = (
            "This image matched more than one application record, so it was compared to none of "
            "them. Pairing a label with the wrong record produces a confident wrong answer."
        )
        title = "Could not be compared to a record"
        return [
            _check(c, "amber", title, detail, {}, None, f"{_CODE_KEYS.get(c, c)}:ambiguous")
            for c in _FIELD_NAMES
        ]
    readable = len(p.page_tokens) >= s.min_tokens_for_absence
    says = {c: (record.get(_RECORD_COLUMN.get(c, c)) or "").strip() for c in _FIELD_NAMES}
    found: dict[str, matching.FieldMatch] = {}
    for cid, green, amber in (
        ("brand", s.brand_green, s.brand_amber),
        ("class_type", s.class_green, s.class_amber),
    ):
        if says[cid]:
            # A brand may be printed as one word or hyphenated; a class/type designation is not.
            text, score = matching.best_window(
                p.page_text, says[cid], ignore_spacing=cid == "brand"
            )
            found[cid] = matching.compare_similar(
                text, says[cid], score, green=green, amber=amber, readable=readable
            )
    if says["alcohol_content"]:
        found["alcohol_content"] = matching.compare_alcohol(
            fields.parse_abv(p.page_text), fields.parse_abv(says["alcohol_content"])
        )
    if says["net_contents"]:
        found["net_contents"] = matching.compare_volume(
            fields.parse_volumes(p.page_text),
            fields.parse_volume(says["net_contents"]),
            says["net_contents"],
        )
    return [
        _record_check(c, found[c])
        if c in found
        else _na(c, f"The application record states no {_FIELD_NAMES[c].lower()}.")
        for c in _FIELD_NAMES
    ]


def _record_check(check_id: str, m: matching.FieldMatch) -> Check:
    """Wrap a `FieldMatch` as a `Check`. `CITATIONS` maps all four of these to None (§6.6)."""
    name = _FIELD_NAMES[check_id]
    ending = {
        "green": "matches the application record",
        "red": "does not match the application record",
        "amber": "could not be confirmed",
    }[m.state]
    detail = f'The label reads "{m.label or "nothing"}"; the record states "{m.record}".'
    if m.note:
        detail = f"{detail} {m.note}"
    ev: dict[str, Any] = {"label": m.label, "record": m.record}
    if m.score is not None:
        ev["score"] = round(m.score, 1)
    code = f"{_CODE_KEYS.get(check_id, check_id)}:{m.code}" if m.code else None
    return _check(check_id, m.state, f"{name} {ending}", f"{detail} {RECORD_BASIS}", ev, None, code)


# ------------------------------------------------------------ informational panel (§16.22(b))


def _type_size(record: dict | None, page_text: str) -> dict:
    """Type-size minimums for the declared volume, and the note that a photograph cannot
    settle them. Informational only: it never enters the roll-up (§1, §15.1)."""
    volume = fields.parse_volume(((record or {}).get("net_contents") or "").strip())
    if volume is None:
        volume = next(iter(fields.parse_volumes(page_text)), None)
    if volume is None:
        text = (
            "The net contents could not be determined, so the minimum type size 27 CFR 16.22(b) "
            "requires cannot be stated. It is not verifiable from a photograph in any case."
        )
    else:
        mm = min_type_size_mm(volume.ml)
        text = (
            f"{volume.text} container: 27 CFR 16.22(b) requires type of at least "
            f"{format_number(mm)} mm, and 16.22(a)(4) allows at most {max_chars_per_inch(mm)} "
            "characters per inch at that size. Neither is verifiable from an uncalibrated "
            "photograph."
        )
    return {"id": "type_size", "text": text, "citation": CITATIONS["type_size"]}
