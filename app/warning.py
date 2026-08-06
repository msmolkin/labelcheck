"""Find 27 CFR 16.21 in a page of OCR output and judge how well it was read.

Three jobs, none needing ground truth — which is what makes them usable at runtime as an
escalation trigger: re-assemble the detected lines into reading order (the detector emits
them arbitrarily, and the statute only matches once they are back in sequence); score the
text against the statute and against clause (1) alone; decide whether the read is good enough
to skip the second pass. The engine's own confidence appears nowhere, deliberately: measured
(`prep_eval.json -> conf_signal`), garbled reads carry mean confidence 0.987-0.995, dead
inside the range of perfect reads (0.980-0.996), and sweeping that gate from 0.80 to 0.95
changed the escalation set by zero images. What works is semantic — how well the text matches
the law.
"""

import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz

from .config import Settings, get_settings
from .imageops import MIN_SKEW_DEG, estimate_skew_from_boxes
from .ocr import OcrResult
from .statute import CLAUSE_1, STATUTE_TEXT

# "GOVERNMENT WARNING" as the recogniser may hand it back: spaced out, any case. Matching on
# the raw read is deliberate — nothing upstream of the case check may fold case (§6.3).
GW_RE = re.compile(r"g\s*o\s*v\s*e\s*r\s*n\s*m\s*e\s*n\s*t\s+w\s*a\s*r\s*n\s*i\s*n\s*g", re.I)

# Landmark phrases that mark a line as part of the statement, for the pass-2 band crop.
BAND_KEYS = ("government warning", "surgeon general", "during pregnancy", "birth defects",
             "alcoholic beverages", "operate machinery", "health problems",
             "impairs your ability", "consumption of")  # fmt: skip
BAND_KEY_MIN = 70.0  # fuzzy score a line needs against a landmark to join the band
BAND_MIN_W, BAND_MIN_H = 60, 30  # a crop smaller than this is not a paragraph


@dataclass
class WarningRead:
    """One reading of the statement, with the evidence for every downstream decision."""

    full_text: str
    region_text: str
    region_score: float  # partial-fuzzy against the whole statute
    prefix_score: float  # partial-fuzzy against clause (1) alone
    prefix_case: str | None  # "caps" | "title" | "other" | None
    upper_ratio: float
    n_lines: int
    n_tokens: int


def assemble_text(res: OcrResult, deskew: bool) -> str:
    """Join detected lines top-to-bottom, left-to-right. Clustering happens on box centroids,
    which interleave across visual lines on a rotated page — so with `deskew` on we rotate
    them by the negative of the skew the detector implies (a 2x2 matmul over a few dozen
    points, free next to the OCR), about their own centroid: only relative order matters."""
    lines = res.lines
    if not lines:
        return ""
    centres = np.array([[q[:, 0].mean(), q[:, 1].mean()] for q in res.quads], dtype=np.float64)
    heights = np.array([q[:, 1].max() - q[:, 1].min() for q in res.quads], dtype=np.float64)
    if deskew:
        theta = estimate_skew_from_boxes(res.quads)
        if abs(theta) >= MIN_SKEW_DEG:
            rad = np.radians(-theta)
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            centres = (centres - centres.mean(axis=0)) @ rot.T
    tol = max(4.0, float(np.median(heights)) * 0.6)
    order = sorted(range(len(lines)), key=lambda i: (round(centres[i, 1] / tol), centres[i, 0]))
    return " ".join(lines[i].text for i in order)


def upper_ratio(span: str) -> float:
    """Share of the letters in a span that are capitals. Never a string equality test: strict
    equality scores 0.9388 because OCR returns "GoVERNMENT" on 3 of 49 images, where this
    ratio scores 1.000 (one slipped glyph in 17 letters is still 0.941, Title Case is 0.12)."""
    letters = [c for c in span if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def classify_case(ratio: float, s: Settings) -> str:
    """Map an uppercase ratio onto the three states of 16.22(a)(2)."""
    if ratio >= s.case_caps_min:
        return "caps"
    if ratio <= s.case_title_max:
        return "title"
    return "other"


def read_warning(res: OcrResult, deskew: bool) -> WarningRead:
    """Score one OCR result against 27 CFR 16.21. Reads `Settings` for the case cuts only."""
    s = get_settings()
    full = re.sub(r"\s+", " ", assemble_text(res, deskew)).strip()
    region, region_score, prefix_score = "", 0.0, 0.0
    if full:
        aligned = fuzz.partial_ratio_alignment(STATUTE_TEXT.lower(), full.lower())
        region = full[aligned.dest_start : aligned.dest_end]
        region_score = float(aligned.score)
        prefix_score = float(fuzz.partial_ratio(CLAUSE_1.lower(), full.lower()))
    match = GW_RE.search(full)
    ratio = upper_ratio(match.group(0)) if match else 0.0
    return WarningRead(
        full_text=full, region_text=region, region_score=region_score,
        prefix_score=prefix_score, prefix_case=classify_case(ratio, s) if match else None,
        upper_ratio=ratio, n_lines=len(res.lines), n_tokens=len(full.split()),
    )  # fmt: skip


def should_escalate(r: WarningRead, s: Settings) -> tuple[bool, str]:
    """Pass-1 doubt test: run the second pass, and say why. The two-score form keeps genuine
    violations out of the queue for the wrong reason — a label that legitimately omits clause
    (2) scores low against the full statute even on a perfect read, but its clause-(1) score
    stays at 100, so it does not look like OCR damage. `require_caps` is the asymmetric-cost
    rule: a pass-1 read implying a rejection is re-verified before the tool rejects."""
    if r.n_lines == 0:
        return True, "no text detected"
    if r.prefix_score < s.esc_prefix_min:
        return True, f"prefix_score {r.prefix_score:.1f} < {s.esc_prefix_min:.1f}"
    if r.region_score < s.esc_region_min:
        return True, f"region_score {r.region_score:.1f} < {s.esc_region_min:.1f}"
    if s.esc_require_caps and r.prefix_case != "caps":
        return True, f'prefix case is "{r.prefix_case}", not capitals'
    return False, ""


def statute_band(
    img: np.ndarray, res: OcrResult, scale: float, pad_frac: float = 0.05
) -> list[int] | None:
    """Rect `[x0, y0, x1, y1]` in FULL-resolution coordinates around the statement. `scale` is
    the pass-1 frame's size relative to `img`, so pass-1 quads map back onto it. Selection is
    by TEXT — a line joins the band if it fuzzy-matches a landmark phrase. Density-based
    picking was measured and cut: it truncated paragraphs, cost accuracy (0.9285 vs 0.9912)
    and was slower."""
    if not res.lines or scale <= 0:
        return None

    def is_statute_line(text: str) -> bool:
        return max(fuzz.partial_ratio(k, text.lower()) for k in BAND_KEYS) >= BAND_KEY_MIN

    selected = [ln.quad for ln in res.lines if is_statute_line(ln.text)]
    if not selected:
        return None
    xs = np.concatenate([q[:, 0] for q in selected]) / scale
    ys = np.concatenate([q[:, 1] for q in selected]) / scale
    h, w = img.shape[:2]
    x0 = int(max(0, xs.min() - pad_frac * w))
    x1 = int(min(w, xs.max() + pad_frac * w))
    y0 = int(max(0, ys.min() - pad_frac * h))
    y1 = int(min(h, ys.max() + pad_frac * h))
    if x1 - x0 < BAND_MIN_W or y1 - y0 < BAND_MIN_H:
        return None
    return [x0, y0, x1, y1]
