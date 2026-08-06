"""Shared fixtures and the small factories the suite builds its inputs from.

Two rules shape this file.

*The default suite runs no OCR.* Anything that needs the engine is marked `slow`. The
text-level tests replay `fixtures/ocr_pass1.json` — real pass-1 output for all 56 corpus
images, cached once by `fixtures/gen_fixtures.py` — and the bold tests drive
`fixtures/bold/*.png`, synthetic patches drawn with PIL whose word quads are known exactly.

*A test may only touch a frozen interface.* The factories below build the dataclasses of
ARCHITECTURE §3.1. When the real module is importable they build the real class, so a
signature change fails the suite loudly; until then they fall back to a local mirror, which
is what lets the verdict tests run without the vision pipeline and vice versa.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CORPUS = ROOT / "spike" / "corpus"  # git-ignored: present locally, absent in a fresh clone
SAMPLES = ROOT / "app" / "samples"  # committed: the 22 golden images (§8.4)

# `app` is a plain source tree, never an installed package (see pyproject.toml).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set before anything imports app.config, which caches its settings on first read.
os.environ.setdefault("LABELCHECK_DATA_DIR", tempfile.mkdtemp(prefix="labelcheck-tests-"))

STATUTE_TEXT = importlib.import_module("app.statute").STATUTE_TEXT


# ---------------------------------------------------------------------------
# data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def settings():
    """The shipped defaults. Individual tests override a knob with `dataclasses.replace`."""
    from app.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def cached_ocr() -> dict:
    """Pass-1 OCR for every corpus image: {filename: {lines[], page_text, width, height}}."""
    return json.loads((FIXTURES / "ocr_pass1.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def bold_fixtures() -> dict:
    """Synthetic bold patches: {case: {file, lines[], font_px, blur_sigma, spike_evidence}}."""
    return json.loads((FIXTURES / "bold" / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden() -> dict:
    """The expected verdict envelope for the 22 golden labels (§11.9)."""
    return json.loads((Path(__file__).resolve().parent / "golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def ground_truth() -> dict:
    """The full corpus ground truth, keyed by filename. Skips when the corpus is absent."""
    path = CORPUS / "ground_truth.json"
    if not path.is_file():
        pytest.skip("spike/corpus/ is git-ignored and not present")
    return {row["file"]: row for row in json.loads(path.read_text(encoding="utf-8"))}


# ---------------------------------------------------------------------------
# factories for the §3.1 dataclasses
# ---------------------------------------------------------------------------


def _make(module: str, name: str, mirror, kwargs: dict):
    """Build the real dataclass when its module exists, otherwise the local mirror."""
    try:
        cls = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        cls = mirror
    return cls(**kwargs)


@dataclass
class _WarningRead:
    full_text: str
    region_text: str
    region_score: float
    prefix_score: float
    prefix_case: str | None
    upper_ratio: float
    n_lines: int
    n_tokens: int


@dataclass
class _BoldResult:
    state: str
    ratio: float | None
    sharpness: float | None
    cap_px: float | None
    gate: str | None
    likely_violation: str | None
    subtype_signal: str | None
    subtype_value: float | None
    crop_rect: list[int] | None
    n_prefix_tokens: int
    n_body_tokens: int
    ms: float


@dataclass
class _PipelineResult:
    read: object
    bold: object
    page_text: str
    page_tokens: list[str]
    passes: int
    pass2_reason: str | None
    pass2_skipped_reason: str | None
    low_confidence: bool
    band_rect: list[int] | None
    timings_ms: dict = field(default_factory=dict)


GW_SPAN = re.compile(r"g\s*o\s*v\s*e\s*r\s*n\s*m\s*e\s*n\s*t\s+w\s*a\s*r\s*n\s*i\s*n\s*g", re.I)


def upper_ratio_of(text: str) -> float:
    """Share of the matched GOVERNMENT WARNING span that is capitalised (§2.7, §6.3)."""
    match = GW_SPAN.search(text)
    if not match:
        return 0.0
    letters = [c for c in match.group(0) if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def make_read(region_text: str = STATUTE_TEXT, page_text: str | None = None, **override):
    """A `WarningRead` for a page whose statement reads as `region_text`.

    The two scores are computed the way pass 1 computes them (§4.1) so the object is
    internally consistent; a test that needs a specific score passes it explicitly.
    """
    from rapidfuzz import fuzz

    from app.statute import CLAUSE_1

    page = page_text if page_text is not None else region_text
    ratio = upper_ratio_of(region_text)
    values = {
        "full_text": page,
        "region_text": region_text,
        "region_score": float(fuzz.partial_ratio(STATUTE_TEXT.lower(), page.lower())),
        "prefix_score": float(fuzz.partial_ratio(CLAUSE_1.lower(), page.lower())),
        "prefix_case": "caps" if ratio >= 0.90 else ("title" if ratio <= 0.40 else "other"),
        "upper_ratio": ratio,
        "n_lines": max(1, len(page.split()) // 8),
        "n_tokens": len(page.split()),
    }
    values.update(override)
    return _make("app.warning", "WarningRead", _WarningRead, values)


def make_bold(**override):
    """A `BoldResult`. Defaults to the compliant measurement (§5.2 PASS band)."""
    values = {
        "state": "pass",
        "ratio": 1.48,
        "sharpness": 0.59,
        "cap_px": 25.0,
        "gate": None,
        "likely_violation": None,
        "subtype_signal": None,
        "subtype_value": None,
        "crop_rect": None,
        "n_prefix_tokens": 2,
        "n_body_tokens": 20,
        "ms": 98.0,
    }
    values.update(override)
    return _make("app.bold", "BoldResult", _BoldResult, values)


def make_result(page_text: str | None = None, **override):
    """A `PipelineResult` for the verdict engine: one clean pass over a compliant label."""
    read = override.pop("read", None) or make_read()
    page = page_text if page_text is not None else read.full_text
    values = {
        "read": read,
        "bold": override.pop("bold", None) or make_bold(),
        "page_text": page,
        "page_tokens": page.split(),
        "passes": 1,
        "pass2_reason": None,
        "pass2_skipped_reason": None,
        "low_confidence": False,
        "band_rect": None,
        "timings_ms": {
            "decode": 50.0,
            "resize": 5.0,
            "ocr1": 1700.0,
            "score": 3.0,
            "bold": 98.0,
            "prep2": 0.0,
            "ocr2": 0.0,
            "total": 1856.0,
        },
    }
    values.update(override)
    return _make("app.pipeline", "PipelineResult", _PipelineResult, values)


# ---------------------------------------------------------------------------
# OCR results replayed from the cached fixtures
# ---------------------------------------------------------------------------


def make_ocr_result(lines: list[dict], ms: float = 1700.0):
    """Build an `OcrResult` from fixture lines: [{text, score, quad, words?}, ...]."""
    import numpy as np

    from app.ocr import OcrLine, OcrResult, OcrWord

    built = []
    for line in lines:
        words = [
            OcrWord(
                text=w["text"],
                score=float(w.get("score", line.get("score", 0.99))),
                quad=np.asarray(w["quad"], dtype=np.float64),
            )
            for w in line.get("words", [])
        ]
        built.append(
            OcrLine(
                text=line["text"],
                score=float(line.get("score", 0.99)),
                quad=np.asarray(line["quad"], dtype=np.float64),
                words=words,
            )
        )
    return OcrResult(lines=built, ms=ms)


def check_by_id(verdict, check_id: str):
    """The one `Check` with this id, or None. Keeps the tests readable."""
    for check in verdict.checks:
        if check.id == check_id:
            return check
    return None


def make_record(**over) -> dict:
    """An application record in the CSV's canonical field names (§7.6)."""
    record = {
        "record_id": "label_001",
        "brand_name": "Old Tom Distillery",
        "class_type": "Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
    }
    record.update(over)
    return record
