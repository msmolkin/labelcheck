"""The two-pass ladder: an image file in, everything the verdict engine needs out.

Pass 1 always runs, cheaply, at 960 px on the long side. It reads the whole label, which is
where the brand, class, alcohol content and net contents come from. If the recovered warning
text scores well against 27 CFR 16.21, that is the answer and the image costs ~1.9 s.

Pass 2 runs only on doubt (§4.2). It crops the band pass 1 thinks the statement occupies out
of the FULL-resolution frame, deskews it with the pass-1 detection quads, applies CLAHE, and
re-reads it. For a large photograph that crop is an effective magnification of the statement
rather than a downscale, which is why a 960 px pass 1 plus a crop beats a single 1600 px pass
at a third of the cost. The better of the two reads wins on `region_score` alone.

Bold measurement (§4.6) runs last, on the native frame, with pass-1 word quads scaled back.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import bold as bold_stage
from . import imageops, ocr, warning
from .config import Settings
from .ocr import OcrLine, OcrResult
from .warning import WarningRead

# Seed for the pass-2 cost estimate before this process has completed one (§4.5). The
# measured pass-2 OCR range on the reference Mac is 1410-1717 ms plus 8-31 ms of prep; the
# seed is the top of that, so the first escalation is judged pessimistically rather than
# optimistically. It is replaced by measurement after the first completed pass 2.
PASS2_SEED_MS = 1750.0
PASS2_EWMA_ALPHA = 0.3

_pass2_ewma_ms = PASS2_SEED_MS

TIMING_KEYS = ("decode", "resize", "ocr1", "score", "bold", "prep2", "ocr2", "total")


@dataclass
class PipelineResult:
    read: WarningRead
    bold: bold_stage.BoldResult
    page_text: str
    page_tokens: list[str]
    passes: int
    pass2_reason: str | None
    pass2_skipped_reason: str | None
    low_confidence: bool
    band_rect: list[int] | None
    timings_ms: dict[str, float]


def analyse(image_path: Path, s: Settings) -> PipelineResult:
    """Run the ladder on one image. Never raises for a merely difficult photograph."""
    started = time.perf_counter()
    timings = dict.fromkeys(TIMING_KEYS, 0.0)

    with _timed(timings, "decode"):
        full = imageops.load_bgr(str(image_path))
    with _timed(timings, "resize"):
        small = imageops.resize_long_side(full, s.pass1_long_side)
    scale = small.shape[1] / full.shape[1]

    res1 = ocr.run(small, s.pass1_long_side)
    timings["ocr1"] = res1.ms
    with _timed(timings, "score"):
        read = warning.read_warning(res1, s.reading_order_deskew)
        band = warning.statute_band(full, res1, scale)
        escalate, reason = warning.should_escalate(read, s)
    # Held before pass 2 can replace `read`: only pass 1 ever sees the whole label, and the
    # field checks need the brand and the net contents, not the warning paragraph.
    page_text = read.full_text

    passes, pass2_reason, skipped = 1, None, None
    if escalate and s.pass2_mode != "off":
        skipped = _deadline_check(started, s)
        if skipped is None:
            read2 = _pass2(full, res1, band, s, timings)
            passes, pass2_reason = 2, reason
            # Region score only. `spike/warning_match.py` also offered max(region, prefix);
            # the region-only rule is the one every measured accuracy figure was taken
            # under, and shipping both would leave a reviewer guessing which ran.
            if read2.region_score >= read.region_score:
                read = read2

    with _timed(timings, "bold"):
        bold_result = _measure_bold(full, res1, scale, s)

    timings["total"] = (time.perf_counter() - started) * 1000.0
    return PipelineResult(
        read=read,
        bold=bold_result,
        page_text=page_text,
        page_tokens=page_text.split(),
        passes=passes,
        pass2_reason=pass2_reason,
        pass2_skipped_reason=skipped,
        low_confidence=skipped is not None,
        band_rect=band,
        timings_ms={k: round(v, 1) for k, v in timings.items()},
    )


def _deadline_check(started: float, s: Settings) -> str | None:
    """Skip pass 2 rather than blow the budget. An in-flight ONNX call cannot be cancelled,
    so the only honest mechanism is a pre-check against what pass 2 has been costing."""
    elapsed = (time.perf_counter() - started) * 1000.0
    remaining = s.budget_ms - s.pass2_reserve_ms - elapsed
    if _pass2_ewma_ms <= remaining:
        return None
    return (
        f"second pass needs about {_pass2_ewma_ms:.0f} ms and only {max(0.0, remaining):.0f} ms "
        f"of the {s.budget_ms} ms budget remained"
    )


def _pass2(
    full, res1: OcrResult, band: list[int] | None, s: Settings, timings: dict
) -> WarningRead:
    """Crop, deskew, CLAHE, re-read. Every op here earned its place by measurement (§4.4)."""
    global _pass2_ewma_ms
    with _timed(timings, "prep2"):
        crop = full
        if s.pass2_mode == "crop" and band is not None:
            # Clip to the frame and refuse degenerate slivers: a detection quad hanging
            # slightly off the resized frame scales to coordinates outside the full one,
            # and an empty ndarray crashes OpenCV downstream (adversarial-review finding).
            h, w = full.shape[:2]
            x0, y0 = max(0, band[0]), max(0, band[1])
            x1, y1 = min(w, band[2]), min(h, band[3])
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                crop = full[y0:y1, x0:x1]
        crop = imageops.resize_long_side(crop, s.pass2_long_side)
        crop = imageops.rotate(crop, imageops.estimate_skew_from_boxes(res1.quads))
        crop = imageops.clahe_lab(crop)
    res2 = ocr.run(crop, s.pass2_long_side)
    timings["ocr2"] = res2.ms
    spent = timings["prep2"] + res2.ms
    _pass2_ewma_ms = PASS2_EWMA_ALPHA * spent + (1 - PASS2_EWMA_ALPHA) * _pass2_ewma_ms
    return warning.read_warning(res2, s.reading_order_deskew)


def _measure_bold(full, res1: OcrResult, scale: float, s: Settings) -> bold_stage.BoldResult:
    """Measure stroke weight on the native frame, with pass-1 quads scaled onto it.

    Pass-1 quads specifically, not pass-2's: pass 2's geometry has been cropped, resized and
    rotated, so mapping it back is lossy, and pass-1 *detection* is reliable even where
    pass-1 *recognition* is not (`region_found_rate` 1.000 at 960 px).
    """
    frame = imageops.resize_long_side(full, s.bold_long_side)
    factor = (frame.shape[1] / full.shape[1]) / scale
    lines = [
        OcrLine(
            text=ln.text,
            score=ln.score,
            quad=ln.quad * factor,
            words=[ocr.OcrWord(w.text, w.score, w.quad * factor) for w in ln.words],
        )
        for ln in res1.lines
    ]
    return bold_stage.measure(frame, lines, s)


@contextmanager
def _timed(timings: dict, key: str):
    """Add this block's wall time to `timings[key]`."""
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[key] += (time.perf_counter() - started) * 1000.0
