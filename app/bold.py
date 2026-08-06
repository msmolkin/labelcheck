"""27 CFR 16.22(a)(2): is "GOVERNMENT WARNING" bold, and is the rest of the statement not?

A photograph carries no font metadata, so the measurement is relative stroke weight. Every
word of the statement is rectified from its OCR quad into an upright patch, binarised, and
thinned to a one-pixel skeleton; `ink_area / skeleton_length` is then its average stroke
width in pixels. Pool each group by median and take

    ratio = median(stroke width of "GOVERNMENT WARNING") / median(stroke width of the body)

A compliant label has a heavier prefix than body; a violation carries one weight throughout.
The ratio compares two runs of type *within the same photograph at the same size*, so it is
immune to camera distance, label size and exposure — and it is explainable to an agent and to
an appeals process in one sentence, which a CNN verdict is not.

Measured over 49 warning-bearing corpus images (`spike/out/bold_eval.json`): 0.9714 balanced
accuracy; in the three-state rule, 40 confident calls, 0 hard errors, 9 amber. `area_skel`
beat the 25x cheaper `area_perim` not on balanced accuracy — they tie — but on hard errors,
0 against 2, so the thinning is worth paying for. The stage abstains twice before any ratio
band is consulted: under defocus, and below ~11 px cap height. Neither is optional (§5.2).
"""

import statistics
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from rapidfuzz import fuzz

from .config import Settings
from .ocr import OcrLine
from .statute import STATUTE_TEXT

# A line of the label joins the warning block if it fuzzy-matches the statute or this
# opening fragment of it; the anchor helps when the block's first line is the garbled one.
BODY_ANCHOR = "According to the Surgeon General women should not drink alcoholic beverages"
PREFIX_KEY = "governmentwarning"

BLOCK_MIN_SCORE = 72.0  # fuzzy score against the statute that admits a line to the block
PREFIX_MIN_SCORE = 78.0  # fuzzy score that accepts an accumulation as the prefix
MAX_BODY_TOKENS = 20  # cost bound: the median is stable long before 20 words
MAX_REF_TOKENS = 14
REF_TALL_FACTOR = 1.45  # a line taller than this x the median is display type (the brand)
MIN_TOKEN_CHARS = 3  # shorter tokens carry too little ink to measure
MIN_REF_LINE_CHARS = 6
SHARP_TOKENS = 12  # body patches the defocus signal is pooled over
PAD_FRAC = 0.30  # patch padding, as a fraction of glyph height, for a clean border ring
GEOM_SEG_MIN = 0.18  # inter-word gap floor, as a fraction of line height


@dataclass
class BoldResult:
    state: str  # "pass" | "fail" | "amber" | "not_found"
    ratio: float | None = None
    sharpness: float | None = None
    cap_px: float | None = None
    gate: str | None = None  # "defocus" | "tiny" | "ratio" | None
    likely_violation: str | None = None  # "prefix_not_bold" | "body_bold"
    subtype_signal: str | None = None
    subtype_value: float | None = None
    crop_rect: list[int] | None = None  # x0, y0, x1, y1 in measurement-frame coords
    n_prefix_tokens: int = 0
    n_body_tokens: int = 0
    ms: float = 0.0
    note: str = ""


@dataclass
class _Tok:
    """One word to measure, with the height of the line it came from."""

    text: str
    quad: np.ndarray
    line_h: float
    line_i: int = -1


@dataclass
class _Block:
    prefix: list[_Tok] = field(default_factory=list)
    body: list[_Tok] = field(default_factory=list)
    ref: list[_Tok] = field(default_factory=list)
    line_h: float = 0.0
    n_lines: int = 0
    note: str = ""
    prefix_source: str = ""


# --- geometry and text helpers ---------------------------------------------

_ALNUM = set("0123456789abcdefghijklmnopqrstuvwxyz")


def _norm(s: str) -> str:
    """Fold to alphanumerics: OCR punctuation noise must not break a match."""
    return "".join(ch for ch in s.lower() if ch in _ALNUM)


def _centre_y(quad: np.ndarray) -> float:
    return float(np.asarray(quad, dtype=float)[:, 1].mean())


def _quad_hw(quad: np.ndarray) -> tuple[float, float]:
    """(width, height) of a 4-point quad in TL, TR, BR, BL order."""
    a = np.asarray(quad, dtype=float)
    w = (np.linalg.norm(a[1] - a[0]) + np.linalg.norm(a[2] - a[3])) / 2.0
    h = (np.linalg.norm(a[3] - a[0]) + np.linalg.norm(a[2] - a[1])) / 2.0
    return float(w), float(h)


def _quad_xspan(quad: np.ndarray) -> tuple[float, float]:
    a = np.asarray(quad, dtype=float)
    return float(a[:, 0].min()), float(a[:, 0].max())


def _sub_quad(quad: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """Slice a line quad along its text direction, between fractions f0 and f1."""
    tl, tr, br, bl = np.asarray(quad, dtype=np.float32)
    return np.array(
        [tl + (tr - tl) * f0, tl + (tr - tl) * f1, bl + (br - bl) * f1, bl + (br - bl) * f0],
        dtype=np.float32,
    )


def _toks_of(line: OcrLine, line_i: int) -> list[_Tok]:
    h = _quad_hw(line.quad)[1]
    return [_Tok(w.text, w.quad, h, line_i) for w in line.words]


# --- locating the block and the prefix --------------------------------------


def _prefix_from_words(toks: list[_Tok]) -> list[_Tok] | None:
    """Accumulate leading word tokens until they spell GOVERNMENT WARNING. The recogniser
    fragments words ("GOVERNM" + "ENT", "GOVERNMENT" + "WA" + "RNIN" + "G:"), so a
    token-by-token equality test misses the prefix on exactly the images where getting it
    right matters most."""
    best_score, best_toks = 0.0, None
    for start in range(0, min(3, len(toks))):
        acc = ""
        for k in range(start, min(start + 6, len(toks))):
            acc += _norm(toks[k].text)
            if len(acc) < 8:
                continue
            score = fuzz.ratio(acc, PREFIX_KEY)
            if best_toks is None or score > best_score:
                best_score, best_toks = score, toks[start : k + 1]
            if len(acc) >= len(PREFIX_KEY) + 2:
                break
    return best_toks if best_score >= PREFIX_MIN_SCORE else None


def _segment_line_words(img: np.ndarray, quad: np.ndarray) -> list[_Tok]:
    """Split a line strip into word blobs by column-gap analysis: the fallback for when the
    OCR text of the prefix line is too garbled to locate "GOVERNMENT WARNING" by string
    matching. Word gaps are separated from letter gaps by a 1-D threshold over gap widths."""
    patch, pad = _rectify(img, quad, pad_frac=0.12)
    if patch is None:
        return []
    ink, _ = _binarize(patch, pad, mode="otsu")
    if ink is None:
        return []
    core = ink[:, pad:-pad] if pad and patch.shape[1] > 2 * pad else ink
    cols = np.flatnonzero(core.sum(axis=0) > 0)
    if cols.size == 0:
        return []
    runs, start = [], cols[0]
    for a, b in zip(cols[:-1], cols[1:], strict=True):
        if b > a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, cols[-1]))
    if len(runs) < 2:
        return []
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    threshold = max(2.0, GEOM_SEG_MIN * core.shape[0])
    if len(gaps) >= 3:
        g = np.array(sorted(gaps), dtype=np.float32)
        best_var, best_thr = -1.0, threshold
        for i in range(1, len(g)):  # 1-D Otsu: intra-letter gaps vs inter-word gaps
            w0, w1 = i / len(g), 1 - i / len(g)
            var = w0 * w1 * (g[:i].mean() - g[i:].mean()) ** 2
            if var > best_var:
                best_var, best_thr = var, (g[i - 1] + g[i]) / 2
        threshold = max(threshold, best_thr)
    words, current = [], [runs[0]]
    for i, gap in enumerate(gaps):
        if gap >= threshold:
            words.append((current[0][0], current[-1][1]))
            current = [runs[i + 1]]
        else:
            current.append(runs[i + 1])
    words.append((current[0][0], current[-1][1]))
    width, line_h = core.shape[1], _quad_hw(quad)[1]
    return [
        # "?" marks a word located geometrically: there is no text for it, by construction.
        _Tok("?", _sub_quad(quad, max(0.0, (a - 1) / width), min(1.0, (b + 2) / width)), line_h)
        for a, b in words
    ]


def _line_above(lines: list[OcrLine], top: OcrLine, line_h: float) -> OcrLine | None:
    """The line sitting immediately above the block and overlapping it horizontally."""
    ty = _centre_y(top.quad)
    tx0, tx1 = _quad_xspan(top.quad)
    candidates = []
    for ln in lines:
        if ln is top or not (0 < ty - _centre_y(ln.quad) < 2.0 * line_h):
            continue
        x0, x1 = _quad_xspan(ln.quad)
        if min(x1, tx1) - max(x0, tx0) > 0.35 * (tx1 - tx0):
            candidates.append(ln)
    return max(candidates, key=lambda ln: _centre_y(ln.quad)) if candidates else None


def find_warning_block(img: np.ndarray, lines: list[OcrLine]) -> _Block:
    """Select the statement's lines, then split them into prefix, body and reference."""
    statute, anchor = _norm(STATUTE_TEXT), _norm(BODY_ANCHOR)
    block = [
        ln
        for ln in lines
        if len(_norm(ln.text)) >= MIN_REF_LINE_CHARS
        and max(
            fuzz.partial_ratio(_norm(ln.text), statute), fuzz.partial_ratio(_norm(ln.text), anchor)
        )
        >= BLOCK_MIN_SCORE
    ]
    if not block:
        return _Block(note="warning_block_not_found")
    block.sort(key=lambda ln: _centre_y(ln.quad))
    line_h = statistics.median([_quad_hw(ln.quad)[1] for ln in block])
    block_toks = [_toks_of(ln, i) for i, ln in enumerate(block)]

    prefix, prefix_line, source = None, None, ""
    for li, toks in enumerate(block_toks[:2]):  # (a) locate the prefix by its text
        found = _prefix_from_words(toks)
        if found:
            prefix, prefix_line, source = found, li, "text"
            break

    if prefix is None:  # (b) a garbled prefix line sitting just above the matched block
        above = _line_above(lines, block[0], line_h)
        if above is not None:
            segments = _segment_line_words(img, above.quad)
            block.insert(0, above)
            block_toks.insert(0, segments if len(segments) >= 3 else _toks_of(above, 0))
            if len(segments) >= 3:
                prefix, prefix_line, source = segments[:2], 0, "geometric_above"

    if prefix is None:  # (c) geometric segmentation of the block's own first line
        segments = _segment_line_words(img, block[0].quad)
        if len(segments) >= 4:
            prefix, prefix_line, source = segments[:2], 0, "geometric_first_line"
            block_toks[0] = segments

    if prefix is None:
        return _Block(note="prefix_not_found", n_lines=len(block), line_h=line_h)
    for i, toks in enumerate(block_toks):
        for tok in toks:
            tok.line_i = i

    def usable(tok: _Tok) -> bool:
        return len(_norm(tok.text)) >= MIN_TOKEN_CHARS or tok.text == "?"

    off_line = [
        t for i, toks in enumerate(block_toks) if i != prefix_line for t in toks if usable(t)
    ]
    if len(off_line) >= 6:
        body = off_line
    else:  # a one-line statement: take everything the prefix did not claim
        claimed = {id(t) for t in prefix}
        body = [t for toks in block_toks for t in toks if id(t) not in claimed and usable(t)]

    return _Block(
        prefix=prefix,
        body=body,
        ref=_reference_tokens(lines, block),
        line_h=line_h,
        n_lines=len(block),
        prefix_source=source,
    )


def _reference_tokens(lines: list[OcrLine], block: list[OcrLine]) -> list[_Tok]:
    """The label's other copy, as a sample of its regular weight. Used only to choose *which*
    bold rule was broken (§5.3), never to make the call itself. The tallest lines are dropped
    because the brand name is normally the one display-bold element on a label — a heuristic
    about label design, not a statutory fact, and an all-bold label defeats it."""
    block_ids = {id(ln) for ln in block}
    others = [
        ln for ln in lines if id(ln) not in block_ids and len(_norm(ln.text)) >= MIN_REF_LINE_CHARS
    ]
    if not others:
        return []
    median_h = statistics.median([_quad_hw(ln.quad)[1] for ln in others])
    ref: list[_Tok] = []
    for ln in others:
        h = _quad_hw(ln.quad)[1]
        if h > REF_TALL_FACTOR * median_h:
            continue
        ref += [_Tok(w.text, w.quad, h) for w in ln.words if len(_norm(w.text)) >= MIN_TOKEN_CHARS]
    return ref


# --- patch extraction and binarisation --------------------------------------


def _rectify(img: np.ndarray, quad, pad_frac: float = PAD_FRAC):
    """Perspective-warp a word quad into an upright patch, padded from its surroundings. The
    padding gives the binariser a border ring of genuine background, which is what lets the
    threshold sit at the half-amplitude point between paper and ink."""
    src = np.asarray(quad, dtype=np.float32)
    w, h = _quad_hw(quad)
    w, h = max(4.0, w), max(4.0, h)
    pad = max(3.0, pad_frac * h)
    out_w, out_h = int(round(w + 2 * pad)), int(round(h + 2 * pad))
    if out_w <= 0 or out_h <= 0 or out_w > 6000 or out_h > 3000:
        return None, 0
    dst = np.array(
        [[pad, pad], [pad + w, pad], [pad + w, pad + h], [pad, pad + h]], dtype=np.float32
    )
    patch = cv2.warpPerspective(
        img,
        cv2.getPerspectiveTransform(src, dst),
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch, int(round(pad))


def _cc_clean(ink: np.ndarray) -> np.ndarray:
    """Drop connected components too small to be ink (JPEG mosquitoes, paper texture)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    min_area = max(4, int(0.0015 * ink.size))
    keep = np.array([i > 0 and stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(n)])
    return keep[labels]


def _binarize(patch: np.ndarray, pad: int, mode: str = "halfamp"):
    """Ink mask for one patch. Returns `(mask, info)` or `(None, reason)`.

    "halfamp" is the shipped mode: threshold at the half-amplitude point between the border
    ring (background) and a robust ink level — the classic 50 %-crossing rule, which a
    symmetric blur kernel leaves in place, so stroke width survives mild defocus. Ink polarity
    is detected, so dark-background labels measure correctly. The alternative mode exists only
    for the geometric word segmentation above, which needs a mask rather than a width."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    if gray.size < 64:
        return None, "too_small"
    if float(gray.std()) < 6.0:
        return None, "no_contrast"

    ring = np.ones(gray.shape, dtype=bool)
    k = max(1, pad // 2)
    if gray.shape[0] > 2 * k + 2 and gray.shape[1] > 2 * k + 2:
        ring[k:-k, k:-k] = False
    background = float(np.median(gray[ring])) if ring.any() else float(np.median(gray))

    if mode == "otsu":
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_light = background > threshold
    else:
        core = gray[pad:-pad, pad:-pad] if gray.shape[0] > 2 * pad + 2 else gray
        lo, hi = float(np.percentile(core, 2)), float(np.percentile(core, 98))
        bg_light = abs(lo - background) > abs(hi - background)
        foreground = lo if bg_light else hi
        if abs(foreground - background) < 12:
            return None, "no_contrast"
        threshold = (background + foreground) / 2.0

    ink = _cc_clean((gray < threshold) if bg_light else (gray > threshold))
    fraction = float(ink.mean())
    if fraction < 0.02:
        return None, "no_ink"
    if fraction > 0.65:
        return None, "ink_flood"
    return ink, {"ink_frac": fraction, "bg_light": bool(bg_light)}


# --- stroke width -----------------------------------------------------------


def zhang_suen(bw: np.ndarray, max_iter: int = 40) -> np.ndarray:
    """Vectorised Zhang-Suen thinning to a one-pixel skeleton (uint8 0/1)."""
    img = bw.astype(np.uint8).copy()
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4 = p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:]
            p5, p6, p7 = p[2:, 2:], p[2:, 1:-1], p[2:, 0:-2]
            p8, p9 = p[1:-1, 0:-2], p[0:-2, 0:-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            transitions = np.zeros_like(neighbours)
            for i in range(8):
                transitions += ((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8)
            base = (img == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            if step == 0:
                condition = base & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condition = base & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if condition.any():
                img[condition] = 0
                changed = True
        if not changed:
            break
    return img


def _token_stats(ink: np.ndarray, pad: int) -> dict | None:
    """Average stroke width of one word: ink area over skeleton length.

    Width uses the whole padded patch — the configuration every measured ratio in this
    repository was taken under. Ink *height* is measured inside the word quad only, because
    the padding can reach into the line above or below. On the prefix, which the statute
    requires to be capitals, that height is cap height: 27 px on a 38 px face and 15 px on a
    21 px face, matching the "21 px font is about a 15 px cap" figure the abstain was set
    from."""
    u8 = ink.astype(np.uint8)
    area = float(u8.sum())
    if area < 20:
        return None
    skeleton = int(zhang_suen(u8).sum())
    if skeleton < 5:
        return None
    core = ink[pad:-pad, pad:-pad] if min(ink.shape) > 2 * pad + 2 else ink
    ys, _ = np.nonzero(core)
    return {
        "area_skel": area / skeleton,
        "ink_h": float(ys.max() - ys.min() + 1) if ys.size else 0.0,
    }


def _measure_tokens(img: np.ndarray, toks: list[_Tok], max_tokens: int | None = None) -> list[dict]:
    measured = []
    for tok in toks[:max_tokens] if max_tokens else toks:
        patch, pad = _rectify(img, tok.quad)
        if patch is None:
            continue
        ink, _ = _binarize(patch, pad)
        if ink is None:
            continue
        stats = _token_stats(ink, pad)
        if stats is None:
            continue
        stats["line_h"] = tok.line_h
        # Defocus signal, on the patch we already warped, so it costs nothing: Laplacian
        # variance normalised by the patch's own variance. Dividing out contrast is what
        # stops a dark or washed-out label reading as blurry, and — unlike stroke width per
        # glyph height — it is not confounded with boldness.
        g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        stats["sharp"] = float(cv2.Laplacian(g, cv2.CV_32F).var() / max(g.var(), 1e-6))
        measured.append(stats)
    return measured


def _aggregate(tokens: list[dict]) -> dict:
    """Pool per-token measurements by median (robust to a blown patch or two)."""
    out = {
        "area_skel": float(statistics.median([t["area_skel"] for t in tokens])),
        "ink_h": float(statistics.median([t["ink_h"] for t in tokens])),
    }
    # Height-normalised variant, needed whenever two runs of DIFFERENT point size are
    # compared (the warning body against the label's other copy).
    normalised = [t["area_skel"] / t["line_h"] for t in tokens if t.get("line_h")]
    if normalised:
        out["n_area_skel"] = float(statistics.median(normalised))
    return out


def _crop_rect(img: np.ndarray, block: _Block) -> list[int] | None:
    """Prefix words plus the first two body lines — what the UI shows as evidence."""
    body_lines = sorted({t.line_i for t in block.body})[:2]
    quads = [t.quad for t in block.prefix] + [t.quad for t in block.body if t.line_i in body_lines]
    if not quads:
        return None
    points = np.concatenate([np.asarray(q, dtype=float) for q in quads])
    h, w = img.shape[:2]
    return [
        int(max(0, points[:, 0].min())),
        int(max(0, points[:, 1].min())),
        int(min(w, points[:, 0].max())),
        int(min(h, points[:, 1].max())),
    ]


def measure(img: np.ndarray, lines: list[OcrLine], s: Settings) -> BoldResult:
    """Measure 16.22(a)(2) on one frame.

    `img` is the measurement frame and `lines` must carry quads in its coordinates (the
    pipeline scales pass-1 quads onto it). Measuring on the native frame rather than the
    downscaled OCR frame is not a detail: the 960 px frame compresses ratios toward 1.0 and
    moved two of ten reference labels across a band boundary (§2.4)."""
    started = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - started) * 1000.0

    block = find_warning_block(img, lines)
    if block.note:
        return BoldResult(state="not_found", note=block.note, ms=elapsed())

    prefix = _measure_tokens(img, block.prefix)
    body = _measure_tokens(img, block.body, MAX_BODY_TOKENS)
    reference = _measure_tokens(img, block.ref, MAX_REF_TOKENS)
    sharps = [t["sharp"] for t in body[:SHARP_TOKENS]]
    sharpness = float(statistics.median(sharps)) if sharps else None
    if not prefix or len(body) < 2:
        return BoldResult(
            state="not_found",
            note="insufficient_tokens",
            sharpness=sharpness,
            n_prefix_tokens=len(prefix),
            n_body_tokens=len(body),
            ms=elapsed(),
        )

    p, b = _aggregate(prefix), _aggregate(body)
    r = _aggregate(reference) if len(reference) >= 3 else None
    ratio = p["area_skel"] / b["area_skel"] if b["area_skel"] else None
    cap_px = p["ink_h"]  # cap height of "GOVERNMENT WARNING"; the body is the same face

    # Order is the rule (§5.2): abstain on unmeasurable ink BEFORE reading the ratio bands.
    # A defocused violation inflates past the pass threshold (measured: 1.374 and 1.702 at
    # sigma >= 3.0), which without this gate is a confident false PASS.
    gate: str | None = None
    if sharpness is not None and sharpness < s.bold_t_sharp:
        state, gate = "amber", "defocus"
    elif cap_px < s.bold_min_cap_px:
        state, gate = "amber", "tiny"
    elif ratio is None:
        state, gate = "amber", "ratio"
    elif ratio >= s.bold_t_pass:
        state = "pass"
    elif ratio <= s.bold_t_fail:
        state = "fail"
    else:
        state, gate = "amber", "ratio"

    result = BoldResult(
        state=state,
        ratio=ratio,
        sharpness=sharpness,
        cap_px=cap_px,
        gate=gate,
        crop_rect=_crop_rect(img, block),
        n_prefix_tokens=len(prefix),
        n_body_tokens=len(body),
        ms=elapsed(),
    )
    if state in ("fail", "amber"):
        _add_subtype(result, block, b, r, s)
    return result


def _add_subtype(result: BoldResult, block: _Block, body: dict, ref: dict | None, s: Settings):
    """Decide which of the two bold requirements is the likely one to have been broken. 92.9 %
    accurate on the corpus, which is why the verdict layer reports it as the *likely*
    violation and states the red as "one of the two bold requirements in 16.22(a)(2)"."""
    if ref and ref.get("n_area_skel") and body.get("n_area_skel"):
        value = body["n_area_skel"] / ref["n_area_skel"]
        signal, body_is_bold = "body_over_ref", value >= s.bold_t_ref
    else:  # no regular-weight copy on the label: fall back to stroke width per line height
        value = body["area_skel"] / block.line_h if block.line_h else 0.0
        signal, body_is_bold = "body_norm_abs", value >= s.bold_t_abs
    result.subtype_signal = signal
    result.subtype_value = float(value)
    result.likely_violation = "body_bold" if body_is_bold else "prefix_not_bold"
