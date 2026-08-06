#!/usr/bin/env python
"""Regenerate the cached test fixtures. Run by hand; its output is committed.

Two products, both of which exist so that the default test suite runs in seconds with no
OCR engine and no corpus:

  ocr_pass1.json   one entry per `spike/corpus/` image: the pass-1 OCR lines (text, score,
                   quad) at the shipped pass-1 configuration, plus the reading-order page
                   text. Text-level tests replay these instead of running the engine.

  bold/*.png       small synthetic label patches drawn with PIL — a bold prefix over a
  bold/manifest.json  regular body, an all-regular pair, an all-bold body, a blurred
                   violation, and a tiny-type block — together with the exact word quads
                   used to draw them, so `app.bold.measure()` can be driven directly.

This generator is the ONE place allowed to import from `spike/` (§13 lifts the engine
configuration from there, and the bold measurement below is used only to record evidence
in the manifest — never to derive a test expectation). No test imports spike.

    .venv/bin/python tests/fixtures/gen_fixtures.py --ocr --bold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORPUS = ROOT / "spike" / "corpus"
BOLD_DIR = HERE / "bold"

# The pass-1 configuration from ARCHITECTURE §2.1 / §2.5. Mirrored here rather than
# imported from app/, because a fixture generator that follows the application would
# regenerate itself into agreement with a bug.
PASS1_LONG_SIDE = 960
OCR_THREADS = 2
CLS_THRESH = 0.99

STATUTE = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
    "alcoholic beverages during pregnancy because of the risk of birth defects. "
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
    "operate machinery, and may cause health problems."
)

# Georgia, because its bold and regular weights separate cleanly at these sizes: the drawn
# compliant/violation patches land well inside the §5.2 bands instead of on a boundary.
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Georgia.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"


# ---------------------------------------------------------------------------
# A. cached pass-1 OCR
# ---------------------------------------------------------------------------


def build_ocr_fixture(out_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "spike"))
    import cv2
    import prep_ops as P
    from rapidocr import RapidOCR

    cv2.setNumThreads(OCR_THREADS)
    engine = RapidOCR(
        params={
            "Cls.cls_thresh": CLS_THRESH,
            "Det.limit_type": "max",
            "Det.limit_side_len": PASS1_LONG_SIDE,
            "EngineConfig.onnxruntime.intra_op_num_threads": OCR_THREADS,
            "EngineConfig.onnxruntime.inter_op_num_threads": OCR_THREADS,
        }
    )

    out: dict[str, dict] = {}
    files = sorted(p for p in CORPUS.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    for i, path in enumerate(files, 1):
        img = P.resize_long_side(P.load_bgr(str(path)), PASS1_LONG_SIDE)
        res = engine(img, return_word_box=True)
        lines = []
        boxes = list(res.boxes) if res.boxes is not None else []
        for text, score, box in zip(res.txts or [], res.scores or [], boxes, strict=False):
            quad = np.asarray(box, dtype=np.float64).reshape(4, 2)
            lines.append(
                {
                    "text": str(text),
                    "score": round(float(score), 4),
                    "quad": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
                }
            )
        out[path.name] = {
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "lines": lines,
            "page_text": reading_order_text(lines),
        }
        print(f"[{i:>2}/{len(files)}] {path.name}: {len(lines)} lines", flush=True)

    out_path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KiB)")


def reading_order_text(lines: list[dict]) -> str:
    """Top-to-bottom, left-to-right join — the un-deskewed form (spike/warning_match.py).

    Cached so that tests which only need a realistic page of label text (the record-match
    and warning-presence checks) do not depend on the pipeline's own assembly.
    """
    if not lines:
        return ""
    quads = [np.asarray(ln["quad"], dtype=np.float64) for ln in lines]
    ys = [float(q[:, 1].mean()) for q in quads]
    xs = [float(q[:, 0].mean()) for q in quads]
    hs = [float(q[:, 1].max() - q[:, 1].min()) for q in quads]
    tol = max(4.0, float(np.median(hs)) * 0.6)
    order = sorted(range(len(lines)), key=lambda i: (round(ys[i] / tol), xs[i]))
    return " ".join(lines[i]["text"] for i in order)


# ---------------------------------------------------------------------------
# B. synthetic bold patches
# ---------------------------------------------------------------------------

WARNING_LINES = [
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not",
    "drink alcoholic beverages during pregnancy because of the risk of birth defects.",
    "(2) Consumption of alcoholic beverages impairs your ability to drive a car or",
    "operate machinery, and may cause health problems.",
]
PREFIX_WORDS = 2  # "GOVERNMENT" "WARNING:" are the words 16.22(a)(2) requires in bold
REFERENCE_LINE = "Distilled and Bottled by Old Tom Distillery, Bardstown, Kentucky"

# The recogniser fragments long words, so one case ships the prefix pre-split the way the
# engine really returns it (§5.1 step 2).
FRAGMENTS = {"GOVERNMENT": ["GOVERNM", "ENT"], "WARNING:": ["WA", "RNING:"]}

# (prefix bold, body bold, font px, blur sigma, draw a reference line, fragment the prefix)
BOLD_CASES = {
    # prefix bold, body regular: the compliant arrangement 16.22(a)(2) requires
    "compliant": (True, False, 30, 0.0, True, False),
    # nothing is bold: same weight throughout
    "prefix_not_bold": (False, False, 30, 0.0, True, False),
    # everything is bold, including the body the regulation requires to stay regular
    "body_bold": (True, True, 30, 0.0, True, False),
    # the same-weight violation, defocused: the false-PASS the sharpness gate exists to stop
    "blurred_violation": (False, False, 30, 2.0, True, False),
    # type so small that stroke width carries no signal
    "tiny": (True, False, 9, 0.0, False, False),
    # compliant, but the prefix arrives as four fragments instead of two words
    "fragmented_prefix": (True, False, 30, 0.0, True, True),
}


def draw_case(name: str, spec: tuple) -> dict:
    """Render one synthetic block and return its OCR-shaped lines with exact word quads."""
    prefix_bold, body_bold, size, sigma, with_reference, fragment = spec
    regular = ImageFont.truetype(FONT_REGULAR, size)
    bold = ImageFont.truetype(FONT_BOLD, size)
    margin, leading = 30, int(size * 1.6)

    canvas = Image.new("RGB", (10, 10), "white")
    probe = ImageDraw.Draw(canvas)
    width = margin * 2 + max(
        int(probe.textlength(t, font=bold)) for t in WARNING_LINES + [REFERENCE_LINE]
    )
    height = margin * 2 + leading * (len(WARNING_LINES) + (2 if with_reference else 0))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    lines: list[dict] = []
    y = margin
    word_index = 0
    for text in WARNING_LINES:
        x = margin
        words = text.split(" ")
        quads = []
        for word in words:
            in_prefix = word_index < PREFIX_WORDS
            font = bold if (in_prefix and prefix_bold) or (not in_prefix and body_bold) else regular
            draw.text((x, y), word, font=font, fill="black")
            pieces = FRAGMENTS[word] if (fragment and in_prefix) else [word]
            cut = x
            for piece in pieces:
                end = cut + probe.textlength(piece, font=font)
                quads.append({"text": piece, "quad": box_quad(cut, y, end, y + size * 1.2)})
                cut = end
            x += probe.textlength(word + " ", font=font)
            word_index += 1
        lines.append({"text": text, "score": 0.99, "words": quads, "quad": union_quad(quads)})
        y += leading

    if with_reference:
        y += leading // 2
        draw.text((margin, y), REFERENCE_LINE, font=regular, fill="black")
        w = probe.textlength(REFERENCE_LINE, font=regular)
        words = []
        x = margin
        for word in REFERENCE_LINE.split(" "):
            words.append(
                {
                    "text": word,
                    "quad": box_quad(
                        x, y, x + probe.textlength(word, font=regular), y + size * 1.2
                    ),
                }
            )
            x += probe.textlength(word + " ", font=regular)
        lines.append(
            {
                "text": REFERENCE_LINE,
                "score": 0.99,
                "words": words,
                "quad": box_quad(margin, y, margin + w, y + size * 1.2),
            }
        )

    if sigma:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    BOLD_DIR.mkdir(parents=True, exist_ok=True)
    img.save(BOLD_DIR / f"{name}.png")
    return {"file": f"{name}.png", "lines": lines, "font_px": size, "blur_sigma": sigma}


def box_quad(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [
        [round(x0, 1), round(y0, 1)],
        [round(x1, 1), round(y0, 1)],
        [round(x1, 1), round(y1, 1)],
        [round(x0, 1), round(y1, 1)],
    ]


def union_quad(words: list[dict]) -> list[list[float]]:
    pts = np.asarray([w["quad"] for w in words], dtype=np.float64).reshape(-1, 2)
    return box_quad(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())


def build_bold_fixture(out_path: Path) -> None:
    manifest = {name: draw_case(name, spec) for name, spec in BOLD_CASES.items()}
    for name, entry in manifest.items():
        entry.update(measure_with_spike(BOLD_DIR / entry["file"], entry["lines"]))
        print(f"{name:>18}: {entry['spike_evidence']}", flush=True)
    out_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


def measure_with_spike(path: Path, lines: list[dict]) -> dict:
    """Record what the spike's stroke-weight code measures on the drawn patch.

    Evidence only. It tells a future reader that the fixture still reproduces the effect
    it was drawn for (notably: a blurred violation whose ratio inflates past T_PASS, which
    is the false-PASS the defocus gate exists to stop). Tests assert the design rule, not
    these numbers.
    """
    sys.path.insert(0, str(ROOT / "spike"))
    import eval_bold as EB  # noqa: N812
    import prep_ops as P  # noqa: N812

    img = P.load_bgr(str(path))
    shaped = [
        {
            "text": ln["text"],
            "score": ln["score"],
            "quad": np.asarray(ln["quad"], dtype=np.float64),
            "words": [
                {"text": w["text"], "quad": np.asarray(w["quad"], dtype=np.float64)}
                for w in ln["words"]
            ],
        }
        for ln in lines
    ]
    rec = EB.measure_image(path, shaped, upscale=1, mode="halfamp", max_body=20, img=img)
    est = (rec.get("est") or {}).get("area_skel") or {}
    # aggregate() exposes ink height only through body_hnorm = stroke / ink_h
    ink_h = None
    if est.get("body") and est.get("body_hnorm"):
        ink_h = est["body"] / est["body_hnorm"]
    return {
        "spike_evidence": {
            "ratio": _round(est.get("ratio")),
            "sharpness": _round(rec.get("sharpness")),
            "body_over_ref": _round(est.get("body_over_ref")),
            "median_body_ink_h_px": _round(ink_h),
            "n_prefix_tokens": rec.get("n_prefix_tokens"),
            "n_body_tokens": rec.get("n_body_tokens"),
            "prefix_texts": rec.get("prefix_texts"),
            "note": rec.get("note") or "",
        }
    }


def _round(v):
    return None if v is None else round(float(v), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ocr", action="store_true", help="re-run the corpus through the engine")
    ap.add_argument("--bold", action="store_true", help="redraw the synthetic bold patches")
    args = ap.parse_args()
    if not (args.ocr or args.bold):
        ap.error("choose --ocr and/or --bold")
    if args.ocr:
        if not CORPUS.is_dir():
            ap.error(f"{CORPUS} is absent (it is git-ignored); the corpus is required")
        build_ocr_fixture(HERE / "ocr_pass1.json")
    if args.bold:
        build_bold_fixture(BOLD_DIR / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
