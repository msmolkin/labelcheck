"""RapidOCR access: one cached engine per detection resolution, one result shape.

The engine parameters below are not configurable. They are the configuration every accuracy
number in this repository was taken under, and two of them (`Cls.cls_thresh`, the thread
counts) break the tool quietly if changed. Resolution is the knob — `pass1_long_side` /
`pass2_long_side` — and it is applied by resizing the image, because tuning the detector's
own `limit_side_len` produced identical accuracy at every setting.

Models ship inside the `rapidocr` wheel, so nothing here touches the network; that is what
`tests/test_no_network.py` exists to prove.
"""

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .config import get_settings


@dataclass
class OcrWord:
    """One recognised word with its own quadrilateral (from `return_word_box=True`)."""

    text: str
    score: float
    quad: np.ndarray  # (4, 2) float64, TL TR BR BL


@dataclass
class OcrLine:
    """One detected text line and the words inside it."""

    text: str
    score: float
    quad: np.ndarray
    words: list[OcrWord] = field(default_factory=list)


@dataclass
class OcrResult:
    lines: list[OcrLine]
    ms: float

    @property
    def texts(self) -> list[str]:
        return [ln.text for ln in self.lines]

    @property
    def quads(self) -> list[np.ndarray]:
        return [ln.quad for ln in self.lines]


@lru_cache(maxsize=4)
def get_engine(long_side: int):
    """Build (once per resolution) the RapidOCR engine the whole app shares."""
    from rapidocr import RapidOCR

    threads = get_settings().ocr_threads
    return RapidOCR(
        params={
            # spike/env_check.py: at the stock 0.9 threshold the angle classifier
            # 180-flips horizontal lines on pristine renders and returns garbage.
            "Cls.cls_thresh": 0.99,
            "Det.limit_type": "max",
            "Det.limit_side_len": long_side,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": threads,
        }
    )


def run(img: np.ndarray, long_side: int) -> OcrResult:
    """OCR one BGR frame. Word boxes are always on: measured free (2021 ms with, 2239 ms
    without, on the same image), and the bold stage needs them."""
    engine = get_engine(long_side)
    started = time.perf_counter()
    res = engine(img, return_word_box=True)
    return _to_result(res, (time.perf_counter() - started) * 1000.0)


def _quad(raw) -> np.ndarray:
    return np.asarray(raw, dtype=np.float64).reshape(4, 2)


def _to_result(res, ms: float) -> OcrResult:
    """Convert a RapidOCR output object into our own dataclasses."""
    if res.boxes is None or res.txts is None:
        # RapidOCR can detect boxes and still return no recognitions; treating that as an
        # empty page beats zipping mismatched lists (adversarial-review finding).
        return OcrResult(lines=[], ms=ms)
    per_line = res.word_results or [()] * len(res.boxes)
    zipped = zip(res.boxes, res.txts, res.scores, per_line, strict=False)
    lines = [
        OcrLine(
            text=txt,
            score=float(score),
            quad=_quad(quad),
            words=[OcrWord(w[0], float(w[1]), _quad(w[2])) for w in words],
        )
        for quad, txt, score, words in zipped
    ]
    return OcrResult(lines=lines, ms=ms)


def warmup() -> None:
    """Load the models before the first real request (cold load costs ~900 ms)."""
    import cv2

    s = get_settings()
    canvas = np.full((96, 480, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "GOVERNMENT WARNING", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,) * 3, 2)
    for long_side in {s.pass1_long_side, s.pass2_long_side}:
        run(canvas, long_side)
