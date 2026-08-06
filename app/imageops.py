"""The image operations that earned a place in the pipeline by measurement.

Five functions, each a pure BGR-uint8 transform except the loader, which owns EXIF. The
spike implemented and measured a dozen more; none of them ship. Their deltas in mean warning
accuracy, against the same pipeline without the op, are why (`spike/out/prep_eval.json`):
glare repair -0.5795 (and the slowest op measured, 364-646 ms), tight-mask glare repair
+0.0041 (cut on cost), global binarisation -0.0339, adaptive binarisation -0.0262,
sharpening -0.0043, percentile contrast stretch -0.0026, flat-field flatten -0.0464 against
CLAHE on the harsh corpus, 1.5x upscale -0.0015 with case fidelity falling 0.980 -> 0.959,
and a pixel-based skew estimator +0.0055 against +0.0355 for the quad-based one below.

Threading is pinned by `app.config` at import, before OpenCV reads it.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

# Below this a rotation is noise: the warp costs more than the ordering it fixes.
MIN_SKEW_DEG = 0.4


def load_bgr(path: str, apply_exif: bool = True) -> np.ndarray:
    """Read an image to BGR uint8, honouring the EXIF orientation tag.

    Phone cameras store the sensor frame plus tag 274; `cv2.imread` ignores it, so a portrait
    photo arrives sideways and the detector sees rotated text. Worth +0.4558 mean warning
    accuracy on sideways-stored JPEGs (0.9806 vs 0.5249, n=6) for about 1.5 ms.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    im = Image.open(io.BytesIO(raw))
    if apply_exif:
        im = ImageOps.exif_transpose(im)
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def resize_long_side(img: np.ndarray, target: int) -> np.ndarray:
    """Downscale so the long side is `target` pixels. Never upscales."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= target:
        return img
    s = target / float(longest)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def estimate_skew_from_boxes(boxes, max_abs_deg: float = 20.0) -> float:
    """Skew angle in degrees from OCR detection quads (positive = content rotated CCW).

    Width-weighted median of the top and bottom edge angles. Free once a detection pass has
    run, and more reliable than any pixel heuristic because the detector has already decided
    what is text: it recovered rotated pages the pixel estimator missed entirely.
    """
    if boxes is None or len(boxes) == 0:
        return 0.0
    angles: list[float] = []
    weights: list[float] = []
    for box in boxes:
        p = np.asarray(box, dtype=np.float64).reshape(-1, 2)
        if p.shape[0] < 4:
            continue
        for start, end in ((p[0], p[1]), (p[3], p[2])):  # top edge, bottom edge
            dx, dy = end[0] - start[0], end[1] - start[1]
            if abs(dx) < 20:  # too short to carry a reliable angle
                continue
            ang = float(np.degrees(np.arctan2(dy, dx)))
            if abs(ang) <= max_abs_deg:
                angles.append(ang)
                weights.append(abs(dx))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    sorted_angles = np.array(angles)[order]
    cumulative = np.cumsum(np.array(weights)[order])
    return float(sorted_angles[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def rotate(img: np.ndarray, angle_deg: float, min_abs: float = MIN_SKEW_DEG) -> np.ndarray:
    """Rotate by -angle so a +angle skew is undone, expanding the canvas to fit."""
    if abs(angle_deg) < min_abs:
        return img
    h, w = img.shape[:2]
    centre = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += new_w / 2.0 - centre[0]
    m[1, 2] += new_h / 2.0 - centre[1]
    return cv2.warpAffine(
        img, m, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def clahe_lab(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Local contrast on the LAB L channel, leaving colour alone.

    The one photometric op that paid: +0.018 on the glare class and +0.005 overall on the
    realistic corpus, +0.131 on glare and +0.036 overall on the harsh corpus, and case
    fidelity 0.980 -> 1.000, for 8-14 ms. The clip and tile values are the measured ones.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)
