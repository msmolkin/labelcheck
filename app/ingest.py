"""Upload validation, storage and record pairing — the §8.1 ladder, in order.

The re-encode in step 4 is the load-bearing step: it discards EXIF (and any payload
hidden in it), collapses polyglot files that are simultaneously a valid JPEG and a valid
something-else, and normalises orientation once at the boundary rather than in every
downstream stage.

Uploaded filenames are display metadata and nothing more. Storage names are
`uploads/<batch_id>/<uuid4>.jpg`, so a filename of `../../etc/passwd` is stored under an
ordinary UUID and only ever travels back out as a JSON string.
"""

import io
import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from .config import Settings

# A decode past this is a decompression bomb, not a label photograph: 36 MP covers any
# phone or DSLR label shot (a 12 MP photo is typical, 24 MP is a pro body), and one RGB
# decode stays near ~110 MB — the earlier 80 MP ceiling allowed ~1 GB resident on a
# 3.7 GB shared box under MemoryMax=1500M (adversarial-review finding).
MAX_PIXELS = 36_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# Below this on either side there is nothing to analyse — and degenerate 1-px inputs turn
# into divide-happy edge cases downstream.
MIN_SIDE = 64

# iPhones shoot HEIC by default; naming the format in the rejection saves the round trip
# of a confused agent asking why their photo "isn't an image".
_HEIC_BRANDS = (b"heic", b"heix", b"hevc", b"mif1", b"msf1", b"avif")

# Magic-byte prefixes of the three formats worth accepting. Animated and vector formats
# (GIF, SVG) carry the most decoder surface, and TIFF and PDF add a lot of parser for a
# workflow nobody has; all are rejected before a decoder sees them.
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_UNSAFE_NAME = re.compile(r"[\x00-\x1f\x7f]")


class IngestError(Exception):
    """A rejected upload, carrying the HTTP status and a message a person can act on."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def sniff_format(head: bytes) -> str | None:
    """Identify JPEG, PNG or WebP from the first bytes. Anything else returns None."""
    if head.startswith(JPEG_MAGIC):
        return "jpeg"
    if head.startswith(PNG_MAGIC):
        return "png"
    return "webp" if head[:4] == b"RIFF" and head[8:12] == b"WEBP" else None


def display_name(filename: str | None) -> str:
    """Reduce an uploaded filename to something safe to store and show. Path separators
    go because the result is displayed, never joined to a directory; HTML-significant
    characters stay, because the frontend renders it with `textContent` and mangling the
    name would misreport what was uploaded."""
    name = _UNSAFE_NAME.sub("", (filename or "").replace("\\", "/").split("/")[-1]).strip()
    return name[:200] or "upload"


def db_path(s: Settings) -> Path:
    """The one SQLite file. Layout lives here because this module writes it (§8.1, §8.2)."""
    return s.data_dir / "labelcheck.db"


def batch_dir(s: Settings, batch_id: str) -> Path:
    return s.data_dir / "uploads" / batch_id


def crops_dir(s: Settings) -> Path:
    return s.data_dir / "crops"


_USED_CACHE: dict[str, tuple[float, int]] = {}
_USED_CACHE_TTL_S = 15.0


def used_bytes(data_dir: Path) -> int:
    """Bytes currently held under the data directory, for the §7.5 disk quota.

    Cached for 15 s: the full-tree walk is O(files) and this runs on every upload, so a
    300-image batch would otherwise pay a quadratic re-scan (adversarial-review finding).
    The quota was always a soft, best-effort bound — the purge is what actually reclaims
    space — so a 15-second-stale reading changes nothing about its guarantees.
    """
    import time

    key = str(data_dir)
    cached = _USED_CACHE.get(key)
    now = time.monotonic()
    if cached and now - cached[0] < _USED_CACHE_TTL_S:
        return cached[1]
    total = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file())
    _USED_CACHE[key] = (now, total)
    return total


def store_upload(data: bytes, filename: str, dest_dir: Path, s: Settings) -> dict:
    """Run the §8.1 ladder and write the re-encoded image. Raises `IngestError`."""
    name = display_name(filename)
    if len(data) > s.max_upload_bytes:
        limit_mb = s.max_upload_bytes // (1024 * 1024)
        raise IngestError(413, f"{name} is larger than the {limit_mb} MB limit for one image.")
    if sniff_format(data[:32]) is None:
        if len(data) > 12 and data[4:8] == b"ftyp" and data[8:12].rstrip() in _HEIC_BRANDS:
            raise IngestError(
                415, f"{name} is a HEIC photo (the iPhone default). Export it as JPEG or PNG."
            )
        raise IngestError(415, f"{name} is not a JPEG, PNG or WebP image.")

    image = _decode(data, name)
    if min(image.width, image.height) < MIN_SIDE:
        raise IngestError(
            415, f"{name} is only {image.width}x{image.height} — too small to read a label from."
        )
    # `thumbnail` never upscales, so a small label photo keeps its own resolution. 2400 px
    # leaves the pass-2 band crop room to magnify while bounding decode cost (§8.1).
    image.thumbnail((s.ingest_long_side, s.ingest_long_side), Image.LANCZOS)
    if used_bytes(s.data_dir) + len(data) > s.max_disk_bytes:
        raise IngestError(507, "The server is out of storage. Older batches are purged hourly.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    stored = dest_dir / f"{uuid.uuid4().hex}.jpg"
    image.save(stored, "JPEG", quality=s.ingest_jpeg_quality, optimize=True)
    return {"filename": name, "stored_path": str(stored), "bytes": stored.stat().st_size,
            "width": image.width, "height": image.height, "exif_normalised": 1}  # fmt: skip


def _decode(data: bytes, name: str) -> Image.Image:
    """Steps 3 and 4: verify, decode, normalise orientation, flatten, strip metadata."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > MAX_PIXELS:
                raise IngestError(415, f"{name} is too large to open safely ({width}x{height}).")
            probe.verify()  # structural check; consumes the object, so re-open to decode
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened)
            image.load()
    except IngestError:
        raise
    except Image.DecompressionBombError:
        # Pillow's own guard, tripped before the pixel test above when the header lies.
        raise IngestError(415, f"{name} is too large to open safely.") from None
    except Exception:
        raise IngestError(415, f"{name} could not be opened as an image.") from None

    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[3])
        image = flat
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pair_record(filename: str, records: list[dict]) -> tuple[str | None, str]:
    """Associate an image with an application record (§7.7); first match wins. Never pairs
    by brand similarity: a wrong pairing gives a confident wrong verdict, which is worse
    than no verdict at all."""
    stem = Path(filename).stem.lower()
    lower = filename.lower()
    exact = [r for r in records if (r.get("filename") or "").strip().lower() == lower]
    if exact:
        return (exact[0]["record_id"], "exact") if len(exact) == 1 else (None, "ambiguous")
    by_stem = [r for r in records if str(r["record_id"]).lower() == stem]
    if by_stem:
        return (by_stem[0]["record_id"], "stem") if len(by_stem) == 1 else (None, "ambiguous")
    parts = set(re.split(r"[-_. ]+", stem))
    by_token = [r for r in records if str(r["record_id"]).lower() in parts]
    if len(by_token) == 1:
        return by_token[0]["record_id"], "token"
    if len(by_token) > 1:
        return None, "ambiguous"
    return None, "none"
