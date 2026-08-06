"""The single worker: one image at a time, in the web process (§7.1).

`ThreadPoolExecutor(max_workers=1)` rather than a worker pool or a child process. The box
is 2 vCPU shared with nginx, Sarah's requirement is per-image latency rather than
throughput, and every measured number in the design was taken at one worker with two ONNX
threads. ONNX Runtime and OpenCV release the GIL during the heavy work, so the event loop
stays responsive; the models load once; there is no image pickling. Swapping in a
`ProcessPoolExecutor` is a one-line change if isolation is ever needed.

The analysis modules are imported inside the worker function, not at module scope, so the
OCR stack is loaded once on the worker thread and the HTTP surface can answer `/healthz`
while it is still warming.
"""

import json
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from . import db
from .config import Settings
from .ingest import crops_dir, db_path

log = logging.getLogger("labelcheck.queue")

# §8.2: crops are evidence for the reviewer, not archival copies.
CROP_QUALITY = 85
CROP_LONG_SIDE = 900


def pending_count(conn: sqlite3.Connection) -> int:
    """Images queued or in flight, for `/healthz`."""
    sql = "SELECT COUNT(*) FROM images WHERE status IN ('queued','running')"
    return conn.execute(sql).fetchone()[0]


def image_result(row: dict, result, verdict) -> dict:
    """The stable half of the §7.4 `ImageResult`; the API overlays live DB timings."""
    return {
        "image_id": row["id"],
        "filename": row["filename"],
        "record_id": row["record_id"],
        "pairing": row["pairing"],
        "passes": result.passes,
        "pass2_reason": result.pass2_reason,
        "pass2_skipped_reason": result.pass2_skipped_reason,
        "checks": [asdict(check) for check in verdict.checks],
        "informational": list(verdict.informational),
        "timings_ms": result.timings_ms,
        "page_text": result.page_text,
        "warning_region_text": result.read.region_text,
    }


class Worker:
    """Owns the executor, the worker thread's database connection, and warm-up state."""

    def __init__(self, s: Settings) -> None:
        self.s = s
        self.engine_ready = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labelcheck-worker")
        self._local = threading.local()

    def conn(self) -> sqlite3.Connection:
        """One connection per thread; the worker keeps its own for the whole run."""
        if getattr(self._local, "conn", None) is None:
            self._local.conn = db.connect(db_path(self.s))
        return self._local.conn

    def start(self) -> None:
        """Warm the engine, then re-queue anything a previous process left running."""
        self._pool.submit(self._warmup)
        conn = db.connect(db_path(self.s))
        orphans = db.requeue_orphans(conn)
        if orphans:
            log.warning("requeued %d image(s) left running by a previous process", orphans)
        for _ in range(pending_count(conn)):
            self.submit()
        conn.close()

    def stop(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def submit(self) -> None:
        """Ask the worker to take one more image off the queue."""
        self._pool.submit(self._process_one)

    def _warmup(self) -> None:
        from . import ocr  # lazy: loads onnxruntime on the worker thread (~900 ms cold)

        ocr.warmup()
        self.engine_ready = True

    def _process_one(self) -> None:
        """Claim the oldest queued image and record its verdict. Never raises."""
        conn = self.conn()
        row = db.claim_next_queued(conn)
        if row is None:
            return
        try:
            payload, overall, low_confidence = self._analyse(row)
            db.finish_image(conn, row["id"], status="done", overall=overall,
                            low_confidence=low_confidence,
                            result_json=json.dumps(payload))  # fmt: skip
        except Exception as exc:  # one bad image must never take the worker down
            log.exception("analysis failed for image %s", row["id"])
            message = f"This image could not be checked (internal error: {type(exc).__name__})."
            db.finish_image(conn, row["id"], status="error", overall="error", error=message)

    def _analyse(self, row: dict) -> tuple[dict, str, bool]:
        from . import pipeline, verdict  # lazy: see the module docstring

        result = pipeline.analyse(Path(row["stored_path"]), self.s)
        record = db.record_for_image(self.conn(), row["id"])
        # `pairing` lives on the image row, not on the record: an image the API marked
        # `ambiguous` (§7.7) keeps its `record_id`, so the join still returns a normal-looking
        # record and the four field checks would run confidently on an untrustworthy pair.
        # Carrying the pairing across is what lets `verdict` amber all four instead.
        record = dict(record, pairing=row["pairing"]) if record else None
        decision = verdict.evaluate(result, record, self.s)
        if any(check.state in ("amber", "red") for check in decision.checks):
            self._write_crops(row, result)
        return image_result(row, result, decision), decision.overall, result.low_confidence

    def _write_crops(self, row: dict, result) -> None:
        """Persist the band and bold crops (§8.2). A clean batch writes nothing."""
        from PIL import Image

        out = crops_dir(self.s)
        out.mkdir(parents=True, exist_ok=True)
        with Image.open(row["stored_path"]) as full:
            # The bold stage measures a frame capped at `bold_long_side`; rescale its rect
            # back onto the stored frame before cropping.
            bold_scale = max(full.size) / min(max(full.size), self.s.bold_long_side)
            for name, rect, scale in (("band", result.band_rect, 1.0),
                                      ("bold", result.bold.crop_rect, bold_scale)):  # fmt: skip
                if not rect:
                    continue
                box = tuple(int(round(v * scale)) for v in rect)
                crop = full.crop(box).convert("RGB")
                crop.thumbnail((CROP_LONG_SIDE, CROP_LONG_SIDE), Image.LANCZOS)
                crop.save(out / f"{row['id']}-{name}.jpg", "JPEG", quality=CROP_QUALITY)
