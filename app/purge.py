"""Retention purge — deletes batches past the TTL, with their files (§8.3).

Run as `python -m app.purge` from a systemd timer every 15 minutes. A 60-minute TTL
checked four times an hour keeps the "deleted every hour" promise within 15 minutes of
slack, where an hourly timer could leave data for 119. The API also calls `purge()`
lazily when a batch is created, so a stopped timer cannot silently retain uploads while
traffic continues.

The two `SELECT id` statements below read the database directly: `app/db.py` exposes the
queries the request path needs, and a bare id listing is only ever wanted here.
"""

import json
import shutil
import sqlite3
import sys

from . import db
from .config import Settings, get_settings
from .ingest import batch_dir, crops_dir, db_path


def purge(s: Settings) -> dict:
    """Delete expired batches, their uploads and crops, then reclaim database pages."""
    conn = db.connect(db_path(s))
    db.init_schema(conn)
    expired = db.expired_batches(conn, s.purge_ttl_min)
    files = 0
    deleted = 0
    for batch in expired:
        # Never delete under a batch that is still being worked: the worker would hit
        # missing files mid-analysis and an agent would watch live results vanish
        # (adversarial-review finding). It ages out on the next cycle once it finishes.
        busy = conn.execute(
            "SELECT COUNT(*) FROM images WHERE batch_id=? AND status IN ('queued','running')",
            (batch["id"],),
        ).fetchone()[0]
        if busy:
            continue
        deleted += 1
        rows = conn.execute("SELECT id FROM images WHERE batch_id=?", (batch["id"],)).fetchall()
        for row in rows:
            for crop in crops_dir(s).glob(f"{row['id']}-*.jpg"):
                crop.unlink(missing_ok=True)
                files += 1
        shutil.rmtree(batch_dir(s, batch["id"]), ignore_errors=True)
        db.delete_batch(conn, batch["id"])
    files += _sweep_orphans(conn, s)
    # auto_vacuum=INCREMENTAL is applied at connect time, so this really returns pages.
    conn.execute("PRAGMA incremental_vacuum")
    conn.commit()
    conn.close()
    return {"batches_deleted": deleted, "files_deleted": files, "ttl_min": s.purge_ttl_min}


def _sweep_orphans(conn: sqlite3.Connection, s: Settings) -> int:
    """Remove upload directories and crops with no surviving batch or image row.

    A process killed between writing a file and committing its row would otherwise leave
    an image on disk that no TTL ever reaches.
    """
    removed = 0
    uploads, crops = s.data_dir / "uploads", crops_dir(s)
    live_batches = {r["id"] for r in conn.execute("SELECT id FROM batches")}
    for path in sorted(uploads.iterdir()) if uploads.is_dir() else []:
        if path.is_dir() and path.name not in live_batches:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    live_images = {r["id"] for r in conn.execute("SELECT id FROM images")}
    for crop in sorted(crops.glob("*.jpg")) if crops.is_dir() else []:
        if crop.name.rsplit("-", 1)[0] not in live_images:
            crop.unlink(missing_ok=True)
            removed += 1
    return removed


def main() -> int:
    print(json.dumps(purge(get_settings())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
