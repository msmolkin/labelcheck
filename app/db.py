"""SQLite storage: schema, connection factory, and every query the app runs.

Four conventions to know before reading further. All timestamps are epoch milliseconds, so
`queued_ms` and `processing_ms` are plain subtractions. Every query returns plain dicts —
no ORM, no model layer. Open one connection per thread; `connect()` is cheap, and the
request threads and the single worker thread each hold their own. And `finished_seq` is
assigned inside the completion transaction as `MAX(finished_seq)+1` scoped to the batch:
only the worker assigns it, so it is gapless and monotonic, which is what makes the
results-polling cursor exact (§7.2).
"""

import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  id TEXT PRIMARY KEY, created_at INTEGER NOT NULL,
  expected_count INTEGER, closed INTEGER NOT NULL DEFAULT 0,
  records_source TEXT, label TEXT);

CREATE TABLE IF NOT EXISTS images (
  id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  filename TEXT NOT NULL, stored_path TEXT NOT NULL,
  bytes INTEGER, width INTEGER, height INTEGER, exif_normalised INTEGER,
  record_id TEXT, pairing TEXT,              -- exact|stem|token|none|ambiguous
  status TEXT NOT NULL,                      -- queued|running|done|error
  queued_at INTEGER, started_at INTEGER, finished_at INTEGER,
  processing_ms INTEGER, queued_ms INTEGER,
  overall TEXT, low_confidence INTEGER DEFAULT 0,
  result_json TEXT, error TEXT,
  finished_seq INTEGER);
CREATE INDEX IF NOT EXISTS ix_images_cursor ON images(batch_id, finished_seq);
CREATE INDEX IF NOT EXISTS ix_images_queue  ON images(status, queued_at);

CREATE TABLE IF NOT EXISTS records (
  batch_id TEXT NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
  record_id TEXT NOT NULL, brand_name TEXT, class_type TEXT,
  alcohol_content TEXT, net_contents TEXT, filename TEXT,
  PRIMARY KEY (batch_id, record_id));

CREATE TABLE IF NOT EXISTS adjudications (
  image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  check_id TEXT NOT NULL, decision TEXT NOT NULL,   -- confirm|reject
  note TEXT, decided_at INTEGER NOT NULL,
  PRIMARY KEY (image_id, check_id));
"""

# auto_vacuum must be set before the first table exists or the purge's incremental_vacuum
# is a silent no-op (§8.3); the rest are the durability/concurrency pragmas of §7.3.
PRAGMAS = ("auto_vacuum=INCREMENTAL", "journal_mode=WAL", "synchronous=NORMAL",
           "busy_timeout=5000", "foreign_keys=ON")  # fmt: skip
RECORD_COLS = ("brand_name", "class_type", "alcohol_content", "net_contents", "filename")
IMAGE_COLS = ("filename", "stored_path", "bytes", "width", "height", "exif_normalised",
              "record_id", "pairing")  # fmt: skip


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id() -> str:
    return uuid.uuid4().hex


def _all(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def connect(db_path: Path) -> sqlite3.Connection:
    """Open one connection with the pragmas of §7.3 applied."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False so a connection may be handed to the worker thread; each
    # thread still owns exactly one connection.
    conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    for pragma in PRAGMAS:
        conn.execute("PRAGMA " + pragma)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the schema. Idempotent: running it a second time changes nothing."""
    conn.executescript(SCHEMA)
    conn.commit()


def requeue_orphans(conn: sqlite3.Connection) -> int:
    """Crash recovery: images left `running` by a killed process go back in the queue."""
    with conn:
        cur = conn.execute(
            "UPDATE images SET status='queued', started_at=NULL WHERE status='running'"
        )
    return cur.rowcount


def create_batch(conn: sqlite3.Connection, **meta) -> str:
    """Start a batch. Optional meta: expected_count, records_source, label."""
    batch_id = new_id()
    cols = ("expected_count", "records_source", "label")
    with conn:
        conn.execute(
            "INSERT INTO batches (id, created_at, expected_count, records_source, label)"
            " VALUES (?,?,?,?,?)",
            (batch_id, now_ms(), *(meta.get(c) for c in cols)),
        )
    return batch_id


def close_batch(conn: sqlite3.Connection, batch_id: str, expected_count: int | None) -> None:
    """Mark a batch complete so the client knows no further uploads are coming."""
    with conn:
        conn.execute(
            "UPDATE batches SET closed=1, expected_count=COALESCE(?, expected_count) WHERE id=?",
            (expected_count, batch_id),
        )


def upsert_records(conn: sqlite3.Connection, batch_id: str, rows: Iterable[dict]) -> int:
    """Insert or replace the application records for a batch; returns the row count."""
    values = [(batch_id, str(r["record_id"]), *(r.get(c) for c in RECORD_COLS)) for r in rows]
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO records (batch_id, record_id, brand_name, class_type,"
            " alcohol_content, net_contents, filename) VALUES (?,?,?,?,?,?,?)",
            values,
        )
    return len(values)


def add_image(conn: sqlite3.Connection, batch_id: str, **meta) -> str:
    """Queue one ingested image; meta carries the IMAGE_COLS fields.

    `filename` is display metadata only — it is never used to build a path (§8.1).
    """
    image_id = new_id()
    with conn:
        conn.execute(
            f"INSERT INTO images (id, batch_id, {', '.join(IMAGE_COLS)}, status, queued_at)"
            f" VALUES ({', '.join('?' * (len(IMAGE_COLS) + 4))})",
            (image_id, batch_id, *(meta.get(c) for c in IMAGE_COLS), "queued", now_ms()),
        )
    return image_id


def claim_next_queued(conn: sqlite3.Connection) -> dict | None:
    """Atomically take the oldest queued image and mark it running."""
    started = now_ms()
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM images WHERE status='queued' ORDER BY queued_at, rowid LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        waited = started - row["queued_at"]
        conn.execute(
            "UPDATE images SET status='running', started_at=?, queued_ms=? WHERE id=?",
            (started, waited, row["id"]),
        )
    return dict(row) | {"status": "running", "started_at": started, "queued_ms": waited}


def finish_image(
    conn: sqlite3.Connection, image_id: str, *, status: str = "done", overall: str | None = None,
    low_confidence: bool = False, result_json: str | None = None, error: str | None = None,
) -> int:  # fmt: skip
    """Record a completed or failed image; returns its `finished_seq` cursor value."""
    done = now_ms()
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        batch_id, started = conn.execute(
            "SELECT batch_id, started_at FROM images WHERE id=?", (image_id,)
        ).fetchone()
        seq = conn.execute(
            "SELECT COALESCE(MAX(finished_seq), 0) + 1 FROM images WHERE batch_id=?", (batch_id,)
        ).fetchone()[0]
        params = (status, done, done - started if started else None, overall,
                  int(low_confidence), result_json, error, seq, image_id)  # fmt: skip
        conn.execute(
            "UPDATE images SET status=?, finished_at=?, processing_ms=?, overall=?,"
            " low_confidence=?, result_json=?, error=?, finished_seq=? WHERE id=?",
            params,
        )
    return seq


def get_image(conn: sqlite3.Connection, image_id: str) -> dict | None:
    return _one(conn, "SELECT * FROM images WHERE id=?", (image_id,))


def results_since(
    conn: sqlite3.Connection, batch_id: str, since: int = 0, limit: int = 200
) -> list[dict]:
    """Rows that finished after the cursor, oldest first. Never skips, never repeats."""
    sql = "SELECT * FROM images WHERE batch_id=? AND finished_seq > ? ORDER BY finished_seq LIMIT ?"
    return _all(conn, sql, (batch_id, since, limit))


def batch_counts(conn: sqlite3.Connection, batch_id: str) -> dict:
    """The §7.4 counts block: queue state plus the verdict distribution."""
    row = conn.execute(
        "SELECT COUNT(*) AS total, SUM(status='queued') AS queued,"
        " SUM(status='running') AS running, SUM(overall='green') AS green,"
        " SUM(overall='amber') AS amber, SUM(overall='red') AS red,"
        " SUM(status='error') AS error FROM images WHERE batch_id=?",
        (batch_id,),
    ).fetchone()
    return {key: int(row[key] or 0) for key in row.keys()}


def processing_times(conn: sqlite3.Connection, batch_id: str) -> list[int]:
    """Sorted processing times, for the median/p95/max block the footer shows."""
    sql = ("SELECT processing_ms FROM images WHERE batch_id=? AND processing_ms IS NOT NULL"
           " ORDER BY processing_ms")  # fmt: skip
    return [row["processing_ms"] for row in _all(conn, sql, (batch_id,))]


def record_for_image(conn: sqlite3.Connection, image_id: str) -> dict | None:
    """The application record paired with this image, or None when it has no pair."""
    sql = ("SELECT r.* FROM records r JOIN images i ON i.batch_id = r.batch_id"
           " AND i.record_id = r.record_id WHERE i.id=?")  # fmt: skip
    return _one(conn, sql, (image_id,))


def record_adjudication(
    conn: sqlite3.Connection, image_id: str, check_id: str, decision: str, note: str | None = None
) -> None:
    """Store an agent's decision alongside — never over — the machine verdict (§6.8)."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO adjudications (image_id, check_id, decision, note,"
            " decided_at) VALUES (?,?,?,?,?)",
            (image_id, check_id, decision, note, now_ms()),
        )


def adjudications_for_image(conn: sqlite3.Connection, image_id: str) -> list[dict]:
    sql = ("SELECT check_id, decision, note, decided_at FROM adjudications WHERE image_id=?"
           " ORDER BY check_id")  # fmt: skip
    return _all(conn, sql, (image_id,))


def agent_decisions(conn: sqlite3.Connection, image_ids: list[str]) -> dict[str, str]:
    """One decision per image for a whole page of results, in a single statement.

    An image can carry a decision per check; the row-level answer is `reject` if the agent
    rejected anything at all, because a rejection on one finding rejects the label (§6.8).
    """
    if not image_ids:
        return {}
    sql = (f"SELECT image_id, decision FROM adjudications WHERE image_id IN"
           f" ({','.join('?' * len(image_ids))}) ORDER BY check_id")  # fmt: skip
    decided: dict[str, str] = {}
    for row in conn.execute(sql, tuple(image_ids)):
        if decided.get(row["image_id"]) != "reject":
            decided[row["image_id"]] = row["decision"]
    return decided


def expired_batches(conn: sqlite3.Connection, ttl_min: int) -> list[dict]:
    """Batches past the retention window (§8.3)."""
    sql = "SELECT id, created_at FROM batches WHERE created_at <= ? ORDER BY created_at"
    return _all(conn, sql, (now_ms() - ttl_min * 60_000,))


def delete_batch(conn: sqlite3.Connection, batch_id: str) -> None:
    """Delete a batch; images, records and adjudications cascade (foreign_keys=ON)."""
    with conn:
        conn.execute("DELETE FROM batches WHERE id=?", (batch_id,))
