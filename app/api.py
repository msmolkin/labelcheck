"""The HTTP surface: routes, security headers, static files (§7.4).

No cookies, no sessions, no CORS — there is no cross-origin surface to get wrong. The
response headers below, above all the Content-Security-Policy, are a deliverable rather
than a detail: they are how a reviewer verifies "no CDN, no analytics, no phone-home"
with one `curl -I`. Errors are written for a compliance agent, not a developer.
"""

import csv
import io
import json
import logging
import math
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from . import __version__, db, ingest, statute
from .config import get_settings
from .ingest import IngestError, batch_dir, crops_dir, db_path
from .purge import purge
from .queue import Worker, pending_count
from .ratelimit import RateLimiter

log = logging.getLogger("labelcheck.api")

# Verbatim from §7.4. `default-src 'none'` forces every resource class to be named, so
# nothing can be pulled from a third-party origin by accident.
CSP = (
    "default-src 'none'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=()",
}
GONE = "That is no longer available. Uploaded images and results are deleted every hour."
PINNED = ("rapidocr", "onnxruntime", "opencv-python-headless")  # echoed by /healthz
TEMPLATE_CSV_NAME = "labelcheck-records-template.csv"

APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "samples"
STATIC_DIR = APP_DIR.parent / "static"
ROOT_FILES = {"favicon.svg", "favicon-32.png", "apple-touch-icon.png", "og.png", "robots.txt"}

SETTINGS = get_settings()
LIMITER = RateLimiter(SETTINGS.rate_per_min, SETTINGS.rate_per_hour, SETTINGS.rate_burst)
# Uploads get their own per-IP bucket (§7.5). The general one is sized for a page load, and
# the documented workflow is 300 images in one go: on the shared bucket the batch throttles
# itself after 20 files. Same hourly ceiling, its own minute dimension.
UPLOAD_LIMITER = RateLimiter(SETTINGS.upload_per_min, SETTINGS.rate_per_hour, SETTINGS.upload_burst)
# Batch creation gets a small bucket of its own: with only the general bucket (burst 20),
# one anonymous client could open the global MAX_ACTIVE_BATCHES in a single breath and
# lock everyone else out of the tool for the purge TTL (adversarial-review finding).
BATCH_LIMITER = RateLimiter(10, 40, burst=6)
WORKER = Worker(SETTINGS)
STARTED_AT = time.monotonic()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    conn = db.connect(db_path(SETTINGS))
    db.init_schema(conn)
    conn.close()
    WORKER.start()  # warms the OCR engine and re-queues whatever a crash left running
    yield
    WORKER.stop()


# The interactive docs are off: a second, unreviewed surface whose assets the
# Content-Security-Policy would block anyway.
app = FastAPI(
    title="labelcheck", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
)  # noqa: E501


def _installed(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def get_conn():
    """One SQLite connection per request, closed on the way out."""
    conn = db.connect(db_path(SETTINGS))
    try:
        yield conn
    finally:
        conn.close()


# Annotated aliases rather than call-valued defaults, so route signatures stay ordinary
# Python. The three JSON request bodies are the small models below.
Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Upload = Annotated[UploadFile, File()]


class CloseBody(BaseModel):
    expected_count: int | None = None


class AdjudicateBody(BaseModel):
    check_id: str
    decision: str
    note: str | None = None


class SampleBody(BaseModel):
    size: int = 8


def fail(status_code: int, message: str) -> HTTPException:
    """Every error the user can see is a sentence, not a status-code name."""
    return HTTPException(status_code=status_code, detail=message)


# Exact route shapes, not suffixes: `endswith("/results")` would exempt any invented
# /api/<anything>/results path from metering (adversarial-review finding). Batch ids are
# uuid4().hex, so the shape is pinned to 32 hex characters.
_POLL_PATH = re.compile(r"^/api/batches/[0-9a-f]{32}/results$")
_UPLOAD_PATH = re.compile(r"^/api/batches/[0-9a-f]{32}/images$")


def _is_poll(request: Request) -> bool:
    """The results cursor is exempt from the rate limit, and only it: §7.2 polls twice a
    second for the quarter of an hour a 300-image batch runs, which would exhaust §7.5's
    90-a-minute bucket in seconds and 429 the page against itself. The bucket exists to
    protect the single OCR worker and the disk, and an indexed `finished_seq > ?` scan
    costs neither. Uploads, batch creation and the sample trigger all stay limited."""
    return request.method in ("GET", "HEAD") and bool(_POLL_PATH.match(request.url.path))


def _is_upload(request: Request) -> bool:
    """The image-upload route, which is metered on its own bucket. Uploads are the only
    request the user makes hundreds of in a row, and they are the one request whose rate the
    single OCR worker already bounds downstream — the queue absorbs them, the disk quota caps
    them, and 429-ing a file the agent selected loses work. The statute endpoint and the demo
    trigger stay on the general bucket."""
    return request.method == "POST" and bool(_UPLOAD_PATH.match(request.url.path))


@app.middleware("http")
async def guard(request: Request, call_next) -> Response:
    """Rate-limit the API and stamp the security headers onto every response.

    `X-Forwarded-For` handling: nginx *appends* its peer's address to whatever the client
    sent, so the FIRST entry is attacker-controlled — keying the bucket on it would let
    every request mint a fresh identity (adversarial-review blocker). The LAST entry is
    the one nginx itself wrote, and the unit only accepts localhost connections, so nginx
    is the only peer that can reach us at all."""
    if request.scope["method"] == "HEAD":
        # `curl -I` is the documented header check (§7.4) and health check (§10.5), so
        # HEAD routes as GET; the server drops the body and returns headers only.
        request.scope["method"] = "GET"
    # Reject oversized bodies from the header, before the multipart parser spools them
    # to disk (adversarial-review finding). nginx's client_max_body_size is the hard
    # edge; this is the polite early answer for an honest client, sized for the image
    # plus multipart framing.
    if request.method == "POST" and request.url.path.startswith("/api/"):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > SETTINGS.max_upload_bytes + 65536:
            limit_mb = SETTINGS.max_upload_bytes // (1024 * 1024)
            return JSONResponse(
                {"error": f"That upload is larger than the {limit_mb} MB limit."},
                status_code=413,
                headers=dict(SECURITY_HEADERS),
            )
    response = None
    if request.url.path.startswith("/api/") and not _is_poll(request):
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
        client = forwarded or (request.client.host if request.client else "-")
        limiter = LIMITER
        if _is_upload(request):
            limiter = UPLOAD_LIMITER
        elif request.method == "POST" and request.url.path == "/api/batches":
            limiter = BATCH_LIMITER
        wait = limiter.check(client)
        if wait:
            response = JSONResponse(
                {"error": "Too many requests. Please wait a moment and try again."},
                status_code=429,
                headers={"Retry-After": str(math.ceil(wait))},
            )
    response = response or await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    if request.url.path.startswith("/assets/"):
        response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
    return response


@app.exception_handler(IngestError)
async def ingest_error(_request: Request, exc: IngestError) -> JSONResponse:
    return JSONResponse({"error": exc.message}, status_code=exc.status_code)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> Response:
    page = STATIC_DIR / "404.html"
    if exc.status_code == 404 and not request.url.path.startswith("/api/") and page.is_file():
        return FileResponse(page, status_code=404)
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.get("/")
def index() -> Response:
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page)
    return PlainTextResponse("labelcheck is running. The frontend has not been built yet.\n")


@app.get("/healthz")
def healthz(conn: Conn) -> dict:
    """Liveness plus the tuned knobs. That echo is how the docs prove the deploy box was
    benchmarked rather than assumed (§12)."""
    installed = {p.split("-")[0]: _installed(p) for p in PINNED}
    return {
        "status": "ok", "engine_ready": WORKER.engine_ready, "version": __version__,
        "queue_depth": pending_count(conn),
        "active_batches": conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0],
        "uptime_s": round(time.monotonic() - STARTED_AT, 1),
        "budget_ms": SETTINGS.budget_ms, "pass1_long_side": SETTINGS.pass1_long_side,
        "pass2_long_side": SETTINGS.pass2_long_side, **installed,
    }  # fmt: skip


@app.get("/api/statute")
def get_statute() -> dict:
    """The statutory text the tool compares against, so the UI never retypes it."""
    rules = [{"citation": c, "requirement": r, "checked": v} for c, r, v in statute.FORMAT_RULES]
    return {
        "text": statute.STATUTE_TEXT, "sha256": statute.STATUTE_SHA256,
        "clause_1": statute.CLAUSE_1, "clause_2": statute.CLAUSE_2, "prefix": statute.PREFIX,
        "tokens": list(statute.STATUTE_TOKENS), "format_rules": rules,
        "citations": statute.CITATIONS,
    }  # fmt: skip


@app.get("/api/records-template.csv")
def records_template() -> Response:
    """A blank records CSV with the canonical header and two filled example rows. The header
    is printed on the page as well, but an agent building the file in Excel should not have to
    retype it — and a downloaded template cannot acquire a typo the import then refuses."""
    from . import csvio

    disposition = f'attachment; filename="{TEMPLATE_CSV_NAME}"'
    return Response(csvio.template_csv(), media_type="text/csv",
                    headers={"Content-Disposition": disposition})  # fmt: skip


@app.post("/api/batches")
def create_batch(
    conn: Conn,
    records: Annotated[UploadFile | None, File()] = None,
    label: Annotated[str | None, Form()] = None,
) -> dict:
    """Start a batch, optionally importing the application records that go with it.

    A plain `def`: FastAPI runs it on the threadpool, so the CSV parse and the lazy purge
    never block the event loop (adversarial-review finding — an `async def` doing sync
    work stalls every concurrent request, including /healthz)."""
    try:
        purge(SETTINGS)  # lazy retention sweep: a stopped timer cannot retain uploads
    except Exception:
        log.exception("lazy purge failed; continuing")
    if (
        conn.execute("SELECT COUNT(*) FROM batches WHERE closed=0").fetchone()[0]
        >= SETTINGS.max_active_batches
    ):  # noqa: E501
        raise fail(429, "Too many batches are open right now. Please try again in a few minutes.")

    rows, warnings = [], []
    if records is not None:
        from . import csvio  # owns the alias / BOM / CRLF tolerance of §7.6

        data = records.file.read(SETTINGS.max_csv_bytes + 1)
        if len(data) > SETTINGS.max_csv_bytes:
            raise fail(413, f"The records file is larger than {SETTINGS.max_csv_bytes // 1024} KB.")
        try:
            rows, warnings = csvio.read_records(data)
        except Exception as exc:
            raise fail(400, str(exc)) from None
    source = records.filename if records else None
    batch_id = db.create_batch(conn, label=label, records_source=source)
    if rows:
        db.upsert_records(conn, batch_id, rows)
    ids = [r["record_id"] for r in rows]
    return {"batch_id": batch_id, "records_imported": len(rows), "record_ids": ids,
            "warnings": warnings}  # fmt: skip


@app.post("/api/batches/{batch_id}/images")
def upload_image(batch_id: str, conn: Conn, file: Upload) -> dict:
    """Validate, re-encode and queue one image. The client uploads files one at a time.

    A plain `def` for the same reason as `create_batch`: the decode + LANCZOS resize +
    JPEG re-encode are pure CPU, and on the threadpool they cannot freeze the event loop
    for every polling client mid-batch."""
    _batch_or_404(conn, batch_id)
    if db.batch_counts(conn, batch_id)["total"] >= SETTINGS.max_batch_images:
        raise fail(413, f"A batch holds at most {SETTINGS.max_batch_images} images.")
    data = file.file.read(SETTINGS.max_upload_bytes + 1)
    stored = ingest.store_upload(data, file.filename or "", batch_dir(SETTINGS, batch_id), SETTINGS)

    records = [dict(r) for r in conn.execute("SELECT * FROM records WHERE batch_id=?", (batch_id,))]
    record_id, pairing = ingest.pair_record(stored["filename"], records)
    taken = "SELECT COUNT(*) FROM images WHERE batch_id=? AND record_id=?"
    if record_id and conn.execute(taken, (batch_id, record_id)).fetchone()[0]:
        # A record matching two images cannot be trusted for either of them (§7.7).
        with conn:
            conn.execute("UPDATE images SET pairing='ambiguous' WHERE batch_id=? AND record_id=?",
                         (batch_id, record_id))  # fmt: skip
        pairing = "ambiguous"
    image_id = db.add_image(conn, batch_id, record_id=record_id, pairing=pairing, **stored)
    WORKER.submit()
    return {"image_id": image_id, "filename": stored["filename"], "record_id": record_id,
            "pairing": pairing, "queue_position": pending_count(conn)}  # fmt: skip


@app.post("/api/batches/{batch_id}/close")
def close_batch(batch_id: str, body: CloseBody, conn: Conn) -> dict:
    _batch_or_404(conn, batch_id)
    db.close_batch(conn, batch_id, body.expected_count)
    return {"ok": True}


@app.get("/api/batches/{batch_id}/results")
def batch_results(batch_id: str, conn: Conn, since: int = 0, limit: int = 200) -> dict:
    """Rows that finished after the cursor (§7.2). Never skips a row, never repeats one."""
    batch = _batch_or_404(conn, batch_id)
    rows = db.results_since(conn, batch_id, since, max(1, min(limit, 500)))
    counts = db.batch_counts(conn, batch_id)
    times = db.processing_times(conn, batch_id)
    timing = {"median_ms": None, "p95_ms": None, "max_ms": None}
    if times:
        timing = {"median_ms": times[len(times) // 2], "max_ms": times[-1],
                  "p95_ms": times[max(0, math.ceil(0.95 * len(times)) - 1)]}  # fmt: skip
    # One query for the page, not one per row: this endpoint is polled twice a second for the
    # length of a 300-image batch, so an N+1 here is 600 needless queries a second.
    decided = db.agent_decisions(conn, [row["id"] for row in rows])
    cursor = rows[-1]["finished_seq"] if rows else since
    done = bool(batch["closed"]) and counts["queued"] == 0 and counts["running"] == 0
    if done:
        # The page and the counts are separate reads: an image finishing between them
        # would let done=true ship alongside a page that misses its row, and the client
        # stops polling (adversarial-review finding). Done therefore also requires that
        # every finished row is at or before the cursor being returned.
        last = conn.execute(
            "SELECT COALESCE(MAX(finished_seq), 0) FROM images WHERE batch_id=?", (batch_id,)
        ).fetchone()[0]
        done = last <= cursor
    return {
        "cursor": cursor, "done": done, "counts": counts, "timing": timing,
        "results": [_merge(row, decided.get(row["id"])) for row in rows],
    }  # fmt: skip


@app.get("/api/images/{image_id}")
def get_image(image_id: str, conn: Conn) -> dict:
    row = db.get_image(conn, image_id)
    if row is None:
        raise fail(404, GONE)
    adjudications = db.adjudications_for_image(conn, image_id)
    decided = db.agent_decisions(conn, [image_id]).get(image_id)
    return _merge(row, decided) | {"adjudications": adjudications}


@app.get("/api/images/{image_id}/crop/{kind}")
def get_crop(image_id: str, kind: str, conn: Conn) -> Response:
    """The evidence images: the warning band, the bold measurement, or the whole label."""
    row = db.get_image(conn, image_id)
    if row is None or kind not in ("band", "bold", "full"):
        raise fail(404, GONE)
    crop = crops_dir(SETTINGS) / f"{image_id}-{kind}.jpg"
    path = Path(row["stored_path"]) if kind == "full" else crop
    if not path.is_file():
        raise fail(404, "No cropped evidence was saved for this check.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/images/{image_id}/adjudicate")
def adjudicate(image_id: str, body: AdjudicateBody, conn: Conn) -> dict:
    """Record an agent's decision beside the machine verdict, which never changes (§6.8).

    `check_id` is stored as given and is deliberately not validated against `CHECK_IDS`: the
    review dialog posts the literal `"overall"` for an all-green image, where the agent is
    agreeing with the whole verdict rather than adjudicating one finding. A closed vocabulary
    here would reject exactly the case a reviewer approves most often.
    """
    if db.get_image(conn, image_id) is None:
        raise fail(404, GONE)
    if body.decision not in ("confirm", "reject"):
        raise fail(400, 'A decision must be either "confirm" or "reject".')
    db.record_adjudication(conn, image_id, body.check_id, body.decision, body.note)
    return {"ok": True, "adjudications": db.adjudications_for_image(conn, image_id)}


@app.get("/api/batches/{batch_id}/export.csv")
def export_csv(batch_id: str, conn: Conn) -> Response:
    """Both columns, machine and agent: overwriting a finding destroys the trail (§6.8).

    The verdict was stored as JSON when the image finished, so it is rebuilt into the
    dataclasses `csvio` writes from — the database is the boundary, not the object graph.
    """
    from . import csvio
    from .verdict import Check, Verdict

    _batch_or_404(conn, batch_id)
    rows = db.results_since(conn, batch_id, 0, SETTINGS.max_batch_images)
    decided = db.agent_decisions(conn, [row["id"] for row in rows])
    items = []
    for row in rows:
        meta = _merge(row, decided.get(row["id"]))
        verdict = Verdict(
            overall=meta["overall"] or "error",
            checks=[Check(**check) for check in meta.get("checks", [])],
            informational=meta.get("informational", []),
        )
        items.append((meta, verdict))
    out = io.StringIO()
    csvio.write_results(items, out)
    disposition = f'attachment; filename="labelcheck-{batch_id[:8]}.csv"'
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": disposition})  # fmt: skip


@app.post("/api/demo/sample-batch")
def sample_batch(body: SampleBody, conn: Conn) -> dict:
    """Load the bundled samples through the ordinary ingest path, so the demo measures the
    same work a real upload does (§8.4). `size` is 8 for the quick set, 22 for all of them.
    The bundled records.csv is read with the stdlib reader, not `csvio`: it ships with this
    app under the canonical header, so there are no aliases to tolerate."""
    if body.size not in (8, 22):
        raise fail(400, "The sample batch comes in two sizes: 8 (quick set) or 22 (all).")
    manifest = json.loads((SAMPLES_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    files = [i["file"] for i in manifest["images"] if body.size == 22 or i["quick_set"]]
    with (SAMPLES_DIR / "records.csv").open(newline="", encoding="utf-8") as handle:
        records = [r for r in csv.DictReader(handle) if f"{r['record_id']}.jpg" in files]

    batch_id = db.create_batch(conn, label=f"Bundled samples ({len(files)})",
                               records_source="app/samples/records.csv")  # fmt: skip
    db.upsert_records(conn, batch_id, records)
    for name in files:
        data = (SAMPLES_DIR / name).read_bytes()
        stored = ingest.store_upload(data, name, batch_dir(SETTINGS, batch_id), SETTINGS)
        record_id, pairing = ingest.pair_record(name, records)
        db.add_image(conn, batch_id, record_id=record_id, pairing=pairing, **stored)
        WORKER.submit()
    db.close_batch(conn, batch_id, len(files))
    return {"batch_id": batch_id, "images": len(files)}


def _batch_or_404(conn: sqlite3.Connection, batch_id: str) -> dict:
    row = conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
    if row is None:
        raise fail(404, GONE)
    return dict(row)


def _merge(row: dict, agent_decision: str | None = None) -> dict:
    """One §7.4 `ImageResult`: the stored analysis plus the live columns from the row.

    `error` and `agent_decision` are always present, null when there is nothing to report:
    the frontend reads both on every row, and a key that appears only sometimes is a key a
    client gets wrong. The decision is passed in rather than queried here so the callers can
    fetch a whole page of them in one statement.
    """
    stored = json.loads(row["result_json"]) if row["result_json"] else {}
    return stored | {
        "image_id": row["id"], "filename": row["filename"], "record_id": row["record_id"],
        "pairing": row["pairing"], "status": row["status"], "overall": row["overall"],
        "low_confidence": bool(row["low_confidence"]), "processing_ms": row["processing_ms"],
        "queued_ms": row["queued_ms"], "error": row["error"],
        "agent_decision": agent_decision,
    }  # fmt: skip


if (STATIC_DIR / "dist").is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "dist"), name="assets")


@app.get("/{name}")
def root_file(name: str) -> Response:
    # The handful of files §9.3 requires at the site root, served from disk, never a CDN.
    if name not in ROOT_FILES or not (STATIC_DIR / name).is_file():
        raise fail(404, "Not found.")
    return FileResponse(STATIC_DIR / name)
