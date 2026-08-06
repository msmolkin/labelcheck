"""§11.6 — the HTTP surface: what it accepts, what it refuses, and what it promises.

No OCR runs here: `pipeline.analyse` is replaced with a canned result so the queue, the
cursor and the export can be exercised in milliseconds. Everything else — ingest, the
security headers, the rate limiter, the CSV paths — is the shipped code.

The client is `tests/asgi.py`, which drives the application through the ASGI interface and
opens no socket.
"""

import io
import re
import struct
import time
import uuid
import zlib

import pytest
from asgi import ASGIClient
from conftest import make_result
from PIL import Image

from app.config import get_settings

SETTINGS = get_settings()

EXPORT_COLUMNS = (
    "filename,record_id,pairing,machine_verdict,agent_decision,warning_present,warning_text,"
    "warning_case,warning_bold,brand,class_type,alcohol_content,net_contents,processing_ms,"
    "low_confidence,notes"
).split(",")

CANONICAL_CSV = (
    "record_id,brand_name,class_type,alcohol_content,net_contents\n"
    "label_001,Old Tom Distillery,Straight Bourbon Whiskey,45% Alc./Vol. (90 Proof),750 mL\n"
)
ALIASED_CSV = (
    "﻿id,brand,class/type,abv,net contents\r\n"
    "label_001,Old Tom Distillery,Straight Bourbon Whiskey,45% Alc./Vol. (90 Proof),750 mL\r\n"
)


def png_bytes(size=(240, 160), colour=(250, 250, 250)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


def decompression_bomb() -> bytes:
    """A 1x1 PNG whose header claims 20000x20000 — 400 Mpx against an 80 Mpx ceiling."""
    buffer = io.BytesIO()
    Image.new("L", (1, 1)).save(buffer, "PNG")
    data = bytearray(buffer.getvalue())
    struct.pack_into(">II", data, 16, 20_000, 20_000)  # IHDR width, height
    struct.pack_into(">I", data, 29, zlib.crc32(bytes(data[12:29])) & 0xFFFFFFFF)
    return bytes(data)


@pytest.fixture(scope="module")
def api():
    """The running application, with the OCR stack stubbed out."""
    from app import ocr, pipeline

    real_analyse, real_warmup = pipeline.analyse, ocr.warmup
    pipeline.analyse = lambda path, s: make_result()
    ocr.warmup = lambda: None

    from app.api import app as fastapi_app

    client = ASGIClient(fastapi_app)
    client.startup()
    try:
        yield client
    finally:
        client.shutdown()
        pipeline.analyse, ocr.warmup = real_analyse, real_warmup


@pytest.fixture
def client(api, request):
    """A fresh client address per test, so one test's requests cannot rate-limit another."""
    index = abs(hash(request.node.name)) % 250 + 1
    return api.with_ip(f"10.0.0.{index}")


def new_batch(client, records: bytes | None = None) -> str:
    files = [("records", "records.csv", "text/csv", records)] if records else []
    response = client.post_form("/api/batches", fields={"label": "test"}, files=files)
    assert response.status == 200, response.text
    return response.json()["batch_id"]


def upload(client, batch_id: str, filename="label_001.png", data=None, content_type="image/png"):
    return client.post_form(
        f"/api/batches/{batch_id}/images",
        files=[("file", filename, content_type, data if data is not None else png_bytes())],
    )


def wait_for_results(client, batch_id: str, expected: int, timeout: float = 20.0) -> list[dict]:
    """Poll the cursor the way the frontend does, collecting rows exactly once."""
    collected, cursor, deadline = [], 0, time.monotonic() + timeout
    while len(collected) < expected and time.monotonic() < deadline:
        payload = client.get(f"/api/batches/{batch_id}/results", since=cursor).json()
        collected.extend(payload["results"])
        cursor = payload["cursor"]
        if len(collected) < expected:
            time.sleep(0.05)
    return collected


def test_healthz_reports_readiness_and_the_tuned_knobs(client):
    """§11.6.47 — the knob echo is how the docs prove the box was benchmarked, not assumed."""
    response = client.get("/healthz")
    assert response.status == 200
    body = response.json()
    for key in (
        "status",
        "engine_ready",
        "version",
        "rapidocr",
        "onnxruntime",
        "opencv",
        "queue_depth",
        "active_batches",
        "uptime_s",
        "budget_ms",
        "pass1_long_side",
        "pass2_long_side",
    ):
        assert key in body, f"/healthz is missing {key}"
    assert body["pass1_long_side"] == SETTINGS.pass1_long_side
    assert body["budget_ms"] == SETTINGS.budget_ms
    assert isinstance(body["engine_ready"], bool)


def test_an_oversized_upload_is_refused(client):
    """§11.6.48 — the first rung of the §8.1 ladder."""
    batch_id = new_batch(client)
    response = upload(client, batch_id, data=b"\x89PNG\r\n\x1a\n" + b"0" * (13 * 1024 * 1024))
    assert response.status == 413
    assert re.search(r"\d+\s*(MB|MiB)", response.text), (
        f"the message should tell the user the limit in human units: {response.text}"
    )


def test_a_text_file_renamed_as_an_image_is_refused_in_plain_english(client):
    """§11.6.49 — magic-byte sniffing, and an error a compliance agent can act on."""
    batch_id = new_batch(client)
    response = upload(client, batch_id, "notes.jpg", b"this is not an image", "image/jpeg")
    assert response.status == 415
    message = response.json()["error"].lower()
    assert "jpeg" in message or "png" in message, message
    assert "unsupported media type" not in message, "status-code names are not error messages"


def test_a_decompression_bomb_is_refused_without_decoding_it(client):
    """§11.6.50 — 400 Mpx declared in the header, 80 Mpx allowed. Nothing is decoded."""
    batch_id = new_batch(client)
    response = upload(client, batch_id, "bomb.png", decompression_bomb())
    assert response.status == 415
    assert response.json()["error"]


def test_a_hostile_filename_never_becomes_a_path(client):
    """§11.6.51 — storage is uploads/<batch>/<uuid>.jpg; the name is display data only."""
    from app.ingest import batch_dir

    batch_id = new_batch(client)
    hostile = "../../etc/passwd<script>alert(1)</script>.png"
    body = upload(client, batch_id, hostile).json()
    uuid.UUID(body["image_id"])
    shown = body["filename"]
    assert "/" not in shown and ".." not in shown, f"a display name may not carry a path: {shown}"
    assert shown, "an upload always keeps a display name"

    stored = list(batch_dir(SETTINGS, batch_id).iterdir())
    assert stored, "the image should have been written"
    for path in stored:
        uuid.UUID(path.stem)
        assert path.suffix == ".jpg"


def test_the_rate_limiter_refuses_a_flood_with_retry_after(client):
    """§11.6.52 — in-process, per IP (§7.5). No shared nginx config is touched."""
    limit = SETTINGS.rate_per_min
    statuses = [client.get("/api/statute").status for _ in range(limit + 25)]
    assert 429 in statuses, f"{limit + 25} requests in a minute should have been throttled"
    throttled = client.get("/api/statute")
    assert throttled.status == 429
    assert int(throttled.headers["retry-after"]) >= 0


def test_rate_limit_key_ignores_the_spoofable_first_forwarded_entry(client, monkeypatch):
    """Adversarial-review blocker: nginx APPENDS its peer to X-Forwarded-For, so the first
    entry is attacker-controlled — keying the bucket on it lets every request mint a fresh
    identity. The key must come from the last (nginx-written) entry."""
    from app import api
    from app.ratelimit import RateLimiter

    monkeypatch.setattr(api, "LIMITER", RateLimiter(6, 600, burst=2))
    statuses = [
        client.get("/api/statute", headers={"x-forwarded-for": f"spoof-{i}, 10.0.0.9"}).status
        for i in range(6)
    ]
    assert 429 in statuses, f"spoofed first entries must all key to one real client: {statuses}"


def test_uploads_ride_their_own_bucket_past_a_throttled_general_lane(client, monkeypatch):
    """§7.5 (integration amendment) — a 300-image batch must never 429 itself.

    Uploads use a dedicated per-IP bucket, so exhausting the general bucket cannot block
    the headline workflow; and the general lane stays throttled while uploads proceed.
    """
    from app import api
    from app.ratelimit import RateLimiter

    monkeypatch.setattr(api, "LIMITER", RateLimiter(6, 600, burst=3))
    monkeypatch.setattr(
        api,
        "UPLOAD_LIMITER",
        RateLimiter(SETTINGS.upload_per_min, SETTINGS.rate_per_hour, SETTINGS.upload_burst),
    )
    batch_id = new_batch(client)
    statuses = [client.get("/api/statute").status for _ in range(8)]
    assert 429 in statuses, "the general bucket should throttle first"
    results = [upload(client, batch_id, f"img_{i}.png").status for i in range(10)]
    assert all(s < 400 for s in results), f"uploads must not ride the general lane: {results}"


def test_the_polling_cursor_returns_every_row_exactly_once(client, monkeypatch):
    """§11.6.53 — resumability is the reason §7.2 chose polling over SSE.

    Twenty images are uploaded while the poll runs; the cursor must produce a gapless,
    duplicate-free sequence, because that is what makes a mid-batch page reload harmless.

    The per-IP bucket is lifted for this test only: its burst is 20, so twenty uploads plus
    the batch creation would be throttled here for a reason that has nothing to do with the
    cursor. The limiter has its own test above.
    """
    from app import api
    from app.ratelimit import RateLimiter

    monkeypatch.setattr(api, "LIMITER", RateLimiter(1_000_000, 1_000_000, burst=1_000_000))
    batch_id = new_batch(client)
    for index in range(20):
        assert upload(client, batch_id, f"label_{index:03d}.png").status == 200
    client.post_json(f"/api/batches/{batch_id}/close", {"expected_count": 20})

    rows = wait_for_results(client, batch_id, 20)
    assert len(rows) == 20, f"expected 20 results, saw {len(rows)}"
    assert len({row["image_id"] for row in rows}) == 20, "the cursor repeated a row"

    payload = client.get(f"/api/batches/{batch_id}/results", since=0).json()
    assert payload["done"] is True
    assert payload["counts"]["green"] + payload["counts"]["amber"] + payload["counts"]["red"] == 20
    assert payload["timing"]["median_ms"] is not None


def test_the_export_carries_both_verdicts_and_the_documented_columns(client):
    """§11.6.54 — §7.6's column list, including `agent_decision` beside `machine_verdict`."""
    batch_id = new_batch(client, CANONICAL_CSV.encode())
    image_id = upload(client, batch_id).json()["image_id"]
    wait_for_results(client, batch_id, 1)
    client.post_json(
        f"/api/images/{image_id}/adjudicate", {"check_id": "warning_bold", "decision": "confirm"}
    )

    response = client.get(f"/api/batches/{batch_id}/export.csv")
    assert response.status == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers.get("content-disposition", "")
    lines = response.text.strip().splitlines()
    assert lines[0].strip().split(",") == EXPORT_COLUMNS
    assert len(lines) == 2
    row = dict(zip(EXPORT_COLUMNS, lines[1].split(","), strict=False))
    assert row["filename"] == "label_001.png"
    assert row["record_id"] == "label_001"
    assert row["machine_verdict"] in ("green", "amber", "red")
    assert row["agent_decision"], "the agent's decision belongs beside the machine's"


def test_records_import_tolerates_the_shapes_a_real_export_arrives_in(client):
    """§11.6.55 — alias headers, a UTF-8 BOM and CRLF endings all parse to the same rows."""
    canonical = client.post_form(
        "/api/batches",
        fields={"label": "canonical"},
        files=[("records", "records.csv", "text/csv", CANONICAL_CSV.encode())],
    ).json()
    aliased = client.post_form(
        "/api/batches",
        fields={"label": "aliased"},
        files=[("records", "records.csv", "text/csv", ALIASED_CSV.encode())],
    ).json()
    assert canonical["records_imported"] == aliased["records_imported"] == 1
    assert canonical["record_ids"] == aliased["record_ids"] == ["label_001"]

    unknown = client.post_form(
        "/api/batches",
        fields={"label": "unknown"},
        files=[("records", "records.csv", "text/csv", b"foo,bar\n1,2\n")],
    )
    assert unknown.status == 400
    assert "brand_name" in unknown.text, "the error must name the expected columns"


def test_the_security_headers_are_on_every_response(client):
    """§11.6.56 — the CSP is the one-line proof of "no CDN, no analytics, no phone-home".

    A reviewer can verify Marcus's requirement with `curl -I`, which is why it is asserted
    verbatim rather than by substring.
    """
    expected_csp = (
        "default-src 'none'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )
    for path in ("/", "/api/statute", "/healthz"):
        response = client.get(path)
        assert response.status == 200, path
        assert response.headers["content-security-policy"] == expected_csp, path
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "geolocation=()" in response.headers["permissions-policy"]
    assert "set-cookie" not in {k.lower() for k in client.get("/").headers}
