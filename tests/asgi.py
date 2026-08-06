"""A minimal ASGI client, so the API tests need no HTTP library.

Starlette's own `TestClient` requires `httpx`, which is not in the runtime closure and has
no business being installed on a box whose whole security story is "this process cannot
open a socket" (§10.2). Driving the application through the ASGI interface directly is
about eighty lines, adds no dependency, and opens no socket at all.

It speaks enough of the protocol for these tests: the lifespan handshake (so start-up runs
for real), one request/response exchange with a complete body, and multipart uploads.
"""

from __future__ import annotations

import asyncio
import json as jsonlib
import uuid
from dataclasses import dataclass
from urllib.parse import urlencode


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self):
        return jsonlib.loads(self.body)


class ASGIClient:
    """One event loop, one application, one simulated client address."""

    def __init__(self, app, client_ip: str = "127.0.0.1", loop=None):
        self.app = app
        self.client_ip = client_ip
        self.loop = loop or asyncio.new_event_loop()
        self._lifespan = None

    def with_ip(self, client_ip: str) -> ASGIClient:
        """A view of the same running application from a different address."""
        return ASGIClient(self.app, client_ip, self.loop)

    # -- lifespan ---------------------------------------------------------
    def startup(self) -> None:
        async def run():
            self._to_app = asyncio.Queue()
            self._from_app = asyncio.Queue()
            scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
            self._lifespan = asyncio.ensure_future(
                self.app(scope, self._to_app.get, self._from_app.put)
            )
            await self._to_app.put({"type": "lifespan.startup"})
            message = await self._from_app.get()
            assert message["type"] == "lifespan.startup.complete", message

        self.loop.run_until_complete(run())

    def shutdown(self) -> None:
        async def run():
            await self._to_app.put({"type": "lifespan.shutdown"})
            await self._from_app.get()
            await self._lifespan

        self.loop.run_until_complete(run())
        self.loop.close()

    # -- requests ---------------------------------------------------------
    def request(self, method: str, path: str, *, headers=None, body: bytes = b"") -> Response:
        path, _, query = path.partition("?")
        sent = []
        header_pairs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
        if body:
            header_pairs.append((b"content-length", str(len(body)).encode()))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode(),
            "root_path": "",
            "headers": header_pairs,
            "client": (self.client_ip, 54321),
            "server": ("labelcheck.test", 80),
        }

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            sent.append(message)

        self.loop.run_until_complete(self.app(scope, receive, send))
        start = next(m for m in sent if m["type"] == "http.response.start")
        payload = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        return Response(
            status=start["status"],
            headers={k.decode(): v.decode() for k, v in start["headers"]},
            body=payload,
        )

    def get(self, path: str, **params) -> Response:
        if params:
            path = f"{path}?{urlencode(params)}"
        return self.request("GET", path)

    def post_json(self, path: str, payload) -> Response:
        return self.request(
            "POST",
            path,
            headers={"content-type": "application/json"},
            body=jsonlib.dumps(payload).encode(),
        )

    def post_form(self, path: str, *, fields=None, files=None) -> Response:
        """`files` is [(field, filename, content_type, bytes)]."""
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in (fields or {}).items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
                + str(value).encode()
                + b"\r\n"
            )
        for name, filename, content_type, data in files or []:
            head = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            )
            parts.append(head.encode() + data + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return self.request(
            "POST",
            path,
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
            body=b"".join(parts),
        )
