"""Per-IP token bucket, held in this process only (§7.5).

Two buckets per client: a per-minute bucket that allows a short burst (a page load fires
several requests at once, and the frontend uploads three files in parallel) and a
per-hour bucket that bounds sustained use. A request needs a token in both.

This is deliberately not nginx `limit_req`: `limit_req_zone` has to be declared in the
`http{}` block of the shared `nginx.conf`, which also serves 78 unrelated apps on the
same box. Editing that file to rate-limit a prototype is not a trade this project makes.

The client address is salted and hashed before it is used as a key, with a random salt
per process, so nothing here can be turned back into an address. Nothing is logged.
"""

import hashlib
import os
import threading
import time

from .config import get_settings

_IDLE_DROP_S = 3600.0  # refilled buckets this old are forgotten, bounding memory


class RateLimiter:
    """Token buckets keyed by a salted hash of the client address."""

    def __init__(self, per_min: int, per_hour: int, burst: int | None = None) -> None:
        # The burst is a `Settings` field (`rate_burst`), so a deployment retunes it in
        # /etc/labelcheck.env with everything else rather than by editing this module.
        burst = get_settings().rate_burst if burst is None else burst
        self.minute_capacity, self.minute_rate = float(max(burst, 1)), per_min / 60.0
        self.hour_capacity, self.hour_rate = float(max(per_hour, 1)), per_hour / 3600.0
        self._salt = os.urandom(16)
        self._lock = threading.Lock()
        self._buckets: dict[str, list[float]] = {}  # key -> [minute, hour, last_seen]

    def check(self, client: str) -> float:
        """Spend one token. Returns 0.0 when allowed, else the seconds to wait."""
        key = hashlib.blake2b(client.encode(), key=self._salt, digest_size=16).hexdigest()
        now = time.monotonic()
        with self._lock:
            full = (self.minute_capacity, self.hour_capacity, now)
            minute, hour, last = self._buckets.get(key) or full
            minute = min(self.minute_capacity, minute + (now - last) * self.minute_rate)
            hour = min(self.hour_capacity, hour + (now - last) * self.hour_rate)
            if minute < 1.0 or hour < 1.0:
                self._buckets[key] = [minute, hour, now]
                short = (1.0 - minute) / self.minute_rate if minute < 1.0 else 0.0
                long = (1.0 - hour) / self.hour_rate if hour < 1.0 else 0.0
                return max(1.0, short, long)
            self._buckets[key] = [minute - 1.0, hour - 1.0, now]
            if len(self._buckets) > 1024:
                for stale in [k for k, b in self._buckets.items() if now - b[2] > _IDLE_DROP_S]:
                    del self._buckets[stale]
            return 0.0
