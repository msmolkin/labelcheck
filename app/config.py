"""Settings for labelcheck: every threshold in the system, read once from the environment.

Keeping them here puts the whole tuning surface on one screen and lets the on-box benchmark
retune a deployment by writing `/etc/labelcheck.env` instead of editing code. Each knob comes
from `LABELCHECK_<FIELD_NAME_UPPERCASED>`; each default is the value measured during the
design pass (§ marks refer to `internal/ARCHITECTURE.md`). Importing this module also pins
the BLAS/OpenMP thread counts, so import it before `cv2` or `onnxruntime`.
"""

import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

MIB = 1024 * 1024
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
PASS2_MODES = ("crop", "full", "off")
TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """One immutable bundle of every tunable value. The defaults are measured."""

    data_dir: Path = Path(".data")  # /var/lib/labelcheck in production (§10.2)
    port: int = 8712
    ocr_threads: int = 2  # one worker x two ONNX threads on a 2-vCPU box (§7.1)
    pass1_long_side: int = 960  # two-pass OCR ladder (§2.5, §4)
    pass2_long_side: int = 960
    pass2_mode: str = "crop"
    bold_long_side: int = 1600  # bold measures the native frame, not the 960 one (§2.4)
    bold_min_cap_px: int = 11  # under this, stroke ratio carries no signal (§5.2)
    ingest_long_side: int = 2400  # §8.1 re-encode cap: 2.5x headroom for the band crop
    ingest_jpeg_quality: int = 92
    budget_ms: int = 5000  # per-image latency target (§12)
    pass2_reserve_ms: int = 400  # bold (<=151) + pass-2 prep (<=31) + verdict + slack
    esc_prefix_min: float = 75.0  # escalation policy; no confidence term by design (§2.3, §4.2)
    esc_region_min: float = 95.0
    esc_require_caps: bool = True
    text_trust_min: float = 95.0  # under this the tool may not reject at all (§6.2)
    text_region_min: float = 90.0
    case_caps_min: float = 0.90  # uppercase ratio, never string equality (§2.7, §6.3)
    case_title_max: float = 0.40
    bold_t_pass: float = 1.20  # stroke-width ratio bands (§5.2)
    bold_t_fail: float = 1.05
    bold_t_sharp: float = 0.20  # defocus gate, evaluated before the bands
    bold_t_ref: float = 1.20  # body-vs-reference cut choosing the violation subtype (§5.3)
    bold_t_abs: float = 0.0785  # subtype fallback when the label has no reference line
    anchor_min: float = 80.0  # "government warning" fuzzy presence gate (§6.1)
    min_tokens_for_absence: int = 12  # fewer tokens means unreadable photo, not absent warning
    noise_green_max_ratio: float = 0.15  # share of the 43 statute tokens allowed to be noise
    noise_word_min_sim: float = 80.0  # per-word similarity under which noise becomes a defect
    brand_green: float = 92.0  # record-match thresholds (§6.5)
    brand_amber: float = 80.0
    class_green: float = 90.0
    class_amber: float = 75.0
    reading_order_deskew: bool = True  # new and unmeasured; golden run prints the A/B (§14.9)
    max_upload_bytes: int = 12 * MIB  # intake limits (§7.5, §8.1)
    max_batch_images: int = 400
    max_csv_bytes: int = 2 * MIB
    max_disk_bytes: int = 4 * 1024 * MIB
    purge_ttl_min: int = 60  # §8.3: uploads and results are deleted every hour
    rate_per_min: int = 90
    rate_per_hour: int = 900
    rate_burst: int = 20  # §7.5 general bucket: a page load fires several requests at once
    max_active_batches: int = 20  # §7.5 cap on open batches, one agent working at a time
    # Image upload has the shape of the workflow, not of a page load: the documented headline
    # run is 300 files at 3 in flight, which the general bucket would 429 after 20. Its own
    # bucket holds a whole batch and then some, and refills at twice the general rate (§8.1).
    upload_burst: int = 320
    upload_per_min: int = 120

    def __post_init__(self) -> None:
        if self.pass2_mode not in PASS2_MODES:
            raise ValueError(f"pass2_mode must be one of {PASS2_MODES}, got {self.pass2_mode!r}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read `LABELCHECK_*` once, coercing each value to its declared type."""
    overrides = {}
    for field in fields(Settings):
        raw = os.environ.get("LABELCHECK_" + field.name.upper())
        if raw is None or raw.strip() == "":
            continue
        if field.type is bool:
            overrides[field.name] = raw.strip().lower() in TRUTHY
        elif field.type is Path:
            overrides[field.name] = Path(raw).expanduser()
        else:
            overrides[field.name] = field.type(raw)
    return Settings(**overrides)


def apply_thread_env(threads: int | None = None) -> None:
    """Pin OMP/OpenBLAS/MKL threads before OpenCV or ONNX Runtime read them at import.

    Called at module scope for that reason: getting it wrong oversubscribes a 2-core box.
    """
    count = str(threads if threads is not None else get_settings().ocr_threads)
    for var in THREAD_ENV_VARS:
        os.environ[var] = count


apply_thread_env()
