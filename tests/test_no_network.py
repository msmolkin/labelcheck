"""§11.7 — the offline guarantee, and the two engine parameters that are hardcoded.

Marcus's requirement is that nothing leaves the box. `IPAddressDeny=any` in the systemd
unit makes that kernel-enforced in production; these tests make it a property of the code,
so it cannot regress between deployments.

The concrete thing being guarded is `rapidocr/utils/download_file.py`. It fetches a model
when one is missing, and it is the single code path in the whole dependency tree that would
reach the network. It never fires because the models ship inside the wheel — but "never
fires" is a claim, and this file is what turns it into a test.
"""

import ast
import io
import re
import socket
import ssl
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
STATIC_DIR = ROOT / "static"
SAMPLES_DIR = APP_DIR / "samples"

FORBIDDEN_IMPORTS = {"urllib", "urllib3", "requests", "httpx", "aiohttp", "socket", "ftplib"}
# XML namespace URIs are identifiers, never fetched: an SVG's xmlns is not a network call.
NAMESPACE_URI = re.compile(r"https?://www\.w3\.org/\S*")


@pytest.mark.slow
def test_the_whole_pipeline_runs_with_the_network_sabotaged(monkeypatch, settings):
    """§11.7.57 — every socket entry point raises, and the tool still works end to end.

    Constructing the engine, analysing three labels, importing records and exporting the
    results all have to succeed with the network physically unavailable to the process.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("the application attempted a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(ssl.SSLContext, "wrap_socket", refuse)

    from app import csvio, ocr, pipeline, verdict

    ocr.warmup()  # builds the engine: the download path would fire here if it ever did

    images = sorted(SAMPLES_DIR.glob("label_*.jpg"))[:3]
    assert len(images) == 3, "app/samples/ should ship the golden images"
    records = {row["record_id"]: row for row in csvio.read_records(SAMPLES_DIR / "records.csv")[0]}

    rows = []
    for path in images:
        result = pipeline.analyse(path, settings)
        decision = verdict.evaluate(result, records.get(path.stem), settings)
        assert decision.overall in ("green", "amber", "red")
        rows.append(({"filename": path.name, "record_id": path.stem}, decision))

    out = io.StringIO()
    csvio.write_results(rows, out)
    assert len(out.getvalue().strip().splitlines()) == len(rows) + 1


def test_no_module_under_app_can_open_a_connection():
    """§11.7.58 — an AST scan of `app/`, plus a scan of `static/` for remote assets.

    Adding `import requests` anywhere in the application fails this test, which is the
    point: the offline property has to be defended at review time, not at incident time.
    """
    offenders = []
    for path in sorted(APP_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {name}")
    assert not offenders, "the application must not be able to open a connection: " + "; ".join(
        offenders
    )

    remote = []
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.is_dir() or path.suffix in (".png", ".jpg", ".ico", ".woff2"):
            continue
        text = NAMESPACE_URI.sub("", path.read_text(encoding="utf-8", errors="replace"))
        for match in re.findall(r"https?://\S+", text):
            remote.append(f"{path.relative_to(ROOT)}: {match}")
    assert not remote, "the frontend must fetch nothing from a third-party origin: " + "; ".join(
        remote
    )


def test_the_engine_parameters_that_must_not_move_are_hardcoded():
    """§11 acceptance 5 — `Cls.cls_thresh` is 0.99 in the source and is not a knob.

    At the stock 0.9 threshold the angle classifier 180-flips horizontal lines on pristine
    renders and returns garbage (`spike/env_check.py`). Nobody may "tune" it from the
    environment, so it is asserted in the shipped source and absent from `Settings`.
    """
    source = (APP_DIR / "ocr.py").read_text(encoding="utf-8")
    assert re.search(r"""["']Cls\.cls_thresh["']\s*:\s*0\.99""", source), (
        "app/ocr.py must construct the engine with Cls.cls_thresh = 0.99 (§2.1)"
    )
    assert re.search(r"""["']Det\.limit_type["']\s*:\s*["']max["']""", source)

    from app.config import Settings

    for field in Settings.__dataclass_fields__:
        assert "cls_thresh" not in field, "the classifier threshold is not a configuration knob"
