#!/usr/bin/env python
"""Print what the tool currently says about the golden set, next to what is expected.

This is a review aid, not a fixture generator: **it never writes `tests/golden.json`.**
Those expectations are derived from `spike/corpus/ground_truth.json` by the mapping
documented at the top of that file, so that the golden test measures the tool against the
labels' ground truth rather than against yesterday's behaviour. A tool that rewrote them
would turn the whole suite into a tautology — the point of this script is that a human
reads the diff and decides.

    .venv/bin/python tools/regen_golden.py            # app/samples/ (the committed images)
    .venv/bin/python tools/regen_golden.py --dir spike/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "tests" / "golden.json"
MARKS = {"ok": "  ", "outside": "!!", "narrower": " ~"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="app/samples", help="directory of label images")
    parser.add_argument("--json", type=Path, help="also write the observations here")
    args = parser.parse_args()

    from app import csvio, pipeline, verdict
    from app.config import get_settings

    settings = get_settings()
    directory = (ROOT / args.dir).resolve()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["labels"]
    expected = {name.rsplit(".", 1)[0]: entry for name, entry in golden.items()}

    rows, _ = csvio.read_records(directory / "records.csv")
    records = {row["record_id"]: row for row in rows}

    observations, misses = [], 0
    print(f"{'label':14s} {'expected':16s} {'observed':8s}  checks")
    print("-" * 96)
    for path in sorted(directory.glob("label_*.*")):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg") or path.stem not in expected:
            continue
        entry = expected[path.stem]
        started = time.perf_counter()
        result = pipeline.analyse(path, settings)
        decision = verdict.evaluate(result, records.get(path.stem), settings)
        elapsed = (time.perf_counter() - started) * 1000.0

        if decision.overall not in entry["allowed"]:
            mark, misses = MARKS["outside"], misses + 1
        elif len(entry["allowed"]) < len(entry["derived_allowed"]):
            mark = MARKS["narrower"]
        else:
            mark = MARKS["ok"]
        states = " ".join(f"{c.id.split('_')[-1]}={c.state}" for c in decision.checks)
        allowed = "|".join(entry["allowed"])
        print(f"{mark}{path.stem:12s} {allowed:16s} {decision.overall:8s}  {states}")
        observations.append(
            {
                "file": path.name,
                "allowed": entry["allowed"],
                "observed": decision.overall,
                "inside_allowed": decision.overall in entry["allowed"],
                "basis": entry["basis"],
                "checks": {c.id: c.state for c in decision.checks},
                "passes": result.passes,
                "ms": round(elapsed, 1),
            }
        )

    print("-" * 96)
    print(f"{len(observations)} labels, {misses} outside their allowed set.")
    print("Expectations were NOT modified. If a change here is correct, edit tests/golden.json")
    print("by hand and say in the commit message why the ground-truth mapping now differs.")
    if args.json:
        args.json.write_text(json.dumps(observations, indent=1) + "\n", encoding="utf-8")
        print(f"observations written to {args.json}")
    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
