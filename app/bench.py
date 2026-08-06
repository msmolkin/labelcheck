"""Measure this box, then choose the resolution knobs for it. `python -m app.bench`.

Every latency figure in the design was taken on one contended Intel Mac; a deploy box is not
that machine. So the install runs this before the service is enabled: it processes the
bundled samples, prints per-stage p50/p95/max, and steps the pass-1 resolution down until the
measured p95 fits inside 80 % of the budget. The chosen knobs go to `/etc/labelcheck.env` and
are echoed on `/healthz`, which is how the documentation can claim the box was measured
rather than assumed. The ladder only moves in the safe direction on its own — a slower box
gets less resolution and loses a little accuracy; raising resolution on a fast box is
offered, with the accuracy it buys, but never applied automatically.
"""

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings, get_settings
from .ocr import warmup
from .pipeline import TIMING_KEYS, analyse

# The reference machine (ARCHITECTURE §12): contended Intel Mac, 2 ONNX threads. The ratio
# of this box's pass-1 OCR to that figure is the single number that says whether the box is
# 1x, 2x or 3x slower per core.
MAC_REFERENCE_OCR1_MS = 1875.0
MAC_REFERENCE_TOTAL_P95_MS = 4410.0

PASS1_LADDER = (960, 832, 736)  # §12 step 3
PASS2_LADDER = (960, 832, 736)
TARGET_FRACTION = 0.8  # p95 must fit inside this share of the budget
FAST_BOX_P95_MS = 2500  # under this, 1280 px pass 1 is worth offering (§12 step 4)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[k]


def run_once(paths: list[Path], s: Settings, repeats: int) -> dict:
    """Process every image `repeats` times and summarise each pipeline stage."""
    samples: dict[str, list[float]] = {k: [] for k in TIMING_KEYS}
    for _ in range(repeats):
        for path in paths:
            timings = analyse(path, s).timings_ms
            for key in TIMING_KEYS:
                samples[key].append(timings[key])
    return {
        key: {
            "p50": round(statistics.median(v), 1),
            "p95": round(percentile(v, 95), 1),
            "max": round(max(v), 1),
        }
        for key, v in samples.items()
    }


def tune(paths: list[Path], s: Settings, repeats: int, target_ms: float) -> tuple[Settings, list]:
    """Walk the resolution ladder until the measured p95 fits the target."""
    rungs = []
    for pass1 in PASS1_LADDER:
        s = replace(s, pass1_long_side=pass1)
        table = run_once(paths, s, repeats)
        rungs.append({"settings": _knobs(s), "stages": table, "p95_ms": table["total"]["p95"]})
        _print_rung(rungs[-1])
        if table["total"]["p95"] <= target_ms:
            return s, rungs
    for pass2 in PASS2_LADDER[1:]:  # still over budget: shrink the second pass too
        s = replace(s, pass2_long_side=pass2, pass2_mode="crop")
        table = run_once(paths, s, repeats)
        rungs.append({"settings": _knobs(s), "stages": table, "p95_ms": table["total"]["p95"]})
        _print_rung(rungs[-1])
        if table["total"]["p95"] <= target_ms:
            return s, rungs
    return s, rungs


def _knobs(s: Settings) -> dict:
    return {
        "pass1_long_side": s.pass1_long_side,
        "pass2_long_side": s.pass2_long_side,
        "pass2_mode": s.pass2_mode,
        "bold_long_side": s.bold_long_side,
    }


def _print_rung(rung: dict) -> None:
    knobs = rung["settings"]
    print(
        f"\npass1={knobs['pass1_long_side']} pass2={knobs['pass2_long_side']} "
        f"({knobs['pass2_mode']})   stage p50 / p95 / max, milliseconds"
    )
    for key, v in rung["stages"].items():
        print(f"  {key:<7} {v['p50']:>8.1f} {v['p95']:>8.1f} {v['max']:>8.1f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="measure this box and recommend knobs")
    here = Path(__file__).parent
    parser.add_argument("--dir", type=Path, default=here / "samples", help="images to measure")
    parser.add_argument("--images", type=int, default=8, help="how many of them to use")
    parser.add_argument("--repeats", type=int, default=3, help="passes over the set")
    parser.add_argument("--json", type=Path, help="write the machine-readable report here")
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        print(f"no images in {args.dir} — pass --dir <directory of label images>", file=sys.stderr)
        return 2
    paths = paths[: args.images]

    s = get_settings()
    target = TARGET_FRACTION * s.budget_ms
    print(f"labelcheck bench: {len(paths)} images x {args.repeats}, {s.ocr_threads} OCR threads")
    print(f"budget {s.budget_ms} ms, tuning until p95 <= {target:.0f} ms")
    warmup()  # never bill the one-off model load to the first image

    chosen, rungs = tune(paths, s, args.repeats, target)
    final = rungs[-1]
    ratio = final["stages"]["ocr1"]["p50"] / MAC_REFERENCE_OCR1_MS
    fits = final["p95_ms"] <= target
    env = [f"LABELCHECK_{k.upper()}={v}" for k, v in _knobs(chosen).items()]

    print(f"\nthis box is {ratio:.2f}x the reference machine per pass-1 OCR")
    print(
        f"(reference: pass-1 {MAC_REFERENCE_OCR1_MS:.0f} ms, "
        f"total p95 {MAC_REFERENCE_TOTAL_P95_MS:.0f} ms)"
    )
    if not fits:
        print(
            f"WARNING: p95 {final['p95_ms']:.0f} ms is still over {target:.0f} ms at the "
            "lowest resolution on the ladder; this box is too slow for the stated budget"
        )
    elif final["p95_ms"] < FAST_BOX_P95_MS and chosen.pass1_long_side == PASS1_LADDER[0]:
        print("this box has room: pass1_long_side=1280 would raise the exact-match rate from "
              "0.9592 to 0.9796 for about 1.3x the pass-1 cost. Not applied.")  # fmt: skip
    print("\n# /etc/labelcheck.env")
    print("\n".join(env))

    if args.json:
        report = {
            "images": [p.name for p in paths], "repeats": args.repeats,
            "ocr_threads": s.ocr_threads, "budget_ms": s.budget_ms,
            "target_p95_ms": round(target, 1), "reference_ratio": round(ratio, 3),
            "rungs": rungs, "stages": final["stages"], "p95_ms": final["p95_ms"],
            "fits": fits, "chosen": _knobs(chosen), "env": env,
        }  # fmt: skip
        args.json.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json}")
    return 0 if fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
