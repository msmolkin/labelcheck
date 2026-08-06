"""Build app/samples/ from spike/corpus/ per ARCHITECTURE §8.4 and the §11.9 list.

Run once from the repo root:  .venv/bin/python tools/gen_samples.py
"""

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CORPUS = Path("spike/corpus")
OUT = Path("app/samples")
QUALITY = 88

# (corpus file, what the image demonstrates).  The 22 of ARCHITECTURE §11.9, in the order
# that section lists them: eight tightenings, three known-hard cases, eleven coverage picks.
IMAGES = [
    (
        "label_001.png",
        "Reference case: compliant, clean, 38 px Helvetica. Must come back green.",
    ),
    (
        "label_056.png",
        "Compliant statement on a curved bottle. Must not be rejected for curvature.",
    ),
    (
        "label_002.png",
        'Title-case "Government Warning" on a clean shot - the canonical rejection.',
    ),
    (
        "label_006.png",
        "Prefix not bold, plus an alcohol-content mismatch against the record.",
    ),
    (
        "label_023.png",
        "No warning statement at all, shot at an angle, plus an ABV mismatch.",
    ),
    ("label_031.png", "Clause (2) missing under glare, plus a brand mismatch."),
    (
        "label_039.png",
        "Title case on a dark background - the uppercase-ratio rule's proof case.",
    ),
    ("label_045.jpg", "Whole statement bold, JPEG artefacts, 27 px type."),
    (
        "label_026.png",
        "Hard: 19 px type at a steep angle - worst word recall in the corpus.",
    ),
    (
        "label_008.png",
        "Hard: compliant at 21 px, inside the tiny-type abstain band by design.",
    ),
    (
        "label_037.png",
        "Hard: compliant but defocused - the bold gate abstains rather than guess.",
    ),
    ("label_003.png", "Statement reworded, clean shot, Georgia."),
    ("label_007.png", "Whole statement bold on a clean shot."),
    ("label_012.png", "Clause (2) missing, page rotated 5 degrees."),
    ("label_014.png", "Prefix not bold, rotated 5 degrees, plus a brand mismatch."),
    (
        "label_017.png",
        "18 px type rotated 15 degrees - reading order has to re-assemble it.",
    ),
    ("label_020.png", "Statement reworded, rotated 15 degrees, dark background."),
    ("label_027.png", "18 px type under glare."),
    ("label_030.png", "Statement reworded under glare."),
    ("label_036.png", "19 px type, defocused."),
    ("label_042.png", "No warning statement, dark degradation, plus an ABV mismatch."),
    ("label_050.jpg", "Clause (2) missing, JPEG artefacts."),
]

# The eight-image quick demo: two greens, two ambers, four reds, and all four warning
# violation types (title case, prefix not bold, whole statement bold, statement missing).
QUICK = [
    "label_001.png",
    "label_056.png",
    "label_002.png",
    "label_006.png",
    "label_007.png",
    "label_023.png",
    "label_031.png",
    "label_037.png",
]


def main() -> None:
    ground = {g["file"]: g for g in json.loads((CORPUS / "ground_truth.json").read_text())}
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    entries = []
    for source, demonstrates in IMAGES:
        stem = Path(source).stem
        dest = OUT / f"{stem}.jpg"
        with Image.open(CORPUS / source) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.save(dest, "JPEG", quality=QUALITY, optimize=True)  # no EXIF written
            width, height = im.size
        g = ground[source]
        entries.append(
            {
                "file": dest.name,
                "record_id": stem,
                "source_file": source,
                "width": width,
                "height": height,
                "bytes": dest.stat().st_size,
                "demonstrates": demonstrates,
                "warning_status": g["warning_status"],
                "degradation": g["degradation"],
                "font_family": g["font_family"],
                "dark_background": g["dark_background"],
                "expected_record_match": g["expected_record_match"],
                "quick_set": source in QUICK,
            }
        )

    wanted = {e["record_id"] for e in entries}
    with (CORPUS / "records.csv").open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["record_id"] in wanted]
    order = {e["record_id"]: i for i, e in enumerate(entries)}
    rows.sort(key=lambda r: order[r["record_id"]])
    with (OUT / "records.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "description": (
            "Twenty-two bundled label images that let the tool be demonstrated in a fresh "
            "clone with no corpus present, plus the application records they pair with."
        ),
        "source": (
            "spike/corpus (56 synthetic labels, git-ignored), re-encoded to JPEG quality 88 "
            "at native size with all metadata stripped - the same transform ingest applies."
        ),
        "encoding": {
            "format": "JPEG",
            "quality": QUALITY,
            "resize": "none",
            "metadata": "stripped",
        },
        "pairing": "Each image stem equals its record_id in records.csv (pairing='stem').",
        "quick_set": [e["file"] for e in entries if e["quick_set"]],
        "counts": {
            "images": len(entries),
            "quick_set": sum(e["quick_set"] for e in entries),
            "bytes": sum(e["bytes"] for e in entries),
        },
        "images": entries,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"{len(entries)} images + records.csv + MANIFEST.json = {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
