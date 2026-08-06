"""Generate the site's raster assets locally, so nothing is fetched from a design tool.

Run from the repository root with the project interpreter::

    .venv/bin/python static/__fixtures__/gen_assets.py

Writes ``static/favicon-32.png``, ``static/apple-touch-icon.png`` and ``static/og.png``.
``static/favicon.svg`` is hand-written and is the source the raster icons imitate: a dark
green rounded square with a white check mark, which stays legible at 16 px.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GREEN = (20, 83, 45)
INK = (22, 24, 29)
MUTED = (74, 79, 87)
WHITE = (255, 255, 255)
BORDER = (185, 190, 198)
AMBER = (122, 65, 0)
RED = (143, 20, 20)

STATIC = Path(__file__).resolve().parent.parent
FONT_DIR = Path("/System/Library/Fonts/Supplemental")


def load_font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    """Pick a system sans face. The generator is a build-time tool, not app code, so a
    macOS path with a PIL fallback is acceptable; the PNGs it produces are committed."""
    candidates = [FONT_DIR / ("Arial Bold.ttf" if bold else "Arial.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def check_icon(size: int) -> Image.Image:
    """Rounded square with a white check mark, drawn at 4x and downsampled."""
    scale = 4
    box = size * scale
    img = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, box - 1, box - 1], radius=int(box * 0.22), fill=GREEN)
    stroke = max(2, int(box * 0.11))
    points = [
        (box * 0.24, box * 0.53),
        (box * 0.43, box * 0.72),
        (box * 0.78, box * 0.30),
    ]
    draw.line(points, fill=WHITE, width=stroke, joint="curve")
    for point in points:
        draw.ellipse(
            [
                point[0] - stroke / 2,
                point[1] - stroke / 2,
                point[0] + stroke / 2,
                point[1] + stroke / 2,
            ],
            fill=WHITE,
        )
    return img.resize((size, size), Image.LANCZOS)


def write_icons() -> None:
    check_icon(32).save(STATIC / "favicon-32.png")
    # Apple home-screen icons are composited on white; no transparency, no rounding of ours.
    touch = Image.new("RGB", (180, 180), WHITE)
    touch.paste(check_icon(180), (0, 0), check_icon(180))
    touch.save(STATIC / "apple-touch-icon.png")


def write_og_card() -> None:
    """A typographic 1200x630 share card: title, one line of what it does, the four
    statutory checks, and the three verdict states the tool can return."""
    img = Image.new("RGB", (1200, 630), WHITE)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1199, 629], outline=BORDER, width=3)
    draw.rectangle([0, 0, 1199, 13], fill=GREEN)

    icon = check_icon(96)
    img.paste(icon, (72, 84), icon)

    draw.text((196, 92), "labelcheck", font=load_font(True, 88), fill=INK)
    draw.text(
        (72, 236),
        "Alcohol beverage label verification",
        font=load_font(True, 44),
        fill=GREEN,
    )
    draw.text(
        (72, 300),
        "Reads a photograph of a label and checks the health warning statement\n"
        "required by 27 CFR Part 16, then compares the brand, class/type, alcohol\n"
        "content and net contents against the application record.",
        font=load_font(False, 32),
        fill=MUTED,
        spacing=14,
    )

    small = load_font(True, 28)
    states = [("● PASS", GREEN), ("▲ NEEDS REVIEW", AMBER), ("■ FAIL", RED)]
    x = 72
    for text, colour in states:
        width = draw.textlength(text, font=small)
        draw.rounded_rectangle([x, 500, x + width + 44, 560], radius=30, outline=colour, width=3)
        draw.text((x + 22, 512), text, font=small, fill=colour)
        x += width + 68

    img.save(STATIC / "og.png", optimize=True)


def main() -> int:
    write_icons()
    write_og_card()
    for name in ("favicon-32.png", "apple-touch-icon.png", "og.png"):
        path = STATIC / name
        print(f"{path.relative_to(STATIC.parent)}  {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
