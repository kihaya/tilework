#!/usr/bin/env python3
"""Generate the Tilework app icon SVGs.

Usage: python3 brand/icon/generate.py [OUT_DIR]   (default: brand/icon)

Every geometry and color value lives here, so this file is the source of
truth for the icon; the SVGs next to it are its output. Rasterize them with
render_png.js. See README.md in this directory for the design rationale.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE

# Palette. Blades step darker clockwise from the top, which is what reads as
# rotation; the four blades stand for the :cpu / :metal / :cuda / :hip backends
# and the ivory core for the one codebase they share.
INK = "#15171D"
RUBY_RAMP = ("#F07A63", "#E5304E", "#B01E3E", "#861A36")  # coral, ruby, crimson, wine
IVORY = "#ECE7DD"

DARK_BG = "#0B0C0F"
DARK_RAMP = ("#F48A74", "#EE4A64", "#C42A4B", "#9A2242")
DARK_CORE = "#D8D3C9"

# Tinted appearance: the system tints a grayscale image, so only luminance
# survives. The ramp keeps the same clockwise ordering.
GRAY_BG = "#000000"
GRAY_RAMP = ("#E6E6E6", "#B3B3B3", "#808080", "#595959")
GRAY_CORE = "#FFFFFF"


def squircle(n=5, samples=96):
    """Superellipse |x|^n + |y|^n = 1 in a 100x100 box (macOS body outline)."""
    pts = []
    for i in range(samples):
        t = 2 * math.pi * i / samples
        c, s = math.cos(t), math.sin(t)
        x = 50 + 50 * math.copysign(abs(c) ** (2 / n), c)
        y = 50 + 50 * math.copysign(abs(s) ** (2 / n), s)
        pts.append(f"{x:.2f} {y:.2f}")
    return "M" + "L".join(pts) + "Z"


def blades(top, right, bottom, left):
    # 100 units = 1024 px. Blades 40.5 x 22.5 (415 x 230 px), gap 3.5 (36 px).
    return (
        f'<rect x="16.75" y="16.75" width="40.5" height="22.5" rx="4" fill="{top}"/>'
        f'<rect x="60.75" y="16.75" width="22.5" height="40.5" rx="4" fill="{right}"/>'
        f'<rect x="42.75" y="60.75" width="40.5" height="22.5" rx="4" fill="{bottom}"/>'
        f'<rect x="16.75" y="42.75" width="22.5" height="40.5" rx="4" fill="{left}"/>'
    )


def core(color):
    # 14.5 units (148 px) square at the center.
    return f'<rect x="42.75" y="42.75" width="14.5" height="14.5" rx="3" fill="{color}"/>'


def svg(body, defs=""):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 100 100">'
        + (f"<defs>{defs}</defs>" if defs else "")
        + body
        + "</svg>\n"
    )


def full_bleed(bg, ramp, core_color):
    # iOS / iPadOS: a full-bleed square. The system applies the rounded mask.
    return svg(f'<rect width="100" height="100" fill="{bg}"/>' + blades(*ramp) + core(core_color))


MACOS_SHADOW = (
    '<filter id="s" x="-20%" y="-20%" width="140%" height="140%">'
    '<feDropShadow dx="0" dy="1.2" stdDeviation="1" flood-color="#000" flood-opacity="0.3"/>'
    "</filter>"
)

FILES = {
    "tilework-icon-ios.svg": full_bleed(INK, RUBY_RAMP, IVORY),
    "tilework-icon-ios-dark.svg": full_bleed(DARK_BG, DARK_RAMP, DARK_CORE),
    "tilework-icon-ios-tinted.svg": full_bleed(GRAY_BG, GRAY_RAMP, GRAY_CORE),
    # macOS: an 824 px body on the 1024 canvas (100 px margin) with the shape
    # and shadow baked in.
    "tilework-icon-macos.svg": svg(
        '<g filter="url(#s)"><g transform="translate(9.77 9.77) scale(0.8047)">'
        f'<path d="{squircle()}" fill="{INK}"/>' + blades(*RUBY_RAMP) + core(IVORY) + "</g></g>",
        MACOS_SHADOW,
    ),
    # Icon Composer layers (transparent). The background is a solid INK fill
    # set in Icon Composer itself.
    "layers/blades.svg": svg(blades(*RUBY_RAMP)),
    "layers/core.svg": svg(core(IVORY)),
}


def main():
    for name, text in FILES.items():
        path = os.path.join(OUT, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(path)


if __name__ == "__main__":
    main()
