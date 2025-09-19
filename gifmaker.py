#!/usr/bin/env python3
"""
png_jpg_sequence_to_gif.py

Create an animated GIF from images named like:
  frame_000000.png, frame_000005.jpg, frame_000010.png, ...
Works with PNG and JPG/JPEG, supports gaps, and avoids common GIF artifacts.

Usage:
  python png_jpg_sequence_to_gif.py /path/to/frames -o out.gif --fps 20
  # or specify per-frame duration instead of fps:
  python png_jpg_sequence_to_gif.py /path/to/frames -o out.gif --duration-ms 50

Common quality flags:
  --no-dither      # avoid speckled color noise (default: no dither)
  --no-opt         # avoid palette/partial-frame optimize (default: no optimize)
  --size 800x450   # optionally scale frames
  --bg 000000      # matte color for alpha (hex, default ffffff)

Requires: Pillow
  pip install Pillow
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image

FNAME_RE = re.compile(r"^frame_(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)


def find_frames(dir_path: Path) -> List[Tuple[int, Path]]:
    frames: List[Tuple[int, Path]] = []
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if m:
            idx = int(m.group(1))
            frames.append((idx, p))
    frames.sort(key=lambda t: t[0])
    return frames


def load_prepare_rgb(p: Path, out_size: Optional[Tuple[int, int]], bg_rgb: Tuple[int, int, int]) -> Image.Image:
    with Image.open(p) as im:
        # Resize (if requested)
        if out_size and im.size != out_size:
            im = im.resize(out_size, Image.Resampling.LANCZOS)

        # Flatten alpha onto background for stable GIF results
        if im.mode in ("RGBA", "LA"):
            base = Image.new("RGBA", im.size, bg_rgb + (255,))
            im = Image.alpha_composite(base, im.convert("RGBA")).convert("RGB")
        else:
            im = im.convert("RGB")
        return im


def build_master_palette(images_rgb: List[Image.Image], sample_n: int = 64, thumb_w: int = 320) -> Image.Image:
    """
    Build a single global palette from a sampled 'contact sheet' of frames.
    This avoids frame-to-frame palette changes (which cause flicker/shadows).
    """
    if not images_rgb:
        raise ValueError("No images to build palette from")

    n = min(sample_n, len(images_rgb))
    # Pick evenly spaced frames
    idxs = [round(i * (len(images_rgb) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]

    thumbs: List[Image.Image] = []
    for i in idxs:
        t = images_rgb[i].copy()
        if t.width > thumb_w:
            t.thumbnail((thumb_w, int(t.height * (thumb_w / t.width))), Image.Resampling.LANCZOS)
        thumbs.append(t)

    # Stack vertically into a sheet
    w = max(t.width for t in thumbs)
    h = sum(t.height for t in thumbs)
    sheet = Image.new("RGB", (w, h))
    y = 0
    for t in thumbs:
        sheet.paste(t, (0, y))
        y += t.height

    # Quantize sheet to 256 colors WITHOUT dithering to create stable palette
    # (MEDIANCUT is widely available; LIBIMAGEQUANT is great if your Pillow has it)
    try:
        method = Image.Quantize.MEDIANCUT  # Pillow >=9
    except Exception:
        method = 0  # fallback

    palette_img = sheet.quantize(colors=256, method=method, dither=Image.Dither.NONE)
    return palette_img


def main():
    ap = argparse.ArgumentParser(description="Convert frame_*.png/jpg sequence to GIF (with global palette).")
    ap.add_argument("directory", type=str, help="Directory containing frame_*.png/jpg")
    ap.add_argument("-o", "--output", type=str, default="animation.gif", help="Output GIF path")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--fps", type=float, default=10.0, help="Frames per second (default: 10)")
    group.add_argument("--duration-ms", type=int, help="Duration per frame in milliseconds")
    ap.add_argument("--loop", type=int, default=0, help="Number of loops (0=infinite)")
    ap.add_argument("--size", type=str, default="", help="Resize to WxH (e.g., 512x512).")
    ap.add_argument("--bg", type=str, default="ffffff", help="Hex RGB background for alpha flattening (default ffffff)")
    ap.add_argument("--samples", type=int, default=64, help="Frames to sample for global palette (default 64)")
    ap.add_argument("--no-dither", action="store_true", help="Disable dithering (recommended to avoid speckle)")
    ap.add_argument("--no-opt", action="store_true", help="Disable GIF optimization (recommended to avoid trails)")
    args = ap.parse_args()

    dir_path = Path(args.directory).expanduser().resolve()
    if not dir_path.exists():
        raise SystemExit(f"Directory not found: {dir_path}")

    frames = find_frames(dir_path)
    if not frames:
        raise SystemExit(f"No files matching frame_*.png/jpg found in: {dir_path}")

    min_idx, max_idx = frames[0][0], frames[-1][0]
    expected = max_idx - min_idx + 1
    if len(frames) != expected:
        missing = expected - len(frames)
        print(f"Note: gaps detected ({missing} missing). Using present frames only.")

    # Duration
    if args.duration_ms is not None:
        duration_ms = int(args.duration_ms)
    else:
        if args.fps <= 0:
            raise SystemExit("--fps must be > 0")
        duration_ms = max(1, int(round(1000.0 / args.fps)))

    # Parse size
    out_size: Optional[Tuple[int, int]] = None
    if args.size:
        try:
            w_str, h_str = args.size.lower().split("x")
            out_size = (int(w_str), int(h_str))
        except Exception:
            raise SystemExit("Invalid --size. Use WxH, e.g., 512x512")

    # Parse background
    bg_hex = args.bg.strip().lstrip("#")
    if len(bg_hex) != 6 or any(c not in "0123456789abcdefABCDEF" for c in bg_hex):
        raise SystemExit("Invalid --bg. Use 6-hex digits, e.g., ffffff")
    bg_rgb = tuple(int(bg_hex[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore

    # Load all frames as RGB (resized, flattened as needed)
    images_rgb: List[Image.Image] = []
    for _, p in frames:
        images_rgb.append(load_prepare_rgb(p, out_size, bg_rgb))

    # If no resize requested, lock to first frame’s size
    if out_size is None:
        out_size = images_rgb[0].size

    # Build a single, stable global palette from sampled frames
    palette_img = build_master_palette(images_rgb, sample_n=max(1, args.samples))

    # Quantize each frame to that palette (no dithering to avoid speckle/shadows)
    dither_mode = Image.Dither.NONE if args.no_dither or True else Image.Dither.FLOYDSTEINBERG  # default to NONE
    quantized_frames: List[Image.Image] = [im.quantize(palette=palette_img, dither=dither_mode) for im in images_rgb]

    # Save GIF
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use disposal=2 to clear between frames; keep optimize off by default to avoid ghost trails
    optimize = False if args.no_opt or True else True  # default to False
    quantized_frames[0].save(
        out_path,
        save_all=True,
        append_images=quantized_frames[1:],
        format="GIF",
        duration=duration_ms,
        loop=args.loop,
        disposal=2,
        optimize=optimize,
    )
    print(f"Saved GIF → {out_path} ({len(quantized_frames)} frames @ {duration_ms} ms/frame, loop={args.loop})")


if __name__ == "__main__":
    main()
