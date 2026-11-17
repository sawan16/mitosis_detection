#!/usr/bin/env python3
"""
Prepare ICPR-2012 mitosis data (A = Aperio, H = Hamamatsu) for detection training.

Input : a folder that contains the extracted A00_v2 ... A04_v2 / H00_v2 ... H04_v2 folders
        (any nesting; files are found recursively). Each frame X00_01.jpg has an optional
        X00_01.csv: one line per mitosis, listing all its pixels as x,y,x,y,...
Output: <out>/patches/*.jpg          512x512 patches (JPEG q95)
        <out>/tiles.csv              one row per patch (scanner, slide, frame, x0, y0, n_boxes)
        <out>/patch_boxes.csv        boxes in patch coordinates
        <out>/frames.csv             one row per full frame (path, size, n_mitosis)
        <out>/boxes.csv              boxes in full-frame coordinates (use for full-frame evaluation)
        <out>/check/*.jpg            overlays to visually verify the box orientation

SPLITTING: always split by the `slide` column (e.g. A00, H03), never by patch or frame.

Usage:
    python prepare_patches.py --root D:/data/icpr2012 --out D:/data/icpr_patches --check 6
"""
import argparse
import csv
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
NAME = re.compile(r"^([AH])(\d\d)_(\d\d)$")


def read_boxes(csv_path, pad):
    """Each line = one mitosis as x,y,x,y,... pixels -> (x1, y1, x2, y2)."""
    boxes = []
    if not csv_path.exists():
        return boxes
    for line in csv_path.read_text(errors="ignore").splitlines():
        vals = [int(float(v)) for v in line.replace(";", ",").split(",") if v.strip()]
        if len(vals) < 4:
            continue
        if len(vals) % 2:
            vals = vals[:-1]
        xs, ys = vals[0::2], vals[1::2]
        boxes.append((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
    return boxes


def clip(b, w, h):
    return (max(0, b[0]), max(0, b[1]), min(w - 1, b[2]), min(h - 1, b[3]))


def area(b):
    return max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)


def inter(b, t):
    return (max(b[0], t[0]), max(b[1], t[1]), min(b[2], t[2]), min(b[3], t[3]))


def positions(length, patch, stride):
    pos = list(range(0, max(length - patch, 0) + 1, stride))
    if pos[-1] != length - patch and length > patch:
        pos.append(length - patch)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=".jpg", help="image extension to use (.jpg or .bmp)")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--pad", type=int, default=2, help="pixels added around the mask extent")
    ap.add_argument("--neg-ratio", type=float, default=3.0, help="negative patches per positive patch")
    ap.add_argument("--neg-empty-frame", type=int, default=4, help="negatives sampled from frames without mitosis")
    ap.add_argument("--min-visible", type=float, default=0.8,
                    help="a box is kept in a patch only if this fraction is visible; "
                         "patches with a more heavily cut box are skipped")
    ap.add_argument("--check", type=int, default=4, help="number of overlay images to write")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root, out = Path(args.root), Path(args.out)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    (out / "check").mkdir(parents=True, exist_ok=True)

    frames = []
    for p in sorted(root.rglob(f"*{args.ext}")):
        m = NAME.match(p.stem)
        if m:
            frames.append((p, m.group(1), m.group(1) + m.group(2), p.stem))
    if not frames:
        raise SystemExit(f"No frames like A00_01{args.ext} found under {root}")

    f_frames = open(out / "frames.csv", "w", newline="")
    f_boxes = open(out / "boxes.csv", "w", newline="")
    f_tiles = open(out / "tiles.csv", "w", newline="")
    f_pb = open(out / "patch_boxes.csv", "w", newline="")
    w_frames, w_boxes = csv.writer(f_frames), csv.writer(f_boxes)
    w_tiles, w_pb = csv.writer(f_tiles), csv.writer(f_pb)
    w_frames.writerow(["scanner", "slide", "frame", "path", "width", "height", "n_mitosis"])
    w_boxes.writerow(["scanner", "slide", "frame", "x1", "y1", "x2", "y2", "w", "h"])
    w_tiles.writerow(["patch", "scanner", "slide", "frame", "x0", "y0", "size", "n_boxes"])
    w_pb.writerow(["patch", "x1", "y1", "x2", "y2"])

    mitoses = defaultdict(int)
    sizes_w, sizes_h = [], []
    n_pos_patches = n_neg_patches = skipped_partial = 0
    checks_done = 0

    for path, scanner, slide, frame in frames:
        img = Image.open(path).convert("RGB")
        W, H = img.size
        boxes = [clip(b, W, H) for b in read_boxes(path.with_suffix(".csv"), args.pad)]
        mitoses[scanner] += len(boxes)
        mitoses[slide] += len(boxes)
        w_frames.writerow([scanner, slide, frame, str(path), W, H, len(boxes)])
        for b in boxes:
            bw, bh = b[2] - b[0] + 1, b[3] - b[1] + 1
            sizes_w.append(bw)
            sizes_h.append(bh)
            w_boxes.writerow([scanner, slide, frame, *b, bw, bh])

        pos_tiles, neg_tiles = [], []
        for y0 in positions(H, args.patch, args.stride):
            for x0 in positions(W, args.patch, args.stride):
                tile = (x0, y0, x0 + args.patch - 1, y0 + args.patch - 1)
                kept, bad = [], False
                for b in boxes:
                    ib = inter(b, tile)
                    frac = area(ib) / area(b) if ib[2] >= ib[0] and ib[3] >= ib[1] else 0.0
                    if frac >= args.min_visible:
                        kept.append((ib[0] - x0, ib[1] - y0, ib[2] - x0, ib[3] - y0))
                    elif frac > 0:
                        bad = True
                if bad:
                    skipped_partial += 1
                    continue
                (pos_tiles if kept else neg_tiles).append((x0, y0, kept))

        k = round(args.neg_ratio * len(pos_tiles)) if pos_tiles else args.neg_empty_frame
        chosen = pos_tiles + rng.sample(neg_tiles, min(k, len(neg_tiles)))
        for x0, y0, kept in chosen:
            name = f"{frame}_{x0}_{y0}.jpg"
            img.crop((x0, y0, x0 + args.patch, y0 + args.patch)).save(out / "patches" / name, quality=95)
            w_tiles.writerow([name, scanner, slide, frame, x0, y0, args.patch, len(kept)])
            for b in kept:
                w_pb.writerow([name, *b])
        n_pos_patches += len(pos_tiles)
        n_neg_patches += len(chosen) - len(pos_tiles)

        if boxes and checks_done < args.check:
            s = 0.5
            ov = img.resize((int(W * s), int(H * s)))
            d = ImageDraw.Draw(ov)
            for b in boxes:
                d.rectangle([int(v * s) for v in b], outline=(0, 255, 0), width=2)
            ov.save(out / "check" / f"{frame}_check.jpg", quality=90)
            checks_done += 1
        print(f"{frame}: {W}x{H}, {len(boxes)} mitoses, {len(pos_tiles)} pos tiles")

    for f in (f_frames, f_boxes, f_tiles, f_pb):
        f.close()

    print("\n===== SUMMARY =====")
    print("frames:", len(frames))
    for key in sorted(k for k in mitoses if len(k) == 1):
        print(f"scanner {key}: {mitoses[key]} mitoses")
    for key in sorted(k for k in mitoses if len(k) == 3):
        print(f"  slide {key}: {mitoses[key]} mitoses")
    print(f"patches: {n_pos_patches} positive, {n_neg_patches} negative; "
          f"{skipped_partial} tiles skipped (partially cut box)")
    if sizes_w:
        for label, v in (("width", sizes_w), ("height", sizes_h)):
            q = statistics.quantiles(v, n=4) if len(v) > 3 else [min(v)] * 3
            print(f"box {label}: min {min(v)}, Q1 {q[0]:.0f}, median {q[1]:.0f}, Q3 {q[2]:.0f}, max {max(v)}")
    print(f"Open {out / 'check'} and confirm the green boxes sit on the mitotic figures.")


if __name__ == "__main__":
    main()
