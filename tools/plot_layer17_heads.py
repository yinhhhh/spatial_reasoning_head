import argparse
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize per-head layer17 patch attention.")
    parser.add_argument("--attn-file", type=str, required=True, help="Path to diff_17_start*_end*.npy")
    parser.add_argument("--image-path", type=str, required=True, help="Original image path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save per-head images")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    parser.add_argument("--draw-grid", action="store_true", help="Draw patch grid lines")
    parser.add_argument("--title-prefix", type=str, default="layer17", help="Prefix for title text")
    return parser.parse_args()


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def make_colormap(gray_uint8):
    x = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 - np.abs(2 * x - 1) * 1.5, 0, 1)
    b = np.clip(1.5 * (1 - x), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def parse_span_from_name(path):
    m = re.search(r"start(-?\d+)_end(-?\d+)", Path(path).name)
    if not m:
        raise ValueError(f"Cannot parse token span from filename: {path}")
    start_idx = int(m.group(1))
    end_idx = int(m.group(2))
    if start_idx < 0 or end_idx < 0:
        raise ValueError(f"Invalid image-token span in filename: start={start_idx}, end={end_idx}")
    return start_idx, end_idx


def draw_grid(draw, width, height, side):
    step_x = width / side
    step_y = height / side
    for gx in range(1, side):
        x = int(round(gx * step_x))
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255), width=1)
    for gy in range(1, side):
        y = int(round(gy * step_y))
        draw.line([(0, y), (width, y)], fill=(255, 255, 255), width=1)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(args.attn_file)  # [1, heads, seq_len]
    arr = arr[0]  # [heads, seq_len]
    arr = np.where(arr < -1e20, -1e20, arr)
    probs = softmax(arr)
    start_idx, end_idx = parse_span_from_name(args.attn_file)
    patch_probs = probs[:, start_idx : end_idx + 1]  # [heads, patches]

    num_heads, n_patch = patch_probs.shape
    side = int(round(math.sqrt(n_patch)))
    if side * side != n_patch:
        raise ValueError(f"Patch count is not a square: {n_patch}")

    image = Image.open(args.image_path).convert("RGB")
    per_head_paths = []

    for h in range(num_heads):
        patch_map = patch_probs[h].reshape(side, side)
        patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
        patch_u8 = (patch_map * 255).astype(np.uint8)
        heat = Image.fromarray(make_colormap(patch_u8)).resize(image.size, Image.NEAREST)

        overlay = Image.blend(image, heat, alpha=args.alpha)
        if args.draw_grid:
            draw = ImageDraw.Draw(overlay)
            draw_grid(draw, overlay.width, overlay.height, side)

        canvas = Image.new("RGB", (overlay.width, overlay.height + 42), (255, 255, 255))
        canvas.paste(overlay, (0, 42))
        header = ImageDraw.Draw(canvas)
        header.text((10, 10), f"{args.title_prefix} | head={h}", fill=(0, 0, 0))

        out_path = out_dir / f"head_{h:02d}.png"
        canvas.save(out_path)
        per_head_paths.append(out_path)

    # Build contact sheet (32 heads -> 4x8 by default)
    cols = 8
    rows = math.ceil(num_heads / cols)
    thumb_w = 240
    thumb_h = int((image.height + 42) * (thumb_w / image.width))
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (245, 245, 245))

    for idx, p in enumerate(per_head_paths):
        r = idx // cols
        c = idx % cols
        tile = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.BILINEAR)
        sheet.paste(tile, (c * thumb_w, r * thumb_h))

    sheet_path = out_dir / "all_heads_sheet.png"
    sheet.save(sheet_path)

    print(f"Saved {num_heads} head images to {out_dir}")
    print(f"Patch grid: {side}x{side} ({n_patch} patches)")
    print(f"Sheet: {sheet_path}")


if __name__ == "__main__":
    main()
