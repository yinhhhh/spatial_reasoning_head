import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Create discrete patch overlays with threshold highlights.")
    parser.add_argument("--attn-dir", type=str, required=True)
    parser.add_argument("--dataset-json", type=str, default="data/controlled_images_dataset.json")
    parser.add_argument("--prompt-jsonl", type=str, default="prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    parser.add_argument("--sampled-idx", type=str, default="output/sampled_idx_Controlled_Images_A.npy")
    parser.add_argument("--test-sample-count", type=int, default=6)
    parser.add_argument("--sample-local-idx", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--criterion", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.45)
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


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def pick_layer_file(sample_dir, layer):
    for p in sorted(sample_dir.glob(f"diff_{layer}_start*_end*.npy")):
        m = re.search(r"start(-?\d+)_end(-?\d+)", p.name)
        if m and int(m.group(1)) >= 0:
            return p, int(m.group(1)), int(m.group(2))
    return None, None, None


def build_patch_map(attn_file, start_idx, end_idx):
    arr = np.load(attn_file)[0]  # [heads, seq_len]
    arr = np.where(arr < -1e20, -1e20, arr)
    probs = softmax(arr)
    patch_probs = probs[:, start_idx : end_idx + 1]
    patch_score = patch_probs.mean(axis=0)
    n_patch = patch_score.shape[0]
    side = int(round(math.sqrt(n_patch)))
    if side * side != n_patch:
        raise ValueError(f"Patch count is not square: {n_patch}")
    patch_map = patch_score.reshape(side, side)
    patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
    return patch_map


def draw_overlay(image_path, patch_map, threshold, title, out_path, alpha):
    image = Image.open(image_path).convert("RGB")
    patch_u8 = (patch_map * 255).astype(np.uint8)
    color_grid = make_colormap(patch_u8)

    # Mark high-attention patches in orange.
    orange = np.array([255, 140, 0], dtype=np.uint8)
    color_grid[patch_map > threshold] = orange

    heat = Image.fromarray(color_grid).resize(image.size, Image.NEAREST)
    overlay = Image.blend(image, heat, alpha=alpha)

    grid_h, grid_w = patch_map.shape
    draw = ImageDraw.Draw(overlay)
    step_x = overlay.width / grid_w
    step_y = overlay.height / grid_h
    for gx in range(1, grid_w):
        x = int(round(gx * step_x))
        draw.line([(x, 0), (x, overlay.height)], fill=(255, 255, 255), width=1)
    for gy in range(1, grid_h):
        y = int(round(gy * step_y))
        draw.line([(0, y), (overlay.width, y)], fill=(255, 255, 255), width=1)

    canvas = Image.new("RGB", (overlay.width, overlay.height + 62), (255, 255, 255))
    canvas.paste(overlay, (0, 62))
    header = ImageDraw.Draw(canvas)
    header.text((10, 8), title, fill=(0, 0, 0))
    header.text((10, 32), f"orange if patch_score > {threshold:.2f}", fill=(0, 0, 0))
    canvas.save(out_path)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attn_root = Path(args.attn_dir)

    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    prompts = load_jsonl(args.prompt_jsonl)
    sampled_idx = np.load(args.sampled_idx).tolist()
    unsampled = sorted(set(range(len(prompts))) - set(sampled_idx))
    eval_indices = unsampled[: args.test_sample_count]

    records = []
    for local_idx in args.sample_local_idx:
        if local_idx >= len(eval_indices):
            continue
        global_idx = eval_indices[local_idx]
        sample_dir = attn_root / str(local_idx)
        attn_file, start_idx, end_idx = pick_layer_file(sample_dir, args.layer)
        if attn_file is None:
            continue

        patch_map = build_patch_map(attn_file, start_idx, end_idx)
        title = f"{args.criterion} | sample={local_idx} | layer={args.layer}"
        out_file = out_dir / f"{args.criterion}_sample{local_idx}_th{args.threshold:.2f}_layer{args.layer}.png"
        draw_overlay(
            image_path=dataset[global_idx]["image_path"],
            patch_map=patch_map,
            threshold=args.threshold,
            title=title,
            out_path=out_file,
            alpha=args.alpha,
        )

        records.append(
            {
                "criterion": args.criterion,
                "threshold": args.threshold,
                "local_sample_idx": local_idx,
                "global_dataset_idx": global_idx,
                "image_path": dataset[global_idx]["image_path"],
                "prompt": prompts[global_idx]["question"],
                "attn_file": str(attn_file),
                "token_span": [start_idx, end_idx],
                "patch_shape": list(patch_map.shape),
                "output_image": str(out_file),
            }
        )

    summary = out_dir / f"{args.criterion}_th{args.threshold:.2f}_summary.json"
    summary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} images -> {out_dir}")


if __name__ == "__main__":
    main()
