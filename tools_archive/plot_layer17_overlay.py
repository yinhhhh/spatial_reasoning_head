import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Create layer-17 patch attention overlays.")
    parser.add_argument("--attn-dir", type=str, required=True, help="Attention root dir, e.g. output/Controlled_Images_A_weight1.01")
    parser.add_argument("--criterion", type=str, required=True, help="Label to print on images, e.g. max_prob or margin")
    parser.add_argument("--dataset-json", type=str, default="data/controlled_images_dataset.json")
    parser.add_argument("--prompt-jsonl", type=str, default="prompts/Controlled_Images_A_with_answer_four_options.jsonl")
    parser.add_argument("--sampled-idx", type=str, default="output/sampled_idx_Controlled_Images_A.npy")
    parser.add_argument("--test-sample-count", type=int, default=6)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def make_colormap(gray_uint8):
    # Simple blue->cyan->yellow->red color map.
    x = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 - np.abs(2 * x - 1) * 1.5, 0, 1)
    b = np.clip(1.5 * (1 - x), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def load_prompts(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def pick_layer_file(sample_dir, layer):
    candidates = sorted(sample_dir.glob(f"diff_{layer}_start*_end*.npy"))
    # Prefer files with a concrete image-token span (start >= 0).
    for p in candidates:
        m = re.search(r"start(-?\d+)_end(-?\d+)", p.name)
        if m and int(m.group(1)) >= 0:
            return p, int(m.group(1)), int(m.group(2))
    return None, None, None


def overlay_one(image_path, attn_file, start_idx, end_idx, out_path, title):
    arr = np.load(attn_file)  # [1, heads, seq_len]
    arr = arr[0]  # [heads, seq_len]
    arr = np.where(arr < -1e20, -1e20, arr)
    probs = softmax(arr)  # [heads, seq_len]

    patch_probs = probs[:, start_idx : end_idx + 1]
    patch_score = patch_probs.mean(axis=0)
    n_patch = patch_score.shape[0]
    side = int(round(math.sqrt(n_patch)))
    if side * side != n_patch:
        raise ValueError(f"Patch count is not a square: {n_patch}")

    patch_map = patch_score.reshape(side, side)
    patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
    patch_img = Image.fromarray((patch_map * 255).astype(np.uint8))

    image = Image.open(image_path).convert("RGB")
    patch_img = patch_img.resize(image.size, Image.BICUBIC)
    heat_rgb = make_colormap(np.array(patch_img))
    heat = Image.fromarray(heat_rgb)

    overlay = Image.blend(image, heat, alpha=0.40)

    canvas_h = overlay.height + 56
    canvas = Image.new("RGB", (overlay.width, canvas_h), (255, 255, 255))
    canvas.paste(overlay, (0, 56))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    canvas.save(out_path)


def main():
    args = parse_args()
    attn_root = Path(args.attn_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
    prompts = load_prompts(args.prompt_jsonl)
    sampled_idx = np.load(args.sampled_idx).tolist()

    total = len(prompts)
    unsampled = sorted(set(range(total)) - set(sampled_idx))
    eval_indices = unsampled[: args.test_sample_count]

    n = min(args.num_examples, len(eval_indices))
    summary = []
    for local_idx in range(n):
        global_idx = eval_indices[local_idx]
        sample_dir = attn_root / str(local_idx)
        attn_file, start_idx, end_idx = pick_layer_file(sample_dir, args.layer)
        if attn_file is None:
            continue

        image_path = dataset[global_idx]["image_path"]
        prompt = prompts[global_idx].get("question", "N/A")
        out_path = out_dir / f"{args.criterion}_sample{local_idx}_layer{args.layer}.png"
        title = f"{args.criterion} | sample={local_idx} | layer={args.layer}"
        overlay_one(image_path, attn_file, start_idx, end_idx, out_path, title)

        summary.append(
            {
                "criterion": args.criterion,
                "local_sample_idx": local_idx,
                "global_dataset_idx": global_idx,
                "image_path": image_path,
                "prompt": prompt,
                "attn_file": str(attn_file),
                "token_span": [start_idx, end_idx],
                "output_image": str(out_path),
            }
        )

    (out_dir / f"{args.criterion}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(summary)} overlays to {out_dir}")


if __name__ == "__main__":
    main()
