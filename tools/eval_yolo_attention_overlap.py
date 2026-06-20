import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate overlap between YOLO boxes and layer attention maps (max_prob vs margin)."
    )
    parser.add_argument("--attn-dir-max", type=str, required=True, help="Attention dir for max_prob run.")
    parser.add_argument("--attn-dir-margin", type=str, required=True, help="Attention dir for margin run.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=["Controlled_Images_A", "Controlled_Images_B", "COCO_QA_one_obj", "COCO_QA_two_obj", "VG_QA_one_obj", "VG_QA_two_obj"],
    )
    parser.add_argument("--sampled-idx", type=str, required=True)
    parser.add_argument("--test-sample-count", type=int, default=6)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--max-boxes", type=int, default=2, help="Keep at most top-K YOLO boxes by confidence.")
    parser.add_argument(
        "--box-expand-ratio",
        type=float,
        default=0.0,
        help="Expand each YOLO box by this ratio on width/height (e.g. 0.2 means +20% on each side).",
    )
    parser.add_argument(
        "--new-adaptive-beta",
        type=float,
        default=0.0,
        help="Adaptive boost strength for NEW map; 0 disables boosting.",
    )
    parser.add_argument(
        "--new-adaptive-cap",
        type=float,
        default=0.95,
        help="Upper cap of adaptive lambda for NEW map.",
    )
    parser.add_argument(
        "--new-top-fraction",
        type=float,
        default=0.0,
        help="If >0, equalize top fraction patches in NEW map toward max (e.g. 0.2 means top 20%).",
    )
    parser.add_argument(
        "--new-top-mix",
        type=float,
        default=1.0,
        help="Mix factor for top-fraction equalization: 1.0 means set selected patches to max.",
    )
    parser.add_argument(
        "--only-local-idx",
        type=str,
        default="",
        help='Optional comma-separated local sample indices to evaluate, e.g. "11".',
    )
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def pick_attn_file(sample_dir: Path, layer: int):
    candidates = sorted(sample_dir.glob(f"diff_{layer}_start*_end*.npy"))
    for p in candidates:
        m = re.search(r"start(-?\d+)_end(-?\d+)", p.name)
        if m and int(m.group(1)) >= 0:
            return p, int(m.group(1)), int(m.group(2))
    return None, None, None


def load_patch_prob(attn_file: Path, start_idx: int, end_idx: int):
    arr = np.load(attn_file)[0]  # [heads, seq_len]
    arr = np.where(arr < -1e20, -1e20, arr)
    probs = softmax(arr)  # [heads, seq_len]
    patch = probs[:, start_idx : end_idx + 1]  # [heads, patches]
    patch = patch.mean(axis=0)  # head average
    side = int(round(math.sqrt(patch.shape[0])))
    if side * side != patch.shape[0]:
        raise ValueError(f"Patch number is not square: {patch.shape[0]}")
    patch_grid = patch.reshape(side, side)
    patch_grid = patch_grid / (patch_grid.sum() + 1e-12)  # normalize patch mass to 1
    return patch_grid


def patch_mask_from_boxes(side: int, img_w: int, img_h: int, boxes_xyxy: np.ndarray):
    if boxes_xyxy.size == 0:
        return np.zeros((side, side), dtype=bool)
    x_centers = (np.arange(side) + 0.5) * (img_w / side)
    y_centers = (np.arange(side) + 0.5) * (img_h / side)
    xv, yv = np.meshgrid(x_centers, y_centers)  # [side, side]
    mask = np.zeros((side, side), dtype=bool)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.tolist()
        inside = (xv >= x1) & (xv <= x2) & (yv >= y1) & (yv <= y2)
        mask |= inside
    return mask


def expand_boxes_xyxy(boxes_xyxy: np.ndarray, img_w: int, img_h: int, expand_ratio: float):
    if boxes_xyxy.size == 0 or expand_ratio <= 0:
        return boxes_xyxy
    out = boxes_xyxy.copy()
    for i in range(out.shape[0]):
        x1, y1, x2, y2 = out[i].tolist()
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        dx = w * expand_ratio
        dy = h * expand_ratio
        nx1 = max(0.0, x1 - dx)
        ny1 = max(0.0, y1 - dy)
        nx2 = min(float(img_w - 1), x2 + dx)
        ny2 = min(float(img_h - 1), y2 + dy)
        out[i] = np.array([nx1, ny1, nx2, ny2], dtype=out.dtype)
    return out


def adaptive_boost_to_max(grid: np.ndarray, beta: float, cap: float):
    if beta <= 0:
        return grid, 0.0
    m = float(grid.max())
    mean_v = float(grid.mean())
    # Sample-adaptive lambda: larger when max is much larger than mean.
    lam = beta * max(0.0, (m / (mean_v + 1e-12) - 1.0))
    lam = min(max(0.0, lam), cap)
    boosted = grid + lam * (m - grid)
    boosted = boosted / (boosted.sum() + 1e-12)
    return boosted, lam


def equalize_top_fraction(grid: np.ndarray, fraction: float, mix: float):
    """Pull top-fraction patches toward global max.

    fraction=0.2, mix=1.0 => top 20% patches are set to max exactly.
    """
    if fraction <= 0:
        return grid, 0
    flat = grid.reshape(-1)
    n = flat.shape[0]
    k = int(round(n * fraction))
    k = max(1, min(n, k))
    order = np.argsort(flat)[::-1]
    sel = order[:k]
    out = flat.copy()
    m = float(flat.max())
    mix = min(1.0, max(0.0, float(mix)))
    out[sel] = out[sel] + mix * (m - out[sel])
    out = out.reshape(grid.shape)
    out = out / (out.sum() + 1e-12)
    return out, k


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
    }


def make_colormap(gray_uint8):
    x = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 - np.abs(2 * x - 1) * 1.5, 0, 1)
    b = np.clip(1.5 * (1 - x), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def draw_overlay(image_path, patch_grid, boxes_xyxy, save_path, title):
    image = Image.open(image_path).convert("RGB")
    side = patch_grid.shape[0]
    patch_u8 = (patch_grid / (patch_grid.max() + 1e-12) * 255).astype(np.uint8)
    heat = Image.fromarray(make_colormap(patch_u8)).resize(image.size, Image.NEAREST)
    overlay = Image.blend(image, heat, alpha=0.40)
    draw = ImageDraw.Draw(overlay)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
    # patch grid
    step_x = overlay.width / side
    step_y = overlay.height / side
    for gx in range(1, side):
        x = int(round(gx * step_x))
        draw.line([(x, 0), (x, overlay.height)], fill=(255, 255, 255), width=1)
    for gy in range(1, side):
        y = int(round(gy * step_y))
        draw.line([(0, y), (overlay.width, y)], fill=(255, 255, 255), width=1)

    canvas = Image.new("RGB", (overlay.width, overlay.height + 48), (255, 255, 255))
    canvas.paste(overlay, (0, 48))
    header = ImageDraw.Draw(canvas)
    header.text((10, 12), title, fill=(0, 0, 0))
    canvas.save(save_path)


def load_records_and_paths(dataset_name: str):
    if dataset_name == "Controlled_Images_A":
        records = json.loads(Path("data/controlled_images_dataset.json").read_text(encoding="utf-8"))
        return [r["image_path"] for r in records]
    if dataset_name == "Controlled_Images_B":
        records = json.loads(Path("data/controlled_clevr_dataset.json").read_text(encoding="utf-8"))
        return [r["image_path"] for r in records]
    if dataset_name == "COCO_QA_one_obj":
        records = json.loads(Path("data/coco_qa_one_obj.json").read_text(encoding="utf-8"))
        return [f"data/val2017/{str(r[0]).zfill(12)}.jpg" for r in records]
    if dataset_name == "COCO_QA_two_obj":
        records = json.loads(Path("data/coco_qa_two_obj.json").read_text(encoding="utf-8"))
        return [f"data/val2017/{str(r[0]).zfill(12)}.jpg" for r in records]
    if dataset_name == "VG_QA_one_obj":
        records = json.loads(Path("data/vg_qa_one_obj.json").read_text(encoding="utf-8"))
        return [f"data/vg_images/{r[0]}.jpg" for r in records]
    if dataset_name == "VG_QA_two_obj":
        records = json.loads(Path("data/vg_qa_two_obj.json").read_text(encoding="utf-8"))
        return [f"data/vg_images/{r[0]}.jpg" for r in records]
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_records_and_paths(args.dataset_name)
    sampled_idx = np.load(args.sampled_idx).tolist()
    unsampled = sorted(set(range(len(image_paths))) - set(sampled_idx))
    eval_indices = unsampled[: args.test_sample_count]
    only_local = set()
    if args.only_local_idx.strip():
        for t in args.only_local_idx.split(","):
            t = t.strip()
            if t:
                only_local.add(int(t))

    model = YOLO(args.yolo_model)

    per_sample = []
    max_vals = []
    margin_vals = []

    for local_idx, global_idx in enumerate(eval_indices):
        if only_local and local_idx not in only_local:
            continue
        image_path = image_paths[global_idx]

        attn_max_file, s1, e1 = pick_attn_file(Path(args.attn_dir_max) / str(local_idx), args.layer)
        attn_margin_file, s2, e2 = pick_attn_file(Path(args.attn_dir_margin) / str(local_idx), args.layer)
        if attn_max_file is None or attn_margin_file is None:
            continue

        grid_max = load_patch_prob(attn_max_file, s1, e1)
        grid_margin = load_patch_prob(attn_margin_file, s2, e2)
        if grid_max.shape != grid_margin.shape:
            continue
        side = grid_max.shape[0]

        results = model.predict(source=image_path, conf=args.yolo_conf, verbose=False)
        det = results[0]
        if det.boxes is None:
            boxes = np.zeros((0, 4), dtype=float)
            cls_ids = []
            confs = []
            img_h, img_w = det.orig_shape
        else:
            boxes = det.boxes.xyxy.detach().cpu().numpy() if len(det.boxes) else np.zeros((0, 4), dtype=float)
            cls_ids = det.boxes.cls.detach().cpu().numpy().astype(int).tolist() if len(det.boxes) else []
            confs = det.boxes.conf.detach().cpu().numpy().tolist() if len(det.boxes) else []
            img_h, img_w = det.orig_shape

        # Keep top-K boxes by confidence if requested.
        if args.max_boxes > 0 and len(confs) > args.max_boxes:
            order = np.argsort(np.asarray(confs))[::-1][: args.max_boxes]
            boxes = boxes[order]
            cls_ids = [cls_ids[i] for i in order.tolist()]
            confs = [confs[i] for i in order.tolist()]

        boxes = expand_boxes_xyxy(boxes, img_w, img_h, args.box_expand_ratio)

        adaptive_lambda = 0.0
        top_equalized_count = 0
        if args.new_adaptive_beta > 0:
            grid_margin, adaptive_lambda = adaptive_boost_to_max(
                grid_margin, args.new_adaptive_beta, args.new_adaptive_cap
            )
        if args.new_top_fraction > 0:
            grid_margin, top_equalized_count = equalize_top_fraction(
                grid_margin, args.new_top_fraction, args.new_top_mix
            )

        patch_mask = patch_mask_from_boxes(side, img_w, img_h, boxes)
        overlap_max = float(grid_max[patch_mask].sum()) if patch_mask.any() else 0.0
        overlap_margin = float(grid_margin[patch_mask].sum()) if patch_mask.any() else 0.0
        diff = overlap_margin - overlap_max

        max_vals.append(overlap_max)
        margin_vals.append(overlap_margin)

        names = model.names if hasattr(model, "names") else {}
        label_names = [str(names.get(c, c)) for c in cls_ids]

        per_sample.append(
            {
                "local_sample_idx": local_idx,
                "global_dataset_idx": global_idx,
                "image_path": image_path,
                "num_boxes": int(len(boxes)),
                "boxes_xyxy": boxes.tolist(),
                "box_classes": label_names,
                "box_confs": confs,
                "patch_grid_side": side,
                "box_patch_coverage_ratio": float(patch_mask.mean()),
                "box_expand_ratio": args.box_expand_ratio,
                "overlap_max_prob": overlap_max,
                "overlap_margin": overlap_margin,
                "margin_minus_max": diff,
                "new_adaptive_lambda": adaptive_lambda,
                "new_top_fraction": args.new_top_fraction,
                "new_top_mix": args.new_top_mix,
                "new_top_equalized_count": int(top_equalized_count),
            }
        )

        if args.save_visualizations:
            viz_dir = out_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            draw_overlay(
                image_path=image_path,
                patch_grid=grid_max,
                boxes_xyxy=boxes,
                save_path=viz_dir / f"sample{local_idx:02d}_baseline.png",
                title=f"{args.dataset_name} sample={local_idx} baseline overlap={overlap_max:.4f}",
            )
            draw_overlay(
                image_path=image_path,
                patch_grid=grid_margin,
                boxes_xyxy=boxes,
                save_path=viz_dir / f"sample{local_idx:02d}_new.png",
                title=(
                    f"{args.dataset_name} sample={local_idx} new overlap={overlap_margin:.4f} "
                    f"lam={adaptive_lambda:.3f} topk={top_equalized_count}"
                ),
            )

    max_arr = np.asarray(max_vals, dtype=float)
    margin_arr = np.asarray(margin_vals, dtype=float)
    diff_arr = margin_arr - max_arr

    if diff_arr.size > 0 and np.any(np.abs(diff_arr) > 1e-12):
        try:
            stat, p_value = wilcoxon(diff_arr, alternative="greater")
            wilcoxon_res = {"statistic": float(stat), "p_value": float(p_value), "alternative": "margin > max_prob"}
        except ValueError:
            wilcoxon_res = {"statistic": None, "p_value": None, "alternative": "margin > max_prob"}
    else:
        wilcoxon_res = {"statistic": None, "p_value": None, "alternative": "margin > max_prob"}

    summary = {
        "num_samples": int(diff_arr.size),
        "dataset_name": args.dataset_name,
        "yolo_model": args.yolo_model,
        "yolo_conf": args.yolo_conf,
        "max_boxes": args.max_boxes,
        "box_expand_ratio": args.box_expand_ratio,
        "new_adaptive_beta": args.new_adaptive_beta,
        "new_adaptive_cap": args.new_adaptive_cap,
        "new_top_fraction": args.new_top_fraction,
        "new_top_mix": args.new_top_mix,
        "layer": args.layer,
        "overlap_max_prob": summarize(max_vals),
        "overlap_margin": summarize(margin_vals),
        "margin_minus_max": summarize(diff_arr.tolist()),
        "margin_better_count": int((diff_arr > 0).sum()),
        "max_better_count": int((diff_arr < 0).sum()),
        "tie_count": int((np.abs(diff_arr) <= 1e-12).sum()),
        "wilcoxon_greater": wilcoxon_res,
    }

    (out_dir / "per_sample.json").write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
