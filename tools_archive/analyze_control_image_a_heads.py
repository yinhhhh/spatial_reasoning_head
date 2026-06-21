import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Head-level YOLO overlap and AUROC analysis on Controlled_Images_A "
            "using baseline attention dumps."
        )
    )
    parser.add_argument("--attn-dir", type=str, required=True, help="Attention dump dir, e.g. output/Controlled_Images_A_weight1.00")
    parser.add_argument("--result-jsonl", type=str, required=True, help="Run result jsonl file, e.g. outputs/control_image_a_baseline100/res.json")
    parser.add_argument("--sample-count", type=int, default=100, help="Number of unsampled test samples to analyze")
    parser.add_argument("--layer", type=int, default=17, help="Layer index for diff_*.npy")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--bbox-scale", type=float, default=1.2, help="Scale factor for YOLO box size (1.2 means 20%% larger area by w/h)")
    parser.add_argument("--top-k", type=int, default=8, help="Top heads to report by AUROC")
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def load_last_json(path: Path):
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    if not rows:
        raise RuntimeError(f"No JSON row found in {path}")
    return rows[-1]


def pick_attn_file(sample_dir: Path, layer: int):
    for p in sorted(sample_dir.glob(f"diff_{layer}_start*_end*.npy")):
        m = re.search(r"start(-?\d+)_end(-?\d+)", p.name)
        if not m:
            continue
        start_idx = int(m.group(1))
        end_idx = int(m.group(2))
        if start_idx >= 0 and end_idx >= start_idx:
            return p, start_idx, end_idx
    return None, None, None


def load_patch_probs_per_head(attn_file: Path, start_idx: int, end_idx: int):
    arr = np.load(attn_file)[0]  # [heads, seq_len]
    arr = np.where(arr < -1e20, -1e20, arr)
    row_sums = arr.sum(axis=-1)
    is_prob_like = (
        float(arr.min()) >= -1e-8
        and np.all(np.isfinite(arr))
        and np.all(np.abs(row_sums - 1.0) < 1e-2)
    )
    probs = arr if is_prob_like else softmax(arr)  # [heads, seq_len]
    patch = probs[:, start_idx : end_idx + 1]  # [heads, n_patch]
    patch = patch / (patch.sum(axis=1, keepdims=True) + 1e-12)
    side = int(round(math.sqrt(patch.shape[1])))
    if side * side != patch.shape[1]:
        raise ValueError(f"Patch number is not square: {patch.shape[1]}")
    return patch, side


def scale_box_xyxy(box: np.ndarray, img_w: int, img_h: int, scale: float):
    if scale <= 0:
        raise ValueError("scale must be positive")
    x1, y1, x2, y2 = box.tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1e-6, x2 - x1) * scale
    h = max(1e-6, y2 - y1) * scale
    nx1 = max(0.0, cx - 0.5 * w)
    ny1 = max(0.0, cy - 0.5 * h)
    nx2 = min(float(img_w - 1), cx + 0.5 * w)
    ny2 = min(float(img_h - 1), cy + 0.5 * h)
    return np.array([nx1, ny1, nx2, ny2], dtype=float)


def patch_mask_for_single_box(side: int, img_w: int, img_h: int, box_xyxy: np.ndarray):
    x_centers = (np.arange(side) + 0.5) * (img_w / side)
    y_centers = (np.arange(side) + 0.5) * (img_h / side)
    xv, yv = np.meshgrid(x_centers, y_centers)
    x1, y1, x2, y2 = box_xyxy.tolist()
    return (xv >= x1) & (xv <= x2) & (yv >= y1) & (yv <= y2)


def binary_auc(scores: np.ndarray, labels: np.ndarray):
    labels = labels.astype(int)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def save_bar(values: np.ndarray, title: str, ylabel: str, out_path: Path):
    x = np.arange(values.shape[0])
    plt.figure(figsize=(12, 4.8))
    plt.bar(x, values, width=0.8)
    plt.xticks(x, [str(i) for i in x], fontsize=8)
    plt.title(title)
    plt.xlabel("Head index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path("data/controlled_images_dataset.json").read_text(encoding="utf-8"))
    image_paths = [r["image_path"] for r in dataset]

    sampled_idx = np.load("output/sampled_idx_Controlled_Images_A.npy").tolist()
    unsampled = sorted(set(range(len(image_paths))) - set(sampled_idx))
    eval_global_indices = unsampled[: args.sample_count]

    last_res = load_last_json(Path(args.result_jsonl))
    correct_local = set(last_res.get("correct_id", []))

    yolo = YOLO(args.yolo_model)
    attn_dir = Path(args.attn_dir)

    per_head_overlaps = [[] for _ in range(32)]
    per_head_scores = [[] for _ in range(32)]
    labels = []
    used_local_indices = []
    skipped = {"missing_attn": 0, "not_single_box": 0, "empty_mask": 0}

    for local_idx, global_idx in enumerate(eval_global_indices):
        sample_dir = attn_dir / str(local_idx)
        attn_file, start_idx, end_idx = pick_attn_file(sample_dir, args.layer)
        if attn_file is None:
            skipped["missing_attn"] += 1
            continue

        head_patch_probs, side = load_patch_probs_per_head(attn_file, start_idx, end_idx)  # [32, n_patch]
        image_path = image_paths[global_idx]
        det = yolo.predict(source=image_path, conf=args.yolo_conf, verbose=False)[0]
        if det.boxes is None or len(det.boxes) != 1:
            skipped["not_single_box"] += 1
            continue

        box = det.boxes.xyxy.detach().cpu().numpy()[0]
        img_h, img_w = det.orig_shape
        scaled_box = scale_box_xyxy(box, img_w, img_h, args.bbox_scale)
        mask = patch_mask_for_single_box(side, img_w, img_h, scaled_box).reshape(-1)
        if not mask.any():
            skipped["empty_mask"] += 1
            continue

        y = 1 if local_idx in correct_local else 0
        labels.append(y)
        used_local_indices.append(local_idx)
        for h in range(32):
            ov = float(head_patch_probs[h, mask].sum())
            per_head_overlaps[h].append(ov)
            per_head_scores[h].append(ov)

    label_arr = np.asarray(labels, dtype=int)
    mean_overlap = np.asarray(
        [float(np.mean(v)) if len(v) > 0 else float("nan") for v in per_head_overlaps],
        dtype=float,
    )
    auc_per_head = []
    for h in range(32):
        if len(per_head_scores[h]) != len(label_arr) or len(label_arr) == 0:
            auc_per_head.append(float("nan"))
            continue
        auc = binary_auc(np.asarray(per_head_scores[h], dtype=float), label_arr)
        auc_per_head.append(float("nan") if auc is None else auc)
    auc_per_head = np.asarray(auc_per_head, dtype=float)

    top_candidates = [
        {"head": int(h), "auroc": float(a)}
        for h, a in enumerate(auc_per_head.tolist())
        if not np.isnan(a)
    ]
    top_candidates.sort(key=lambda x: x["auroc"], reverse=True)
    top_heads = top_candidates[: args.top_k]

    save_bar(
        values=np.nan_to_num(mean_overlap, nan=0.0),
        title=f"Controlled_Images_A (n={len(label_arr)} valid): Mean overlap per head",
        ylabel="Mean overlap in expanded YOLO ROI",
        out_path=out_dir / "head_overlap_bar.png",
    )
    save_bar(
        values=np.nan_to_num(auc_per_head, nan=0.0),
        title=f"Controlled_Images_A (n={len(label_arr)} valid): AUROC per head",
        ylabel="AUROC (overlap -> correctness)",
        out_path=out_dir / "head_auroc_bar.png",
    )

    summary = {
        "dataset": "Controlled_Images_A",
        "requested_sample_count": args.sample_count,
        "valid_sample_count": int(len(label_arr)),
        "label_positive_count": int((label_arr == 1).sum()),
        "label_negative_count": int((label_arr == 0).sum()),
        "layer": args.layer,
        "bbox_scale": args.bbox_scale,
        "yolo_conf": args.yolo_conf,
        "attn_dir": str(attn_dir),
        "result_jsonl": args.result_jsonl,
        "used_local_indices": used_local_indices,
        "skipped": skipped,
        "mean_overlap_per_head": mean_overlap.tolist(),
        "auroc_per_head": auc_per_head.tolist(),
        "top_heads_by_auroc": top_heads,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
