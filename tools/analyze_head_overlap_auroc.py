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
    parser = argparse.ArgumentParser(description="Analyze per-head YOLO overlap and AUROC.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=[
            "Controlled_Images_A",
            "COCO_QA_one_obj",
            "COCO_QA_two_obj",
            "VG_QA_one_obj",
            "VG_QA_two_obj",
        ],
    )
    parser.add_argument("--attn-dir", type=str, required=True)
    parser.add_argument("--result-jsonl", type=str, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--max-boxes", type=int, default=2)
    parser.add_argument("--box-scale", type=float, default=1.2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--title-prefix", type=str, default="")
    return parser.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def load_last_json(path: Path):
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s:
            rows.append(json.loads(s))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[-1]


def load_image_paths(dataset_name: str):
    if dataset_name == "Controlled_Images_A":
        records = json.loads(Path("data/controlled_images_dataset.json").read_text(encoding="utf-8"))
        return [r["image_path"] for r in records]
    if dataset_name == "COCO_QA_one_obj":
        records = json.loads(Path("data/coco_qa_one_obj.json").read_text(encoding="utf-8"))
        return [f"data/val2017/{str(r[0]).zfill(12)}.jpg" for r in records]
    if dataset_name == "COCO_QA_two_obj":
        records = json.loads(Path("data/coco_qa_two_obj.json").read_text(encoding="utf-8"))
        return [f"data/val2017/{str(r[0]).zfill(12)}.jpg" for r in records]
    if dataset_name == "VG_QA_one_obj":
        records = json.loads(Path("data/vg_qa_one_obj.json").read_text(encoding="utf-8"))
        return [f"data/vg_images/{str(r[0])}.jpg" for r in records]
    if dataset_name == "VG_QA_two_obj":
        records = json.loads(Path("data/vg_qa_two_obj.json").read_text(encoding="utf-8"))
        return [f"data/vg_images/{str(r[0])}.jpg" for r in records]
    raise ValueError(dataset_name)


def pick_attn_file(sample_dir: Path, layer: int):
    for p in sorted(sample_dir.glob(f"diff_{layer}_start*_end*.npy")):
        m = re.search(r"start(-?\d+)_end(-?\d+)", p.name)
        if not m:
            continue
        s = int(m.group(1))
        e = int(m.group(2))
        if s >= 0 and e >= s:
            return p, s, e
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
    probs = arr if is_prob_like else softmax(arr)
    patch = probs[:, start_idx : end_idx + 1]
    patch = patch / (patch.sum(axis=1, keepdims=True) + 1e-12)
    n_patch = patch.shape[1]
    side = int(round(math.sqrt(n_patch)))
    if side * side != n_patch:
        # Some dumps include one extra token in image span; fallback by dropping tail token.
        side2 = int(round(math.sqrt(max(0, n_patch - 1))))
        if side2 * side2 == (n_patch - 1):
            patch = patch[:, :-1]
            n_patch = patch.shape[1]
            side = side2
        else:
            raise ValueError(f"Patch number not square: {n_patch}")
    return patch, side


def expand_box(box, img_w: int, img_h: int, box_scale: float):
    x1, y1, x2, y2 = box.tolist()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1e-6, x2 - x1) * box_scale
    h = max(1e-6, y2 - y1) * box_scale
    nx1 = max(0.0, cx - 0.5 * w)
    ny1 = max(0.0, cy - 0.5 * h)
    nx2 = min(float(img_w - 1), cx + 0.5 * w)
    ny2 = min(float(img_h - 1), cy + 0.5 * h)
    return np.array([nx1, ny1, nx2, ny2], dtype=float)


def patch_mask_from_boxes(side: int, img_w: int, img_h: int, boxes_xyxy: np.ndarray):
    if boxes_xyxy.size == 0:
        return np.zeros((side, side), dtype=bool)
    x_centers = (np.arange(side) + 0.5) * (img_w / side)
    y_centers = (np.arange(side) + 0.5) * (img_h / side)
    xv, yv = np.meshgrid(x_centers, y_centers)
    mask = np.zeros((side, side), dtype=bool)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.tolist()
        mask |= (xv >= x1) & (xv <= x2) & (yv >= y1) & (yv <= y2)
    return mask


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

    image_paths = load_image_paths(args.dataset_name)
    sampled_idx = np.load(f"output/sampled_idx_{args.dataset_name}.npy").tolist()
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
    skipped = {"missing_attn": 0, "no_box": 0, "empty_mask": 0}

    for local_idx, global_idx in enumerate(eval_global_indices):
        sample_dir = attn_dir / str(local_idx)
        attn_file, start_idx, end_idx = pick_attn_file(sample_dir, args.layer)
        if attn_file is None:
            skipped["missing_attn"] += 1
            continue

        head_patch_probs, side = load_patch_probs_per_head(attn_file, start_idx, end_idx)

        image_path = image_paths[global_idx]
        det = yolo.predict(source=image_path, conf=args.yolo_conf, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            skipped["no_box"] += 1
            continue
        boxes = det.boxes.xyxy.detach().cpu().numpy()
        confs = det.boxes.conf.detach().cpu().numpy()
        if args.max_boxes > 0 and len(boxes) > args.max_boxes:
            order = np.argsort(confs)[::-1][: args.max_boxes]
            boxes = boxes[order]
        img_h, img_w = det.orig_shape
        boxes = np.asarray([expand_box(b, img_w, img_h, args.box_scale) for b in boxes], dtype=float)
        mask = patch_mask_from_boxes(side, img_w, img_h, boxes).reshape(-1)
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

    labels_arr = np.asarray(labels, dtype=int)
    mean_overlap = np.asarray(
        [float(np.mean(v)) if len(v) > 0 else float("nan") for v in per_head_overlaps],
        dtype=float,
    )
    auc_per_head = []
    for h in range(32):
        if len(per_head_scores[h]) != len(labels_arr):
            auc_per_head.append(float("nan"))
            continue
        auc = binary_auc(np.asarray(per_head_scores[h], dtype=float), labels_arr)
        auc_per_head.append(float("nan") if auc is None else auc)
    auc_per_head = np.asarray(auc_per_head, dtype=float)

    top = [
        {"head": int(h), "auroc": float(a)}
        for h, a in enumerate(auc_per_head.tolist())
        if not np.isnan(a)
    ]
    top.sort(key=lambda x: x["auroc"], reverse=True)
    top = top[: args.top_k]

    prefix = args.title_prefix.strip() or args.dataset_name
    save_bar(
        np.nan_to_num(mean_overlap, nan=0.0),
        f"{prefix}: mean overlap per head (n={len(labels_arr)})",
        "Mean overlap",
        out_dir / "head_overlap_bar.png",
    )
    save_bar(
        np.nan_to_num(auc_per_head, nan=0.0),
        f"{prefix}: AUROC per head (n={len(labels_arr)})",
        "AUROC (overlap -> correctness)",
        out_dir / "head_auroc_bar.png",
    )

    summary = {
        "dataset": args.dataset_name,
        "requested_sample_count": args.sample_count,
        "valid_sample_count": int(len(labels_arr)),
        "label_positive_count": int((labels_arr == 1).sum()),
        "label_negative_count": int((labels_arr == 0).sum()),
        "layer": args.layer,
        "box_scale": args.box_scale,
        "max_boxes": args.max_boxes,
        "yolo_conf": args.yolo_conf,
        "attn_dir": str(attn_dir),
        "result_jsonl": args.result_jsonl,
        "used_local_indices": used_local_indices,
        "skipped": skipped,
        "mean_overlap_per_head": mean_overlap.tolist(),
        "auroc_per_head": auc_per_head.tolist(),
        "top_heads_by_auroc": top,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
