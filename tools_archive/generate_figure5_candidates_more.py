import argparse
import json
import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Figure-5 candidate images from candidate_pairs_more.json."
    )
    parser.add_argument("--root", type=str, default="/root/AdaptVis")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/root/autodl-tmp/AdaptVis_storage/fig5_search_samples",
        help="Directory containing pure_attn/ours_attn and candidate_pairs_more.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/AdaptVis/report/figures/figure5_candidates_more",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--box-scale", type=float, default=1.2)
    parser.add_argument(
        "--draw-yolo-box",
        action="store_true",
        help="Draw YOLO rectangle on each panel. Default is off.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def pick_file(sample_dir: Path, layer: int = 17):
    for x in sorted(sample_dir.glob(f"diff_{layer}_start*_end*.npy")):
        m = re.search(r"start(-?\d+)_end(-?\d+)", x.name)
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        if s >= 0 and e >= s:
            return x, s, e
    return None, None, None


def expand_box(box, width, height, scale=1.2):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max(1e-6, x2 - x1) * scale
    h = max(1e-6, y2 - y1) * scale
    return np.array(
        [max(0, cx - w / 2), max(0, cy - h / 2), min(width - 1, cx + w / 2), min(height - 1, cy + h / 2)],
        dtype=float,
    )


def read_disc(attn_root: Path, local_idx: int, head: int):
    f, s, e = pick_file(attn_root / str(local_idx))
    if f is None:
        return None
    arr = np.load(f)[0][head]
    if arr.min() < 0 or abs(arr.sum() - 1) > 1e-2:
        ex = np.exp(arr - arr.max())
        arr = ex / ex.sum()
    p = arr[s : e + 1]
    if p.size == 577:
        p = p[:-1]
    side = int(round(np.sqrt(p.size)))
    if side * side != p.size:
        return None
    p = p.reshape(side, side)
    p = (p - p.min()) / (p.max() - p.min() + 1e-12)
    bins = np.linspace(0, 1, 9)
    d = np.digitize(p, bins, right=True)
    d = (d - d.min()) / (d.max() - d.min() + 1e-12)
    return d


def main():
    args = parse_args()

    root = Path(args.root)
    work = Path(args.work_dir)
    pure_attn = work / "pure_attn"
    ours_attn = work / "ours_attn"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((work / "candidate_pairs_more.json").read_text(encoding="utf-8"))
    pairs = meta["pairs"]
    records = json.loads((root / "data/controlled_images_dataset.json").read_text(encoding="utf-8"))
    image_paths = [r["image_path"] for r in records]
    yolo = YOLO("yolov8n.pt")
    cmap = plt.get_cmap("turbo", 8)

    for i, pair in enumerate(pairs, 1):
        fig, axs = plt.subplots(2, 2, figsize=(9.2, 8.2))
        for row_i, key in enumerate(["up", "down"]):
            info = pair[key]
            local_idx, global_idx, head = info["local"], info["global"], info["head"]
            image_path = root / image_paths[global_idx]
            img = Image.open(image_path).convert("RGB")
            width, height = img.size

            det = yolo.predict(source=str(image_path), conf=args.yolo_conf, verbose=False)[0]
            boxes = det.boxes.xyxy.detach().cpu().numpy()
            conf = det.boxes.conf.detach().cpu().numpy()
            box = expand_box(boxes[np.argsort(conf)[::-1][0]], width, height, args.box_scale)
            x1, y1, x2, y2 = box

            for col_i, (name, attn_root) in enumerate([("Pure", pure_attn), ("Ours", ours_attn)]):
                disc = read_disc(attn_root, local_idx, head)
                if disc is None:
                    continue
                ax = axs[row_i, col_i]
                ax.imshow(img)
                ax.imshow(disc, cmap=cmap, alpha=0.45, interpolation="nearest", extent=[0, width, height, 0])
                if args.draw_yolo_box:
                    ax.add_patch(
                        patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2)
                    )
                tag = "Largest Increase" if key == "up" else "Largest Decrease"
                ax.set_title(f"{tag} h{head} s{global_idx} | {name}", fontsize=9)
                ax.axis("off")

        fig.tight_layout()
        out = out_dir / f"figure5_candidate_more_{i}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {len(pairs)} candidates to {out_dir}")


if __name__ == "__main__":
    main()
